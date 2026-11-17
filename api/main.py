"""HTTP API in front of the quantcore engine.

Stateless on purpose: it prices what it is sent and remembers nothing. All the
maths lives in `quantcore`. This file only checks the request, calls the
engine, and shapes the answer for the front end.

Run locally:
    set QUANTCORE_DEV_ORIGINS=1
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import math
import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, model_validator

import quantcore as qc
from quantcore import probability, scenarios
from quantcore.grid import price_axis

MAX_LEGS = 8

# Browsers may call this API only from the site itself. Local development
# origins are allowed only when QUANTCORE_DEV_ORIGINS=1.
SITE_ORIGINS = r"https://([a-z0-9-]+\.)?blessedshoh\.dev"
DEV_ORIGINS = r"http://(localhost|127\.0\.0\.1)(:\d+)?"


def allowed_origins() -> str:
    if os.environ.get("QUANTCORE_DEV_ORIGINS") == "1":
        return f"({SITE_ORIGINS})|({DEV_ORIGINS})"
    return SITE_ORIGINS


# --- what the front end sends -------------------------------------------------


class LegIn(BaseModel):
    kind: Literal["call", "put", "stock"]
    quantity: float = Field(ge=-10_000, le=10_000, description="Contracts or shares. Negative is short.")
    strike: float | None = Field(default=None, gt=0, le=1_000_000)
    days: float | None = Field(default=None, ge=0, le=1825, description="Calendar days to expiry.")
    vol: float | None = Field(default=None, ge=0.01, le=5.0, description="Implied volatility, 0.64 for 64%.")
    multiplier: float = Field(default=100.0, gt=0, le=10_000)
    entry_price: float | None = Field(default=None, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def check_leg(self):
        if self.quantity == 0:
            raise ValueError("quantity cannot be zero")
        if self.kind != "stock" and (self.strike is None or self.days is None or self.vol is None):
            raise ValueError("an option leg needs strike, days and vol")
        return self


class GridRequest(BaseModel):
    spot: float = Field(gt=0, le=1_000_000)
    rate: float = Field(default=0.0, ge=-0.05, le=0.5)
    div_yield: float = Field(default=0.0, ge=0.0, le=0.5)
    legs: list[LegIn] = Field(min_length=1, max_length=MAX_LEGS)
    range_pct: float = Field(default=0.25, ge=0.02, le=0.9, description="Half-width of the price axis around spot.")


# --- what it gets back --------------------------------------------------------


class Axes(BaseModel):
    prices: list[float]
    days: list[float]
    vol_scales: list[float]


class Cube(BaseModel):
    shape: tuple[int, int, int] = Field(description="(vol_scales, days, prices)")
    values: list[float] = Field(description="Profit, flattened in row-major order.")


class GreeksOut(BaseModel):
    """Position Greeks at the current spot. Each is indexed [vol_scale][day]."""

    delta: list[list[float]]
    gamma: list[list[float]]
    theta_per_day: list[list[float]]
    vega_per_point: list[list[float]]


class LegOut(BaseModel):
    entry_price: float


class Summary(BaseModel):
    entry_cost: float = Field(description="Positive is a net debit, negative a net credit.")
    first_expiry_days: float | None
    breakevens: list[float] = Field(description="At the first expiry.")
    max_profit: float | None = Field(description="At the first expiry. null means unlimited.")
    max_loss: float | None = Field(description="At the first expiry, as a negative number. null means unlimited.")
    legs: list[LegOut]


class GridResponse(BaseModel):
    axes: Axes
    profit: Cube
    greeks: GreeksOut
    summary: Summary


class RiskRequest(GridRequest):
    drift: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Your expected annual return, for a real-world view. Leave empty for risk-neutral."
    )
    horizon_days: float = Field(default=0.0, ge=0, le=1825, description="When the scenario shocks happen, in days from today.")


class Distribution(BaseModel):
    """Risk-neutral density of the price on the first expiry, on the same price axis as the grid."""

    days: float
    vol: float
    prices: list[float]
    density: list[float]
    below_range: float = Field(description="Probability of finishing below the first price.")
    above_range: float = Field(description="Probability of finishing above the last price.")


class ScenarioTable(BaseModel):
    price_shocks: list[float] = Field(description="Fractions of today's price: -0.1 is a 10% drop.")
    vol_shifts: list[float] = Field(description="Volatility changes: 0.1 is +10 points.")
    horizon_days: float
    profit: list[list[float]] = Field(description="Indexed [vol_shift][price_shock].")


class RiskResponse(BaseModel):
    representative_vol: float | None
    chance_of_profit: float | None = Field(description="Risk-neutral, at the first expiry.")
    chance_of_profit_with_drift: float | None
    expected_profit: float | None = Field(description="Risk-neutral average profit at the first expiry.")
    distribution: Distribution | None
    scenarios: ScenarioTable


def _position(request: GridRequest) -> qc.Position:
    return qc.Position(
        spot=request.spot,
        rate=request.rate,
        div_yield=request.div_yield,
        legs=[qc.Leg(**leg.model_dump()) for leg in request.legs],
    )


# --- the app ------------------------------------------------------------------

app = FastAPI(
    title="quantcore API",
    version=qc.__version__,
    description="Option positions priced with Black-Scholes. Stateless.",
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
    max_age=86400,
)


def _finite_or_none(x: float) -> float | None:
    return round(float(x), 2) if math.isfinite(x) else None


def _rows(table: np.ndarray, digits: int) -> list[list[float]]:
    return np.round(table, digits).tolist()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": qc.__version__}


@app.post("/options/grid", response_model=GridResponse)
def options_grid(request: GridRequest) -> GridResponse:
    """Profit at every price, date and volatility, plus the headline numbers."""
    try:
        position = _position(request)
        grid = qc.profit_grid(position, range_pct=request.range_pct)
        greeks = qc.greeks_grid(position, grid.days, grid.vol_scales)
        best, worst = qc.profit_range(position)
        summary = Summary(
            entry_cost=round(qc.entry_cost(position), 2),
            first_expiry_days=qc.first_expiry(position),
            breakevens=[round(x, 4) for x in qc.breakevens(position)],
            max_profit=_finite_or_none(best),
            max_loss=_finite_or_none(worst),
            legs=[LegOut(entry_price=round(qc.entry_price(leg, position), 4)) for leg in position.legs],
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return GridResponse(
        axes=Axes(
            prices=np.round(grid.prices, 4).tolist(),
            days=np.round(grid.days, 6).tolist(),
            vol_scales=np.round(grid.vol_scales, 6).tolist(),
        ),
        profit=Cube(shape=grid.profit.shape, values=np.round(grid.profit, 2).ravel().tolist()),
        greeks=GreeksOut(
            delta=_rows(greeks["delta"], 4),
            gamma=_rows(greeks["gamma"], 6),
            theta_per_day=_rows(greeks["theta_per_day"], 4),
            vega_per_point=_rows(greeks["vega_per_point"], 4),
        ),
        summary=summary,
    )


@app.post("/options/risk", response_model=RiskResponse)
def options_risk(request: RiskRequest) -> RiskResponse:
    """Chance of profit, the price distribution and the scenario grid for a position."""
    try:
        position = _position(request)
        vol = probability.representative_vol(position)
        days = qc.first_expiry(position)
        distribution = None
        if vol is not None and days:
            prices = price_axis(position, request.range_pct)
            edges = probability.cdf([prices[0], prices[-1]], position.spot, days, vol, position.rate, position.div_yield)
            distribution = Distribution(
                days=days,
                vol=round(vol, 6),
                prices=np.round(prices, 4).tolist(),
                density=np.round(
                    probability.density(prices, position.spot, days, vol, position.rate, position.div_yield), 8
                ).tolist(),
                below_range=round(float(edges[0]), 6),
                above_range=round(float(1 - edges[1]), 6),
            )
        table = scenarios.scenario_grid(position, days_forward=request.horizon_days)
        chance = probability.chance_of_profit(position)
        with_drift = probability.chance_of_profit(position, drift=request.drift) if request.drift is not None else None
        expected = probability.expected_profit(position)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    def rounded(x, digits):
        return None if x is None else round(x, digits)

    return RiskResponse(
        representative_vol=rounded(vol, 6),
        chance_of_profit=rounded(chance, 6),
        chance_of_profit_with_drift=rounded(with_drift, 6),
        expected_profit=rounded(expected, 2),
        distribution=distribution,
        scenarios=ScenarioTable(
            price_shocks=table.price_shocks.tolist(),
            vol_shifts=table.vol_shifts.tolist(),
            horizon_days=table.days_forward,
            profit=np.round(table.profit, 2).tolist(),
        ),
    )


class ExplainRequest(GridRequest):
    price_move_pct: float = Field(default=0.05, gt=-0.9, le=2.0, description="Price move as a fraction of today's price.")
    days_passed: float = Field(default=0.0, ge=0, le=1825)
    vol_shift: float = Field(default=0.0, ge=-1.0, le=1.0, description="Volatility change: 0.05 is +5 points.")


class ExplainResponse(BaseModel):
    """A change in value split into what each Greek predicts, plus what is left over."""

    price_move: float
    delta: float
    gamma: float
    theta: float
    vega: float
    unexplained: float
    actual: float


@app.post("/options/explain", response_model=ExplainResponse)
def options_explain(request: ExplainRequest) -> ExplainResponse:
    try:
        position = _position(request)
        move = position.spot * request.price_move_pct
        parts = scenarios.explain(position, move, request.days_passed, request.vol_shift)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ExplainResponse(
        price_move=round(move, 4),
        **{name: round(getattr(parts, name), 2) for name in ("delta", "gamma", "theta", "vega", "unexplained", "actual")},
    )
