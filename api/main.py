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
        position = qc.Position(
            spot=request.spot,
            rate=request.rate,
            div_yield=request.div_yield,
            legs=[qc.Leg(**leg.model_dump()) for leg in request.legs],
        )
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
