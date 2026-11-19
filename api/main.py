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
from quantcore import arbitrage, binomial, earnings, probability, replicate, scenarios, work
from quantcore import chain as option_chain
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


class EventIn(BaseModel):
    days: float = Field(ge=1, le=1825, description="Calendar days until the earnings announcement.")
    move: float = Field(ge=0.005, le=0.5, description="One-day standard deviation of the jump in the log price: 0.08 is about ±8%.")


class GridRequest(BaseModel):
    spot: float = Field(gt=0, le=1_000_000)
    rate: float = Field(default=0.0, ge=-0.05, le=0.5)
    div_yield: float = Field(default=0.0, ge=0.0, le=0.5)
    legs: list[LegIn] = Field(min_length=1, max_length=MAX_LEGS)
    range_pct: float = Field(default=0.25, ge=0.02, le=0.9, description="Half-width of the price axis around spot.")
    exercise: Literal["european", "american"] = Field(
        default="european", description="american values every option with a binomial tree that allows early exercise."
    )
    event: EventIn | None = Field(default=None, description="An earnings announcement. Leave empty for none.")


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


class EventLegOut(BaseModel):
    index: int
    base_vol: float
    vol_before: float
    vol_after: float
    weight: float = Field(description="|size| x vega today: how much the position depends on this leg.")


class EventSummary(BaseModel):
    days: float
    move: float
    legs: list[EventLegOut] = Field(description="Legs still alive at the announcement.")


class Summary(BaseModel):
    entry_cost: float = Field(description="Positive is a net debit, negative a net credit.")
    first_expiry_days: float | None
    breakevens: list[float] = Field(description="At the first expiry.")
    max_profit: float | None = Field(description="At the first expiry. null means unlimited.")
    max_loss: float | None = Field(description="At the first expiry, as a negative number. null means unlimited.")
    legs: list[LegOut]
    event: EventSummary | None = None


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
        exercise=request.exercise,
        event=None if request.event is None else qc.Event(**request.event.model_dump()),
    )


def _event_summary(position: qc.Position) -> EventSummary | None:
    if position.event is None:
        return None
    return EventSummary(
        days=position.event.days,
        move=position.event.move,
        legs=[
            EventLegOut(index=i, base_vol=round(b, 6), vol_before=round(before, 6), vol_after=round(after, 6), weight=round(w, 4))
            for i, b, before, after, w in earnings.event_vols(position)
        ],
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
            event=_event_summary(position),
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


# --- show your work -------------------------------------------------------------


class LegWorkOut(BaseModel):
    kind: str
    quantity: float
    size: float
    strike: float | None
    days: float | None
    years: float | None
    vol: float | None
    d1: float | None
    d2: float | None
    n_d1: float | None = Field(description="N(d1) for a call, N(-d1) for a put.")
    n_d2: float | None = Field(description="N(d2) for a call, N(-d2) for a put.")
    spot_discount: float | None = Field(description="e^(-qT)")
    strike_discount: float | None = Field(description="e^(-rT)")
    european_price: float | None
    american_price: float | None
    entry_price: float
    entry_price_stated: bool
    cost: float
    greeks: dict[str, float] = Field(description="Per share.")
    contribution: dict[str, float] = Field(description="Size times the per-share Greek.")
    base_vol: float | None = Field(default=None, description="Everyday volatility once the earnings jump is taken out.")
    vol_after_event: float | None = Field(default=None, description="Volatility left after the announcement.")


class IntervalOut(BaseModel):
    low: float
    high: float | None = Field(description="null means no upper limit.")
    profitable: bool
    probability: float


class ChanceWorkOut(BaseModel):
    days: float
    vol: float
    vol_weights: list[tuple[int, float, float]] = Field(description="(leg index, vol, weight = |size| x vega).")
    mean_log: float
    sd_log: float
    intervals: list[IntervalOut]
    chance: float


class WorkResponse(BaseModel):
    """The intermediate values behind the cost, Greeks and chance of profit, for today."""

    exercise: str
    legs: list[LegWorkOut]
    entry_cost: float
    greeks: dict[str, float]
    chance: ChanceWorkOut | None
    event: EventIn | None = None


def _r(x, digits):
    if x is None:
        return None
    if isinstance(x, float) and math.isinf(x):
        return None
    return round(float(x), digits)


@app.post("/options/work", response_model=WorkResponse)
def options_work(request: GridRequest) -> WorkResponse:
    """Every step behind the headline numbers, so each can be checked by hand."""
    try:
        work_out = work.show_work(_position(request))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    def leg_out(leg) -> LegWorkOut:
        return LegWorkOut(
            kind=leg.kind,
            quantity=leg.quantity,
            size=leg.size,
            strike=leg.strike,
            days=leg.days,
            years=_r(leg.years, 6),
            vol=leg.vol,
            d1=_r(leg.d1, 6),
            d2=_r(leg.d2, 6),
            n_d1=_r(leg.n_d1, 6),
            n_d2=_r(leg.n_d2, 6),
            spot_discount=_r(leg.spot_discount, 6),
            strike_discount=_r(leg.strike_discount, 6),
            european_price=_r(leg.european_price, 4),
            american_price=_r(leg.american_price, 4),
            entry_price=round(leg.entry_price, 4),
            entry_price_stated=leg.entry_price_stated,
            cost=round(leg.cost, 2),
            greeks={k: round(v, 6) for k, v in leg.greeks.items()},
            contribution={k: round(v, 4) for k, v in leg.contribution.items()},
            base_vol=_r(leg.base_vol, 6),
            vol_after_event=_r(leg.vol_after_event, 6),
        )

    chance = work_out.chance
    return WorkResponse(
        exercise=work_out.exercise,
        legs=[leg_out(leg) for leg in work_out.legs],
        entry_cost=round(work_out.entry_cost, 2),
        greeks={k: round(v, 4) for k, v in work_out.greeks.items()},
        chance=None
        if chance is None
        else ChanceWorkOut(
            days=chance.days,
            vol=round(chance.vol, 6),
            vol_weights=[(i, v, round(w, 4)) for i, v, w in chance.vol_weights],
            mean_log=round(chance.mean_log, 6),
            sd_log=round(chance.sd_log, 6),
            intervals=[
                IntervalOut(low=round(p.low, 4), high=_r(p.high, 4), profitable=p.profitable, probability=round(p.probability, 6))
                for p in chance.intervals
            ],
            chance=round(chance.chance, 6),
        ),
        event=None if work_out.event is None else EventIn(days=work_out.event[0], move=work_out.event[1]),
    )


# --- earnings: the move the market is pricing -------------------------------------


class ImpliedMoveRequest(BaseModel):
    near_days: float = Field(gt=0, le=1825, description="Days to the nearer expiry. Both expiries fall after the announcement.")
    near_vol: float = Field(ge=0.01, le=5.0)
    far_days: float = Field(gt=0, le=1825)
    far_vol: float = Field(ge=0.01, le=5.0)


class ImpliedMoveResponse(BaseModel):
    move: float = Field(description="Standard deviation of the one-day jump in the log price.")
    expected_abs_move: float = Field(description="Average size of the jump, up or down: move x sqrt(2/pi).")
    base_vol: float = Field(description="Everyday volatility shared by both expiries.")
    event_variance_share: float = Field(description="Part of the nearer expiry's variance that is the jump.")


@app.post("/options/implied-move", response_model=ImpliedMoveResponse)
def options_implied_move(request: ImpliedMoveRequest) -> ImpliedMoveResponse:
    """The earnings move and everyday volatility implied by two expiries after the announcement."""
    try:
        result = earnings.implied_move(request.near_days, request.near_vol, request.far_days, request.far_vol)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ImpliedMoveResponse(
        move=round(result.move, 6),
        expected_abs_move=round(result.expected_abs_move, 6),
        base_vol=round(result.base_vol, 6),
        event_variance_share=round(result.event_variance_share, 6),
    )


# --- binomial explorer ----------------------------------------------------------


class SingleOption(BaseModel):
    spot: float = Field(gt=0, le=1_000_000)
    strike: float = Field(gt=0, le=1_000_000)
    days: float = Field(gt=0, le=1825)
    vol: float = Field(ge=0.01, le=5.0)
    rate: float = Field(default=0.0, ge=-0.05, le=0.5)
    div_yield: float = Field(default=0.0, ge=0.0, le=0.5)
    kind: Literal["call", "put"] = "put"
    american: bool = True


class TreeRequest(SingleOption):
    steps: int = Field(default=5, ge=1, le=12)


class TreeResponse(BaseModel):
    prices: list[list[float]] = Field(description="Row i holds the i + 1 nodes after i steps, lowest price first.")
    values: list[list[float]]
    exercise: list[list[bool]]
    p: float
    u: float
    d: float
    dt_days: float
    european_value: float = Field(description="Black-Scholes value, for comparison with the tree's root.")
    american_value: float = Field(description="Accurate (201-step Leisen-Reimer) American value.")


@app.post("/options/tree", response_model=TreeResponse)
def options_tree(request: TreeRequest) -> TreeResponse:
    """Every node of a small Cox-Ross-Rubinstein tree, for drawing."""
    years = request.days / 365.0
    market = (request.spot, request.strike, years, request.vol, request.rate, request.div_yield)
    try:
        tree = binomial.build_tree(*market, request.kind, request.american, request.steps)
        accurate = binomial.price(*market, request.kind, True, 201)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return TreeResponse(
        prices=[[round(x, 4) for x in row] for row in tree.prices],
        values=[[round(x, 4) for x in row] for row in tree.values],
        exercise=tree.exercise,
        p=round(tree.p, 6),
        u=round(tree.u, 6),
        d=round(tree.d, 6),
        dt_days=round(tree.dt * 365.0, 6),
        european_value=round(float(qc.black_scholes.price(*market, request.kind)), 4),
        american_value=round(accurate, 4),
    )


class ConvergenceRequest(SingleOption):
    max_steps: int = Field(default=100, ge=5, le=200)


class ConvergenceResponse(BaseModel):
    steps: list[int]
    crr: list[float]
    leisen_reimer: list[float]
    black_scholes: float = Field(description="European Black-Scholes value.")
    reference: float = Field(description="Leisen-Reimer value with 1,001 steps: the target the trees approach.")


@app.post("/options/convergence", response_model=ConvergenceResponse)
def options_convergence(request: ConvergenceRequest) -> ConvergenceResponse:
    """Tree prices for 1 to max_steps steps, both methods, to show how they approach the answer."""
    years = request.days / 365.0
    market = (request.spot, request.strike, years, request.vol, request.rate, request.div_yield)
    try:
        series = binomial.convergence(*market, request.kind, request.american, range(1, request.max_steps + 1))
        reference = binomial.price(*market, request.kind, request.american, 1001)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ConvergenceResponse(
        steps=series["steps"],
        crr=[round(x, 6) for x in series["crr"]],
        leisen_reimer=[round(x, 6) for x in series["leisen_reimer"]],
        black_scholes=round(series["black_scholes"], 6),
        reference=round(reference, 6),
    )


# --- option chain and arbitrage --------------------------------------------------


class ChainRequest(BaseModel):
    spot: float = Field(gt=0, le=1_000_000)
    days: float = Field(gt=0, le=1825)
    rate: float = Field(default=0.0, ge=-0.05, le=0.5)
    div_yield: float = Field(default=0.0, ge=0.0, le=0.5)
    atm_vol: float = Field(ge=0.01, le=5.0)
    skew: float = Field(default=0.0, ge=-5.0, le=5.0)
    curvature: float = Field(default=0.0, ge=-100.0, le=100.0)
    width: float = Field(default=0.25, ge=0.05, le=0.6, description="Strikes from spot * (1 - width) to spot * (1 + width).")


class ChainRow(BaseModel):
    strike: float
    vol: float
    call: float
    put: float
    call_delta: float
    put_delta: float


class ChainResponse(BaseModel):
    forward: float
    rows: list[ChainRow]


@app.post("/options/chain", response_model=ChainResponse)
def options_chain(request: ChainRequest) -> ChainResponse:
    """Model prices for every listed strike, each at its own volatility on the smile."""
    try:
        strikes = option_chain.default_strikes(request.spot, request.width)
        chain = option_chain.model_chain(
            request.spot, request.days, request.atm_vol, request.skew, request.curvature, request.rate, request.div_yield, strikes
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    rows = [
        ChainRow(strike=float(k), vol=round(float(v), 6), call=round(float(c), 4), put=round(float(p), 4),
                 call_delta=round(float(cd), 4), put_delta=round(float(pd), 4))
        for k, v, c, p, cd, pd in zip(chain.strikes, chain.vols, chain.calls, chain.puts, chain.call_deltas, chain.put_deltas)
    ]
    return ChainResponse(forward=round(chain.forward, 4), rows=rows)


class QuoteIn(BaseModel):
    strike: float = Field(gt=0, le=1_000_000)
    call_bid: float | None = Field(default=None, ge=0, le=1_000_000)
    call_ask: float | None = Field(default=None, ge=0, le=1_000_000)
    put_bid: float | None = Field(default=None, ge=0, le=1_000_000)
    put_ask: float | None = Field(default=None, ge=0, le=1_000_000)


class ArbitrageRequest(BaseModel):
    spot: float = Field(gt=0, le=1_000_000)
    days: float = Field(gt=0, le=1825)
    rate: float = Field(default=0.0, ge=-0.05, le=0.5)
    div_yield: float = Field(default=0.0, ge=0.0, le=0.5)
    american: bool = False
    quotes: list[QuoteIn] = Field(min_length=1, max_length=80)


class TradeLegOut(BaseModel):
    kind: Literal["call", "put", "stock", "cash"]
    quantity: float = Field(description="Per share. Positive is bought; for cash, money lent today.")
    strike: float | None
    price: float


class ViolationOut(BaseModel):
    rule: str
    title: str
    detail: str
    profit: float = Field(description="Locked in per share, in today's money.")
    strikes: list[float]
    trade: list[TradeLegOut]


class SolverOut(BaseModel):
    arbitrage: bool
    credit: float
    trade: list[TradeLegOut]
    worst_payoff: float


class ArbitrageResponse(BaseModel):
    violations: list[ViolationOut]
    solver: SolverOut | None = Field(description="null for American options, where the expiry-only solver does not apply.")
    assumptions: list[str]


def _legs(trade) -> list[TradeLegOut]:
    return [
        TradeLegOut(kind=leg.kind, quantity=round(float(leg.quantity), 4), strike=leg.strike, price=round(float(leg.price), 4))
        for leg in trade
    ]


@app.post("/options/arbitrage", response_model=ArbitrageResponse)
def options_arbitrage(request: ArbitrageRequest) -> ArbitrageResponse:
    """Rule-by-rule checks and, for European options, the solver's search for a free lunch."""
    try:
        quotes = [arbitrage.Quote(**q.model_dump()) for q in request.quotes]
        report = arbitrage.check(request.spot, request.days, quotes, request.rate, request.div_yield, request.american)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    solver = None
    if report.solver is not None:
        solver = SolverOut(
            arbitrage=report.solver.arbitrage,
            credit=round(report.solver.credit, 4),
            trade=_legs(report.solver.trade),
            worst_payoff=round(report.solver.worst_payoff, 6),
        )
    return ArbitrageResponse(
        violations=[
            ViolationOut(rule=v.rule, title=v.title, detail=v.detail, profit=round(v.profit, 4), strikes=list(v.strikes), trade=_legs(v.trade))
            for v in report.violations
        ],
        solver=solver,
        assumptions=list(report.assumptions),
    )


# --- payoff replication ------------------------------------------------------------


class ReplicateRequest(BaseModel):
    spot: float = Field(gt=0, le=1_000_000)
    days: float = Field(gt=0, le=1825)
    rate: float = Field(default=0.0, ge=-0.05, le=0.5)
    div_yield: float = Field(default=0.0, ge=0.0, le=0.5)
    atm_vol: float = Field(ge=0.01, le=5.0)
    skew: float = Field(default=0.0, ge=-5.0, le=5.0)
    curvature: float = Field(default=0.0, ge=-100.0, le=100.0)
    target_prices: list[float] = Field(min_length=2, max_length=200, description="Where the drawn payoff is given.")
    target_payoffs: list[float] = Field(min_length=2, max_length=200, description="Payoff per share at expiry at each price.")
    mode: Literal["match", "cheapest"] = "match"
    max_legs: int | None = Field(default=8, ge=1, le=8, description="Most option and stock positions to use. Cash is free.")

    @model_validator(mode="after")
    def same_length(self):
        if len(self.target_prices) != len(self.target_payoffs):
            raise ValueError("target_prices and target_payoffs must be the same length")
        if any(abs(y) > 10 * self.spot for y in self.target_payoffs):
            raise ValueError("payoffs must stay within ten times the price")
        return self


class HoldingOut(BaseModel):
    kind: Literal["call", "put", "stock", "cash"]
    quantity: float = Field(description="Per share. Stock: shares at expiry. Cash: amount received at expiry.")
    strike: float | None
    price: float = Field(description="Cost today per unit.")


class ReplicateResponse(BaseModel):
    holdings: list[HoldingOut]
    cost: float = Field(description="Today, per share. Negative means the portfolio pays you.")
    max_error: float = Field(description="Largest gap from the drawing, per share.")
    prices: list[float]
    target: list[float]
    fitted: list[float]


@app.post("/options/replicate", response_model=ReplicateResponse)
def options_replicate(request: ReplicateRequest) -> ReplicateResponse:
    """The portfolio of listed options, stock and cash closest to (or never below) a drawn payoff."""
    try:
        result = replicate.replicate(
            request.target_prices, request.target_payoffs, request.spot, request.days, request.atm_vol, request.skew,
            request.curvature, request.rate, request.div_yield, request.mode, max_legs=request.max_legs,
        )
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ReplicateResponse(
        holdings=[
            HoldingOut(kind=h.kind, quantity=round(h.quantity, 4), strike=h.strike, price=round(h.price, 4))
            for h in result.holdings
            if abs(h.quantity) >= 5e-5
        ],
        cost=round(result.cost, 4),
        max_error=round(result.max_error, 4),
        prices=np.round(result.prices, 4).tolist(),
        target=np.round(result.target, 4).tolist(),
        fitted=np.round(result.fitted, 4).tolist(),
    )
