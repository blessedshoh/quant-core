"""Multi-leg option positions: cost, value, profit, break-evens and limits.

A position is a current spot price plus a list of legs. A leg is a call, a put
or the stock itself, held long (positive quantity) or short (negative).

Sign convention for money
-------------------------
entry_cost   positive means you paid (net debit), negative means you received (net credit)
value        what the position is worth at some price and date
profit       value - entry_cost

Dates are measured as `days_forward`: calendar days from today. The model is
only meaningful up to the first expiry in the position, because after that an
expired leg has settled and no longer moves with the price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import binomial
from . import black_scholes as bs

OPTION_KINDS = ("call", "put")
MIN_VOL = 1e-4  # a shocked volatility never goes below this
LEG_KINDS = OPTION_KINDS + ("stock",)
EXERCISE_STYLES = ("european", "american")


@dataclass(frozen=True)
class Leg:
    """One line of a position.

    kind         "call", "put" or "stock"
    quantity     contracts for options, shares for stock. Positive is long, negative is short.
    strike       strike price (options only)
    days         calendar days until expiry, counted from today (options only)
    vol          implied volatility as a decimal (options only)
    multiplier   shares per contract. Ignored for stock.
    entry_price  price per share actually paid or received. Left empty, the model price today is used.
    """

    kind: str
    quantity: float
    strike: float | None = None
    days: float | None = None
    vol: float | None = None
    multiplier: float = 100.0
    entry_price: float | None = None

    def __post_init__(self):
        if self.kind not in LEG_KINDS:
            raise ValueError(f"kind must be one of {LEG_KINDS}, got {self.kind!r}")
        if self.quantity == 0:
            raise ValueError("quantity cannot be zero")
        if self.is_option:
            if self.strike is None or self.strike <= 0:
                raise ValueError("an option leg needs a positive strike")
            if self.days is None or self.days < 0:
                raise ValueError("an option leg needs days to expiry of zero or more")
            if self.vol is None or self.vol <= 0:
                raise ValueError("an option leg needs a positive volatility")
            if self.multiplier <= 0:
                raise ValueError("multiplier must be positive")
        if self.entry_price is not None and self.entry_price < 0:
            raise ValueError("entry_price cannot be negative")

    @property
    def is_option(self) -> bool:
        return self.kind in OPTION_KINDS

    @property
    def size(self) -> float:
        """Signed number of shares this leg controls."""
        return self.quantity * (self.multiplier if self.is_option else 1.0)


@dataclass(frozen=True)
class Position:
    """A spot price, the legs held against it, and the market assumptions.

    exercise     "european": every option is valued with Black-Scholes.
                 "american": every option is valued with a Leisen-Reimer binomial
                 tree that allows early exercise.
    tree_steps   steps in that tree. The default gives prices accurate to well
                 under a cent per share; the profit grid uses fewer for speed.
    """

    spot: float
    legs: tuple[Leg, ...]
    rate: float = 0.0
    div_yield: float = 0.0
    exercise: str = "european"
    tree_steps: int = 201

    def __post_init__(self):
        object.__setattr__(self, "legs", tuple(self.legs))
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if not self.legs:
            raise ValueError("a position needs at least one leg")
        if self.exercise not in EXERCISE_STYLES:
            raise ValueError(f"exercise must be one of {EXERCISE_STYLES}")
        if self.tree_steps < 1:
            raise ValueError("tree_steps must be at least 1")

    @property
    def american(self) -> bool:
        return self.exercise == "american"

    @property
    def strikes(self) -> list[float]:
        return sorted({leg.strike for leg in self.legs if leg.is_option})


def first_expiry(position: Position) -> float | None:
    """Days until the earliest option expiry, or None for a stock-only position."""
    days = [leg.days for leg in position.legs if leg.is_option]
    return min(days) if days else None


def leg_price(leg: Leg, position: Position, prices, days_forward=0.0, vol_scale=1.0, vol_shift=0.0):
    """Model price PER SHARE of one leg at the given underlying prices.

    days_forward   how many days from today the valuation date is
    vol_scale      multiplier applied to the leg's volatility (1.0 leaves it unchanged)
    vol_shift      amount added to the volatility after scaling: 0.05 is +5 volatility
                   points, the way risk desks quote volatility shocks
    """
    prices, days_forward, vol_scale, vol_shift = np.broadcast_arrays(
        np.asarray(prices, dtype=float),
        np.asarray(days_forward, dtype=float),
        np.asarray(vol_scale, dtype=float),
        np.asarray(vol_shift, dtype=float),
    )
    if not leg.is_option:
        return prices.copy()
    remaining_years = np.maximum(leg.days - days_forward, 0.0) / bs.DAYS_PER_YEAR
    vol = np.maximum(leg.vol * vol_scale + vol_shift, MIN_VOL)
    if position.american:
        return binomial.price_many(
            prices, leg.strike, remaining_years, vol, position.rate, position.div_yield, leg.kind, True, position.tree_steps
        )
    return bs.price(prices, leg.strike, remaining_years, vol, position.rate, position.div_yield, leg.kind)


def entry_price(leg: Leg, position: Position) -> float:
    """Price per share the leg was opened at: the stated one, else the model price today."""
    if leg.entry_price is not None:
        return float(leg.entry_price)
    return float(leg_price(leg, position, position.spot))


def entry_cost(position: Position) -> float:
    """Money paid to open the position. Negative means a net credit was received."""
    return sum(leg.size * entry_price(leg, position) for leg in position.legs)


def value(position: Position, prices, days_forward=0.0, vol_scale=1.0, vol_shift=0.0):
    """What the whole position is worth at the given prices, date and volatility."""
    return sum(leg.size * leg_price(leg, position, prices, days_forward, vol_scale, vol_shift) for leg in position.legs)


def profit(position: Position, prices, days_forward=0.0, vol_scale=1.0, vol_shift=0.0):
    """Profit or loss against the entry cost."""
    return value(position, prices, days_forward, vol_scale, vol_shift) - entry_cost(position)


def profit_at_expiry(position: Position, prices):
    """Profit or loss on the day the first leg expires."""
    return profit(position, prices, days_forward=first_expiry(position) or 0.0)


def _scan_prices(position: Position, low_factor: float, high_factor: float, points: int) -> np.ndarray:
    """A wide, geometrically spaced set of prices that always includes spot and every strike."""
    anchor = max([position.spot] + position.strikes)
    scan = np.geomspace(position.spot * low_factor, anchor * high_factor, points)
    return np.unique(np.concatenate([scan, [position.spot], position.strikes]))


def breakevens(position: Position, days_forward=None, vol_scale=1.0) -> list[float]:
    """Underlying prices at which profit is zero. By default, at the first expiry.

    The profit curve is scanned for sign changes and each one is then refined
    with Brent's root finder.
    """
    if days_forward is None:
        days_forward = first_expiry(position) or 0.0
    xs = _scan_prices(position, 1e-4, 10.0, 4001)
    ys = np.asarray(profit(position, xs, days_forward, vol_scale), dtype=float)

    def curve(x: float) -> float:
        return float(profit(position, x, days_forward, vol_scale))

    roots: list[float] = []
    previous = None  # index of the last point where profit was not exactly zero
    for i, y in enumerate(ys):
        if y == 0.0:
            continue
        if previous is not None and ys[previous] * y < 0:
            roots.append(brentq(curve, xs[previous], xs[i], xtol=1e-10, rtol=1e-12))
        previous = i
    return roots


def profit_range(position: Position) -> tuple[float, float]:
    """(largest profit, largest loss) at the first expiry.

    math.inf means the profit is unlimited and -math.inf means the loss is
    unlimited. That is decided by the slope of the profit curve far above the
    highest strike: only calls and stock still move there.
    """
    expiry = first_expiry(position) or 0.0
    xs = _scan_prices(position, 1e-6, 50.0, 4001)
    ys = np.asarray(profit(position, xs, expiry), dtype=float)
    best, worst = float(ys.max()), float(ys.min())

    far = float(xs[-1])
    slope = float(profit(position, 2.0 * far, expiry) - profit(position, far, expiry)) / far
    tolerance = 1e-9 * sum(abs(leg.size) for leg in position.legs)
    if slope > tolerance:
        best = math.inf
    elif slope < -tolerance:
        worst = -math.inf
    return best, worst


def net_greeks(position: Position, prices=None, days_forward=0.0, vol_scale=1.0) -> dict[str, np.ndarray]:
    """Greeks of the whole position, in money terms.

    delta            profit per +1.00 in the underlying (share-equivalents)
    gamma            change in delta per +1.00 in the underlying
    theta_per_day    profit from one calendar day passing
    vega_per_point   profit per +1 volatility point (0.01) on every leg
    """
    prices = position.spot if prices is None else prices
    prices, days_forward, vol_scale = np.broadcast_arrays(
        np.asarray(prices, dtype=float),
        np.asarray(days_forward, dtype=float),
        np.asarray(vol_scale, dtype=float),
    )
    total = {name: np.zeros(prices.shape) for name in ("delta", "gamma", "theta_per_day", "vega_per_point")}
    for leg in position.legs:
        if not leg.is_option:
            total["delta"] += leg.size
            continue
        if position.american:
            # A tree has no closed-form Greeks, so bump and reprice: the same check
            # the tests use to verify the Black-Scholes formulas.
            for name, value in _bumped_greeks(leg, position, prices, days_forward, vol_scale).items():
                total[name] += leg.size * value
            continue
        remaining_years = np.maximum(leg.days - days_forward, 0.0) / bs.DAYS_PER_YEAR
        market = (prices, leg.strike, remaining_years, leg.vol * vol_scale, position.rate, position.div_yield)
        total["delta"] += leg.size * bs.delta(*market, leg.kind)
        total["gamma"] += leg.size * bs.gamma(*market)
        total["theta_per_day"] += leg.size * bs.theta(*market, leg.kind) / bs.DAYS_PER_YEAR
        total["vega_per_point"] += leg.size * bs.vega(*market) / 100.0
    return total


def _bumped_greeks(leg: Leg, position: Position, prices, days_forward, vol_scale) -> dict[str, np.ndarray]:
    """Greeks of one leg (per share) by repricing it with each input nudged."""
    def at(s=prices, day=days_forward, shift=0.0):
        return np.asarray(leg_price(leg, position, s, day, vol_scale, shift), dtype=float)

    h = 0.01 * prices  # 1% of the price: wide enough to step over the tree's small kinks
    up, mid, down = at(s=prices + h), at(), at(s=prices - h)
    remaining = np.maximum(leg.days - days_forward, 0.0)
    step = np.minimum(0.25, remaining)  # a quarter of a day, never past expiry
    later = at(day=days_forward + step)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(step > 0, (later - mid) / step, 0.0)
    return {
        "delta": (up - down) / (2 * h),
        "gamma": (up - 2 * mid + down) / (h * h),
        "theta_per_day": theta,
        "vega_per_point": at(shift=0.005) - at(shift=-0.005),
    }
