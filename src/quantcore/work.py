"""Show your work: the headline numbers of a position, broken into the steps that make them.

Nothing here is new maths. Every number is computed by the same functions the
rest of the engine uses; this module only keeps the intermediate values that
those functions normally throw away, so a reader can check each step by hand:

    per leg      inputs -> d1, d2 -> N(d1), N(d2) -> price per share -> cost
    position     entry cost = sum of leg costs; Greeks = sum of size x leg Greeks
    probability  representative volatility -> lognormal moments -> chance of
                 finishing in each stretch between break-evens

All values are for today, at each leg's own volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from . import black_scholes as bs
from . import probability
from .position import Position, entry_cost, entry_price, first_expiry, leg_price, net_greeks

GREEK_NAMES = ("delta", "gamma", "theta_per_day", "vega_per_point")


@dataclass(frozen=True)
class LegWork:
    """One leg, every step from inputs to its share of the position."""

    kind: str
    quantity: float
    size: float  # signed shares controlled: quantity x multiplier for options
    strike: float | None
    days: float | None
    years: float | None  # T = days / 365
    vol: float | None
    d1: float | None
    d2: float | None
    n_d1: float | None  # N(d1) for a call, N(-d1) for a put: the terms the price formula uses
    n_d2: float | None  # N(d2) for a call, N(-d2) for a put
    spot_discount: float | None  # e^(-qT)
    strike_discount: float | None  # e^(-rT)
    european_price: float | None  # Black-Scholes, per share
    american_price: float | None  # binomial tree with early exercise, per share; None for european positions
    entry_price: float  # per share: the stated price, or the model price today
    entry_price_stated: bool
    cost: float  # size x entry price
    greeks: dict[str, float]  # per share
    contribution: dict[str, float]  # size x per-share Greek


@dataclass(frozen=True)
class ChanceWork:
    """How the chance of profit at the first expiry is put together."""

    days: float
    vol: float
    vol_weights: list[tuple[int, float, float]]  # (leg index, vol, weight)
    mean_log: float  # mean of ln S_T
    sd_log: float  # standard deviation of ln S_T
    intervals: list[probability.Interval]
    chance: float


@dataclass(frozen=True)
class Work:
    spot: float
    rate: float
    div_yield: float
    exercise: str
    legs: list[LegWork]
    entry_cost: float
    greeks: dict[str, float]
    chance: ChanceWork | None


def _leg_work(position: Position, index: int) -> LegWork:
    leg = position.legs[index]
    paid = entry_price(leg, position)
    cost = leg.size * paid

    if not leg.is_option:
        greeks = {"delta": 1.0, "gamma": 0.0, "theta_per_day": 0.0, "vega_per_point": 0.0}
        return LegWork(
            kind=leg.kind, quantity=leg.quantity, size=leg.size, strike=None, days=None, years=None, vol=None,
            d1=None, d2=None, n_d1=None, n_d2=None, spot_discount=None, strike_discount=None,
            european_price=None, american_price=None, entry_price=paid, entry_price_stated=leg.entry_price is not None,
            cost=cost, greeks=greeks, contribution={name: leg.size * g for name, g in greeks.items()},
        )

    years = leg.days / bs.DAYS_PER_YEAR
    market = (position.spot, leg.strike, years, leg.vol, position.rate, position.div_yield)
    d1, d2 = (float(x) for x in bs.d1_d2(*market))
    sign = 1.0 if leg.kind == "call" else -1.0
    european = float(bs.price(*market, leg.kind))
    american = float(leg_price(leg, position, position.spot)) if position.american else None

    # Per-share Greeks: the leg's share of the position's Greeks, divided back by its size.
    alone = Position(position.spot, (leg,), position.rate, position.div_yield, position.exercise, position.tree_steps)
    per_share = {name: float(value) / leg.size for name, value in net_greeks(alone).items()}

    return LegWork(
        kind=leg.kind,
        quantity=leg.quantity,
        size=leg.size,
        strike=float(leg.strike),
        days=float(leg.days),
        years=years,
        vol=float(leg.vol),
        d1=d1,
        d2=d2,
        n_d1=float(ndtr(sign * d1)),
        n_d2=float(ndtr(sign * d2)),
        spot_discount=float(np.exp(-position.div_yield * years)),
        strike_discount=float(np.exp(-position.rate * years)),
        european_price=european,
        american_price=american,
        entry_price=paid,
        entry_price_stated=leg.entry_price is not None,
        cost=cost,
        greeks=per_share,
        contribution={name: leg.size * value for name, value in per_share.items()},
    )


def _chance_work(position: Position) -> ChanceWork | None:
    days = first_expiry(position)
    intervals = probability.outcome_intervals(position)
    if not days or intervals is None:
        return None
    vol = probability.representative_vol(position)
    mean, sd = probability.log_price_moments(position.spot, days, vol, position.rate, position.div_yield)
    return ChanceWork(
        days=float(days),
        vol=vol,
        vol_weights=probability.vol_weights(position),
        mean_log=mean,
        sd_log=sd,
        intervals=intervals,
        chance=sum(piece.probability for piece in intervals if piece.profitable),
    )


def show_work(position: Position) -> Work:
    """Every intermediate value behind the position's cost, Greeks and chance of profit today."""
    return Work(
        spot=position.spot,
        rate=position.rate,
        div_yield=position.div_yield,
        exercise=position.exercise,
        legs=[_leg_work(position, i) for i in range(len(position.legs))],
        entry_cost=entry_cost(position),
        greeks={name: float(value) for name, value in net_greeks(position).items()},
        chance=_chance_work(position),
    )


__all__ = ["ChanceWork", "LegWork", "Work", "show_work"]
