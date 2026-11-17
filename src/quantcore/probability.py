"""Where the price might finish, and the chance a position makes money.

Black-Scholes assumes the log of the price at a future date is normally distributed:

    ln S_T ~ Normal( ln S + (mu - q - sigma^2 / 2) T ,  sigma^2 T )

With mu equal to the risk-free rate r, this is the RISK-NEUTRAL distribution:
the one option prices imply. With mu set to an expected return of your own, it
becomes a REAL-WORLD forecast. The two differ, and the difference matters: the
"chance of profit" a broker quotes is almost always the risk-neutral one, which
assumes the stock earns only the risk-free rate.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from . import black_scholes as bs
from .position import Position, breakevens, first_expiry, profit_at_expiry


def log_price_moments(spot, days, vol, rate=0.0, div_yield=0.0, drift=None) -> tuple[float, float]:
    """Mean and standard deviation of ln S_T, `days` calendar days from now.

    drift   expected annual return of the stock. None means the risk-free rate,
            which gives the risk-neutral distribution.
    """
    if spot <= 0 or days < 0 or vol < 0:
        raise ValueError("spot must be positive, and days and vol cannot be negative")
    years = days / bs.DAYS_PER_YEAR
    growth = rate if drift is None else drift
    mean = math.log(spot) + (growth - div_yield - 0.5 * vol * vol) * years
    return mean, vol * math.sqrt(years)


def cdf(prices, spot, days, vol, rate=0.0, div_yield=0.0, drift=None):
    """Probability that the price finishes at or below each of `prices`."""
    mean, sd = log_price_moments(spot, days, vol, rate, div_yield, drift)
    prices = np.asarray(prices, dtype=float)
    with np.errstate(divide="ignore"):
        logs = np.log(prices)
    if sd == 0:
        return np.where(logs >= mean, 1.0, 0.0)
    return ndtr((logs - mean) / sd)


def density(prices, spot, days, vol, rate=0.0, div_yield=0.0, drift=None):
    """Probability density of the finishing price (the lognormal curve)."""
    mean, sd = log_price_moments(spot, days, vol, rate, div_yield, drift)
    if sd == 0:
        raise ValueError("with no time or no volatility the price is certain and has no density")
    x = np.asarray(prices, dtype=float)
    return np.exp(-0.5 * ((np.log(x) - mean) / sd) ** 2) / (x * sd * math.sqrt(2.0 * math.pi))


def representative_vol(position: Position) -> float | None:
    """One volatility for the whole position, to describe where the price may go.

    Each option leg's volatility is weighted by how much the position's value
    depends on it (size times vega), so the leg that matters most counts most.
    None for a position with no options.
    """
    vols, weights = [], []
    for leg in position.legs:
        if not leg.is_option:
            continue
        years = leg.days / bs.DAYS_PER_YEAR
        vega = float(bs.vega(position.spot, leg.strike, years, leg.vol, position.rate, position.div_yield))
        vols.append(leg.vol)
        weights.append(abs(leg.size) * vega)
    if not vols:
        return None
    if sum(weights) == 0:
        return float(np.mean(vols))
    return float(np.average(vols, weights=weights))


def chance_of_profit(position: Position, vol=None, drift=None) -> float | None:
    """Probability that the position shows a profit on the day the first leg expires.

    The price axis is cut at the break-even points. Each piece where the expiry
    payoff is positive contributes the probability of the price finishing inside
    it. None for a position with no expiry (stock only).
    """
    days = first_expiry(position)
    if days is None:
        return None
    if days == 0:
        return 1.0 if float(profit_at_expiry(position, position.spot)) > 0 else 0.0
    vol = representative_vol(position) if vol is None else vol
    edges = [0.0] + breakevens(position) + [math.inf]
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = high / 2 if low == 0 else (low * 2 if math.isinf(high) else (low + high) / 2)
        if float(profit_at_expiry(position, inside)) > 0:
            mass = cdf([low, high], position.spot, days, vol, position.rate, position.div_yield, drift)
            total += float(mass[1] - mass[0])
    return total


def expected_profit(position: Position, vol=None, drift=None, points: int = 8001) -> float | None:
    """Average profit at the first expiry under the chosen distribution.

    Integrated numerically on a fine grid of log prices, eight standard
    deviations either side of the centre. Under the risk-neutral distribution,
    with every leg at the same volatility, this is just the entry cost grown at
    the risk-free rate minus the entry cost: options are fairly priced, so on
    average they earn only the financing rate. That makes it a good self-check.
    """
    days = first_expiry(position)
    if not days:
        return None
    vol = representative_vol(position) if vol is None else vol
    mean, sd = log_price_moments(position.spot, days, vol, position.rate, position.div_yield, drift)
    logs = np.linspace(mean - 8 * sd, mean + 8 * sd, points)
    weights = np.exp(-0.5 * ((logs - mean) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))
    payoff = np.asarray(profit_at_expiry(position, np.exp(logs)), dtype=float)
    return float(np.trapezoid(payoff * weights, logs))
