"""A model option chain with a volatility smile.

Real option prices imply a different volatility at every strike. The pattern is
called the smile (or skew, when it slopes). This module draws one from three
numbers, using log-moneyness k = ln(K / F), where F is the forward price:

    sigma(k) = atm_vol + skew * k + curvature * k^2

atm_vol     volatility at the forward
skew        slope: negative makes low strikes dearer, the usual equity pattern
curvature   bend: positive lifts both wings into a smile

The chain is priced with Black-Scholes at each strike's own volatility. Bend the
smile too far and the prices stop being consistent with any probability
distribution, which is exactly what the arbitrage checker looks for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs

MIN_VOL = 0.01


def strike_step(spot: float) -> float:
    """Strike spacing that looks like a listed chain at this price level."""
    if spot < 25:
        return 0.5
    if spot < 100:
        return 1.0
    if spot < 500:
        return 5.0
    return 10.0


def default_strikes(spot: float, width: float = 0.25, step: float | None = None) -> np.ndarray:
    """Strikes on the listed grid from spot * (1 - width) to spot * (1 + width)."""
    step = step or strike_step(spot)
    low = math.ceil(spot * (1 - width) / step) * step
    high = math.floor(spot * (1 + width) / step) * step
    return np.round(np.arange(low, high + step / 2, step), 6)


def smile_vol(strikes, forward: float, atm_vol: float, skew: float = 0.0, curvature: float = 0.0):
    """Volatility at each strike, never below 1%."""
    k = np.log(np.asarray(strikes, dtype=float) / forward)
    return np.maximum(atm_vol + skew * k + curvature * k * k, MIN_VOL)


@dataclass(frozen=True)
class Chain:
    spot: float
    days: float
    rate: float
    div_yield: float
    forward: float
    strikes: np.ndarray
    vols: np.ndarray
    calls: np.ndarray
    puts: np.ndarray
    call_deltas: np.ndarray
    put_deltas: np.ndarray


def model_chain(
    spot: float,
    days: float,
    atm_vol: float,
    skew: float = 0.0,
    curvature: float = 0.0,
    rate: float = 0.0,
    div_yield: float = 0.0,
    strikes=None,
) -> Chain:
    """Black-Scholes prices for every strike, each at its own smile volatility."""
    if spot <= 0 or days <= 0 or atm_vol <= 0:
        raise ValueError("spot, days and the at-the-money volatility must be positive")
    years = days / bs.DAYS_PER_YEAR
    strikes = default_strikes(spot) if strikes is None else np.asarray(strikes, dtype=float)
    forward = spot * math.exp((rate - div_yield) * years)
    vols = smile_vol(strikes, forward, atm_vol, skew, curvature)
    market = (spot, strikes, years, vols, rate, div_yield)
    return Chain(
        spot=spot,
        days=days,
        rate=rate,
        div_yield=div_yield,
        forward=forward,
        strikes=strikes,
        vols=vols,
        calls=bs.price(*market, kind="call"),
        puts=bs.price(*market, kind="put"),
        call_deltas=bs.delta(*market, kind="call"),
        put_deltas=bs.delta(*market, kind="put"),
    )
