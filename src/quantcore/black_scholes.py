"""Black-Scholes-Merton prices and Greeks for European options.

Conventions (the same everywhere in this package)
-------------------------------------------------
spot        current price of the underlying, S
strike      strike price, K
time        time to expiry in YEARS, T  (days / 365)
vol         annualised volatility as a decimal, sigma  (0.64 means 64%)
rate        continuously compounded risk-free rate, r
div_yield   continuously compounded dividend yield, q
kind        "call" or "put"

Units of the Greeks
-------------------
delta   change in option value per +1.00 in spot
gamma   change in delta per +1.00 in spot
vega    change in value per +1.00 in vol (+100 vol points); divide by 100 for one point
theta   change in value per YEAR of calendar time passing; divide by 365 for one day
rho     change in value per +1.00 in the rate (+100%); divide by 100 for one percent

Every function takes plain numbers or numpy arrays and broadcasts them against
each other, so one call can price a single option or a whole grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr  # the standard normal CDF, N(x)

DAYS_PER_YEAR = 365.0


def _pdf(x):
    """Standard normal density, phi(x)."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _sign(kind: str) -> float:
    """+1 for a call, -1 for a put. One formula then covers both."""
    if kind == "call":
        return 1.0
    if kind == "put":
        return -1.0
    raise ValueError(f'kind must be "call" or "put", got {kind!r}')


def _inputs(spot, strike, time, vol, rate, div_yield):
    """Turn every input into a float array and reject values that make no sense."""
    spot, strike, time, vol, rate, div_yield = (
        np.asarray(v, dtype=float) for v in (spot, strike, time, vol, rate, div_yield)
    )
    if np.any(spot <= 0) or np.any(strike <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(time < 0):
        raise ValueError("time to expiry cannot be negative")
    if np.any(vol < 0):
        raise ValueError("volatility cannot be negative")
    return spot, strike, time, vol, rate, div_yield


def d1_d2(spot, strike, time, vol, rate=0.0, div_yield=0.0):
    """The two numbers every Black-Scholes formula is built from.

        d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T))
        d2 = d1 - sigma sqrt(T)

    N(d2) is the risk-neutral probability that a call finishes in the money.

    When no randomness is left (expiry has arrived, or volatility is zero) the
    option is certain to finish in or out of the money, so d1 and d2 are
    +infinity or -infinity. That makes N(d) exactly 1 or 0 and the price
    formulas below collapse to intrinsic value without a special case.
    """
    spot, strike, time, vol, rate, div_yield = _inputs(spot, strike, time, vol, rate, div_yield)
    total_vol = vol * np.sqrt(time)  # sigma sqrt(T)
    log_forward_moneyness = np.log(spot / strike) + (rate - div_yield) * time  # ln(F/K)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = log_forward_moneyness / total_vol + 0.5 * total_vol
    certain = np.where(
        log_forward_moneyness > 0,
        np.inf,
        np.where(log_forward_moneyness < 0, -np.inf, 0.0),
    )
    d1 = np.where(total_vol > 0, d1, certain)
    d2 = d1 - total_vol
    return d1, d2


def price(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call"):
    """Black-Scholes-Merton value of a European option, per share.

        call = S e^(-qT) N(d1)  - K e^(-rT) N(d2)
        put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

    Both lines are the same expression with a sign flipped, which is how the
    code below is written.
    """
    sign = _sign(kind)
    d1, d2 = d1_d2(spot, strike, time, vol, rate, div_yield)
    spot, strike, time, vol, rate, div_yield = _inputs(spot, strike, time, vol, rate, div_yield)
    value = sign * (
        spot * np.exp(-div_yield * time) * ndtr(sign * d1)
        - strike * np.exp(-rate * time) * ndtr(sign * d2)
    )
    # Rounding can leave a far out-of-the-money option at -1e-17. A price is never negative.
    return np.maximum(value, 0.0)


def delta(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call"):
    """Sensitivity of the option value to the spot price.

        call delta =  e^(-qT) N(d1)
        put delta  = -e^(-qT) N(-d1)
    """
    sign = _sign(kind)
    d1, _ = d1_d2(spot, strike, time, vol, rate, div_yield)
    _, _, time, _, _, div_yield = _inputs(spot, strike, time, vol, rate, div_yield)
    return sign * np.exp(-div_yield * time) * ndtr(sign * d1)


def gamma(spot, strike, time, vol, rate=0.0, div_yield=0.0):
    """Sensitivity of delta to the spot price. The same for calls and puts.

        gamma = e^(-qT) phi(d1) / (S sigma sqrt(T))

    Exactly at expiry gamma is zero everywhere except at the strike, where it
    is infinite. This function returns 0 there.
    """
    d1, _ = d1_d2(spot, strike, time, vol, rate, div_yield)
    spot, _, time, vol, _, div_yield = _inputs(spot, strike, time, vol, rate, div_yield)
    total_vol = vol * np.sqrt(time)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.exp(-div_yield * time) * _pdf(d1) / (spot * total_vol)
    return np.where(total_vol > 0, value, 0.0)


def vega(spot, strike, time, vol, rate=0.0, div_yield=0.0):
    """Sensitivity of the option value to volatility. The same for calls and puts.

        vega = S e^(-qT) phi(d1) sqrt(T)

    This is per +1.00 of volatility. Divide by 100 for one volatility point.
    """
    d1, _ = d1_d2(spot, strike, time, vol, rate, div_yield)
    spot, _, time, _, _, div_yield = _inputs(spot, strike, time, vol, rate, div_yield)
    return spot * np.exp(-div_yield * time) * _pdf(d1) * np.sqrt(time)


def theta(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call"):
    """Change in option value as calendar time passes, per YEAR.

        call theta = -S e^(-qT) phi(d1) sigma / (2 sqrt(T)) - r K e^(-rT) N(d2)  + q S e^(-qT) N(d1)
        put theta  = -S e^(-qT) phi(d1) sigma / (2 sqrt(T)) + r K e^(-rT) N(-d2) - q S e^(-qT) N(-d1)

    The first term is time decay and is shared by calls and puts. The other two
    are the cost of carry. Divide by 365 for the value lost per day.
    """
    sign = _sign(kind)
    d1, d2 = d1_d2(spot, strike, time, vol, rate, div_yield)
    spot, strike, time, vol, rate, div_yield = _inputs(spot, strike, time, vol, rate, div_yield)
    discounted_spot = spot * np.exp(-div_yield * time)
    discounted_strike = strike * np.exp(-rate * time)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = -discounted_spot * _pdf(d1) * vol / (2.0 * np.sqrt(time))
    decay = np.where(vol * np.sqrt(time) > 0, decay, 0.0)
    carry = -sign * rate * discounted_strike * ndtr(sign * d2) + sign * div_yield * discounted_spot * ndtr(sign * d1)
    return decay + carry


def rho(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call"):
    """Sensitivity of the option value to the interest rate, per +1.00 in the rate.

        call rho =  K T e^(-rT) N(d2)
        put rho  = -K T e^(-rT) N(-d2)
    """
    sign = _sign(kind)
    _, d2 = d1_d2(spot, strike, time, vol, rate, div_yield)
    _, strike, time, _, rate, _ = _inputs(spot, strike, time, vol, rate, div_yield)
    return sign * strike * time * np.exp(-rate * time) * ndtr(sign * d2)


@dataclass(frozen=True)
class Greeks:
    """All five Greeks of one option, in the units listed at the top of this file."""

    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray

    @property
    def theta_per_day(self):
        return self.theta / DAYS_PER_YEAR

    @property
    def vega_per_point(self):
        return self.vega / 100.0


def greeks(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call") -> Greeks:
    """Convenience wrapper: every Greek for the same inputs."""
    args = (spot, strike, time, vol, rate, div_yield)
    return Greeks(
        delta=delta(*args, kind),
        gamma=gamma(*args),
        vega=vega(*args),
        theta=theta(*args, kind),
        rho=rho(*args, kind),
    )
