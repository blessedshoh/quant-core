"""Exercise 1: write Black-Scholes yourself.

Fill in each function below, then run the engine's own tests against your code:

    pytest exercises -m "not edge"      the core maths
    pytest exercises                    everything, including the edge cases

Rules of the game
-----------------
- Work from the formulas in walkthroughs/01_black_scholes.ipynb, not from
  src/quantcore/black_scholes.py. Look there only after your version passes.
- Inputs can be numbers or numpy arrays. If you use numpy functions (np.log,
  np.exp, np.sqrt) and `ndtr` for N(x), arrays work without extra effort.
- Units: time is in years, vol and rates are decimals. theta is per year,
  vega is per 1.00 of volatility, rho is per 1.00 of rate.

Symbols
-------
S spot, K strike, T time, sigma vol, r rate, q dividend yield
N(x) standard normal CDF  -> ndtr(x)
phi(x) standard normal density -> exp(-x^2 / 2) / sqrt(2 pi)
"""

import numpy as np  # noqa: F401
from scipy.special import ndtr  # noqa: F401


def d1_d2(spot, strike, time, vol, rate=0.0, div_yield=0.0):
    """Return (d1, d2).

    d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

    Edge case for later: when sigma sqrt(T) is zero, d1 should be +infinity if
    the forward S e^((r-q)T) is above K, -infinity if below, and 0 if equal.
    """
    raise NotImplementedError("your turn")


def price(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call"):
    """Option value per share.

    call = S e^(-qT) N(d1)  - K e^(-rT) N(d2)
    put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

    Edge cases for later: reject spot <= 0, strike <= 0, time < 0, vol < 0 and
    any kind other than "call" or "put" by raising ValueError.
    """
    raise NotImplementedError("your turn")


def delta(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call"):
    """call: e^(-qT) N(d1)        put: -e^(-qT) N(-d1)"""
    raise NotImplementedError("your turn")


def gamma(spot, strike, time, vol, rate=0.0, div_yield=0.0):
    """e^(-qT) phi(d1) / (S sigma sqrt(T)).  Return 0 when sigma sqrt(T) is zero."""
    raise NotImplementedError("your turn")


def vega(spot, strike, time, vol, rate=0.0, div_yield=0.0):
    """S e^(-qT) phi(d1) sqrt(T)"""
    raise NotImplementedError("your turn")


def theta(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call"):
    """Per year.

    call: -S e^(-qT) phi(d1) sigma / (2 sqrt(T)) - r K e^(-rT) N(d2)  + q S e^(-qT) N(d1)
    put:  -S e^(-qT) phi(d1) sigma / (2 sqrt(T)) + r K e^(-rT) N(-d2) - q S e^(-qT) N(-d1)
    """
    raise NotImplementedError("your turn")


def rho(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call"):
    """call: K T e^(-rT) N(d2)        put: -K T e^(-rT) N(-d2)"""
    raise NotImplementedError("your turn")
