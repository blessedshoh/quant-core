"""Exercise 2: positions, probability and risk.

Fill in each function below, then check your work against the engine:

    pytest exercises/test_risk.py -m "not edge"     the core maths
    pytest exercises/test_risk.py                   everything, including the edge cases

Rules of the game
-----------------
- Work from walkthroughs/02_positions_and_risk.ipynb, not from the engine's
  source. Look at src/quantcore/ only after your version passes.
- Each function is small on purpose. Together they rebuild the numbers in the
  tool's risk view and its "show your work" panel.
- Units: days are calendar days (years = days / 365); vol, rates and drift are
  decimals; vol_shift is a decimal too (0.05 is +5 volatility points).

Symbols
-------
S spot, T years, sigma vol, r rate, q dividend yield, mu drift
N(x) standard normal CDF -> ndtr(x)
"""

import math  # noqa: F401

import numpy as np  # noqa: F401
from scipy.special import ndtr  # noqa: F401

DAYS_PER_YEAR = 365.0


def entry_cost(sizes, prices):
    """Money paid to open a position: sum of size x entry price per share.

    sizes are signed shares controlled (contracts x 100 for options).
    Positive result is a net debit, negative a net credit.
    """
    raise NotImplementedError("your turn")


def net_greeks(sizes, leg_greeks):
    """Position Greeks from per-share leg Greeks.

    leg_greeks is a list of dicts, one per leg, each with the keys
    "delta", "gamma", "theta_per_day" and "vega_per_point".
    Return one dict with the same keys: sum of size x per-share Greek.
    """
    raise NotImplementedError("your turn")


def log_price_moments(spot, days, vol, rate=0.0, div_yield=0.0, drift=None):
    """Return (mean, sd) of ln S_T, `days` calendar days from now.

    mean = ln S + (mu - q - sigma^2 / 2) T        sd = sigma sqrt(T)

    mu is `drift`, or the risk-free rate when drift is None (risk-neutral).
    """
    raise NotImplementedError("your turn")


def prob_between(low, high, spot, days, vol, rate=0.0, div_yield=0.0, drift=None):
    """Probability that S_T finishes between low and high.

    P = N((ln high - mean) / sd) - N((ln low - mean) / sd)

    Edge cases for later: low can be 0 (ln 0 = -infinity, N = 0) and high can be
    math.inf (N = 1).
    """
    raise NotImplementedError("your turn")


def chance_of_profit(breakevens, profitable, spot, days, vol, rate=0.0, div_yield=0.0, drift=None):
    """Chance the position is in profit at expiry.

    breakevens   sorted break-even prices, e.g. [176.7]
    profitable   one bool per stretch of the price axis between them, in order:
                 below the first break-even, between each pair, above the last.
                 So len(profitable) == len(breakevens) + 1.

    Add up prob_between over the profitable stretches.
    """
    raise NotImplementedError("your turn")


def representative_vol(vols, sizes, vegas):
    """One volatility for a position with several option legs.

    sigma = sum(w_i sigma_i) / sum(w_i),   with w_i = |size_i| x vega_i

    Edge case for later: if every weight is zero, return the plain average.
    """
    raise NotImplementedError("your turn")


def explain(greeks, price_move, days_passed=0.0, vol_shift=0.0):
    """What each Greek predicts for a move, as a dict with keys delta, gamma, theta, vega.

    delta  = delta x move
    gamma  = 1/2 x gamma x move^2
    theta  = theta_per_day x days
    vega   = vega_per_point x vol_shift x 100     (vol_shift 0.05 is 5 points)

    greeks is a dict with "delta", "gamma", "theta_per_day" and "vega_per_point".
    """
    raise NotImplementedError("your turn")
