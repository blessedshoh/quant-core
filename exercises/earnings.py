"""Exercise 3: earnings volatility.

Fill in each function below, then check your work against the engine:

    pytest exercises/test_earnings.py -m "not edge"     the core maths
    pytest exercises/test_earnings.py                   everything, including the refusals

Rules of the game
-----------------
- Work from walkthroughs/03_earnings_volatility.ipynb, not from the engine's
  source (src/quantcore/position.py and earnings.py). Look there only after
  your version passes.
- Units: days are calendar days (years = days / 365); vol and move are decimals.

Symbols
-------
sigma   implied volatility you see (includes the earnings jump)
sigma_b everyday volatility
m       standard deviation of the one-day jump in the log price
T, tau  years to expiry, years left
"""

import math  # noqa: F401

DAYS_PER_YEAR = 365.0


def base_vol(vol, days, move):
    """The everyday volatility left once the earnings jump is taken out.

    sigma^2 T = sigma_b^2 T + m^2      so      sigma_b = sqrt((sigma^2 T - m^2) / T)

    Edge case for later: raise ValueError when sigma^2 T <= m^2.
    """
    raise NotImplementedError("your turn")


def effective_vol(base, days_left, move, event_ahead):
    """The volatility to price an option with, `days_left` days before its expiry.

    event still ahead:  sqrt((sigma_b^2 tau + m^2) / tau)
    event passed:       sigma_b
    """
    raise NotImplementedError("your turn")


def implied_move(near_days, near_vol, far_days, far_vol):
    """Return (move, base_vol) implied by two expiries that both fall after the announcement.

    sigma_b^2 = (sigma_2^2 T_2 - sigma_1^2 T_1) / (T_2 - T_1)
    m^2       = sigma_1^2 T_1 - sigma_b^2 T_1

    Edge cases for later: raise ValueError if near_days >= far_days, or if either
    variance comes out zero or negative.
    """
    raise NotImplementedError("your turn")


def expected_abs_move(move):
    """Average size of a normal jump, up or down: m sqrt(2 / pi)."""
    raise NotImplementedError("your turn")
