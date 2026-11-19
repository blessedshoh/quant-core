"""Earnings volatility: the one-day jump hidden inside implied volatility.

An option that expires after an earnings announcement carries two kinds of
uncertainty: ordinary day-to-day moves, and one large jump on the day of the
announcement. Their variances add over the option's life,

    sigma^2 T = sigma_b^2 T + m^2

where sigma is the implied volatility you see, sigma_b the everyday volatility
and m the standard deviation of the jump in the log price. After the
announcement only sigma_b is left, which is why implied volatility "crushes".
`position.base_vol` and `position.effective_vol` apply this to every leg; this
module adds the reports and the inverse problem.

What move is the market pricing?
--------------------------------
Two expiries that both fall after the announcement share the same jump but
spread it over different lengths of time, so the nearer one has the higher
implied volatility. Assuming they also share one everyday volatility:

    sigma_b^2 = (sigma_2^2 T_2 - sigma_1^2 T_1) / (T_2 - T_1)
    m^2       = sigma_1^2 T_1 - sigma_b^2 T_1

For a normal jump, the average size of the move, up or down, is m sqrt(2/pi):
the number traders quote as "the move the straddle implies".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import black_scholes as bs
from .position import Position, base_vol, event_applies


@dataclass(frozen=True)
class ImpliedMove:
    move: float  # standard deviation of the one-day jump in the log price
    expected_abs_move: float  # average size of the jump, up or down: move x sqrt(2/pi)
    base_vol: float  # everyday volatility shared by both expiries
    event_variance_share: float  # part of the nearer expiry's total variance that is the jump


def implied_move(near_days: float, near_vol: float, far_days: float, far_vol: float) -> ImpliedMove:
    """The earnings move and everyday volatility implied by two expiries after the announcement."""
    if not 0 < near_days < far_days:
        raise ValueError("the near expiry must come before the far one")
    if near_vol <= 0 or far_vol <= 0:
        raise ValueError("volatilities must be positive")
    t1, t2 = near_days / bs.DAYS_PER_YEAR, far_days / bs.DAYS_PER_YEAR
    near_variance = near_vol**2 * t1
    base_variance = (far_vol**2 * t2 - near_variance) / (t2 - t1)
    if base_variance <= 0:
        raise ValueError("the earlier expiry's volatility is too high for one shared base volatility")
    jump_variance = near_variance - base_variance * t1
    if jump_variance <= 0:
        raise ValueError("the later expiry is the more expensive one, so these prices imply no earnings premium")
    move = math.sqrt(jump_variance)
    return ImpliedMove(
        move=move,
        expected_abs_move=move * math.sqrt(2.0 / math.pi),
        base_vol=math.sqrt(base_variance),
        event_variance_share=jump_variance / near_variance,
    )


def event_vols(position: Position) -> list[tuple[int, float, float, float, float]]:
    """(leg index, base vol, vol before, vol after, weight) for every leg alive at the announcement.

    weight is |size| x vega today, the same weight the chance of profit uses, so
    a report can single out the leg the position depends on most.
    """
    rows = []
    for index, leg in enumerate(position.legs):
        if not event_applies(leg, position):
            continue
        everyday = base_vol(leg, position)
        vega = float(bs.vega(position.spot, leg.strike, leg.days / bs.DAYS_PER_YEAR, leg.vol, position.rate, position.div_yield))
        rows.append((index, everyday, float(leg.vol), everyday, abs(leg.size) * vega))
    return rows


__all__ = ["ImpliedMove", "event_vols", "implied_move"]
