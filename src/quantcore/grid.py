"""The profit grid: profit at every combination of price, date and volatility.

The web front end asks for one grid per position and then reads from it as the
user drags the date and volatility sliders, so the sliders never wait on the
server. All pricing happens here in Python. The browser only interpolates.

Axes
----
prices       underlying prices, evenly spaced, plus the spot and every strike
             so the kinks in the payoff land exactly on grid points
days         days forward from today up to the first expiry, packed tightly
             toward expiry where value changes fastest
vol_scales   multipliers applied to every leg's volatility. 1.0 is unchanged.

profit has shape (len(vol_scales), len(days), len(prices)).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .position import Position, entry_cost, first_expiry, net_greeks, profit, value

AMERICAN_DAYS = 17
AMERICAN_VOLS = 4
AMERICAN_GRID_STEPS = 31


def price_axis(position: Position, range_pct: float = 0.25, points: int = 81) -> np.ndarray:
    """Prices from spot * (1 - range_pct) to spot * (1 + range_pct), plus spot and in-range strikes."""
    low = max(position.spot * (1.0 - range_pct), position.spot * 0.01)
    high = position.spot * (1.0 + range_pct)
    extras = [position.spot] + [k for k in position.strikes if low <= k <= high]
    return np.unique(np.concatenate([np.linspace(low, high, points), extras]))


def day_axis(position: Position, points: int = 25) -> np.ndarray:
    """Days forward from 0 to the first expiry, packed tightly toward expiry.

    Value changes slowly while expiry is far away and very fast in the last
    days and hours, so the points follow a cubic curve: with 30 days to run,
    the first step is about 3.6 days and the last is about three minutes.

    The cubic was chosen by measurement. Across six strategies the worst
    interpolation error was 15% of the money at stake with even spacing, 2.4%
    with quadratic spacing and under 0.7% with cubic, for the same 25 points.
    `tests/test_grid.py` keeps that promise.
    """
    last = first_expiry(position)
    if not last:
        return np.array([0.0])
    u = np.linspace(0.0, 1.0, points)
    return last * (1.0 - (1.0 - u) ** 3)


def vol_axis(low: float = 0.25, high: float = 3.0, points_each_side: int = 8) -> np.ndarray:
    """Volatility multipliers from `low` to `high`, with exactly 1.0 in the middle."""
    down = np.geomspace(low, 1.0, points_each_side + 1)
    up = np.geomspace(1.0, high, points_each_side + 1)
    down[-1] = up[0] = 1.0
    return np.unique(np.concatenate([down, up]))


def _bracket(axis: np.ndarray, x: float) -> tuple[int, int, float]:
    """Indices of the two grid points around x and how far x sits between them (0 to 1)."""
    x = min(max(float(x), float(axis[0])), float(axis[-1]))
    if len(axis) == 1:
        return 0, 0, 0.0
    upper = int(np.searchsorted(axis, x, side="right"))
    upper = min(max(upper, 1), len(axis) - 1)
    lower = upper - 1
    span = float(axis[upper] - axis[lower])
    weight = 0.0 if span == 0.0 else (x - float(axis[lower])) / span
    return lower, upper, weight


@dataclass(frozen=True)
class Grid:
    prices: np.ndarray
    days: np.ndarray
    vol_scales: np.ndarray
    profit: np.ndarray

    def at(self, price: float, day: float, vol_scale: float = 1.0) -> float:
        """Profit at any point inside the grid, by interpolation along each axis.

        Price and volatility are read in straight lines between grid points.
        The date is read in straight lines of the SQUARE ROOT of the time left,
        because that is how an option's time value actually shrinks. Reading the
        date in plain days would overstate the decay in the final hours.

        This is the same arithmetic the front end performs. It lives here so the
        tests can measure how far interpolation strays from exact pricing.
        """
        v0, v1, wv = _bracket(self.vol_scales, vol_scale)
        d0, d1, _ = _bracket(self.days, day)
        p0, p1, wp = _bracket(self.prices, price)
        cube = self.profit

        last = float(self.days[-1])
        clamped_day = min(max(float(day), float(self.days[0])), last)
        root0, root1 = np.sqrt(last - self.days[d0]), np.sqrt(last - self.days[d1])
        wd = 0.0 if root0 == root1 else float((root0 - np.sqrt(last - clamped_day)) / (root0 - root1))

        def along_price(v: int, d: int) -> float:
            return float(cube[v, d, p0] * (1.0 - wp) + cube[v, d, p1] * wp)

        def along_day(v: int) -> float:
            return along_price(v, d0) * (1.0 - wd) + along_price(v, d1) * wd

        return along_day(v0) * (1.0 - wv) + along_day(v1) * wv


def profit_grid(
    position: Position,
    range_pct: float = 0.25,
    prices=None,
    days=None,
    vol_scales=None,
) -> Grid:
    """Price the position at every (volatility, day, price) combination in one vectorised call."""
    prices = price_axis(position, range_pct) if prices is None else np.asarray(prices, dtype=float)
    cost = entry_cost(position)  # from the accurate tree, even when the grid uses a faster one
    if position.american:
        # Trees are far slower than the Black-Scholes formula, so American grids use
        # fewer dates, fewer volatility levels and a 31-step tree: about 0.3 s per leg,
        # with prices within about 2 cents per share of a 1,601-step tree.
        days = day_axis(position, AMERICAN_DAYS) if days is None else np.asarray(days, dtype=float)
        vol_scales = vol_axis(points_each_side=AMERICAN_VOLS) if vol_scales is None else np.asarray(vol_scales, dtype=float)
        position = replace(position, tree_steps=AMERICAN_GRID_STEPS)
    days = day_axis(position) if days is None else np.asarray(days, dtype=float)
    vol_scales = vol_axis() if vol_scales is None else np.asarray(vol_scales, dtype=float)
    cube = value(position, prices[None, None, :], days[None, :, None], vol_scales[:, None, None]) - cost
    return Grid(prices=prices, days=days, vol_scales=vol_scales, profit=np.asarray(cube, dtype=float))


def greeks_grid(position: Position, days, vol_scales) -> dict[str, np.ndarray]:
    """Position Greeks at the current spot for every (volatility, day) pair. Shape (V, D)."""
    days = np.asarray(days, dtype=float)
    vol_scales = np.asarray(vol_scales, dtype=float)
    return net_greeks(position, position.spot, days[None, :], vol_scales[:, None])
