"""The profit grid the front end reads from, and how accurate interpolating it is."""

import numpy as np
import pytest

from quantcore import Leg, Position, entry_cost, greeks_grid, net_greeks, profit, profit_at_expiry, profit_grid

LONG_CALL = Position(spot=166.0, rate=0.04, legs=[Leg("call", 10, strike=180, days=30, vol=0.64)])
IRON_CONDOR = Position(
    spot=166.0,
    rate=0.04,
    legs=[
        Leg("put", 1, strike=140, days=30, vol=0.70),
        Leg("put", -1, strike=150, days=30, vol=0.66),
        Leg("call", -1, strike=180, days=30, vol=0.62),
        Leg("call", 1, strike=190, days=30, vol=0.60),
    ],
)


def test_axes_cover_the_range_and_hit_spot_and_strikes():
    grid = profit_grid(IRON_CONDOR, range_pct=0.25)

    assert np.all(np.diff(grid.prices) > 0)
    assert grid.prices[0] == pytest.approx(166.0 * 0.75)
    assert grid.prices[-1] == pytest.approx(166.0 * 1.25)
    for must_have in (166.0, 140.0, 150.0, 180.0, 190.0):
        assert must_have in grid.prices

    assert grid.days[0] == 0.0
    assert grid.days[-1] == pytest.approx(30.0)
    assert np.all(np.diff(grid.days) > 0)

    assert 1.0 in grid.vol_scales
    assert grid.vol_scales[0] == pytest.approx(0.25)
    assert grid.vol_scales[-1] == pytest.approx(3.0)
    assert grid.profit.shape == (len(grid.vol_scales), len(grid.days), len(grid.prices))


def test_strikes_outside_the_range_are_left_out():
    grid = profit_grid(IRON_CONDOR, range_pct=0.05)
    assert 140.0 not in grid.prices
    assert 190.0 not in grid.prices


def test_grid_values_are_the_exact_profit_at_each_node():
    grid = profit_grid(LONG_CALL)
    for v, d, p in [(0, 0, 0), (8, 10, 40), (16, 24, 80), (3, 24, 17)]:
        exact = float(profit(LONG_CALL, grid.prices[p], grid.days[d], grid.vol_scales[v]))
        assert grid.profit[v, d, p] == pytest.approx(exact)


def test_last_day_is_the_expiry_payoff_whatever_the_volatility():
    grid = profit_grid(IRON_CONDOR)
    expected = profit_at_expiry(IRON_CONDOR, grid.prices)
    for v in range(len(grid.vol_scales)):
        np.testing.assert_allclose(grid.profit[v, -1, :], expected, atol=1e-9)


CALENDAR = Position(
    spot=166.0,
    rate=0.04,
    legs=[Leg("call", -1, strike=170, days=30, vol=0.60), Leg("call", 1, strike=170, days=60, vol=0.58)],
)
FOUR_DAY_STRANGLE = Position(
    spot=166.0,
    rate=0.04,
    legs=[Leg("call", 1, strike=170, days=4, vol=0.64), Leg("put", 1, strike=165, days=4, vol=0.64)],
)
LONG_DATED = Position(
    spot=166.0,
    rate=0.04,
    legs=[Leg("put", -2, strike=150, days=400, vol=0.50), Leg("call", 1, strike=200, days=400, vol=0.45)],
)
BUTTERFLY = Position(
    spot=166.0,
    rate=0.04,
    legs=[
        Leg("call", 1, strike=160, days=14, vol=0.60),
        Leg("call", -2, strike=166, days=14, vol=0.60),
        Leg("call", 1, strike=172, days=14, vol=0.60),
    ],
)


@pytest.mark.parametrize(
    "pos",
    [LONG_CALL, IRON_CONDOR, CALENDAR, FOUR_DAY_STRANGLE, LONG_DATED, BUTTERFLY],
    ids=["long call", "iron condor", "calendar", "4-day strangle", "400-day risk reversal", "butterfly"],
)
def test_interpolating_the_grid_stays_close_to_exact_pricing(pos):
    """The front end interpolates. Its worst error must stay under 1% of the money at stake.

    Measured worst cases when this was written: 0.08% (long call) to 0.68% (butterfly).
    The worst points are always in the final hours before expiry.
    """
    grid = profit_grid(pos)
    rng = np.random.default_rng(11)
    prices = rng.uniform(grid.prices[0], grid.prices[-1], 3000)
    days = rng.uniform(0.0, grid.days[-1], 3000)
    scales = rng.uniform(grid.vol_scales[0], grid.vol_scales[-1], 3000)

    exact = np.array([float(profit(pos, p, d, s)) for p, d, s in zip(prices, days, scales)])
    approx = np.array([grid.at(p, d, s) for p, d, s in zip(prices, days, scales)])

    at_stake = float(np.max(np.abs(grid.profit)))
    assert np.max(np.abs(approx - exact)) < 0.01 * at_stake


def test_interpolation_is_exact_on_grid_points():
    grid = profit_grid(LONG_CALL)
    assert grid.at(grid.prices[30], grid.days[7], grid.vol_scales[9]) == pytest.approx(grid.profit[9, 7, 30])


def test_points_outside_the_grid_are_clamped_to_its_edge():
    grid = profit_grid(LONG_CALL)
    assert grid.at(1e9, -5.0, 99.0) == pytest.approx(grid.profit[-1, 0, -1])


def test_greeks_grid_matches_net_greeks():
    grid = profit_grid(LONG_CALL)
    greeks = greeks_grid(LONG_CALL, grid.days, grid.vol_scales)

    assert greeks["delta"].shape == (len(grid.vol_scales), len(grid.days))
    direct = net_greeks(LONG_CALL, 166.0, grid.days[5], grid.vol_scales[8])
    for name in ("delta", "gamma", "theta_per_day", "vega_per_point"):
        assert greeks[name][8, 5] == pytest.approx(float(direct[name]))


def test_stock_only_position_has_a_single_day():
    pos = Position(spot=50.0, legs=[Leg("stock", 100)])
    grid = profit_grid(pos)

    assert list(grid.days) == [0.0]
    assert grid.profit.shape == (len(grid.vol_scales), 1, len(grid.prices))
    np.testing.assert_allclose(grid.profit[0, 0, :], (grid.prices - 50.0) * 100)
    assert entry_cost(pos) == pytest.approx(5000.0)
