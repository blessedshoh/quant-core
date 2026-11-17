"""Scenario grid and P/L explain."""

import numpy as np
import pytest

from quantcore import Leg, Position, profit, value
from quantcore.scenarios import explain, scenario_grid

SPOT = 166.0
LONG_CALL = Position(spot=SPOT, rate=0.04, legs=[Leg("call", 10, strike=180, days=30, vol=0.64)])
STRADDLE = Position(
    spot=SPOT,
    rate=0.04,
    legs=[Leg("call", 1, strike=165, days=30, vol=0.6), Leg("put", 1, strike=165, days=30, vol=0.6)],
)


def test_scenario_grid_cells_are_the_profit_at_the_shocked_inputs():
    grid = scenario_grid(LONG_CALL)
    assert grid.profit.shape == (5, 9)
    assert grid.profit[2, 4] == pytest.approx(0.0, abs=1e-9)  # no shock, no time
    for v, p in [(0, 0), (4, 8), (1, 6)]:
        price = SPOT * (1 + grid.price_shocks[p])
        expected = float(profit(LONG_CALL, price, 0.0, 1.0, grid.vol_shifts[v]))
        assert grid.profit[v, p] == pytest.approx(expected)


def test_a_long_call_gains_with_price_and_with_volatility():
    grid = scenario_grid(LONG_CALL)
    assert np.all(np.diff(grid.profit, axis=1) > 0)
    assert np.all(np.diff(grid.profit, axis=0) > 0)


def test_a_horizon_adds_time_decay():
    assert scenario_grid(LONG_CALL, days_forward=7).profit[2, 4] < 0


def test_volatility_shocks_never_go_below_zero():
    grid = scenario_grid(LONG_CALL, vol_shifts=[-5.0])
    assert np.all(np.isfinite(grid.profit))


def test_impossible_price_shocks_are_refused():
    with pytest.raises(ValueError):
        scenario_grid(LONG_CALL, price_shocks=[-1.0])


def test_explain_adds_up_exactly():
    parts = explain(STRADDLE, price_move=12.0, days_passed=5, vol_shift=-0.04)
    assert parts.predicted + parts.unexplained == pytest.approx(parts.actual, abs=1e-9)
    after = float(value(STRADDLE, SPOT + 12.0, 5, 1.0, -0.04))
    assert parts.actual == pytest.approx(after - float(value(STRADDLE, SPOT)))


def test_small_moves_are_almost_fully_explained():
    parts = explain(LONG_CALL, price_move=0.5, days_passed=0.1, vol_shift=0.002)
    assert abs(parts.unexplained) < 0.02 * abs(parts.actual)


def test_each_greek_carries_the_move_it_describes():
    only_price = explain(STRADDLE, price_move=20.0)
    assert only_price.gamma > 0 and only_price.theta == 0 and only_price.vega == 0
    only_time = explain(STRADDLE, price_move=0.0, days_passed=1.0)
    assert only_time.theta < 0
    assert only_time.theta == pytest.approx(only_time.actual, rel=0.02)
    only_vol = explain(STRADDLE, price_move=0.0, vol_shift=0.01)
    assert only_vol.vega == pytest.approx(only_vol.actual, rel=0.01)


def test_the_unexplained_part_grows_like_the_cube_of_the_move():
    """Delta and gamma capture the first two powers of the move; what is left starts at the third."""
    small = explain(LONG_CALL, price_move=10.0).unexplained
    large = explain(LONG_CALL, price_move=40.0).unexplained
    assert abs(large) > 20 * abs(small)
