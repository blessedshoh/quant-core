"""Binomial trees: convergence to Black-Scholes, American exercise, and the explorer tree."""

import numpy as np
import pytest

from quantcore import black_scholes as bs
from quantcore import binomial

MARKET = dict(spot=100.0, strike=105.0, time=0.5, vol=0.3, rate=0.05)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_leisen_reimer_matches_black_scholes_closely(kind):
    tree = binomial.price(**MARKET, kind=kind, steps=101)
    exact = float(bs.price(**MARKET, kind=kind))
    assert tree == pytest.approx(exact, abs=1e-4)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_crr_converges_but_more_slowly(kind):
    exact = float(bs.price(**MARKET, kind=kind))
    coarse = abs(binomial.price(**MARKET, kind=kind, steps=50, method="crr") - exact)
    fine = abs(binomial.price(**MARKET, kind=kind, steps=2000, method="crr") - exact)
    assert fine < 0.005
    assert fine < coarse
    lr = abs(binomial.price(**MARKET, kind=kind, steps=51) - exact)
    assert lr < coarse / 10


def test_leisen_reimer_uses_an_odd_number_of_steps():
    assert binomial.tree_parameters(100, 1, 0.2, 0.01, 0, 100)[0] == 101
    assert binomial.price(**MARKET, steps=100) == binomial.price(**MARKET, steps=101)


def test_spot_can_be_an_array():
    spots = np.array([80.0, 100.0, 120.0])
    many = binomial.price(spots, 105.0, 0.5, 0.3, 0.05, kind="put", american=True)
    single = [binomial.price(s, 105.0, 0.5, 0.3, 0.05, kind="put", american=True) for s in spots]
    np.testing.assert_allclose(many, single)


def test_price_many_broadcasts_every_argument():
    spots = np.array([90.0, 100.0, 110.0])[None, :]
    times = np.array([0.1, 0.5])[:, None]
    grid = binomial.price_many(spots, 100.0, times, 0.3, 0.05, kind="put", american=True)
    assert grid.shape == (2, 3)
    assert grid[1, 2] == pytest.approx(binomial.price(110.0, 100.0, 0.5, 0.3, 0.05, kind="put", american=True))


def test_american_call_without_dividends_is_never_exercised_early():
    """With no dividends it never pays to exercise a call early, so American = European."""
    american = binomial.price(**MARKET, kind="call", american=True)
    european = binomial.price(**MARKET, kind="call", american=False)
    assert american == pytest.approx(european, abs=1e-10)


def test_american_call_with_a_dividend_can_be_worth_more():
    market = {**MARKET, "spot": 140.0}
    american = binomial.price(**market, div_yield=0.08, kind="call", american=True)
    european = binomial.price(**market, div_yield=0.08, kind="call", american=False)
    assert american > european + 0.05


def test_american_put_is_worth_at_least_the_european_and_at_least_exercising_now():
    spots = np.linspace(60, 140, 41)
    american = binomial.price(spots, 105.0, 0.5, 0.3, 0.05, kind="put", american=True)
    european = binomial.price(spots, 105.0, 0.5, 0.3, 0.05, kind="put", american=False)
    assert np.all(american >= european - 1e-10)
    assert np.all(american >= np.maximum(105.0 - spots, 0.0) - 1e-10)
    assert binomial.early_exercise_premium(70.0, 105.0, 0.5, 0.3, 0.05) > 0.3


def test_no_time_or_no_volatility_falls_back_to_exact_values():
    assert binomial.price(90.0, 100.0, 0.0, 0.3, kind="put", american=True) == pytest.approx(10.0)
    assert binomial.price(110.0, 100.0, 1.0, 0.0, 0.05, kind="call") == pytest.approx(
        float(bs.price(110.0, 100.0, 1.0, 0.0, 0.05))
    )


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        binomial.price(**MARKET, steps=0)
    with pytest.raises(ValueError):
        binomial.price(**MARKET, method="trinomial")
    with pytest.raises(ValueError):
        binomial.build_tree(**MARKET, steps=40)


def test_the_explorer_tree_agrees_with_the_pricer():
    tree = binomial.build_tree(**MARKET, kind="put", american=True, steps=6)
    assert len(tree.prices) == 7 and len(tree.prices[6]) == 7
    assert tree.values[0][0] == pytest.approx(
        binomial.price(**MARKET, kind="put", american=True, steps=6, method="crr")
    )
    assert tree.prices[1][1] / tree.prices[0][0] == pytest.approx(tree.u)
    assert 0 < tree.p < 1
    # Deep in the money, early in the tree, a put is exercised.
    assert any(any(row[: len(row) // 2]) for row in tree.exercise[1:-1])


def test_a_european_explorer_tree_never_marks_early_exercise():
    tree = binomial.build_tree(**MARKET, kind="put", american=False, steps=6)
    assert not any(any(row) for row in tree.exercise[:-1])


def test_convergence_series():
    series = binomial.convergence(**MARKET, steps=range(1, 31))
    assert len(series["crr"]) == len(series["leisen_reimer"]) == 30
    assert abs(series["leisen_reimer"][-1] - series["black_scholes"]) < abs(series["crr"][-1] - series["black_scholes"])


# --- American positions --------------------------------------------------------

from quantcore import Leg, Position, entry_cost, net_greeks, profit_grid  # noqa: E402

PUT_SPREAD = dict(
    spot=166.0,
    rate=0.04,
    legs=[Leg("put", 1, strike=180, days=60, vol=0.5), Leg("put", -1, strike=160, days=60, vol=0.5)],
)


def test_american_positions_cost_more_when_early_exercise_has_value():
    european = Position(**PUT_SPREAD)
    american = Position(**PUT_SPREAD, exercise="american")
    long_put = Position(spot=166.0, rate=0.04, legs=[Leg("put", 1, strike=200, days=90, vol=0.5)], exercise="american")
    assert entry_cost(long_put) > entry_cost(Position(**{**PUT_SPREAD, "legs": long_put.legs}))
    assert entry_cost(american) == pytest.approx(entry_cost(european), rel=0.05)


def test_unknown_exercise_style_is_refused():
    with pytest.raises(ValueError):
        Position(**PUT_SPREAD, exercise="bermudan")


def test_american_greeks_agree_with_european_ones_where_early_exercise_is_worthless():
    """An American call on a stock with no dividends is a European call, so its Greeks must match."""
    legs = [Leg("call", 1, strike=170, days=45, vol=0.5)]
    european = net_greeks(Position(spot=166.0, rate=0.04, legs=legs))
    american = net_greeks(Position(spot=166.0, rate=0.04, legs=legs, exercise="american"))
    assert float(american["delta"]) == pytest.approx(float(european["delta"]), abs=0.6)
    assert float(american["gamma"]) == pytest.approx(float(european["gamma"]), rel=0.05)
    assert float(american["theta_per_day"]) == pytest.approx(float(european["theta_per_day"]), rel=0.05)
    assert float(american["vega_per_point"]) == pytest.approx(float(european["vega_per_point"]), rel=0.02)


def test_american_grid_is_fast_and_starts_at_zero():
    import time

    position = Position(**PUT_SPREAD, exercise="american")
    started = time.perf_counter()
    grid = profit_grid(position)
    assert time.perf_counter() - started < 5.0
    assert grid.profit.shape == (9, 17, len(grid.prices))
    unchanged = grid.at(166.0, 0.0, 1.0)
    assert abs(unchanged) < 0.03 * 100  # within 3 cents a share of zero on day one


def test_american_grid_reads_accurately_between_points():
    from quantcore import profit

    position = Position(**PUT_SPREAD, exercise="american")
    grid = profit_grid(position)
    rng = np.random.default_rng(4)
    points = zip(
        rng.uniform(grid.prices[0], grid.prices[-1], 120),
        rng.uniform(0, grid.days[-1], 120),
        rng.uniform(grid.vol_scales[0], grid.vol_scales[-1], 120),
    )
    worst = max(abs(grid.at(p, d, v) - float(profit(position, p, d, v))) for p, d, v in points)
    assert worst < 0.02 * float(np.max(np.abs(grid.profit)))
