"""What "correct" means for the Black-Scholes functions.

Every test receives `impl`, the module under test. In this folder that is
`quantcore.black_scholes`. The `exercises` folder runs these same tests against
your own implementation.
"""

import numpy as np
import pytest

# Hull, "Options, Futures, and Other Derivatives": the standard worked example.
HULL = dict(spot=42.0, strike=40.0, time=0.5, vol=0.20, rate=0.10)
# Hull's index-option example, which adds a dividend yield.
HULL_INDEX = dict(spot=930.0, strike=900.0, time=2.0 / 12.0, vol=0.20, rate=0.08, div_yield=0.03)


def random_markets(n=300, seed=7):
    rng = np.random.default_rng(seed)
    return dict(
        spot=rng.uniform(20, 400, n),
        strike=rng.uniform(20, 400, n),
        time=rng.uniform(0.01, 3.0, n),
        vol=rng.uniform(0.05, 1.2, n),
        rate=rng.uniform(0.0, 0.08, n),
        div_yield=rng.uniform(0.0, 0.05, n),
    )


def bumped(market, name, amount):
    return {**market, name: market[name] + amount}


# --- prices -----------------------------------------------------------------


def test_textbook_call_and_put(impl):
    assert impl.price(**HULL, kind="call") == pytest.approx(4.76, abs=0.005)
    assert impl.price(**HULL, kind="put") == pytest.approx(0.81, abs=0.005)


def test_textbook_call_with_dividend_yield(impl):
    assert impl.price(**HULL_INDEX, kind="call") == pytest.approx(51.83, abs=0.02)


def test_put_call_parity(impl):
    """call - put = S e^(-qT) - K e^(-rT), whatever the volatility."""
    m = random_markets()
    call = impl.price(**m, kind="call")
    put = impl.price(**m, kind="put")
    forward_value = m["spot"] * np.exp(-m["div_yield"] * m["time"]) - m["strike"] * np.exp(-m["rate"] * m["time"])
    np.testing.assert_allclose(call - put, forward_value, atol=1e-9)


def test_prices_stay_inside_no_arbitrage_bounds(impl):
    m = random_markets()
    discounted_spot = m["spot"] * np.exp(-m["div_yield"] * m["time"])
    discounted_strike = m["strike"] * np.exp(-m["rate"] * m["time"])
    call = impl.price(**m, kind="call")
    put = impl.price(**m, kind="put")
    assert np.all(call <= discounted_spot + 1e-9)
    assert np.all(call >= np.maximum(discounted_spot - discounted_strike, 0.0) - 1e-9)
    assert np.all(put <= discounted_strike + 1e-9)
    assert np.all(put >= np.maximum(discounted_strike - discounted_spot, 0.0) - 1e-9)


def test_more_volatility_never_lowers_the_price(impl):
    m = random_markets()
    for kind in ("call", "put"):
        assert np.all(impl.price(**bumped(m, "vol", 0.01), kind=kind) >= impl.price(**m, kind=kind) - 1e-12)


# --- Greeks -----------------------------------------------------------------


def test_delta_matches_a_small_move_in_spot(impl):
    m = random_markets()
    h = 1e-4 * m["spot"]
    for kind in ("call", "put"):
        numeric = (impl.price(**bumped(m, "spot", h), kind=kind) - impl.price(**bumped(m, "spot", -h), kind=kind)) / (2 * h)
        np.testing.assert_allclose(impl.delta(**m, kind=kind), numeric, atol=1e-6)


def test_call_and_put_delta_differ_by_the_discount_factor(impl):
    m = random_markets()
    gap = impl.delta(**m, kind="call") - impl.delta(**m, kind="put")
    np.testing.assert_allclose(gap, np.exp(-m["div_yield"] * m["time"]), atol=1e-12)


def test_gamma_matches_the_change_in_delta(impl):
    m = random_markets()
    h = 1e-4 * m["spot"]
    numeric = (impl.delta(**bumped(m, "spot", h), kind="call") - impl.delta(**bumped(m, "spot", -h), kind="call")) / (2 * h)
    np.testing.assert_allclose(impl.gamma(**m), numeric, atol=1e-7)


def test_vega_matches_a_small_move_in_volatility(impl):
    m = random_markets()
    h = 1e-5
    numeric = (impl.price(**bumped(m, "vol", h), kind="call") - impl.price(**bumped(m, "vol", -h), kind="call")) / (2 * h)
    np.testing.assert_allclose(impl.vega(**m), numeric, rtol=1e-6, atol=1e-6)


def test_theta_matches_time_passing(impl):
    """Time passing means time-to-expiry shrinking, hence the minus sign."""
    m = random_markets()
    h = 1e-6
    for kind in ("call", "put"):
        numeric = -(impl.price(**bumped(m, "time", h), kind=kind) - impl.price(**bumped(m, "time", -h), kind=kind)) / (2 * h)
        np.testing.assert_allclose(impl.theta(**m, kind=kind), numeric, rtol=1e-5, atol=1e-5)


def test_rho_matches_a_small_move_in_the_rate(impl):
    m = random_markets()
    h = 1e-6
    for kind in ("call", "put"):
        numeric = (impl.price(**bumped(m, "rate", h), kind=kind) - impl.price(**bumped(m, "rate", -h), kind=kind)) / (2 * h)
        np.testing.assert_allclose(impl.rho(**m, kind=kind), numeric, rtol=1e-6, atol=1e-6)


def test_arrays_broadcast_against_each_other(impl):
    spots = np.linspace(80, 120, 5)[:, None]
    strikes = np.array([90.0, 100.0, 110.0])[None, :]
    assert np.shape(impl.price(spots, strikes, 0.5, 0.3, 0.02)) == (5, 3)


# --- edge cases -------------------------------------------------------------


@pytest.mark.edge
def test_at_expiry_the_price_is_intrinsic_value(impl):
    spots = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(impl.price(spots, 100.0, 0.0, 0.5, 0.03, kind="call"), [0.0, 0.0, 20.0])
    np.testing.assert_allclose(impl.price(spots, 100.0, 0.0, 0.5, 0.03, kind="put"), [20.0, 0.0, 0.0])


@pytest.mark.edge
def test_with_zero_volatility_the_price_is_the_discounted_forward_payoff(impl):
    spot, strike, time, rate, div_yield = 100.0, 95.0, 1.0, 0.05, 0.01
    expected = spot * np.exp(-div_yield * time) - strike * np.exp(-rate * time)
    assert impl.price(spot, strike, time, 0.0, rate, div_yield, kind="call") == pytest.approx(expected)
    assert impl.price(spot, strike, time, 0.0, rate, div_yield, kind="put") == pytest.approx(0.0)


@pytest.mark.edge
def test_greeks_are_finite_at_expiry(impl):
    spots = np.array([80.0, 100.0, 120.0])
    for kind in ("call", "put"):
        assert np.all(np.isfinite(impl.delta(spots, 100.0, 0.0, 0.5, kind=kind)))
        assert np.all(np.isfinite(impl.theta(spots, 100.0, 0.0, 0.5, kind=kind)))
    assert np.all(impl.gamma(spots, 100.0, 0.0, 0.5) == 0.0)
    assert np.all(impl.vega(spots, 100.0, 0.0, 0.5) == 0.0)
    np.testing.assert_allclose(impl.delta(spots, 100.0, 0.0, 0.5, kind="call"), [0.0, 0.5, 1.0])


@pytest.mark.edge
@pytest.mark.parametrize(
    "bad",
    [
        dict(spot=-1.0, strike=100.0, time=1.0, vol=0.2),
        dict(spot=100.0, strike=0.0, time=1.0, vol=0.2),
        dict(spot=100.0, strike=100.0, time=-0.1, vol=0.2),
        dict(spot=100.0, strike=100.0, time=1.0, vol=-0.2),
    ],
)
def test_impossible_inputs_are_rejected(impl, bad):
    with pytest.raises(ValueError):
        impl.price(**bad)


@pytest.mark.edge
def test_unknown_option_kind_is_rejected(impl):
    with pytest.raises(ValueError):
        impl.price(100.0, 100.0, 1.0, 0.2, kind="straddle")
