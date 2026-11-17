"""Price distribution and chance of profit, checked against closed forms and simulation."""

import math

import numpy as np
import pytest
from scipy.special import ndtr

from quantcore import Leg, Position, breakevens, entry_cost
from quantcore.position import profit_at_expiry
from quantcore.probability import (
    cdf,
    chance_of_profit,
    density,
    expected_profit,
    log_price_moments,
    representative_vol,
)

SPOT, RATE, VOL = 166.0, 0.04, 0.64
LONG_CALL = Position(spot=SPOT, rate=RATE, legs=[Leg("call", 10, strike=180, days=30, vol=VOL)])
IRON_CONDOR = Position(
    spot=SPOT,
    rate=RATE,
    legs=[
        Leg("put", 1, strike=140, days=30, vol=VOL),
        Leg("put", -1, strike=150, days=30, vol=VOL),
        Leg("call", -1, strike=180, days=30, vol=VOL),
        Leg("call", 1, strike=190, days=30, vol=VOL),
    ],
)


def test_density_integrates_to_one_and_has_the_forward_as_its_mean():
    prices = np.linspace(1, 1500, 300001)
    pdf = density(prices, SPOT, 30, VOL, RATE)
    assert np.trapezoid(pdf, prices) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(prices * pdf, prices) == pytest.approx(SPOT * math.exp(RATE * 30 / 365), rel=1e-6)


def test_cdf_matches_the_textbook_n_of_d2():
    """P(S_T > K) under the risk-neutral measure is N(d2)."""
    years = 30 / 365
    d2 = (math.log(SPOT / 180) + (RATE - 0.5 * VOL**2) * years) / (VOL * math.sqrt(years))
    assert 1 - float(cdf(180, SPOT, 30, VOL, RATE)) == pytest.approx(float(ndtr(d2)), abs=1e-12)


def test_cdf_edges():
    assert float(cdf(0.0, SPOT, 30, VOL)) == 0.0
    assert float(cdf(math.inf, SPOT, 30, VOL)) == 1.0
    np.testing.assert_allclose(cdf([150, 166, 170], SPOT, 0, VOL), [0.0, 1.0, 1.0])


def test_drift_moves_the_distribution():
    risk_neutral, _ = log_price_moments(SPOT, 30, VOL, RATE)
    bullish, _ = log_price_moments(SPOT, 30, VOL, RATE, drift=0.30)
    assert bullish > risk_neutral


def test_chance_of_profit_for_a_long_call_is_the_chance_of_finishing_above_break_even():
    [point] = breakevens(LONG_CALL)
    expected = 1 - float(cdf(point, SPOT, 30, VOL, RATE))
    assert chance_of_profit(LONG_CALL) == pytest.approx(expected, abs=1e-12)
    assert 0.15 < expected < 0.35


def test_chance_of_profit_for_an_iron_condor_is_the_chance_of_finishing_between_break_evens():
    low, high = breakevens(IRON_CONDOR)
    mass = cdf([low, high], SPOT, 30, VOL, RATE)
    assert chance_of_profit(IRON_CONDOR) == pytest.approx(float(mass[1] - mass[0]), abs=1e-12)


def test_chance_of_profit_agrees_with_simulation():
    butterfly = Position(
        spot=SPOT,
        rate=RATE,
        legs=[
            Leg("call", 1, strike=155, days=21, vol=0.5),
            Leg("call", -2, strike=166, days=21, vol=0.5),
            Leg("call", 1, strike=177, days=21, vol=0.5),
        ],
    )
    mean, sd = log_price_moments(SPOT, 21, 0.5, RATE)
    finishes = np.exp(mean + sd * np.random.default_rng(3).standard_normal(400_000))
    simulated = float(np.mean(np.asarray(profit_at_expiry(butterfly, finishes)) > 0))
    assert chance_of_profit(butterfly) == pytest.approx(simulated, abs=0.004)


def test_a_bullish_view_raises_the_chance_for_a_long_call():
    assert chance_of_profit(LONG_CALL, drift=0.40) > chance_of_profit(LONG_CALL)


def test_risk_neutral_expected_profit_is_just_the_financing_cost():
    """Fairly priced options earn the risk-free rate on average, no more."""
    cost = entry_cost(LONG_CALL)
    assert expected_profit(LONG_CALL) == pytest.approx(cost * (math.exp(RATE * 30 / 365) - 1), abs=1.0)


def test_representative_vol_weights_the_leg_that_matters_most():
    mixed = Position(
        spot=SPOT,
        legs=[Leg("call", 10, strike=166, days=30, vol=0.60), Leg("call", 1, strike=250, days=30, vol=1.00)],
    )
    assert 0.60 < representative_vol(mixed) < 0.62
    assert representative_vol(Position(spot=SPOT, legs=[Leg("stock", 100)])) is None


def test_positions_without_options_have_no_chance_of_profit():
    stock = Position(spot=SPOT, legs=[Leg("stock", 100)])
    assert chance_of_profit(stock) is None
    assert expected_profit(stock) is None
