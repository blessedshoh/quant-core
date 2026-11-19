"""Earnings events: implied volatility split into an everyday part and a one-day jump that disappears."""

import math

import numpy as np
import pytest

from quantcore import Event, Leg, Position, entry_cost, net_greeks, profit, value
from quantcore import black_scholes as bs
from quantcore.position import MIN_VOL, base_vol, effective_vol, event_applies

R = 0.04
STRADDLE_LEGS = (Leg("call", 1, strike=165, days=30, vol=0.70), Leg("put", 1, strike=165, days=30, vol=0.70))
EVENT = Event(days=5, move=0.08)


def straddle(event=None, exercise="european"):
    return Position(spot=166.0, rate=R, legs=STRADDLE_LEGS, event=event, exercise=exercise)


PLAIN = [
    Position(spot=166.0, rate=R, legs=[Leg("call", 10, strike=180, days=30, vol=0.64)]),
    Position(spot=166.0, rate=R, legs=[Leg("call", 10, strike=170, days=30, vol=0.64), Leg("call", -10, strike=190, days=30, vol=0.60)]),
    Position(spot=166.0, rate=R, legs=[Leg("put", -5, strike=150, days=45, vol=0.55)]),
    Position(spot=166.0, rate=R, div_yield=0.01, legs=[Leg("stock", 100), Leg("call", -1, strike=180, days=30, vol=0.6)]),
    Position(spot=166.0, rate=R, legs=[Leg("call", -1, strike=170, days=20, vol=0.6), Leg("call", 1, strike=170, days=60, vol=0.55)]),
]


@pytest.mark.parametrize("plain", PLAIN)
def test_no_event_changes_nothing(plain):
    explicit = Position(plain.spot, plain.legs, plain.rate, plain.div_yield, event=None)
    prices = np.array([150.0, 166.0, 190.0])
    for day in (0.0, 10.0):
        assert np.array_equal(profit(plain, prices, day), profit(explicit, prices, day))
        for name, v in net_greeks(plain, prices, day).items():
            assert np.array_equal(v, net_greeks(explicit, prices, day)[name])
    assert entry_cost(plain) == entry_cost(explicit)


def test_event_leaves_today_unchanged():
    with_event, without = straddle(EVENT), straddle()
    assert entry_cost(with_event) == pytest.approx(entry_cost(without), abs=1e-9)
    assert float(value(with_event, 166.0)) == pytest.approx(float(value(without, 166.0)), abs=1e-9)
    g_event, g_plain = net_greeks(with_event), net_greeks(without)
    assert float(g_event["delta"]) == pytest.approx(float(g_plain["delta"]), rel=1e-4, abs=1e-4)
    assert float(g_event["gamma"]) == pytest.approx(float(g_plain["gamma"]), rel=1e-4)


def test_base_vol_formula():
    expected = math.sqrt((0.7**2 * 30 / 365 - 0.08**2) / (30 / 365))
    assert base_vol(STRADDLE_LEGS[0], straddle(EVENT)) == pytest.approx(expected, abs=1e-12)


def test_crush_on_the_day():
    pos = straddle(EVENT)
    b = base_vol(STRADDLE_LEGS[0], pos)
    after = sum(100 * float(bs.price(166.0, 165, 25 / 365, b, R, 0.0, kind)) for kind in ("call", "put"))
    assert float(value(pos, 166.0, 5.0)) == pytest.approx(after, abs=1e-9)
    assert float(value(pos, 166.0, 5.0 - 1e-4)) - float(value(pos, 166.0, 5.0)) > 0


def test_leg_before_event_unaffected():
    leg = Leg("call", 1, strike=165, days=3, vol=0.5)
    pos = Position(spot=166.0, rate=R, legs=[leg, STRADDLE_LEGS[0]], event=EVENT)
    assert not event_applies(leg, pos)
    assert base_vol(leg, pos) == leg.vol


def test_too_low_vol_refused():
    with pytest.raises(ValueError, match="cannot contain"):
        Position(spot=166.0, rate=R, legs=[Leg("call", 1, strike=165, days=10, vol=0.25)], event=EVENT)


@pytest.mark.parametrize("days, move", [(0.5, 0.08), (5, 0.001), (5, 0.6), (2000, 0.08)])
def test_event_limits(days, move):
    with pytest.raises(ValueError):
        Event(days=days, move=move)


def test_monte_carlo_matches_effective_vol():
    pos = straddle(EVENT)
    leg = STRADDLE_LEGS[0]
    b, m, T = base_vol(leg, pos), EVENT.move, 30 / 365
    rng = np.random.default_rng(11)
    n = 1_000_000
    log_s = (
        math.log(166.0)
        + (R - 0.5 * b * b) * T
        + b * math.sqrt(T) * rng.standard_normal(n)
        - 0.5 * m * m
        + m * rng.standard_normal(n)
    )
    discounted = math.exp(-R * T) * np.maximum(np.exp(log_s) - 165, 0.0)
    estimate, se = discounted.mean(), discounted.std() / math.sqrt(n)
    model = float(bs.price(166.0, 165, T, float(effective_vol(leg, pos, 0.0)), R, 0.0, "call"))
    assert abs(estimate - model) < 3 * se


def test_shift_moves_base_only():
    pos = straddle(EVENT)
    leg = STRADDLE_LEGS[0]
    b, m, tau = base_vol(leg, pos), EVENT.move, 30 / 365
    shifted = float(effective_vol(leg, pos, 0.0, 1.0, 0.1))
    assert shifted == pytest.approx(math.sqrt(((b + 0.1) ** 2 * tau + m * m) / tau), abs=1e-12)


def test_effective_vol_without_event_is_the_old_rule():
    leg = PLAIN[0].legs[0]
    for scale, shift in ((1.0, 0.0), (0.25, -0.2), (3.0, 0.1)):
        assert np.array_equal(effective_vol(leg, PLAIN[0], 7.0, scale, shift), np.maximum(leg.vol * scale + shift, MIN_VOL))


def test_theta_near_event_is_finite():
    near = net_greeks(straddle(EVENT), days_forward=5 - 1e-4)["theta_per_day"]
    today_plain = net_greeks(straddle())["theta_per_day"]
    assert abs(float(near)) < 10 * abs(float(today_plain))
