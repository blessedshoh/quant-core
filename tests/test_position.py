"""Positions: cost, profit, break-evens and limits for the standard strategies."""

import math

import numpy as np
import pytest

from quantcore import (
    Leg,
    Position,
    breakevens,
    entry_cost,
    first_expiry,
    net_greeks,
    profit,
    profit_at_expiry,
    profit_range,
    value,
)
from quantcore import black_scholes as bs

SPOT, RATE = 166.0, 0.04


def premium(kind, strike, days, vol):
    return float(bs.price(SPOT, strike, days / 365.0, vol, RATE, kind=kind))


def position(*legs):
    return Position(spot=SPOT, rate=RATE, legs=legs)


def test_long_call():
    pos = position(Leg("call", 10, strike=180, days=30, vol=0.64))
    paid = premium("call", 180, 30, 0.64)
    cost = paid * 1000

    assert entry_cost(pos) == pytest.approx(cost)
    assert float(profit(pos, SPOT)) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(profit_at_expiry(pos, [150.0, 180.0, 200.0]), [-cost, -cost, 20_000 - cost])
    assert breakevens(pos) == pytest.approx([180 + paid], abs=1e-6)
    best, worst = profit_range(pos)
    assert best == math.inf
    assert worst == pytest.approx(-cost)


def test_bull_call_spread():
    pos = position(Leg("call", 1, strike=170, days=30, vol=0.60), Leg("call", -1, strike=190, days=30, vol=0.58))
    debit = (premium("call", 170, 30, 0.60) - premium("call", 190, 30, 0.58)) * 100

    assert entry_cost(pos) == pytest.approx(debit)
    assert breakevens(pos) == pytest.approx([170 + debit / 100], abs=1e-6)
    best, worst = profit_range(pos)
    assert best == pytest.approx(2000 - debit)
    assert worst == pytest.approx(-debit)


def test_short_put_receives_a_credit_and_has_a_large_but_limited_loss():
    pos = position(Leg("put", -1, strike=150, days=30, vol=0.70))
    credit = premium("put", 150, 30, 0.70) * 100

    assert entry_cost(pos) == pytest.approx(-credit)
    assert breakevens(pos) == pytest.approx([150 - credit / 100], abs=1e-6)
    best, worst = profit_range(pos)
    assert best == pytest.approx(credit)
    assert worst == pytest.approx(-(15_000 - credit), abs=0.05)


def test_long_straddle_has_two_breakevens():
    pos = position(Leg("call", 1, strike=165, days=30, vol=0.60), Leg("put", 1, strike=165, days=30, vol=0.60))
    paid = premium("call", 165, 30, 0.60) + premium("put", 165, 30, 0.60)

    assert breakevens(pos) == pytest.approx([165 - paid, 165 + paid], abs=1e-6)
    best, worst = profit_range(pos)
    assert best == math.inf
    assert worst == pytest.approx(-paid * 100)


def test_iron_condor():
    pos = position(
        Leg("put", 1, strike=140, days=30, vol=0.70),
        Leg("put", -1, strike=150, days=30, vol=0.66),
        Leg("call", -1, strike=180, days=30, vol=0.62),
        Leg("call", 1, strike=190, days=30, vol=0.60),
    )
    credit = -entry_cost(pos)

    assert credit > 0
    assert breakevens(pos) == pytest.approx([150 - credit / 100, 180 + credit / 100], abs=1e-6)
    best, worst = profit_range(pos)
    assert best == pytest.approx(credit)
    assert worst == pytest.approx(-(1000 - credit))


def test_covered_call_caps_the_upside():
    pos = position(Leg("stock", 100), Leg("call", -1, strike=180, days=30, vol=0.64))
    credit = premium("call", 180, 30, 0.64) * 100

    assert entry_cost(pos) == pytest.approx(SPOT * 100 - credit)
    best, worst = profit_range(pos)
    assert best == pytest.approx((180 - SPOT) * 100 + credit)
    assert worst == pytest.approx(-(SPOT * 100 - credit), abs=0.05)


def test_stock_only_position():
    pos = position(Leg("stock", 100))

    assert first_expiry(pos) is None
    assert float(profit(pos, 170.0)) == pytest.approx(400.0)
    assert breakevens(pos) == pytest.approx([SPOT], abs=1e-6)
    best, worst = profit_range(pos)
    assert best == math.inf
    assert worst == pytest.approx(-SPOT * 100, abs=0.05)


def test_short_call_has_unlimited_loss():
    best, worst = profit_range(position(Leg("call", -1, strike=180, days=30, vol=0.64)))
    assert worst == -math.inf
    assert best == pytest.approx(premium("call", 180, 30, 0.64) * 100)


def test_a_stated_entry_price_replaces_the_model_price():
    pos = position(Leg("call", 2, strike=180, days=30, vol=0.64, entry_price=5.00))
    assert entry_cost(pos) == pytest.approx(1000.0)


def test_calendar_spread_runs_to_the_first_expiry_and_peaks_near_the_strike():
    pos = position(Leg("call", -1, strike=170, days=30, vol=0.60), Leg("call", 1, strike=170, days=60, vol=0.58))
    prices = np.linspace(120, 220, 401)

    assert first_expiry(pos) == 30
    assert abs(prices[np.argmax(profit_at_expiry(pos, prices))] - 170) < 3


def test_value_at_expiry_does_not_depend_on_volatility():
    pos = position(Leg("call", 1, strike=170, days=30, vol=0.60), Leg("put", -1, strike=150, days=30, vol=0.70))
    prices = np.linspace(120, 220, 11)
    np.testing.assert_allclose(value(pos, prices, 30, vol_scale=0.5), value(pos, prices, 30, vol_scale=2.0))


def test_net_greeks_match_small_moves_in_the_position_value():
    legs = (
        Leg("call", 3, strike=170, days=45, vol=0.60),
        Leg("put", -2, strike=150, days=45, vol=0.70),
        Leg("stock", 50),
    )
    pos = position(*legs)
    greeks = net_greeks(pos, days_forward=5.0)

    h = 1e-3
    d_value = (value(pos, SPOT + h, 5.0) - value(pos, SPOT - h, 5.0)) / (2 * h)
    assert float(greeks["delta"]) == pytest.approx(float(d_value), abs=1e-4)

    d_delta = (net_greeks(pos, SPOT + h, 5.0)["delta"] - net_greeks(pos, SPOT - h, 5.0)["delta"]) / (2 * h)
    assert float(greeks["gamma"]) == pytest.approx(float(d_delta), abs=1e-5)

    one_day = (value(pos, SPOT, 5.0 + h) - value(pos, SPOT, 5.0 - h)) / (2 * h)
    assert float(greeks["theta_per_day"]) == pytest.approx(float(one_day), abs=1e-4)

    def with_vols_moved(by):
        moved = tuple(Leg(l.kind, l.quantity, l.strike, l.days, l.vol + by, l.multiplier) if l.is_option else l for l in legs)
        return position(*moved)

    bump = 1e-5
    per_unit = (value(with_vols_moved(bump), SPOT, 5.0) - value(with_vols_moved(-bump), SPOT, 5.0)) / (2 * bump)
    assert float(greeks["vega_per_point"]) == pytest.approx(float(per_unit) / 100, abs=1e-4)


@pytest.mark.parametrize(
    "make",
    [
        lambda: Leg("swap", 1),
        lambda: Leg("call", 0, strike=100, days=30, vol=0.3),
        lambda: Leg("call", 1, days=30, vol=0.3),
        lambda: Leg("put", 1, strike=100, vol=0.3),
        lambda: Leg("put", 1, strike=100, days=30),
        lambda: Leg("call", 1, strike=100, days=30, vol=0.3, entry_price=-1.0),
        lambda: Position(spot=100.0, legs=[]),
        lambda: Position(spot=0.0, legs=[Leg("stock", 1)]),
    ],
)
def test_invalid_legs_and_positions_are_rejected(make):
    with pytest.raises(ValueError):
        make()
