"""Checks exercises/earnings.py against the engine."""

import math

import numpy as np
import pytest

from exercises import earnings as mine
from quantcore import Event, Leg, Position
from quantcore import earnings as engine
from quantcore.position import base_vol, effective_vol


def engine_leg(vol, days, move, event_days=1.0):
    leg = Leg("call", 1, strike=100, days=days, vol=vol)
    return leg, Position(spot=100.0, legs=[leg], event=Event(event_days, move))


@pytest.mark.parametrize("seed", range(20))
def test_base_vol(seed):
    rng = np.random.default_rng(seed)
    vol, days, move = rng.uniform(0.4, 1.2), rng.uniform(10, 120), rng.uniform(0.01, 0.08)
    leg, pos = engine_leg(vol, days, move)
    assert mine.base_vol(vol, days, move) == pytest.approx(base_vol(leg, pos), rel=1e-12)


@pytest.mark.edge
def test_base_vol_refuses_a_move_too_big():
    with pytest.raises(ValueError):
        mine.base_vol(0.25, 10, 0.08)


@pytest.mark.parametrize("seed", range(20))
def test_effective_vol(seed):
    rng = np.random.default_rng(100 + seed)
    vol, days, move = rng.uniform(0.5, 1.2), rng.uniform(20, 120), rng.uniform(0.01, 0.06)
    event_days = rng.uniform(1, 15)
    leg, pos = engine_leg(vol, days, move, event_days)
    b = base_vol(leg, pos)
    for day in (0.0, event_days / 2, event_days, event_days + 2):
        expected = float(effective_vol(leg, pos, day))
        ours = mine.effective_vol(b, days - day, move, day < event_days)
        assert ours == pytest.approx(expected, rel=1e-12), day


def test_effective_vol_today_is_the_entered_vol():
    b = mine.base_vol(0.7, 30, 0.08)
    assert mine.effective_vol(b, 30, 0.08, True) == pytest.approx(0.7, rel=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_implied_move(seed):
    rng = np.random.default_rng(200 + seed)
    b, m = rng.uniform(0.2, 0.6), rng.uniform(0.02, 0.12)
    d1, d2 = rng.uniform(2, 20), rng.uniform(25, 90)
    near = math.sqrt(b * b + m * m / (d1 / 365))
    far = math.sqrt(b * b + m * m / (d2 / 365))
    move, base = mine.implied_move(d1, near, d2, far)
    expected = engine.implied_move(d1, near, d2, far)
    assert move == pytest.approx(expected.move, rel=1e-9)
    assert base == pytest.approx(expected.base_vol, rel=1e-9)


@pytest.mark.edge
def test_implied_move_refusals():
    with pytest.raises(ValueError):
        mine.implied_move(7, 0.3, 35, 0.5)  # the later expiry is dearer: no earnings premium
    with pytest.raises(ValueError):
        mine.implied_move(35, 0.5, 7, 0.3)  # expiries the wrong way round


def test_expected_abs_move():
    for m in (0.02, 0.08, 0.15):
        assert mine.expected_abs_move(m) == pytest.approx(engine.implied_move(7, math.sqrt(0.09 + m * m * 365 / 7), 35, math.sqrt(0.09 + m * m * 365 / 35)).expected_abs_move, rel=1e-9)
