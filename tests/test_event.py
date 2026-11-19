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


# --- the grid, the implied-move calculator -------------------------------------

from quantcore.earnings import event_vols, implied_move  # noqa: E402
from quantcore.grid import EVENT_EPSILON_DAYS, day_axis, profit_grid  # noqa: E402


def test_day_axis_has_the_event_step():
    days = day_axis(straddle(EVENT))
    assert EVENT_EPSILON_DAYS == 1e-4
    assert np.any(np.isclose(days, 5 - 1e-4, atol=1e-12, rtol=0)) and np.any(days == 5)
    assert np.array_equal(day_axis(straddle()), 30 * (1 - (1 - np.linspace(0, 1, 25)) ** 3))


def test_event_after_first_expiry_adds_no_days():
    calendar = Position(
        spot=166.0, rate=R, legs=[Leg("call", -1, strike=170, days=3, vol=0.6), Leg("call", 1, strike=170, days=30, vol=0.6)], event=EVENT
    )
    days = day_axis(calendar)
    assert len(days) == 25 and days[-1] == 3


def test_american_grid_with_event():
    grid = profit_grid(straddle(EVENT, exercise="american"))
    assert np.any(grid.days == 5)
    assert np.all(np.isfinite(grid.profit))


def test_event_vols_report_each_leg():
    rows = event_vols(straddle(EVENT))
    assert [r[0] for r in rows] == [0, 1]
    index, base, before, after, weight = rows[0]
    assert before == 0.70 and after == base == base_vol(STRADDLE_LEGS[0], straddle(EVENT)) and weight > 0
    assert event_vols(straddle()) == []


def test_implied_move_round_trip():
    b, m, t1, t2 = 0.35, 0.07, 7 / 365, 35 / 365
    near, far = math.sqrt(b * b + m * m / t1), math.sqrt(b * b + m * m / t2)
    result = implied_move(7, near, 35, far)
    assert result.move == pytest.approx(m, abs=1e-10)
    assert result.base_vol == pytest.approx(b, abs=1e-10)
    assert result.expected_abs_move == pytest.approx(m * math.sqrt(2 / math.pi), abs=1e-12)
    assert result.event_variance_share == pytest.approx(m * m / (near * near * t1), abs=1e-10)


@pytest.mark.parametrize("args", [(35, 0.4, 7, 0.5), (7, 0.3, 35, 0.5), (7, 0.9, 35, 0.1)])
def test_implied_move_refusals(args):
    with pytest.raises(ValueError):
        implied_move(*args)


# --- Greeks of European event legs: exact formulas, no bumping (final review) ---


def _per_share(pos, name, day=0.0):
    return float(net_greeks(pos, days_forward=day)[name]) / 100  # one contract per leg; sum of call and put


def test_event_vega_and_theta_follow_the_split():
    pos = straddle(EVENT)
    b, s, T, S, K = base_vol(STRADDLE_LEGS[0], pos), 0.70, 30 / 365, 166.0, 165
    vega = sum(float(bs.vega(S, K, T, s, R, 0.0)) for _ in ("call", "put")) * b / s / 100
    gamma = float(bs.gamma(S, K, T, s, R, 0.0))
    # Only everyday variance decays while the jump is still ahead: dV/d(total variance) = S^2 gamma / 2.
    theta = sum(float(bs.theta(S, K, T, s, R, 0.0, k)) + 0.5 * S * S * gamma * (s * s - b * b) for k in ("call", "put")) / 365
    assert _per_share(pos, "vega_per_point") == pytest.approx(vega, rel=1e-9)
    assert _per_share(pos, "theta_per_day") == pytest.approx(theta, rel=1e-9)


def test_event_greeks_at_expiry_have_no_spike():
    pos = Position(spot=150.0, rate=R, legs=(Leg("call", 1, strike=150, days=30, vol=0.7), Leg("put", 1, strike=150, days=30, vol=0.7)), event=EVENT)
    g = net_greeks(pos, days_forward=30.0)
    assert float(g["gamma"]) == 0.0
    assert float(g["theta_per_day"]) == 0.0


def test_event_theta_near_expiry_is_exact():
    pos = straddle(EVENT)
    b = base_vol(STRADDLE_LEGS[0], pos)
    for day in (29.94, 29.998):
        tau = (30 - day) / 365
        expected = sum(float(bs.theta(166.0, 165, tau, b, R, 0.0, k)) for k in ("call", "put")) / 365
        assert _per_share(pos, "theta_per_day", day) == pytest.approx(expected, rel=1e-9)
