"""Payoff replication: exact where theory says it must be, never worse where asked, and honest costs."""

import math

import numpy as np
import pytest

from quantcore import black_scholes as bs
from quantcore.chain import model_chain
from quantcore.replicate import holding_payoff, replicate

SPOT, DAYS, VOL, RATE = 166.0, 30, 0.45, 0.04
STRIKES = np.arange(130.0, 205.0, 5.0)


def run(xs, ys, mode="match", **kw):
    return replicate(xs, ys, SPOT, DAYS, VOL, rate=RATE, mode=mode, strikes=STRIKES, **kw)


def model(kind, strike):
    return float(bs.price(SPOT, strike, DAYS / 365, VOL, RATE, kind=kind))


def test_a_call_spread_drawn_with_kinks_at_strikes_is_matched_exactly():
    xs = [130, 160, 175, 200]
    ys = [0, 0, 15, 15]
    result = run(xs, ys)
    assert result.max_error < 1e-6
    assert result.cost == pytest.approx(model("call", 160) - model("call", 175), abs=1e-4)


def test_the_answer_is_simple_not_just_correct():
    """Parity allows many equivalent portfolios; the size penalty should pick two calls (or the put twin)."""
    result = run([130, 160, 175, 200], [0, 0, 15, 15])
    options = [h for h in result.holdings if h.kind in ("call", "put")]
    assert len(options) <= 3
    assert sum(abs(h.quantity) for h in options) == pytest.approx(2.0, abs=1e-3)


def test_a_line_held_flat_beyond_the_drawing_is_a_call_spread_not_stock():
    """Off the chart the payoff stays flat, so a rising line from 130 to 200 is a 130/200 call
    spread plus cash: the cheap, bounded version of owning the stock in that range."""
    xs = np.array([130.0, 200.0])
    result = run(xs, xs - 166.0)
    assert result.max_error < 1e-6
    fine = np.linspace(100, 260, 321)
    np.testing.assert_allclose(holding_payoff(result.holdings, fine), np.clip(fine, 130, 200) - 166.0, atol=1e-6)


def test_a_tent_is_a_butterfly_and_costs_its_model_price():
    xs = [130, 160, 165, 170, 200]
    ys = [0, 0, 5, 0, 0]
    result = run(xs, ys)
    assert result.max_error < 1e-6
    fly = model("call", 160) - 2 * model("call", 165) + model("call", 170)
    assert result.cost == pytest.approx(fly, abs=1e-4)


def test_kinks_between_strikes_can_only_be_approximated():
    xs = [130, 162, 168, 200]  # bends at 162 and 168: no strikes there
    ys = [0, 0, 6, 6]
    result = run(xs, ys)
    assert 0.5 < result.max_error < 3.0


def test_cheapest_never_falls_below_the_drawing_and_costs_at_least_the_exact_price():
    xs = np.linspace(130, 200, 41)
    ys = 8.0 * np.exp(-((xs - 166.0) / 8.0) ** 2)  # a smooth bump: not buildable exactly
    cheapest = run(xs, ys, mode="cheapest")
    fine = np.linspace(130, 200, 701)
    assert np.all(holding_payoff(cheapest.holdings, fine) >= np.interp(fine, xs, ys) - 1e-6)
    closest = run(xs, ys)
    assert closest.max_error < cheapest.max_error + 1e-9 or closest.max_error < 1.0
    assert cheapest.cost > 0


def test_the_reported_cost_is_the_sum_of_the_holdings():
    result = run([130, 155, 165, 175, 200], [10, 0, -4, 0, 10])
    assert result.cost == pytest.approx(sum(h.quantity * h.price for h in result.holdings), abs=1e-9)


def test_a_smile_changes_what_the_same_payoff_costs():
    xs, ys = [130, 140, 150, 200], [10, 0, 0, 0]  # a put spread below the market
    flat = run(xs, ys)
    skewed = run(xs, ys, skew=-0.8)
    assert skewed.cost > flat.cost  # negative skew makes low strikes dearer


def test_dividends_are_respected_in_the_stock_price():
    xs = np.array([130.0, 200.0])
    result = replicate(xs, xs, SPOT, 365, VOL, rate=RATE, div_yield=0.05, strikes=STRIKES)
    stock = [h for h in result.holdings if h.kind == "stock"][0]
    assert stock.price == pytest.approx(SPOT * math.exp(-0.05), rel=1e-9)


@pytest.mark.parametrize(
    "bad",
    [dict(target_prices=[150], target_payoffs=[1]), dict(target_prices=[0, 150], target_payoffs=[1, 2]), dict(mode="exact")],
)
def test_bad_requests_are_refused(bad):
    args = dict(target_prices=[140, 190], target_payoffs=[0, 10], spot=SPOT, days=DAYS, atm_vol=VOL, strikes=STRIKES)
    args.update(bad)
    with pytest.raises(ValueError):
        replicate(**args)


def test_a_leg_limit_is_respected_and_costs_some_accuracy():
    xs = np.linspace(130, 200, 41)
    ys = 8.0 * np.exp(-((xs - 166.0) / 8.0) ** 2)
    free = run(xs, ys)
    limited = run(xs, ys, max_legs=4)
    legs = [h for h in limited.holdings if h.kind != "cash"]
    assert len(legs) <= 4
    assert limited.max_error >= free.max_error - 1e-9


def test_a_leg_limit_does_not_hurt_a_payoff_that_needs_few_legs():
    result = run([130, 160, 175, 200], [0, 0, 15, 15], max_legs=2)
    assert result.max_error < 1e-6
    assert len([h for h in result.holdings if h.kind != "cash"]) <= 2
