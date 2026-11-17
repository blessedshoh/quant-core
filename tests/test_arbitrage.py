"""The chain, the arbitrage rules and the solver.

The strongest test is the definition itself: every trade reported must cost
nothing or pay you today, and never lose at expiry.
"""

import math

import numpy as np
import pytest

from quantcore import chain as ch
from quantcore.arbitrage import Quote, check, check_rules, solve, trade_cost, trade_payoff

SPOT, DAYS, RATE, DIV = 166.0, 30, 0.04, 0.01
PRICES = np.linspace(0.0, 600.0, 6001)


def quotes_from(chain, spread=0.02, overrides=None):
    """Bid and ask around each model price, at least a cent apart."""
    overrides = overrides or {}
    out = []
    for k, c, p in zip(chain.strikes, chain.calls, chain.puts):
        half_c, half_p = max(c * spread / 2, 0.005), max(p * spread / 2, 0.005)
        q = dict(strike=float(k), call_bid=max(c - half_c, 0.0), call_ask=c + half_c, put_bid=max(p - half_p, 0.0), put_ask=p + half_p)
        q.update(overrides.get(float(k), {}))
        out.append(Quote(**q))
    return out


def assert_is_an_arbitrage(trade, profit):
    """Pays at least `profit` today (in today's money) and never loses at expiry."""
    cost = trade_cost(trade, SPOT)
    assert cost <= -profit + 1e-6
    payoff = trade_payoff(trade, PRICES, RATE, DIV, DAYS)
    assert payoff.min() >= -1e-6


# --- the chain -------------------------------------------------------------------


def test_default_strikes_sit_on_the_listed_grid():
    strikes = ch.default_strikes(166.0)
    assert strikes[0] == 125 and strikes[-1] == 205
    assert np.all(np.diff(strikes) == 5)


def test_a_flat_smile_prices_like_black_scholes_and_obeys_parity():
    chain = ch.model_chain(SPOT, DAYS, 0.4, rate=RATE, div_yield=DIV)
    years = DAYS / 365
    forward_value = SPOT * math.exp(-DIV * years) - chain.strikes * math.exp(-RATE * years)
    np.testing.assert_allclose(chain.calls - chain.puts, forward_value, atol=1e-9)
    np.testing.assert_allclose(chain.vols, 0.4)


def test_negative_skew_makes_low_strikes_dearer_in_volatility():
    chain = ch.model_chain(SPOT, DAYS, 0.4, skew=-0.3)
    assert np.all(np.diff(chain.vols) < 0)


def test_volatility_never_goes_below_one_percent():
    chain = ch.model_chain(SPOT, DAYS, 0.2, curvature=-50.0)
    assert chain.vols.min() >= 0.01


# --- clean prices: nothing found ---------------------------------------------------


@pytest.mark.parametrize("skew, curvature", [(0.0, 0.0), (-0.3, 0.8), (-0.6, 2.0)])
def test_a_sensible_smile_has_no_arbitrage(skew, curvature):
    chain = ch.model_chain(SPOT, DAYS, 0.45, skew=skew, curvature=curvature, rate=RATE, div_yield=DIV)
    report = check(SPOT, DAYS, quotes_from(chain), RATE, DIV)
    assert report.violations == ()
    assert not report.solver.arbitrage
    assert report.solver.credit == 0.0


def test_zero_spread_model_prices_are_exactly_consistent():
    chain = ch.model_chain(SPOT, DAYS, 0.45, skew=-0.2, rate=RATE, div_yield=DIV)
    exact = [Quote(float(k), c, c, p, p) for k, c, p in zip(chain.strikes, chain.calls, chain.puts)]
    assert not solve(SPOT, DAYS, exact, RATE, DIV).arbitrage


# --- each rule, triggered on purpose ---------------------------------------------


def broken(overrides):
    chain = ch.model_chain(SPOT, DAYS, 0.45, rate=RATE, div_yield=DIV)
    return quotes_from(chain, overrides=overrides)


def found(quotes, rule):
    report = check(SPOT, DAYS, quotes, RATE, DIV)
    hits = [v for v in report.violations if v.rule == rule]
    assert hits, f"no {rule} violation found"
    for violation in hits:
        assert_is_an_arbitrage(violation.trade, violation.profit)
    return hits, report


def test_a_call_bid_above_the_stock_price():
    hits, _ = found(broken({150.0: {"call_bid": 175.0, "call_ask": 176.0}}), "bound")
    assert any("costs more than the stock" in v.title for v in hits)


def test_a_put_ask_below_its_intrinsic_value():
    hits, _ = found(broken({200.0: {"put_bid": 20.0, "put_ask": 21.0}}), "bound")
    assert any("Put 200" in v.title for v in hits)


def test_put_call_parity_both_ways():
    rich_call, _ = found(broken({165.0: {"call_bid": 20.0, "call_ask": 20.2}}), "parity")
    assert any("conversion" in v.detail for v in rich_call)
    cheap_call, _ = found(broken({165.0: {"call_bid": 3.0, "call_ask": 3.1}}), "parity")
    assert any("reversal" in v.detail for v in cheap_call)


def test_a_higher_strike_call_priced_above_a_lower_one():
    chain = ch.model_chain(SPOT, DAYS, 0.45, rate=RATE, div_yield=DIV)
    c165 = float(chain.calls[list(chain.strikes).index(165.0)])
    hits, _ = found(broken({170.0: {"call_bid": c165 + 0.5, "call_ask": c165 + 0.6}}), "order")
    assert hits[0].strikes == (165.0, 170.0)


def test_a_call_spread_worth_more_than_its_width():
    chain = ch.model_chain(SPOT, DAYS, 0.45, rate=RATE, div_yield=DIV)
    c175 = float(chain.calls[list(chain.strikes).index(175.0)])
    found(broken({170.0: {"call_bid": c175 + 5.5, "call_ask": c175 + 5.6}}), "spread")


def test_an_extreme_smile_breaks_butterflies_and_the_solver_agrees():
    chain = ch.model_chain(SPOT, DAYS, 0.45, curvature=-25.0, rate=RATE, div_yield=DIV)
    hits, report = found(quotes_from(chain), "butterfly")
    for violation in hits:
        quantities = [abs(leg.quantity) for leg in violation.trade]
        assert min(quantities) == pytest.approx(1.0)
    assert report.solver.arbitrage
    assert report.solver.worst_payoff >= -1e-6
    assert_is_an_arbitrage(report.solver.trade, report.solver.credit)
    assert report.solver.credit >= max(v.profit for v in hits) - 1e-6


def test_the_solver_finds_arbitrage_the_rules_miss():
    """Wide bid-ask spreads at 110 and 130 make every adjacent butterfly look fine, but the
    wide 100/120/140 butterfly skips those strikes and still pays you to hold it."""
    quotes = [
        Quote(100.0, 29.9, 30.0),
        Quote(110.0, 20.0, 26.0),
        Quote(120.0, 15.5, 15.6),
        Quote(130.0, 2.0, 9.0),
        Quote(140.0, 0.4, 0.5),
    ]
    assert check_rules(125.0, 30, quotes) == []
    result = solve(125.0, 30, quotes)
    assert result.arbitrage
    assert result.credit == pytest.approx(15.5 - 0.5 * 30.0 - 0.5 * 0.5, abs=1e-6)  # half a 100/120/140 butterfly
    assert trade_payoff(result.trade, PRICES).min() >= -1e-6
    assert trade_cost(result.trade, 125.0) == pytest.approx(-result.credit, abs=1e-6)


def test_american_options_use_immediate_exercise_and_skip_the_solver():
    chain = ch.model_chain(SPOT, DAYS, 0.45, rate=RATE)
    k = 200.0
    quotes = quotes_from(chain, overrides={k: {"put_bid": 33.0, "put_ask": 33.5}})  # below K - S = 34
    european = check(SPOT, DAYS, quotes, RATE, 0.0, american=False)
    american = check(SPOT, DAYS, quotes, RATE, 0.0, american=True)
    assert american.solver is None
    assert any("Put 200" in v.title for v in american.violations)
    assert any("American" in line for line in american.assumptions)
    assert len(american.violations) >= len([v for v in european.violations if "Put 200" in v.title])


def test_bad_quotes_are_refused():
    with pytest.raises(ValueError):
        Quote(100.0, call_bid=5.0, call_ask=4.0)
    with pytest.raises(ValueError):
        Quote(100.0, put_ask=-1.0)
    with pytest.raises(ValueError):
        check(SPOT, DAYS, [], RATE)
