"""Independent check: the engine against QuantLib, the industry's open-source pricing library.

QuantLib works with calendar dates, so every case here uses a whole number of
days and the Actual/365 day count, which makes its time to expiry identical to
ours (days / 365).
"""

import itertools

import pytest

from quantcore import black_scholes as bs

ql = pytest.importorskip("QuantLib", reason="install with: pip install QuantLib")

TODAY = ql.Date(16, 11, 2026)
DAY_COUNT = ql.Actual365Fixed()

CASES = list(
    itertools.product(
        (80.0, 100.0, 166.0, 250.0),  # spot
        (90.0, 100.0, 180.0),  # strike
        (4, 30, 180, 730),  # days
        (0.12, 0.35, 0.64, 1.10),  # vol
        (0.0, 0.04),  # rate
        (0.0, 0.02),  # dividend yield
        ("call", "put"),
    )
)


def quantlib_option(spot, strike, days, vol, rate, div_yield, kind):
    ql.Settings.instance().evaluationDate = TODAY
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, strike)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(TODAY + int(days)))
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, div_yield, DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, rate, DAY_COUNT)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), vol, DAY_COUNT)),
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


def test_the_grid_of_cases_is_large():
    assert len(CASES) == 1536


@pytest.mark.parametrize("spot, strike, days, vol, rate, div_yield, kind", CASES)
def test_price_and_greeks_agree_with_quantlib(spot, strike, days, vol, rate, div_yield, kind):
    reference = quantlib_option(spot, strike, days, vol, rate, div_yield, kind)
    market = (spot, strike, days / 365.0, vol, rate, div_yield)
    tolerance = dict(abs=1e-6, rel=1e-9)

    assert float(bs.price(*market, kind)) == pytest.approx(reference.NPV(), **tolerance)
    assert float(bs.delta(*market, kind)) == pytest.approx(reference.delta(), **tolerance)
    assert float(bs.gamma(*market)) == pytest.approx(reference.gamma(), **tolerance)
    assert float(bs.vega(*market)) == pytest.approx(reference.vega(), **tolerance)
    assert float(bs.theta(*market, kind)) == pytest.approx(reference.theta(), **tolerance)
    assert float(bs.rho(*market, kind)) == pytest.approx(reference.rho(), **tolerance)


# --- American options: our Leisen-Reimer tree against QuantLib's -----------------

AMERICAN_CASES = list(
    itertools.product((80.0, 100.0, 125.0), (90.0, 110.0), (30, 365), (0.2, 0.5), (0.0, 0.05), (0.0, 0.03), ("call", "put"))
)


@pytest.mark.parametrize("spot, strike, days, vol, rate, div_yield, kind", AMERICAN_CASES)
def test_american_tree_agrees_with_quantlib(spot, strike, days, vol, rate, div_yield, kind):
    from quantcore import binomial

    ql.Settings.instance().evaluationDate = TODAY
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, strike)
    option = ql.VanillaOption(payoff, ql.AmericanExercise(TODAY, TODAY + int(days)))
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, div_yield, DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, rate, DAY_COUNT)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), vol, DAY_COUNT)),
    )
    option.setPricingEngine(ql.BinomialVanillaEngine(process, "lr", 101))
    ours = binomial.price(spot, strike, days / 365.0, vol, rate, div_yield, kind, american=True, steps=101)
    assert ours == pytest.approx(option.NPV(), abs=1e-8)
