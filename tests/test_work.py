"""Show your work: each intermediate value must rebuild the engine's own headline numbers."""

import math

import numpy as np
import pytest

from quantcore import Leg, Position, entry_cost, net_greeks
from quantcore import black_scholes as bs
from quantcore.probability import chance_of_profit, representative_vol
from quantcore.work import show_work

SPOT, RATE, Q = 166.0, 0.04, 0.01
SPREAD = Position(
    spot=SPOT,
    rate=RATE,
    div_yield=Q,
    legs=[
        Leg("call", 10, strike=170, days=30, vol=0.64),
        Leg("call", -10, strike=190, days=45, vol=0.58, entry_price=3.10),
        Leg("stock", 200),
    ],
)
AMERICAN_PUT = Position(spot=100.0, rate=0.05, legs=[Leg("put", 1, strike=110, days=180, vol=0.3)], exercise="american")


def test_leg_steps_rebuild_the_black_scholes_price():
    for leg in show_work(SPREAD).legs[:2]:
        sign = 1.0 if leg.kind == "call" else -1.0
        assert leg.years == pytest.approx(leg.days / 365)
        assert leg.d2 == pytest.approx(leg.d1 - leg.vol * math.sqrt(leg.years))
        rebuilt = sign * (SPOT * leg.spot_discount * leg.n_d1 - leg.strike * leg.strike_discount * leg.n_d2)
        assert rebuilt == pytest.approx(leg.european_price, abs=1e-12)
        assert leg.european_price == pytest.approx(
            float(bs.price(SPOT, leg.strike, leg.years, leg.vol, RATE, Q, leg.kind)), abs=1e-12
        )


def test_entry_prices_and_costs_add_up():
    work = show_work(SPREAD)
    bought, sold, stock = work.legs
    assert not bought.entry_price_stated and bought.entry_price == pytest.approx(bought.european_price)
    assert sold.entry_price_stated and sold.entry_price == 3.10
    assert stock.entry_price == SPOT and stock.size == 200
    assert sum(leg.cost for leg in work.legs) == pytest.approx(work.entry_cost)
    assert work.entry_cost == pytest.approx(entry_cost(SPREAD))


def test_leg_greeks_sum_to_the_position_greeks():
    work = show_work(SPREAD)
    engine = net_greeks(SPREAD)
    for name in ("delta", "gamma", "theta_per_day", "vega_per_point"):
        assert sum(leg.contribution[name] for leg in work.legs) == pytest.approx(float(engine[name]), rel=1e-12, abs=1e-12)
        assert work.greeks[name] == pytest.approx(float(engine[name]))
    first = work.legs[0]
    assert first.greeks["delta"] == pytest.approx(float(bs.delta(SPOT, 170, 30 / 365, 0.64, RATE, Q, "call")))
    assert first.contribution["delta"] == pytest.approx(1000 * first.greeks["delta"])


def test_chance_of_profit_pieces():
    chance = show_work(SPREAD).chance
    assert chance.days == 30
    assert chance.vol == pytest.approx(representative_vol(SPREAD))
    assert sum(p.probability for p in chance.intervals) == pytest.approx(1.0)
    assert chance.chance == pytest.approx(chance_of_profit(SPREAD))
    # The weighted average of the leg volatilities is the volatility used.
    vols = np.array([v for _, v, _ in chance.vol_weights])
    weights = np.array([w for _, _, w in chance.vol_weights])
    assert np.average(vols, weights=weights) == pytest.approx(chance.vol)
    assert chance.mean_log == pytest.approx(math.log(SPOT) + (RATE - Q - chance.vol**2 / 2) * 30 / 365)
    assert chance.sd_log == pytest.approx(chance.vol * math.sqrt(30 / 365))


def test_american_legs_show_both_values():
    leg = show_work(AMERICAN_PUT).legs[0]
    assert leg.american_price > leg.european_price  # a deep in-the-money put is worth exercising early
    assert leg.entry_price == pytest.approx(leg.american_price)


def test_stock_only_has_no_chance_section():
    work = show_work(Position(spot=SPOT, legs=[Leg("stock", 100)]))
    assert work.chance is None
    assert work.greeks["delta"] == 100
