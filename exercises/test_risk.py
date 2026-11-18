"""Checks exercises/risk.py against the engine, on fixed and random positions."""

import math

import numpy as np
import pytest

from exercises import risk as mine
from quantcore import Leg, Position, breakevens, entry_cost, entry_price, net_greeks, probability, scenarios
from quantcore import black_scholes as bs
from quantcore.position import profit_at_expiry

SPREAD = Position(
    spot=166.0,
    rate=0.04,
    div_yield=0.01,
    legs=[Leg("call", 10, strike=170, days=30, vol=0.64), Leg("call", -10, strike=190, days=30, vol=0.60)],
)
CONDOR = Position(
    spot=166.0,
    rate=0.04,
    legs=[
        Leg("put", 1, strike=140, days=30, vol=0.7),
        Leg("put", -1, strike=150, days=30, vol=0.66),
        Leg("call", -1, strike=180, days=30, vol=0.6),
        Leg("call", 1, strike=190, days=30, vol=0.58),
    ],
)
POSITIONS = [SPREAD, CONDOR]


def leg_greeks(position, leg):
    alone = Position(position.spot, (leg,), position.rate, position.div_yield)
    return {name: float(v) / leg.size for name, v in net_greeks(alone).items()}


@pytest.mark.parametrize("position", POSITIONS)
def test_entry_cost(position):
    sizes = [leg.size for leg in position.legs]
    prices = [entry_price(leg, position) for leg in position.legs]
    assert mine.entry_cost(sizes, prices) == pytest.approx(entry_cost(position))


@pytest.mark.parametrize("position", POSITIONS)
def test_net_greeks(position):
    sizes = [leg.size for leg in position.legs]
    ours = mine.net_greeks(sizes, [leg_greeks(position, leg) for leg in position.legs])
    for name, value in net_greeks(position).items():
        assert ours[name] == pytest.approx(float(value), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_log_price_moments(seed):
    rng = np.random.default_rng(seed)
    args = (rng.uniform(10, 500), rng.uniform(1, 400), rng.uniform(0.05, 1.5), rng.uniform(0, 0.1), rng.uniform(0, 0.05))
    drift = rng.uniform(-0.2, 0.3)
    assert mine.log_price_moments(*args) == pytest.approx(probability.log_price_moments(*args))
    assert mine.log_price_moments(*args, drift=drift) == pytest.approx(probability.log_price_moments(*args, drift=drift))


@pytest.mark.parametrize("seed", range(20))
def test_prob_between(seed):
    rng = np.random.default_rng(100 + seed)
    spot = rng.uniform(50, 300)
    low, high = sorted(rng.uniform(0.6, 1.5, 2) * spot)
    args = (spot, rng.uniform(5, 200), rng.uniform(0.1, 1.0), 0.03, 0.01)
    expected = probability.cdf([low, high], *args)
    assert mine.prob_between(low, high, *args) == pytest.approx(float(expected[1] - expected[0]), abs=1e-12)


@pytest.mark.edge
def test_prob_between_open_ends():
    args = (166.0, 30, 0.6, 0.04, 0.0)
    assert mine.prob_between(0.0, math.inf, *args) == pytest.approx(1.0)
    below = mine.prob_between(0.0, 166.0, *args)
    above = mine.prob_between(166.0, math.inf, *args)
    assert below + above == pytest.approx(1.0)
    assert below == pytest.approx(float(probability.cdf([166.0], *args)[0]))


@pytest.mark.parametrize("position", POSITIONS)
def test_chance_of_profit(position):
    edges = breakevens(position)
    pieces = probability.outcome_intervals(position)
    profitable = [piece.profitable for piece in pieces]
    vol = probability.representative_vol(position)
    days = 30
    ours = mine.chance_of_profit(edges, profitable, position.spot, days, vol, position.rate, position.div_yield)
    assert ours == pytest.approx(probability.chance_of_profit(position), abs=1e-12)
    # and with a view on the stock
    ours = mine.chance_of_profit(edges, profitable, position.spot, days, vol, position.rate, position.div_yield, drift=0.15)
    assert ours == pytest.approx(probability.chance_of_profit(position, drift=0.15), abs=1e-12)


@pytest.mark.parametrize("position", POSITIONS)
def test_representative_vol(position):
    rows = probability.vol_weights(position)
    legs = [position.legs[i] for i, _, _ in rows]
    vegas = [float(bs.vega(position.spot, leg.strike, leg.days / 365, leg.vol, position.rate, position.div_yield)) for leg in legs]
    ours = mine.representative_vol([leg.vol for leg in legs], [leg.size for leg in legs], vegas)
    assert ours == pytest.approx(probability.representative_vol(position))


@pytest.mark.edge
def test_representative_vol_with_no_weight():
    assert mine.representative_vol([0.2, 0.4], [1, 1], [0.0, 0.0]) == pytest.approx(0.3)


@pytest.mark.parametrize("move, days, shift", [(8.3, 0, 0.0), (-16.6, 7, 0.05), (4.0, 3, -0.02), (25.0, 14, 0.1)])
def test_explain(move, days, shift):
    greeks = {name: float(v) for name, v in net_greeks(SPREAD).items()}
    ours = mine.explain(greeks, move, days, shift)
    engine = scenarios.explain(SPREAD, move, days, shift)
    for name in ("delta", "gamma", "theta", "vega"):
        assert ours[name] == pytest.approx(getattr(engine, name), rel=1e-12, abs=1e-9)


def test_profit_flags_make_sense():
    """Not your code: a sanity check that the stretches used above really are profitable or not."""
    for position in POSITIONS:
        for piece in probability.outcome_intervals(position):
            mid = piece.high / 2 if piece.low == 0 else (piece.low * 2 if math.isinf(piece.high) else (piece.low + piece.high) / 2)
            assert (float(profit_at_expiry(position, mid)) > 0) == piece.profitable
