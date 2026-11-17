"""Payoff in, portfolio out: build a drawn payoff from listed options.

The idea
--------
A payoff at expiry that is a straight line between listed strikes can be built
EXACTLY from cash, the stock and calls: the cash sets its height at zero, the
stock its starting slope, and at every strike where the slope changes you buy
(or sell) calls equal to the change in slope. Any other shape can only be
approximated, and this module finds the best approximation.

Two questions, both linear programmes over the listed calls and puts, the stock
and cash, each priced from the model chain:

match       Which portfolio comes closest? Minimises the largest gap between its
            payoff and the drawn one, then prefers fewer and smaller positions.
cheapest    Which portfolio is never worse than the drawing, for the least money?
            Its payoff must be at or above the drawn one everywhere.

Calls and puts at one strike are redundant (put-call parity turns one into the
other with stock and cash), so a small penalty on total position size picks the
simplest of the equivalent answers.

Outside the drawn range the payoff is held flat, so a portfolio cannot win by
doing something extreme off the edge of the chart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from . import black_scholes as bs
from .chain import default_strikes, model_chain, strike_step

MODES = ("match", "cheapest")
SIMPLICITY = 1e-4  # weight on total position size: only breaks ties
MAX_QUANTITY = 100.0  # per share, for any single instrument
LEG_LIMIT_BOUND = 25.0  # per share, for positions chosen under a leg limit (a tighter bound solves faster)


@dataclass(frozen=True)
class Holding:
    kind: str  # "call", "put", "stock" or "cash"
    quantity: float  # per share. Stock: shares held at expiry. Cash: the amount received at expiry.
    strike: float | None
    price: float  # cost today per unit. Stock: spot x e^(-qT), since dividends top the holding up. Cash: e^(-rT).


@dataclass(frozen=True)
class Replication:
    mode: str
    holdings: tuple[Holding, ...]
    cost: float  # today, per share; negative means it pays you
    max_error: float  # largest gap from the drawing inside the drawn range
    prices: np.ndarray  # where the fit is reported
    target: np.ndarray
    fitted: np.ndarray


def holding_payoff(holdings, prices) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    total = np.zeros_like(prices)
    for h in holdings:
        if h.kind == "call":
            total += h.quantity * np.maximum(prices - h.strike, 0.0)
        elif h.kind == "put":
            total += h.quantity * np.maximum(h.strike - prices, 0.0)
        elif h.kind == "stock":
            total += h.quantity * prices
        elif h.kind == "cash":
            total += h.quantity
    return total


def replicate(
    target_prices,
    target_payoffs,
    spot: float,
    days: float,
    atm_vol: float,
    skew: float = 0.0,
    curvature: float = 0.0,
    rate: float = 0.0,
    div_yield: float = 0.0,
    mode: str = "match",
    strikes=None,
    max_legs: int | None = None,
) -> Replication:
    """Best portfolio of listed options, stock and cash for a drawn expiry payoff (per share).

    max_legs   at most this many option and stock positions (cash is free). This turns
               the problem into a mixed-integer programme: one yes/no choice per
               instrument, solved exactly by the same HiGHS solver.
    """
    if max_legs is not None and max_legs < 1:
        raise ValueError("max_legs must be at least 1")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    xs = np.asarray(target_prices, dtype=float)
    ys = np.asarray(target_payoffs, dtype=float)
    if xs.ndim != 1 or xs.size < 2 or xs.size != ys.size:
        raise ValueError("the drawing needs at least two points, as matching price and payoff lists")
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if np.any(xs <= 0) or not np.all(np.isfinite(ys)):
        raise ValueError("prices must be positive and payoffs finite")

    low, high = float(xs[0]), float(xs[-1])
    if strikes is None:
        # Listed strikes across the drawing plus one either side: options further out
        # cannot change the payoff inside the drawn range.
        step = strike_step(spot)
        listed = default_strikes(spot, 0.6, step)
        strikes = listed[(listed >= low - step) & (listed <= high + step)]
        if strikes.size < 2:
            strikes = listed
    else:
        strikes = np.asarray(strikes, dtype=float)
    chain = model_chain(spot, days, atm_vol, skew, curvature, rate, div_yield, strikes)
    years = days / bs.DAYS_PER_YEAR
    disc = math.exp(-rate * years)
    carry = math.exp(-div_yield * years)

    far = 3.0 * max(high, float(strikes.max()))
    nodes = np.unique(np.concatenate([[0.0], strikes, xs, [far]]))

    def target_at(prices):
        return np.interp(prices, xs, ys)  # flat beyond the drawn ends

    # Instruments: payoff at each node, slope beyond the last node, price today.
    names, payoffs, slopes, costs = [], [], [], []
    for k, c, p in zip(chain.strikes, chain.calls, chain.puts):
        names.append(("call", float(k), float(c)))
        payoffs.append(np.maximum(nodes - k, 0.0))
        slopes.append(1.0)
        costs.append(float(c))
        names.append(("put", float(k), float(p)))
        payoffs.append(np.maximum(k - nodes, 0.0))
        slopes.append(0.0)
        costs.append(float(p))
    names.append(("stock", None, spot * carry))  # e^(-qT) shares today become one share at expiry
    payoffs.append(nodes.copy())
    slopes.append(1.0)
    costs.append(spot * carry)
    names.append(("cash", None, disc))
    payoffs.append(np.ones_like(nodes))
    slopes.append(0.0)
    costs.append(disc)

    n = len(names)
    pay = np.array(payoffs).T  # nodes x instruments
    slope = np.array(slopes)
    cost = np.array(costs)
    goal = target_at(nodes)

    # Each quantity q = plus - minus, both >= 0, so total size is sum(plus + minus).
    # Variables: [plus (n), minus (n), error (1)].
    def blocks(matrix):
        return np.hstack([matrix, -matrix])

    size_penalty = SIMPLICITY * np.ones(2 * n)
    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    # Beyond the last node the payoff stays flat, like the drawing.
    a_eq.append(np.concatenate([blocks(slope[None, :])[0], [0.0]]))
    b_eq.append(0.0)
    if mode == "match":
        # |payoff - goal| <= error inside the drawing; outside it, payoff follows the flat extension too.
        objective = np.concatenate([size_penalty, [1.0]])
        for row, target_value in zip(pay, goal):
            a_ub.append(np.concatenate([blocks(row[None, :])[0], [-1.0]]))
            b_ub.append(target_value)
            a_ub.append(np.concatenate([-blocks(row[None, :])[0], [-1.0]]))
            b_ub.append(-target_value)
    else:
        # payoff >= goal everywhere; minimise what it costs today.
        objective = np.concatenate([np.concatenate([cost, -cost]) + size_penalty, [0.0]])
        for row, target_value in zip(pay, goal):
            a_ub.append(np.concatenate([-blocks(row[None, :])[0], [0.0]]))
            b_ub.append(-target_value)

    cash_index = n - 1
    cash_room = 10.0 * (abs(ys).max() + far)
    upper = [MAX_QUANTITY] * (2 * n) + [np.inf]
    upper[cash_index] = upper[n + cash_index] = cash_room
    a_ub, a_eq = np.array(a_ub), np.array(a_eq)
    integrality = np.zeros(2 * n + 1)

    if max_legs is not None:
        # One binary "use it" switch per option and the stock; a position can only be
        # non-zero when its switch is on, and at most max_legs switches may be on.
        m = n - 1  # every instrument except cash
        a_ub = np.hstack([a_ub, np.zeros((a_ub.shape[0], m))])
        a_eq = np.hstack([a_eq, np.zeros((a_eq.shape[0], m))])
        link = np.zeros((m, 2 * n + 1 + m))
        for i in range(m):
            link[i, i] = link[i, n + i] = 1.0
            link[i, 2 * n + 1 + i] = -LEG_LIMIT_BOUND
        budget = np.concatenate([np.zeros(2 * n + 1), np.ones(m)])[None, :]
        a_ub = np.vstack([a_ub, link, budget])
        b_ub = list(b_ub) + [0.0] * m + [float(max_legs)]
        objective = np.concatenate([objective, np.zeros(m)])
        upper += [1.0] * m
        integrality = np.concatenate([integrality, np.ones(m)])

    result = milp(
        objective,
        constraints=[LinearConstraint(a_ub, -np.inf, np.array(b_ub)), LinearConstraint(a_eq, np.array(b_eq), np.array(b_eq))],
        bounds=Bounds(np.zeros(len(upper)), np.array(upper)),
        integrality=integrality,
        options={"time_limit": 3.0},
    )
    if result.x is None:
        raise RuntimeError(f"no portfolio found: {result.message}")

    quantities = result.x[:n] - result.x[n : 2 * n]
    holdings = []
    for (kind, strike, price), q in zip(names, quantities):
        if abs(q) > 1e-7:
            holdings.append(Holding(kind, float(q), strike, price))
    total_cost = float(quantities @ cost)

    report = np.unique(np.concatenate([np.linspace(low, high, 121), xs, strikes[(strikes >= low) & (strikes <= high)]]))
    fitted = holding_payoff(holdings, report)
    target = target_at(report)
    return Replication(
        mode=mode,
        holdings=tuple(holdings),
        cost=total_cost,
        max_error=float(np.max(np.abs(fitted - target))),
        prices=report,
        target=target,
        fitted=fitted,
    )

