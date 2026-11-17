"""Binomial trees: American options and a check on Black-Scholes.

The idea
--------
Split the time to expiry into n steps. In each step the price moves up by a
factor u or down by a factor d. Price the option at expiry (its payoff), then
walk back one step at a time: each node is worth the discounted average of the
two nodes after it, using the risk-neutral probability p of an up move,

    p = (e^((r - q) dt) - d) / (u - d)

An AMERICAN option may be exercised at any node, so at each node it is worth
the larger of holding on (the discounted average) and exercising now (the
payoff at that node's price). That one `max` is the whole difference.

Two ways to choose u, d and p
-----------------------------
crr             Cox-Ross-Rubinstein: u = e^(sigma sqrt(dt)), d = 1/u. The textbook
                tree. Its price zig-zags around the true value as n grows.
leisen_reimer   Leisen-Reimer: chooses p so the tree's probabilities match the
                normal distribution in Black-Scholes. It converges far faster
                and smoothly, so ~100 steps give 4 to 5 correct digits. It needs
                an odd number of steps.

Everything vectorises over spot prices, so one call prices a whole price axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs

METHODS = ("crr", "leisen_reimer")


def _peizer_pratt(z, n: int):
    """Peizer-Pratt inversion: the binomial probability that matches N(z) for n steps (vectorised)."""
    z = np.asarray(z, dtype=float)
    t = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    return 0.5 + np.sign(z) * 0.5 * np.sqrt(1.0 - np.exp(-t * t * (n + 1.0 / 6.0)))


def tree_parameters(strike, time, vol, rate, div_yield, steps, method="leisen_reimer"):
    """(steps, dt, u, d, p) for one Cox-Ross-Rubinstein tree, or the step count and dt for Leisen-Reimer.

    Leisen-Reimer rounds `steps` up to an odd number; its u, d and p depend on spot (see _factors).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if steps < 1:
        raise ValueError("a tree needs at least one step")
    if method == "leisen_reimer" and steps % 2 == 0:
        steps += 1
    dt = time / steps
    if method == "crr":
        growth = math.exp((rate - div_yield) * dt)
        u = math.exp(vol * math.sqrt(dt))
        d = 1.0 / u
        return steps, dt, u, d, (growth - d) / (u - d)
    return steps, dt, None, None, None


def _factors(spot, strike, time, vol, rate, div_yield, steps, method):
    """u, d, p for each column. Leisen-Reimer falls back to CRR where its formulas break down
    (very deep in or out of the money, where its probability rounds to exactly 0 or 1)."""
    dt = time / steps
    growth = np.exp((rate - div_yield) * dt)
    crr_u = np.exp(vol * np.sqrt(dt))
    crr_d = 1.0 / crr_u
    crr_p = (growth - crr_d) / (crr_u - crr_d)
    if method == "crr":
        return crr_u, crr_d, crr_p
    d1, d2 = bs.d1_d2(spot, strike, time, vol, rate, div_yield)
    p = _peizer_pratt(d2, steps)
    p_star = _peizer_pratt(d1, steps)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = growth * p_star / p
        d = (growth - p * u) / (1.0 - p)
    good = np.isfinite(u) & np.isfinite(d) & (p > 0) & (p < 1) & (d > 0) & (u > d)
    return np.where(good, u, crr_u), np.where(good, d, crr_d), np.where(good, p, crr_p)


def _payoff(prices, strike, sign):
    return np.maximum(sign * (prices - strike), 0.0)


def _backward(spot, strike, time, vol, rate, div_yield, sign, american, steps, method):
    """Run one tree per column, all columns at once. Every argument is a 1-D array of equal length."""
    u, d, p = _factors(spot, strike, time, vol, rate, div_yield, steps, method)
    discount = np.exp(-rate * time / steps)
    j = np.arange(steps + 1)[:, None]  # number of up moves, one row per node
    prices = spot * u**j * d ** (steps - j)
    values = _payoff(prices, strike, sign)
    for _ in range(steps):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            prices = prices[:-1] / d  # the node one step earlier: up moves unchanged, one fewer down move
            values = np.maximum(values, _payoff(prices, strike, sign))
    return values[0]


def price_many(
    spot,
    strike,
    time,
    vol,
    rate: float = 0.0,
    div_yield: float = 0.0,
    kind: str = "call",
    american: bool = False,
    steps: int = 101,
    method: str = "leisen_reimer",
    chunk: int = 8192,
):
    """Binomial values per share. spot, strike, time and vol may all be arrays; they broadcast together.

    Every point becomes one column of a single vectorised tree, processed in
    chunks to bound memory, which is what keeps a full profit grid fast.
    """
    sign = bs._sign(kind)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if steps < 1:
        raise ValueError("a tree needs at least one step")
    if method == "leisen_reimer" and steps % 2 == 0:
        steps += 1
    spot, strike, time, vol = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (spot, strike, time, vol)))
    if np.any(spot <= 0) or np.any(strike <= 0) or np.any(time < 0) or np.any(vol < 0):
        raise ValueError("spot and strike must be positive; time and vol cannot be negative")
    shape = spot.shape
    spot, strike, time, vol = (a.ravel() for a in (spot, strike, time, vol))
    out = np.empty(spot.shape)

    # No randomness left: the exact value, and for an American option never less than exercising now.
    certain = (time == 0) | (vol == 0)
    if np.any(certain):
        exact = bs.price(spot[certain], strike[certain], time[certain], vol[certain], rate, div_yield, kind)
        if american:
            exact = np.maximum(exact, _payoff(spot[certain], strike[certain], sign))
        out[certain] = exact

    todo = np.flatnonzero(~certain)
    for start in range(0, todo.size, chunk):
        cols = todo[start : start + chunk]
        out[cols] = _backward(spot[cols], strike[cols], time[cols], vol[cols], rate, div_yield, sign, american, steps, method)
    return out.reshape(shape)


def price(
    spot,
    strike: float,
    time: float,
    vol: float,
    rate: float = 0.0,
    div_yield: float = 0.0,
    kind: str = "call",
    american: bool = False,
    steps: int = 101,
    method: str = "leisen_reimer",
):
    """Option value per share from a binomial tree. Returns a float for a single spot, else an array."""
    values = price_many(spot, strike, time, vol, rate, div_yield, kind, american, steps, method)
    return float(values) if np.ndim(values) == 0 else values


def early_exercise_premium(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="put", steps=101):
    """How much more the American option is worth than the European one."""
    american = price(spot, strike, time, vol, rate, div_yield, kind, True, steps)
    european = price(spot, strike, time, vol, rate, div_yield, kind, False, steps)
    return american - european


@dataclass(frozen=True)
class Tree:
    """A small tree for drawing. Row i holds the i + 1 nodes after i steps, lowest price first."""

    prices: list[list[float]]
    values: list[list[float]]
    exercise: list[list[bool]]  # True where exercising beats holding (American only)
    p: float
    u: float
    d: float
    dt: float


def build_tree(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call", american=False, steps=5) -> Tree:
    """Every node of a small Cox-Ross-Rubinstein tree, for the tree explorer.

    CRR is used here because its recombining up/down structure is the one taught
    and the one a person can check by hand.
    """
    if not 1 <= steps <= 12:
        raise ValueError("the explorer draws between 1 and 12 steps")
    if time <= 0 or vol <= 0:
        raise ValueError("a tree needs time to expiry and volatility")
    sign = bs._sign(kind)
    steps, dt, u, d, p = tree_parameters(strike, time, vol, rate, div_yield, steps, "crr")
    discount = math.exp(-rate * dt)
    prices = [[spot * u**j * d ** (i - j) for j in range(i + 1)] for i in range(steps + 1)]
    values = [[0.0] * (i + 1) for i in range(steps + 1)]
    exercise = [[False] * (i + 1) for i in range(steps + 1)]
    for j in range(steps + 1):
        values[steps][j] = max(sign * (prices[steps][j] - strike), 0.0)
        exercise[steps][j] = values[steps][j] > 0
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold = discount * (p * values[i + 1][j + 1] + (1 - p) * values[i + 1][j])
            now = max(sign * (prices[i][j] - strike), 0.0)
            if american and now > hold + 1e-12:
                values[i][j], exercise[i][j] = now, True
            else:
                values[i][j] = hold
    return Tree(prices=prices, values=values, exercise=exercise, p=p, u=u, d=d, dt=dt)


def convergence(spot, strike, time, vol, rate=0.0, div_yield=0.0, kind="call", american=False, steps=range(1, 101)):
    """Tree prices for each number of steps, for both methods, next to Black-Scholes."""
    steps = list(steps)
    return {
        "steps": steps,
        "crr": [price(spot, strike, time, vol, rate, div_yield, kind, american, n, "crr") for n in steps],
        "leisen_reimer": [price(spot, strike, time, vol, rate, div_yield, kind, american, n, "leisen_reimer") for n in steps],
        "black_scholes": float(bs.price(spot, strike, time, vol, rate, div_yield, kind)),
    }
