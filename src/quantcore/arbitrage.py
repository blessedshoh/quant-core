"""Arbitrage checks on an option chain, using real bid and ask prices.

An arbitrage is a trade that pays you today and can never lose. Option prices
that allow one are inconsistent with any probability distribution of the
future price. Two ways to look for one:

The rules
    Each textbook no-arbitrage condition is checked on its own: price bounds,
    put-call parity, prices falling (calls) or rising (puts) with strike, a
    spread never costing more than its widest payoff, and butterflies never
    paying you to hold them. Every violation comes with the exact trade that
    captures it. Buys use the ask, sells use the bid, so a "violation" that only
    exists at mid prices is not reported.

The solver
    One linear programme searches every combination of the quoted options,
    the stock and cash for the portfolio that pays the most today while never
    losing at expiry. If the best such portfolio pays nothing, the prices admit
    a risk-neutral distribution: this is the fundamental theorem of asset
    pricing, checked numerically. It finds arbitrage the separate rules miss,
    such as trades spanning several strikes. It assumes European exercise.

Conventions: quantities are per share (one option on one share). Positive is
bought, negative sold. A "cash" leg is money lent today (negative: borrowed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from . import black_scholes as bs

TOLERANCE = 0.005  # half a cent per share: smaller gaps are rounding, not arbitrage


@dataclass(frozen=True)
class Quote:
    """Bid and ask for the call and the put at one strike. Missing sides are None."""

    strike: float
    call_bid: float | None = None
    call_ask: float | None = None
    put_bid: float | None = None
    put_ask: float | None = None

    def __post_init__(self):
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        for bid, ask in ((self.call_bid, self.call_ask), (self.put_bid, self.put_ask)):
            for value in (bid, ask):
                if value is not None and value < 0:
                    raise ValueError("prices cannot be negative")
            if bid is not None and ask is not None and bid > ask:
                raise ValueError(f"bid above ask at strike {self.strike}")


@dataclass(frozen=True)
class TradeLeg:
    kind: str  # "call", "put", "stock" or "cash"
    quantity: float
    strike: float | None = None
    price: float = 0.0  # per unit: the ask when buying, the bid when selling; 1 for cash


@dataclass(frozen=True)
class Violation:
    rule: str  # "bound", "parity", "order", "spread" or "butterfly"
    title: str
    detail: str
    profit: float  # locked in, per share, in today's money
    strikes: tuple[float, ...]
    trade: tuple[TradeLeg, ...]


@dataclass(frozen=True)
class SolverResult:
    arbitrage: bool
    credit: float  # paid to you today by the best never-losing portfolio
    trade: tuple[TradeLeg, ...]
    worst_payoff: float  # its lowest payoff at expiry, never below zero when arbitrage is True


@dataclass(frozen=True)
class Report:
    violations: tuple[Violation, ...]
    solver: SolverResult | None  # None for American options
    assumptions: tuple[str, ...] = field(default_factory=tuple)


def _scale(trade: list[TradeLeg], profit: float) -> tuple[tuple[TradeLeg, ...], float]:
    """Scale fractional butterfly weights so the smallest option quantity is 1."""
    smallest = min(abs(leg.quantity) for leg in trade if leg.kind in ("call", "put"))
    factor = 1.0 / smallest if smallest > 0 else 1.0
    scaled = tuple(TradeLeg(leg.kind, leg.quantity * factor, leg.strike, leg.price) for leg in trade)
    return scaled, profit * factor


def check_rules(spot, days, quotes, rate=0.0, div_yield=0.0, american=False) -> list[Violation]:
    """Every rule-by-rule violation, each with the trade that captures it."""
    years = days / bs.DAYS_PER_YEAR
    disc = math.exp(-rate * years)  # value today of $1 at expiry
    carry = math.exp(-div_yield * years)  # shares to buy today to hold one at expiry
    sq = spot * carry
    quotes = sorted(quotes, key=lambda q: q.strike)
    found: list[Violation] = []

    def add(rule, title, detail, profit, strikes, trade):
        if profit > TOLERANCE:
            found.append(Violation(rule, title, detail, float(profit), tuple(strikes), tuple(trade)))

    for q in quotes:
        k = q.strike
        call_floor = max(sq - k * disc, spot - k if american else 0.0, 0.0)
        put_floor = max(k * disc - sq, k - spot if american else 0.0, 0.0)
        if q.call_ask is not None and call_floor > 0:
            if american and spot - k >= sq - k * disc:
                trade = [TradeLeg("call", 1, k, q.call_ask), TradeLeg("stock", -1, None, spot), TradeLeg("cash", k, None, 1)]
                how = "buy the call and exercise it at once"
            else:
                trade = [TradeLeg("call", 1, k, q.call_ask), TradeLeg("stock", -carry, None, spot), TradeLeg("cash", k * disc, None, 1)]
                how = "buy the call, short the stock and lend the strike's present value"
            add("bound", f"Call {k:g} is cheaper than it can ever be worth",
                f"The ask {q.call_ask:.2f} is below the lowest possible value {call_floor:.2f}: {how}.",
                call_floor - q.call_ask, [k], trade)
        call_cap = spot if american else sq
        if q.call_bid is not None and q.call_bid > call_cap + TOLERANCE:
            add("bound", f"Call {k:g} costs more than the stock",
                f"The bid {q.call_bid:.2f} is above {call_cap:.2f}: sell the call and hold the stock instead.",
                q.call_bid - call_cap, [k], [TradeLeg("call", -1, k, q.call_bid), TradeLeg("stock", 1 if american else carry, None, spot)])
        if q.put_ask is not None and put_floor > 0:
            if american and k - spot >= k * disc - sq:
                trade = [TradeLeg("put", 1, k, q.put_ask), TradeLeg("stock", 1, None, spot), TradeLeg("cash", -k, None, 1)]
                how = "buy the put and the stock, and exercise at once"
            else:
                trade = [TradeLeg("put", 1, k, q.put_ask), TradeLeg("stock", carry, None, spot), TradeLeg("cash", -k * disc, None, 1)]
                how = "buy the put and the stock, borrowing the strike's present value"
            add("bound", f"Put {k:g} is cheaper than it can ever be worth",
                f"The ask {q.put_ask:.2f} is below the lowest possible value {put_floor:.2f}: {how}.",
                put_floor - q.put_ask, [k], trade)
        put_cap = k if american else k * disc
        if q.put_bid is not None and q.put_bid > put_cap + TOLERANCE:
            add("bound", f"Put {k:g} costs more than its strike",
                f"The bid {q.put_bid:.2f} is above {put_cap:.2f}, the most it can ever pay: sell it and keep the cash.",
                q.put_bid - put_cap, [k], [TradeLeg("put", -1, k, q.put_bid), TradeLeg("cash", put_cap, None, 1)])

        # Put-call parity: call - put = S e^(-qT) - K e^(-rT) for European options.
        if None not in (q.call_bid, q.call_ask, q.put_bid, q.put_ask):
            high = spot - k * disc if american else sq - k * disc
            low = sq - k if american else sq - k * disc
            label = "the American parity range" if american else "put-call parity"
            excess = q.call_bid - q.put_ask - high
            add("parity", f"Strike {k:g}: the call is rich against the put",
                f"Call bid minus put ask is {q.call_bid - q.put_ask:.2f}, above {high:.2f} allowed by {label}: sell the call, "
                "buy the put and the stock (a conversion).",
                excess, [k],
                [TradeLeg("call", -1, k, q.call_bid), TradeLeg("put", 1, k, q.put_ask), TradeLeg("stock", carry, None, spot),
                 TradeLeg("cash", -k * disc, None, 1)])
            shortfall = low - (q.call_ask - q.put_bid)
            add("parity", f"Strike {k:g}: the call is cheap against the put",
                f"Call ask minus put bid is {q.call_ask - q.put_bid:.2f}, below {low:.2f} allowed by {label}: buy the call, "
                "sell the put and short the stock (a reversal).",
                shortfall, [k],
                [TradeLeg("call", 1, k, q.call_ask), TradeLeg("put", -1, k, q.put_bid), TradeLeg("stock", -carry, None, spot),
                 TradeLeg("cash", k * disc, None, 1)])

    width_factor = 1.0 if american else disc
    for a, b in zip(quotes, quotes[1:]):
        k1, k2 = a.strike, b.strike
        if a.call_ask is not None and b.call_bid is not None:
            add("order", f"Call {k2:g} is dearer than call {k1:g}",
                f"A higher strike can never be worth more: buy the {k1:g} call at {a.call_ask:.2f}, sell the {k2:g} call at "
                f"{b.call_bid:.2f}.",
                b.call_bid - a.call_ask, [k1, k2], [TradeLeg("call", 1, k1, a.call_ask), TradeLeg("call", -1, k2, b.call_bid)])
        if a.put_bid is not None and b.put_ask is not None:
            add("order", f"Put {k1:g} is dearer than put {k2:g}",
                f"A lower strike can never be worth more: buy the {k2:g} put at {b.put_ask:.2f}, sell the {k1:g} put at "
                f"{a.put_bid:.2f}.",
                a.put_bid - b.put_ask, [k1, k2], [TradeLeg("put", 1, k2, b.put_ask), TradeLeg("put", -1, k1, a.put_bid)])
        cap = width_factor * (k2 - k1)
        if a.call_bid is not None and b.call_ask is not None:
            add("spread", f"Calls {k1:g}/{k2:g}: the spread pays more than it can lose",
                f"Selling the {k1:g} and buying the {k2:g} collects {a.call_bid - b.call_ask:.2f}; the most it can ever "
                f"cost is {cap:.2f}.",
                a.call_bid - b.call_ask - cap, [k1, k2],
                [TradeLeg("call", -1, k1, a.call_bid), TradeLeg("call", 1, k2, b.call_ask), TradeLeg("cash", cap, None, 1)])
        if b.put_bid is not None and a.put_ask is not None:
            add("spread", f"Puts {k1:g}/{k2:g}: the spread pays more than it can lose",
                f"Selling the {k2:g} and buying the {k1:g} collects {b.put_bid - a.put_ask:.2f}; the most it can ever "
                f"cost is {cap:.2f}.",
                b.put_bid - a.put_ask - cap, [k1, k2],
                [TradeLeg("put", -1, k2, b.put_bid), TradeLeg("put", 1, k1, a.put_ask), TradeLeg("cash", cap, None, 1)])

    for a, b, c in zip(quotes, quotes[1:], quotes[2:]):
        k1, k2, k3 = a.strike, b.strike, c.strike
        lam = (k3 - k2) / (k3 - k1)
        for kind, ask1, bid2, ask3 in (
            ("call", a.call_ask, b.call_bid, c.call_ask),
            ("put", a.put_ask, b.put_bid, c.put_ask),
        ):
            if None in (ask1, bid2, ask3):
                continue
            credit = bid2 - lam * ask1 - (1 - lam) * ask3
            if credit > TOLERANCE:
                trade, profit = _scale(
                    [TradeLeg(kind, lam, k1, ask1), TradeLeg(kind, -1, k2, bid2), TradeLeg(kind, 1 - lam, k3, ask3)], credit
                )
                add("butterfly", f"{kind.title()} butterfly {k1:g}/{k2:g}/{k3:g} pays you to hold it",
                    f"A butterfly never pays less than zero, so it can't cost less than zero. Here the wings cost less than "
                    f"the body, leaving {credit:.2f} per share of body.",
                    profit, [k1, k2, k3], list(trade))
    return found


def solve(spot, days, quotes, rate=0.0, div_yield=0.0) -> SolverResult:
    """The portfolio paying the most today that never loses at expiry (European options).

    Variables: amounts bought and sold of each quoted option (each between 0 and
    1 per share), stock between -1 and 1, and unlimited cash. Payoffs are
    piecewise linear in the final price, so "never loses" only has to hold at 0,
    at every strike, and in the slope beyond the highest strike.
    """
    years = days / bs.DAYS_PER_YEAR
    disc = math.exp(-rate * years)
    carry = math.exp(-div_yield * years)
    quotes = sorted(quotes, key=lambda q: q.strike)
    strikes = [q.strike for q in quotes]
    nodes = np.array([0.0] + strikes + [2.0 * max(strikes) + spot])

    names, costs, payoffs, slopes, bounds = [], [], [], [], []

    def instrument(name, cost, payoff_at_nodes, slope, bound):
        names.append(name)
        costs.append(cost)
        payoffs.append(payoff_at_nodes)
        slopes.append(slope)
        bounds.append(bound)

    for q in quotes:
        call_pay = np.maximum(nodes - q.strike, 0.0)
        put_pay = np.maximum(q.strike - nodes, 0.0)
        if q.call_ask is not None:
            instrument(("call", 1, q.strike, q.call_ask), q.call_ask, call_pay, 1.0, (0, 1))
        if q.call_bid is not None:
            instrument(("call", -1, q.strike, q.call_bid), -q.call_bid, -call_pay, -1.0, (0, 1))
        if q.put_ask is not None:
            instrument(("put", 1, q.strike, q.put_ask), q.put_ask, put_pay, 0.0, (0, 1))
        if q.put_bid is not None:
            instrument(("put", -1, q.strike, q.put_bid), -q.put_bid, -put_pay, 0.0, (0, 1))
    instrument(("stock", 1, None, spot), spot * carry, nodes.copy(), 1.0, (-1, 1))
    cash_limit = 10.0 * (max(strikes) + spot)
    instrument(("cash", 1, None, 1.0), disc, np.ones_like(nodes), 0.0, (-cash_limit, cash_limit))

    payoff_matrix = np.array(payoffs).T  # nodes x instruments
    # Payoff >= 0 at every node, and slope >= 0 beyond the last strike: written as A_ub x <= 0.
    a_ub = np.vstack([-payoff_matrix, -np.array(slopes)[None, :]])
    b_ub = np.zeros(a_ub.shape[0])
    result = linprog(np.array(costs), A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"the solver failed: {result.message}")

    credit = -float(result.fun)
    x = result.x
    trade = []
    net: dict[tuple, float] = {}
    for (kind, side, strike, price), amount in zip(names, x):
        if abs(amount) < 1e-9:
            continue
        if kind == "cash":
            trade.append(TradeLeg("cash", amount * disc, None, 1.0))
            continue
        quantity = side * amount if kind != "stock" else amount * carry
        net[(kind, strike, price)] = net.get((kind, strike, price), 0.0) + quantity
    trade = [TradeLeg(kind, q, strike, price) for (kind, strike, price), q in net.items() if abs(q) > 1e-9] + trade
    worst = float(np.min(payoff_matrix @ x))
    arbitrage = credit > TOLERANCE
    return SolverResult(
        arbitrage=arbitrage,
        credit=credit if arbitrage else 0.0,
        trade=tuple(trade) if arbitrage else (),
        worst_payoff=worst,
    )


def check(spot, days, quotes, rate=0.0, div_yield=0.0, american=False) -> Report:
    """Rule checks, plus the solver for European options."""
    if spot <= 0 or days <= 0:
        raise ValueError("spot and days must be positive")
    quotes = list(quotes)
    if not quotes:
        raise ValueError("at least one quote is needed")
    rules = check_rules(spot, days, quotes, rate, div_yield, american)
    assumptions = [
        "Buys use the ask and sells the bid, so mid-price gaps are not reported.",
        "Interest and dividends are continuous at the rate and yield entered.",
        "Trading costs other than the bid-ask spread, and short-selling limits, are ignored.",
    ]
    if american:
        assumptions.append(
            "American options: bounds include exercising now, parity becomes a range, and the solver is skipped because "
            "early exercise breaks its expiry-only payoffs."
        )
        return Report(tuple(rules), None, tuple(assumptions))
    return Report(tuple(rules), solve(spot, days, quotes, rate, div_yield), tuple(assumptions))


def trade_payoff(trade, prices, rate=0.0, div_yield=0.0, days=0.0):
    """Payoff at expiry of a trade at each final price.

    Cash grows at the risk-free rate. Shares bought today grow in number as their
    dividends are reinvested, so each becomes e^(qT) shares by expiry.
    """
    years = days / bs.DAYS_PER_YEAR
    growth = math.exp(rate * years)
    shares_per_share = math.exp(div_yield * years)
    prices = np.asarray(prices, dtype=float)
    total = np.zeros_like(prices)
    for leg in trade:
        if leg.kind == "call":
            total += leg.quantity * np.maximum(prices - leg.strike, 0.0)
        elif leg.kind == "put":
            total += leg.quantity * np.maximum(leg.strike - prices, 0.0)
        elif leg.kind == "stock":
            total += leg.quantity * shares_per_share * prices
        elif leg.kind == "cash":
            total += leg.quantity * growth
    return total


def trade_cost(trade, spot) -> float:
    """What the trade costs today. Negative means it pays you."""
    cost = 0.0
    for leg in trade:
        if leg.kind == "stock":
            cost += leg.quantity * spot
        elif leg.kind == "cash":
            cost += leg.quantity
        else:
            cost += leg.quantity * leg.price
    return cost
