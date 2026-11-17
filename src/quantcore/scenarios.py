"""Risk the way a fund reports it: a scenario grid and a P/L explain.

Scenario grid
    Profit if the price jumps by each shock in `price_shocks` and volatility
    moves by each shift in `vol_shifts` (in volatility points), all at once.
    It answers "what do I lose if the market drops 10% and volatility jumps 10
    points?", which a payoff chart at one volatility cannot.

P/L explain
    Splits a change in value into the parts the Greeks predict:

        delta x move  +  1/2 gamma x move^2  +  theta x days  +  vega x vol change

    plus whatever is left over. A small leftover means the Greeks describe the
    position well. A large one means the move was too big for a second-order
    approximation, which is itself worth knowing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .position import Position, net_greeks, profit, value

DEFAULT_PRICE_SHOCKS = (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20)
DEFAULT_VOL_SHIFTS = (-0.20, -0.10, 0.0, 0.10, 0.20)


@dataclass(frozen=True)
class Scenarios:
    price_shocks: np.ndarray
    vol_shifts: np.ndarray
    days_forward: float
    profit: np.ndarray  # shape (len(vol_shifts), len(price_shocks))


def scenario_grid(
    position: Position,
    price_shocks=DEFAULT_PRICE_SHOCKS,
    vol_shifts=DEFAULT_VOL_SHIFTS,
    days_forward: float = 0.0,
) -> Scenarios:
    """Profit for every combination of a price shock (fraction of spot) and a volatility shift."""
    price_shocks = np.asarray(price_shocks, dtype=float)
    vol_shifts = np.asarray(vol_shifts, dtype=float)
    if np.any(price_shocks <= -1):
        raise ValueError("a price shock of -100% or worse is not a price")
    prices = position.spot * (1.0 + price_shocks)
    table = profit(position, prices[None, :], days_forward, 1.0, vol_shifts[:, None])
    return Scenarios(price_shocks, vol_shifts, float(days_forward), np.asarray(table, dtype=float))


@dataclass(frozen=True)
class Explain:
    """A change in value split into the parts each Greek predicts."""

    delta: float
    gamma: float
    theta: float
    vega: float
    unexplained: float
    actual: float

    @property
    def predicted(self) -> float:
        return self.delta + self.gamma + self.theta + self.vega


def explain(position: Position, price_move: float, days_passed: float = 0.0, vol_shift: float = 0.0) -> Explain:
    """Attribute the change in value from now to after the given move.

    price_move    change in the underlying price, in money (not a percentage)
    days_passed   calendar days that pass
    vol_shift     change in every leg's volatility: 0.05 is +5 points

    The Greeks are measured now, before the move, as a risk report would be.
    """
    greeks = {name: float(v) for name, v in net_greeks(position).items()}
    parts = {
        "delta": greeks["delta"] * price_move,
        "gamma": 0.5 * greeks["gamma"] * price_move**2,
        "theta": greeks["theta_per_day"] * days_passed,
        "vega": greeks["vega_per_point"] * vol_shift * 100.0,
    }
    before = float(value(position, position.spot))
    after = float(value(position, position.spot + price_move, days_passed, 1.0, vol_shift))
    actual = after - before
    return Explain(**parts, unexplained=actual - sum(parts.values()), actual=actual)


__all__ = ["Explain", "Scenarios", "explain", "scenario_grid"]
