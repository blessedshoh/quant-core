"""quantcore: option pricing and risk, written to be read."""

from . import black_scholes
from .grid import Grid, greeks_grid, profit_grid
from .position import (
    Leg,
    Position,
    breakevens,
    entry_cost,
    entry_price,
    first_expiry,
    net_greeks,
    profit,
    profit_at_expiry,
    profit_range,
    value,
)

__version__ = "0.1.0"

__all__ = [
    "Grid",
    "Leg",
    "Position",
    "black_scholes",
    "breakevens",
    "entry_cost",
    "entry_price",
    "first_expiry",
    "greeks_grid",
    "net_greeks",
    "profit",
    "profit_at_expiry",
    "profit_grid",
    "profit_range",
    "value",
]
