"""Write the fixture the web front end uses to check its grid reading against Grid.at.

The front end (site/options/src/lib/grid.ts) re-implements Grid.at in TypeScript
so the sliders never wait on the server. This script prices two small grids,
rounds them exactly as the API does, and records Grid.at at many points, so the
TypeScript tests can demand the same answers.

    python scripts/export_grid_fixture.py ../site/options/tests/fixtures/grid-parity.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from quantcore import Grid, Leg, Position, profit_grid
from quantcore.grid import day_axis, price_axis, vol_axis

CASES = {
    "iron condor": Position(
        spot=166.0,
        rate=0.04,
        legs=[
            Leg("put", 1, strike=140, days=30, vol=0.70),
            Leg("put", -1, strike=150, days=30, vol=0.66),
            Leg("call", -1, strike=180, days=30, vol=0.62),
            Leg("call", 1, strike=190, days=30, vol=0.60),
        ],
    ),
    "calendar spread": Position(
        spot=166.0,
        rate=0.04,
        legs=[Leg("call", -1, strike=170, days=30, vol=0.60), Leg("call", 1, strike=170, days=60, vol=0.58)],
    ),
}


def rounded_like_the_api(grid: Grid) -> Grid:
    """The API sends prices to 4 decimals, days and volatility to 6, and profit to the cent."""
    return Grid(
        prices=np.round(grid.prices, 4),
        days=np.round(grid.days, 6),
        vol_scales=np.round(grid.vol_scales, 6),
        profit=np.round(grid.profit, 2),
    )


def main(target: Path) -> None:
    rng = np.random.default_rng(5)
    cases = []
    for name, position in CASES.items():
        grid = rounded_like_the_api(
            profit_grid(
                position,
                prices=price_axis(position, 0.25, 21),
                days=day_axis(position, 7),
                vol_scales=vol_axis(points_each_side=2),
            )
        )
        # Random points inside the grid, then points past each edge (clamping) and on grid nodes.
        prices = np.concatenate([rng.uniform(grid.prices[0], grid.prices[-1], 60), [grid.prices[0] - 10, grid.prices[-1] + 10, grid.prices[3]]])
        days = np.concatenate([rng.uniform(0.0, grid.days[-1], 60), [-1.0, grid.days[-1] + 1.0, grid.days[2]]])
        vols = np.concatenate([rng.uniform(grid.vol_scales[0], grid.vol_scales[-1], 60), [0.01, 9.0, grid.vol_scales[1]]])
        samples = [[float(p), float(d), float(v), grid.at(p, d, v)] for p, d, v in zip(prices, days, vols)]
        cases.append(
            {
                "name": name,
                "grid": {
                    "axes": {
                        "prices": grid.prices.tolist(),
                        "days": grid.days.tolist(),
                        "vol_scales": grid.vol_scales.tolist(),
                    },
                    "profit": {"shape": list(grid.profit.shape), "values": grid.profit.ravel().tolist()},
                },
                "samples": samples,
            }
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"made_by": "quant-core/scripts/export_grid_fixture.py", "cases": cases}))
    print(f"wrote {target}: {sum(len(case['samples']) for case in cases)} samples")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
