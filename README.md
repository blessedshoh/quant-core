# quantcore

Option pricing and risk in plain Python: Black-Scholes prices and Greeks, multi-leg positions, and the profit grid that drives an interactive options tool.

## What is here

| Folder | What it holds |
|---|---|
| `src/quantcore/` | The engine. `black_scholes.py` prices one option, `position.py` handles multi-leg positions, `grid.py` builds the price × date × volatility profit grid. |
| `tests/` | What "correct" means. Run with `pytest`. |
| `walkthroughs/` | One notebook per module: the maths first, then the code. |
| `exercises/` | Blank versions of the core functions, checked by the same tests as the engine. |

## How it is checked

- Textbook values from Hull.
- Put-call parity and no-arbitrage bounds on random inputs.
- Every Greek against a finite-difference estimate.
- QuantLib: 1,536 combinations of spot, strike, expiry, volatility, rate and dividend yield, with the price and five Greeks each agreeing to within 0.000001.
- Interpolating the profit grid stays within 1% of the money at stake across six strategies.

## Run it

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pip install QuantLib
pytest
```

QuantLib is optional. Without it the cross-check tests are skipped.

## The API

`api/main.py` puts the engine behind one HTTP call: `POST /options/grid` returns the profit at every price, date and volatility for a position, plus its Greeks and headline numbers. It stores nothing. Browsers may call it only from `*.blessedshoh.dev`.

```
pip install -e ".[api]"
python api/dev.py               # http://127.0.0.1:8000, also allows the local front end
```

It runs on Fly.io as one small machine in Singapore that suspends when idle and resumes on the next request (`fly.toml`). To publish a new version:

```
fly deploy --remote-only
```

## Conventions

Time is in years (days / 365), volatility and rates are decimals, and money is positive when paid. Theta is per year, vega per 1.00 of volatility, rho per 1.00 of rate; the position-level Greeks are reported per day and per volatility point.
