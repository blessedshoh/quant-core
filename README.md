# quantcore

Option pricing and risk in plain Python: Black-Scholes prices and Greeks, binomial trees for American exercise, multi-leg positions, risk-neutral probabilities, scenario analysis, arbitrage checks and payoff replication. It drives the interactive tool at [options.blessedshoh.dev](https://options.blessedshoh.dev), whose [methodology](https://options.blessedshoh.dev/methodology) and [model limits](https://options.blessedshoh.dev/limits) pages describe the models in plain language.

## What is here

| Folder | What it holds |
|---|---|
| `src/quantcore/` | The engine (see below). |
| `tests/` | What "correct" means: 1,906 tests. Run with `pytest`. |
| `walkthroughs/` | One notebook per topic: the maths first, then the code, then the checks. |
| `exercises/` | Blank versions of the core functions, checked against the engine. |
| `api/` | The FastAPI service that puts the engine behind HTTP. |

| Module | What it does |
|---|---|
| `black_scholes.py` | Prices and Greeks of one European option |
| `binomial.py` | Cox-Ross-Rubinstein and Leisen-Reimer trees, American exercise |
| `position.py` | Multi-leg positions: cost, value, profit, break-evens, limits, net Greeks |
| `grid.py` | The price × date × volatility profit grid the front end reads from |
| `probability.py` | Lognormal price distribution, chance of profit, expected profit |
| `scenarios.py` | Scenario grid and P/L explain |
| `work.py` | Every intermediate value behind the headline numbers ("show your work") |
| `chain.py`, `arbitrage.py` | Model option chain with a smile; no-arbitrage rules and a linear-programming search |
| `replicate.py` | Build a drawn payoff from listed options (linear and mixed-integer programming) |

| Walkthrough | Exercise |
|---|---|
| `01_black_scholes.ipynb`: prices and Greeks of one option | `exercises/black_scholes.py` |
| `02_positions_and_risk.ipynb`: positions, chance of profit, scenarios, P/L explain | `exercises/risk.py` |

## How it is checked

- Textbook values from Hull.
- Put-call parity and no-arbitrage bounds on random inputs.
- Every Greek against a finite-difference estimate.
- QuantLib: 1,536 combinations of spot, strike, expiry, volatility, rate and dividend yield, with the price and five Greeks each agreeing to within 0.000001; and 192 American options where the Leisen-Reimer tree agrees with QuantLib's to within 0.00000001.
- Interpolating the profit grid stays within 1% of the money at stake across six strategies.
- Chance of profit against simulation; risk-neutral expected profit equal to the financing cost.
- Every "show your work" value rebuilds the engine's own numbers.

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

`api/main.py` puts the engine behind HTTP. `POST /options/grid` returns the profit at every price, date and volatility for a position, plus its Greeks and headline numbers; other routes serve the risk view (`/options/risk`, `/options/explain`), show your work (`/options/work`), binomial trees, the chain, arbitrage checks and replication. It stores nothing. Browsers may call it only from `*.blessedshoh.dev`.

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
