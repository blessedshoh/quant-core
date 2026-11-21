# Fixed income lab, stage 3: curve PCA, design

Date: 2026-11-21. Touches three repositories, each on branch `fixed-income-lab-stage-3`:
- **bondlab**, the engine; released as tag `v0.3.0`;
- **quant-core**, the API on Fly;
- **site**, with `bonds/` and `hub/`.

## Intent

The lab at `bonds.blessedshoh.dev` is the fixed income part of a buy-side interview portfolio. Stage 3 adds the classic result of Litterman and Scheinkman (1991): almost all daily Treasury curve moves are three shapes, **level, slope and curvature**, with level explaining roughly 90% of the variance.

It must be practical for someone holding a real bond and impressive to an interviewer. Every number must be something the owner can derive on a whiteboard, so fewer features, fully understood.

The practical payoff is what the factors mean for the bond in the pricer:
- how much a typical day's level, slope or curvature move makes or loses on it;
- whether hedging level and slope with two Treasuries beats a DV01 hedge or a key-rate hedge with the same two bonds.

## Decisions made with the owner (2026-11-21)

1. **Window:** it ends on the day picked in the curve area and is **1, 2, 5 or 10 years** long, 5 by default.
2. **Tenor set:** the maturities from **3 months to 30 years** that the Treasury published on **every** day of the window. Maturities left out are named on the page, with the reason.
3. **Hedge part:** a **level/slope hedge with two Treasuries** (2Y and 10Y by default), shown side by side with a DV01 hedge and a key-rate hedge on the same bonds. Each row reports the daily P&L risk left over and the factor exposures left.
4. **Build approach:** the browser fetches the window's yields from a new cached bonds endpoint and posts them to one new API route, which returns everything. This is how the curve, price and hedge tabs already work: the browser holds yields and the API computes.

## Goals

1. **Engine (bondlab):**
   - PCA of daily par-yield changes over a window and tenor set: eigenvalues, explained variance, loadings and daily factor scores;
   - each bond's sensitivity to each par yield, and from it the factor exposures;
   - the factor hedge and the historical P&L risk of any hedged position.
2. **API (quant-core):** route `POST /fixed-income/pca`.
3. **Site (`bonds/`):** a new **Factors** tab. It draws:
   - the level, slope and curvature loadings against maturity;
   - the explained variance;
   - the three factors' cumulative paths over the window;
   - the pricer bond's factor exposures;
   - the hedge comparison table;
   - a show-your-work block.
4. **Pages:** additions to `/methodology` and `/limits`.
5. **Hub:** a `factors` switch in `BOND_FLAGS`, `settings.json`, labels and a usage event.
6. **Release:** bondlab tag `v0.3.0`, with the API Dockerfile pinned to it.

## Non-goals

- Rolling or time-varying PCA, such as a chart of the variance split across decades. Users get this by moving the curve day.
- PCA on zero or forward rates, on levels rather than changes, or on a correlation matrix.
- A curvature hedge with three bonds.
- Forecasting, or mean reversion of factors.
- A walkthrough notebook for PCA. Deferred; listed at the end.
- Short bills (1M, 6W, 2M, 4M). They start too late (2001 to 2025) and are driven by policy and bill supply.

## Anti-goals

- A number on the page that isn't computed in bondlab. Neither the browser nor the API layer computes; the Worker only reads and caches.
- Any change to the numbers bondlab produces today. All existing tests stay green.
- A D1 write of any kind. This stage adds reads only: no new table, no precomputation.
- A request that ties up the shared API: every input is bounded in `_LIMITS`.
- A first view that needs the pricer filled in. The PCA charts show without a bond; exposures appear when one is set.

## Constraints

- **bondlab contract (`bondlab/CLAUDE.md`):**
  - numpy and scipy only;
  - `mypy --strict` with no ignores; ruff ALL;
  - frozen slotted dataclasses; `datetime.date` only; no defaulted conventions;
  - every public function states its formula and cites a source;
  - errors name the offending values;
  - expected test values come from an external source, a closed-form identity or an independent computation.
- **D1 free tier:** 100,000 row writes a day (none used here) and 5,000,000 row reads a day. A 10-year window reads about 25,000 rows, so the endpoint caches.
- **API body cap:** 200 KB today.
- **No new dependencies:** no npm or pip packages. Charts are hand-drawn SVG, as in `LineChart.svelte`.
- **Privacy:** no university email address or university username anywhere. Commits use the repo's GitHub no-reply address.
- **Pushes and deploys:** merges, pushes to `main`, deploys and the bondlab tag each need the owner's OK.
  - bondlab pushes are named refs only. Never `--all` or `--mirror`, and never the `backup/*` branches or `refs/original`.

## Data facts (checked against D1 on 2026-11-21)

| Maturity | Stored days | From |
|---|---|---|
| 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y | 9,228 (all) | 1990-01-02 |
| 3M | 9,225 | 1990-01-02 (missing 2008-12-10, 12-18, 12-24) |
| 20Y | 8,289 | **1993-10-01** |
| 30Y | 8,234 | 1990-01-02 (missing 2002-02-18 to 2006-02-08) |
| 1M / 2M / 4M / 6W | — | 2001 / 2018 / 2022 / 2025 (excluded) |

So a window that touches December 2008 drops 3M, one before October 1993 drops 20Y, and one touching 2002 to 2006 drops 30Y. All three cases are named on the page.

## Approaches considered

| Approach | Trade-off |
|---|---|
| **Browser fetches the history from a cached bonds endpoint and posts it to one API route (chosen)** | Same pattern as the existing tabs. One call returns everything. The payload is about 160 KB for 10 years. |
| The Worker calls the API server-side and caches the PCA | Two calls and two caches: the PCA from the Worker, then exposures from the browser. Splits one result in two for no gain. |
| Nightly precomputation stored in D1 | Can't serve a window ending on any day since 1990. Adds writes on a tier that was exhausted once already. |

## The method (what the owner explains on the whiteboard)

**1. Changes matrix.** Take the window's stored days in order, restricted to the tenor set. That gives yields Y (n days × m tenors).
- Daily changes in basis points: ΔY = 10,000·(Y[t] − Y[t−1]), an (n−1) × m matrix.
- A change across a weekend or holiday is one observation, as in Litterman and Scheinkman's use of consecutive quotes.

**2. Covariance and eigen-decomposition.**
- Centre each column: X = ΔY − mean. Then S = XᵀX / (n−2), the sample covariance in bp².
- Decompose S = V Λ Vᵀ with λ₁ ≥ λ₂ ≥ … ≥ λₘ.
- The loadings of factor k are column vₖ, a unit vector across tenors. Its explained variance is λₖ / Σλ.
- A one-standard-deviation daily move of factor k shifts the par curve by **√λₖ · vₖ bp**.

**3. Signs.** Eigenvectors have no sign, so we fix them to read naturally:
- **level:** loadings sum to a positive number (yields up);
- **slope:** the 30Y (or longest) loading minus the 3M (or shortest) loading is positive (steepener);
- **curvature:** the middle tenor's loading is above the average of the two ends (belly up).
- The page states these conventions.

**4. Factor scores.** Day t's factor-k score is fₜₖ = Xₜ · vₖ in bp. The chart shows each factor's running sum over the window: "how the factors moved".

**5. A bond's par-yield DV01s.** These are gₜ: the price change per 100 face for a +1bp move in the par yield at tenor t, with the others held. Computing them by rebuilding the curve about 20 times costs too much: one bootstrap takes about 0.3 s. So the engine uses the chain rule with existing fast pieces:
- **Curve:** built from the curve day's yields at the PCA tenors only, so the pillars are the tenors.
- **Zero-rate key-rate DV01s, kₛ:** the existing `key_rate_durations`, which shifts one zero pillar and refits under the same scheme.
- **Jacobian Jₜₛ:** how much par yield t moves when zero pillar s moves 1bp. It uses the same pillar shifts, repricing each tenor's quote from the curve:
  - the bill rate (simple ACT/365) for 6 months and under;
  - `treasury_par_yield` beyond 6 months.
- A zero shift moves the price by kₛ = Σₜ gₜ Jₜₛ. So **g solves Jᵀ g = k**, one m × m linear solve.
- Because bootstrapping is the inverse of "quotes from the curve", this matches rebuilding the curve from bumped par yields to first order. A test checks it against full rebuilds.

**6. Factor exposures.** The P&L of a one-standard-deviation daily move of factor k is **eₖ = √λₖ · Σₜ gₜ vₖₜ**, per 100 face, scaled to the position's face on the page. It's a dot product a reader can check on the show-your-work block.

**7. Level/slope hedge.**
- Hedge bonds: par Treasuries at the two chosen tenors, issued on the curve day with the coupon equal to that day's yield, as in the existing hedge tab.
- Their par-yield DV01s, dotted with the loadings, give exposures E (2 × 2 for level and slope).
- Amounts n solve **E n = −e_position**, the same algebra as the existing key-rate hedge.

**8. The comparison, all with the same two bonds:**
- **Unhedged.**
- **DV01 hedge:** the parallel par DV01 Σₜ gₜ. It uses the one hedge bond closest in maturity to the position.
- **Key-rate hedge:** zero par-yield DV01 at the two hedge tenors, using `key_rate_hedge` with par-yield DV01s. It is labelled as par-yield key rates, so it's slightly different from the Hedge tab's zero-rate key rates.
- **Level/slope hedge.**

Each row shows:
- the hedge amounts;
- the level, slope and curvature exposures left;
- **the daily P&L risk left over:** the standard deviation of Σₜ gₜ^hedged ΔYₜ over the window's actual days, which equals √(gᵀ S g).

This replays history through the hedged position's DV01s, so it's measured on the real data with all m components, not judged by the 3-factor model alone. Being in-sample, and linear to first order, are stated on the limits page.

## Architecture

### bondlab, new module `src/bondlab/risk/factors.py`

All value objects are frozen slotted dataclasses. Public outputs are tuples, and numpy stays inside the module.

- `YieldHistory(days, tenors, yields)`: validated on construction.
  - Days strictly increase and there are at least 2.
  - Each row has one decimal yield per tenor and no gaps; a gap is an error naming the day and tenor.
- `common_tenors(days: Sequence[TreasuryDay], candidates: Sequence[Tenor]) -> CommonTenors`. Returns the kept tenors in maturity order, plus each dropped tenor with the first day it's missing. Building the `YieldHistory` is a separate step.
- `curve_pca(history) -> CurvePCA`. Fields:
  - tenors and observation count;
  - all m eigenvalues (bp²) and explained-variance ratios;
  - all m loadings, sign-fixed for the first three;
  - daily scores for the first three factors, with their days.
  - It refuses fewer observations than a stated minimum: 3·m and at least 60.
- `par_yield_dv01s(pricer, curve, quotes) -> tuple[ParYieldDV01, ...]`. This is the chain-rule step from step 5. `quotes` describes each pillar's instrument: bill or par bond.
- `factor_exposures(dv01s, pca) -> tuple[float, ...]`: eₖ for k = 1..3.
- `factor_hedge(position, hedges, pca) -> FactorHedge`: solves for level and slope; errors on a singular system the way `key_rate_hedge` does.
- `pnl_volatility(dv01s, pca) -> float`: √(gᵀ S g), with S rebuilt as V Λ Vᵀ.

Small additions to `data/treasury.py`, if needed:
- a public helper that builds the curve from a given tenor subset;
- the bill quote from a curve (simple ACT/365 from the discount factor), next to `treasury_par_yield`.

Docstrings cite:
- Litterman and Scheinkman (1991), *Journal of Fixed Income* 1(1), 54–61;
- Tuckman and Serrat, *Fixed Income Securities*, 3rd ed. (empirical hedging and PCA chapter; the builder checks the exact chapter number before citing);
- Jolliffe, *Principal Component Analysis*, 2nd ed., ch. 1–2, for the decomposition and sign indeterminacy.

### bondlab, `ui/lab_service.py`

Route `pca`. **Payload:**
```json
{ "history": { "days": ["2021-11-22", ...], "tenors": ["3 Mo", ..., "30 Yr"],
               "yields": [[0.05, ...], ...] },          // percent, null where unpublished
  "scheme": "MonotoneConvex",
  "position": { "preset": "UST", "coupon": 4.25, "maturity": "2035-11-15" } | null,
  "faceAmount": 1000000,
  "hedgeTenors": ["2 Yr", "10 Yr"] }
```
- The curve day is the last day of `history`.
- **Response:**
  - tenors used, tenors dropped with reasons, n;
  - eigenvalues and explained variance;
  - loadings for the first three factors, with each tenor's years to maturity;
  - each factor's daily standard deviation √λ;
  - the cumulative score series (days plus three series);
  - if a position is given: its par-yield DV01s, factor exposures, and the comparison table.
- **New `_LIMITS` entries:**
  - history rows at most **2,700** (10 years and 7%);
  - at most 10 tenors;
  - `hedgeTenors` exactly 2, distinct, and in the tenor set;
  - yields within the existing rate bounds.
- `null` cells are allowed in the payload: the engine's `common_tenors` decides what's dropped. A null in a kept tenor can't happen by construction.

### quant-core

- Add `"pca"` to `ROUTES`.
- If a real 10-year payload is over the 200 KB cap, raise the cap **for this route only**, to the smallest round number above the measured size. The builder measures it.
- Dockerfile `BONDLAB_VERSION=v0.3.0`.
- Tests: route shape, 422 on bad input, 413 on oversize.

### site

- `site/kit/treasury-data.ts`: `yieldsBetween(db, from, to, tenors)` runs one query over the primary key, `WHERE day BETWEEN ? AND ? AND tenor IN (…)`, ordered by day.
- `bonds/src/pages/history.json.ts` with `?day=YYYY-MM-DD&years=1|2|5|10`:
  - validates both;
  - resolves the day to the stored day on or before it;
  - returns `{ day, years, from, tenors, days, yields }` in percent with nulls;
  - caches in the Workers Cache API, keyed on the resolved day and length, for one day. A miss costs one D1 read query and a hit costs none.
  - The ten candidate tenors are fixed in the site code: 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y.
- **The Factors tab:**
  - a `FactorsPanel.svelte`, a `factors` entry in `AREAS` in `LabApp.svelte`;
  - share-link fields `years` and `hedgeTenors`, validated like the others;
  - `fetchPca` in `lib/api.ts`;
  - response types in `lib/types.ts`.
  - Layout, top to bottom:
    ```
    [window: 1y 2y (5y) 10y]   curve day 2026-11-20 · 1,256 daily changes · 10 tenors
    Loadings (line chart, years to maturity →)        Explained variance (bars + %)
    How the factors moved (three cumulative lines over the window)
    Your bond (from Price a bond): exposure per typical day, $ per position
    Hedge comparison table: Unhedged | DV01 | Key-rate | Level/slope
    ▸ Show your work
    ```
  - **Show your work:**
    - the size of ΔY and the tenors dropped;
    - S = XᵀX/(n−2);
    - the first three eigenvalues and their square roots;
    - one exposure worked as the dot product √λ₁ · Σ gₜ v₁ₜ with the actual numbers;
    - the level/slope 2 × 2 system with its numbers.
  - With no bond in the pricer, the bond and hedge parts show a short line pointing to Price a bond.
- **Pages:**
  - `methodology.astro` gets a "Curve factors (PCA)" section: steps 1 to 8 in prose, sources.
  - `limits.astro` gets: in-sample; the window choice changes the answer; linear (first-order) exposures; the tenor set changes over time; holiday gaps count as one day; par yields from the Treasury's fitted curve, not traded bonds; the comparison judged on the same window it was fitted on.
- **Hub:**
  - `BOND_FLAGS` gets `{ key: "factors", label: "Curve factors", help: "Level, slope and curvature from daily Treasury moves, and a factor hedge." }`;
  - `settings.json` `flags.factors: true`;
  - a label in `hub/src/lib/labels.ts`;
  - an event `factors` in `EVENTS`.
  - The existing test that ties `BOND_FLAGS` to `settings.json` must pass.

## Testing

**bondlab** (pytest, mypy --strict, ruff), expected values independent of the implementation:
- **Closed form:** build ΔY = F·diag(σ)·Vᵀ from a known orthonormal V and an F with orthonormal centred columns scaled by √(n−2). PCA must return eigenvalues σ² and loadings ±V exactly, to 1e-10.
- **Independent algorithm:** on a fixture of real Treasury par yields, compare against numpy SVD of the centred change matrix (singular values² / (n−2), right singular vectors up to sign). The engine uses `eigh` on the covariance. The fixture is public domain, from treasury.gov's CSV with provenance noted: 2019 to 2023, the 10 tenors.
- **Published result:** on the same fixture, level explains more than 75% and the first three more than 95%. This is the qualitative Litterman-Scheinkman finding, so the test cites it as a band, not a point.
- **Signs:** the conventions hold on the fixture and on the synthetic case with flipped inputs.
- **Tenor set:** `common_tenors` on hand-built days with each missing pattern (3M in December 2008, 20Y before 1993-10, the 30Y gap) drops exactly those tenors and names the first missing day.
- **Chain rule:** par-yield DV01s match a full rebuild from bumped par yields (central difference, 1bp) to within 1e-4 per 100 face for a 2Y, 10Y and 25Y bond. This test is slow; it's marked and kept small.
- **Identities:**
  - the level/slope hedge leaves level and slope exposure at 0 to 1e-9;
  - `pnl_volatility` equals `numpy.std(ddof=1)` of the replayed daily P&L;
  - a parallel par shift of 1bp changes price by Σ gₜ.

**quant-core:** the route returns the documented shape; bounds give 422; oversize gives 413; the hedge-tenor rules hold.

**site:**
- node tests for share-link validation of the new fields;
- `history.json` validation and caching keys, with a fake DB;
- `svelte-check` for bonds and options;
- the hub build and its flags test.

**Browser:** run locally with the `api`, `bonds` and `hub` launch configs. Check:
- desktop and phone width, both themes;
- windows that drop 3M (2008), 20Y (1992) and 30Y (2004);
- no bond, and a bond;
- a share link round trip.

## Calls made while writing (check these)

- The tab is named **Factors**; the hub label is **Curve factors**.
- The PCA uses covariance (bp²) of daily changes, not correlation, so level shows its true dominance. The hedge needs bp units anyway.
- Exposures use par yields because the PCA is on par yields. The bond is priced on a curve built from the PCA tenors only, which can differ a hair from the Price tab's all-tenor curve. The tab says so.
- A bond maturing beyond the window's longest kept tenor gets no exposures, just a clear message. For example, a 25-year bond in a window that dropped 30Y. Same no-extrapolation rule as the pricer.
- The DV01 row uses the hedge bond nearest the position's maturity.
- The minimum sample is max(60, 3·m) daily changes. A 1-year window, about 250 changes, always passes.
- If the bondlab release has to wait, the site and API changes still go on their branches; nothing deploys before `v0.3.0` is tagged.

## Left to the builder

- Exact chart styling, within the existing `LineChart` look, and the copy.
- Fixture years, within 2019 to 2023 or similar, from treasury.gov's public CSV.
- Module-internal names.
- Whether the bill-quote helper lives in `data/treasury.py` or `risk/factors.py`.
- Order of work: engine, then service route, then API, then history endpoint, then tab, then pages and hub, then browser check, then release and deploy.

## Deferred (for the owner's list)

- Walkthrough 03: curve PCA, with an exercise file.
- A rolling chart of the variance split since 1990.
- A curvature (three-bond) hedge.
