# Fixed income lab, stages 1–2: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put bondlab under git as the fixed income engine, add the owner's workbook models, serve them from the existing API, load the Treasury curve daily, and ship `bonds.blessedshoh.dev` with today's curve, a bond pricer, a hedge calculator and the basics.

**Architecture:** bondlab (numpy/scipy, strict contract) gains textbook, KWF-tree, convertible, hedge and Treasury-parsing modules plus new `ui/service.py` routes; quant-core's FastAPI app mounts `/fixed-income/{route}` straight onto `bondlab.ui.service.handle`. A scheduled Worker writes Treasury par yields into the hub's D1; a new Astro site reads them and sends curve points with every API call.

**Tech Stack:** Python 3.14, numpy, scipy, pytest, Hypothesis, mypy --strict, ruff (bondlab); FastAPI (quant-core); Astro 7, Svelte 5, TypeScript, node:test, Cloudflare Workers + D1 + cron (site).

**Spec:** `quant-core/docs/superpowers/specs/2026-11-20-fixed-income-lab-stage-1-2-design.md`

## Global Constraints

- bondlab `CLAUDE.md` is binding: numpy/scipy only; `mypy --strict`, zero `# type: ignore`; no default convention arguments (presets are named factories that state assumptions); `@dataclass(frozen=True, slots=True)`; `datetime.date` only, no float year fractions as public inputs; docstrings state the formula and cite a specific source; actionable errors naming the values; expected test values from an external source (the owner's workbooks count, with sheet and cell as provenance).
- bondlab's existing 432 tests, `mypy` and `ruff check .` stay green after every task.
- No new runtime dependencies anywhere; no new npm packages.
- No pricing logic in `ui/service.py`, in the API layer or in the browser.
- API stays stateless: curve points travel in the request.
- Pages read stored data only; nothing calls treasury.gov at request time.
- Treasury source: `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/<YEAR>/all?type=daily_treasury_yield_curve&field_tdr_date_value=<YEAR>&page&_format=csv`; attribute "U.S. Department of the Treasury".
- Site address `bonds.blessedshoh.dev`, name "Fixed income lab"; dashboard switches `curve`, `price`, `hedge`, `basics`, on by default.
- Key-rate tenors default: 1, 2, 3, 5, 7, 10, 20, 30 years.

## Review Focus

1. **A Treasury day with missing tenors** (1990s files, the 30-year gap 2002 to early 2006): the curve builds from the tenors present and says which were missing. Test in Task 5.
2. **A bond that matures after the last curve pillar or settles before the first**: a 422 naming both dates, not an extrapolated number. Test in Task 6.
3. **A requested date with no data** (weekend, holiday, before 1990, future): the nearest earlier day with data, labelled with its real date; before 1990 or future refused. Test in Task 8.
4. **The Treasury answering with an HTML error page or an empty year**: the Worker stores nothing and logs the reason. Test in Task 8.
5. **A key-rate hedge that cannot be solved** (two hedges at one tenor, a tenor with no exposure, more tenors than hedges): refused with the reason. Test in Task 5.

---

### Task 1: Move bondlab and put it under git

**Files:**
- Create: `C:\Users\user\Desktop\blessedshoh\bondlab\` (copy of `Desktop\githubPortfolio\Bond calculator` without `.venv`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.hypothesis`, `__pycache__`)
- Create: `bondlab/constraints.txt` (the old venv's `pip freeze`)

- [ ] **Step 1:** Record the baseline in the old folder: `.venv\Scripts\python -m pytest`, `-m mypy`, `-m ruff check .`. Expected: `432 passed, 1 skipped`, `Success: no issues found in 57 source files`, `All checks passed!`.
- [ ] **Step 2:** `robocopy` the folder to the new path with the exclusions above; write `constraints.txt` from the old venv's `pip freeze` (excluding the editable bondlab line).
- [ ] **Step 3:** In the new folder: `python -m venv .venv`; `.venv\Scripts\pip install -c constraints.txt -e ".[dev]"` (and any extras the old venv had: check `pyproject.toml` groups against the freeze). Same versions, nothing newer.
- [ ] **Step 4:** Re-run the three commands from Step 1 in the new folder. Expected: identical results.
- [ ] **Step 5:** `git init -b main`; check `.gitignore` covers `.venv/`, caches and `.claude/settings.local.json`; commit everything as `Initial import of bondlab` with the session's attribution line. The old folder is left untouched for the owner to delete.

### Task 2: Textbook bond preset and the basic models

**Files:**
- Create: `bondlab/src/bondlab/instruments/textbook.py`, `bondlab/src/bondlab/pricing/textbook.py`
- Create: `bondlab/tests/golden/workbook_basics.csv`, `bondlab/tests/test_textbook.py`
- Modify: `bondlab/src/bondlab/__init__.py` (exports)

**Interfaces:**
- Produces:
  - `textbook_bond(*, face: float, coupon_rate: float, years: int, frequency: Frequency, issue_date: date) -> FixedRateBond` — named preset; docstring states: whole coupon periods, settlement on `issue_date` (a coupon date, so accrued is zero), street time in coupon periods, `WeekendOnly` calendar and unadjusted dates chosen so no date ever moves. Cite Fabozzi, *Handbook*, ch. on bond pricing.
  - `textbook_convention(frequency: Frequency) -> YieldConvention` — street convention for that frequency.
  - `price_path(*, face: float, coupon_rate: float, years: int, frequency: Frequency, ytm_by_year: Sequence[float]) -> tuple[float, ...]` — `len(ytm_by_year) == years`; element `k` is the price of a textbook bond with `years − k` years left at `ytm_by_year[k]` (so the first element has the full life and the last has one year left). Covers "Time path" (one yield throughout) and "Yield Change" (the yield switches at the year of change).
  - `bootstrap_textbook(par_yields: Sequence[float], frequency: Frequency) -> tuple[float, ...]` — spot rates (compounded at `frequency`) at each period, from par bonds at those yields, through the existing `bootstrap()` with `LinearOnLogDiscount`.
  - `tangent_estimates(bond, ytm: float, shift: float) -> TangentEstimate` (frozen dataclass: `price`, `modified_duration`, `convexity`, `duration_estimate`, `duration_convexity_estimate`, `true_price`, `error_duration`, `error_duration_convexity`), where estimates are `P(1 − D·Δy)` and `P(1 − D·Δy + ½·C·Δy²)`.
- [ ] **Step 1: Write `workbook_basics.csv`** (columns: `model,face,coupon,years,frequency,ytm,extra,expected,tolerance,provenance`) with these rows, provenance `FixedIncomeDeriv_ All_Models.xlsm, <sheet>!<cell>`:
  - Coupon: zero-coupon 20y at 10% → 14.8643628024 (`Coupon!D15`); 20% coupon 20y at 10% → 185.1356371975 (`G15`); zero at 1% → 81.9544470337 (`C24`); 20% coupon at 1% → 442.8655063591 (`G24`).
  - Maturity: 5% coupon at 10%, 2y → 91.3223140496 (`Maturity!F12`); 20y → 57.4321814012 (`I12`).
  - Time path: 4% coupon at 8%, 20y → 60.7274103702 (`Time path!E15`), with 19 years left → 61.5856031998 (`E27`); 12% coupon → 139.2725896297 (`I15`).
  - Yield change: 2% coupon 20y at 8% → 41.0911155553 (`Yield Change!F13`); after the yield drops to 2% (year of change 7) → 100 (`H30`).
  - Duration: 8% coupon, 20y, annual, YTM 2% → Macaulay 13.2935006610, modified 13.0328437853 (`Duration and Convexity!E17`, `E18`).
  - Tangent: 5% coupon, 50y, annual, YTM 5%, Δy −4% → price 100, modified duration 18.2559254606, convexity 564.1346813318, duration estimate 173.0237018422, duration + convexity 218.1544763487, true 256.7844701244 (`Tangency!D10, D14, D15, D21, D23, D25`).
  - Bootstrap: semiannual par yields 5%, 5.25%, 5.8%, 6%, 6.2%, 6.5% at 0.5…3 years → spot rates 5%, 5.25%, 5.8186247396%, 6.0233762350%, 6.2321030686%, 6.7017439056% (`Bootstrapping!E10:E15`).
  All tolerances `1e-8` except where the workbook value came from Excel's `PRICE`/`DURATION` (`1e-6`) and the bootstrap rows (`2e-6`, because the workbook solved them with Goal Seek: its own check prices are off by up to 0.000970 per 100, `Bootstrapping!J30`).
- [ ] **Step 2:** `tests/test_textbook.py` reads the CSV and asserts each row with its tolerance; plus `test_textbook_bond_has_no_accrued`, `test_price_path_one_price_per_year_left` (length 20; first 60.7274103702, last 96.2962962963 = `Time path!E26`, `E45`) and `test_yield_change_path` (8% for years 0–6 then 2% → element 7 is 100, `Yield Change!H30`).
- [ ] **Step 3:** Run `.venv\Scripts\python -m pytest tests/test_textbook.py` — expected: fails on import.
- [ ] **Step 4:** Implement the Interfaces; reuse `clean_price_from_yield` and `analytic_risk` (no new pricing formula).
- [ ] **Step 5:** Run the test file, the full suite, `mypy`, `ruff`. Expected: all green.
- [ ] **Step 6:** Commit `Textbook bond preset and the workbook basics, pinned to the workbook`.

### Task 3: Kalotay-Williams-Fabozzi rate tree

**Files:**
- Create: `bondlab/src/bondlab/models/kwf_tree.py`; Test: `bondlab/tests/test_kwf_tree.py`, rows appended to `tests/golden/workbook_basics.csv`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True, slots=True) class RateTree: volatility: float; lowest_rates: tuple[float, ...]` — annual levels; the rate at level `i`, node `j` (j = 0 lowest) is `lowest_rates[i] · e^{2σj}`. `level_rates(i) -> tuple[float, ...]`.
  - `calibrate_rate_tree(par_yields: Sequence[float], volatility: float) -> RateTree` — annual-pay par yields for 1..n years; level `i`'s lowest rate solved with `brentq` so a par bond of maturity `i+1` prices at 100.
  - `@dataclass(frozen=True, slots=True) class TreeValuation: value: float; straight_value: float; option_value: float; nodes: tuple[tuple[TreeNode, ...], ...]` with `TreeNode(rate: float, value: float, exercised: bool)`.
  - `value_on_tree(tree: RateTree, *, face: float, coupon_rate: float, years: int, right: ExerciseRight | None, strike: float | None) -> TreeValuation` — backward induction `V = ½[(V_u + C) + (V_d + C)] / (1 + r)`; for a call (issuer) `V = min(V, strike)`, for a put (holder) `V = max(V, strike)`, applied at levels 1..years−1 (not at level 0, as in the workbook); `option_value = straight_value − value` for a call, `value − straight_value` for a put. Reuse the existing `ExerciseRight` type. Cite Fabozzi, *Handbook*, ch. on valuing bonds with embedded options (the KWF binomial model).
- [ ] **Step 1: Failing tests:**
  - `test_workbook_callable`: `RateTree(0.10, (0.035, 0.0407361919503, 0.0324627558493))`, face 100, coupon 4%, 3 years, call at 100 → value 99.8150009406 (`Binomial Tree!H18`), node rates `0.0497552972051` (`K17`) and `0.0484287409295` (`N14`) to 1e-12.
  - `test_nvidia_callable`: `RateTree(0.1416, (0.04858, 0.04429295078145, 0.03702950670663, 0.03191253030166, 0.02753050362923))`, face 1000, coupon 5%, 5 years, call at 985 → 982.2212369016 (`FI_final_ShohruhM.xlsx, Q1!C21`).
  - `test_calibrated_tree_reprices_its_par_curve`: par yields (0.035, 0.042, 0.047, 0.052), σ 0.1 → each par bond prices at 100 to 1e-10.
  - `test_unreachable_call_equals_straight` (strike 1e6 → value == straight_value); `test_put_floor` (put at 100 → value ≥ straight_value, every exercised node has value 100).
  - `test_bad_inputs`: negative volatility, `years` > tree levels, call without strike → `ValueError` naming the value.
- [ ] **Step 2:** Run — fails on import. **Step 3:** Implement. **Step 4:** File, full suite, mypy, ruff green. **Step 5:** Commit `Kalotay-Williams-Fabozzi rate tree with calls and puts, pinned to the workbook`.

### Task 4: Convertible bond tree

**Files:**
- Create: `bondlab/src/bondlab/models/convertible.py`; Test: `bondlab/tests/test_convertible.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True, slots=True) class ConvertibleTerms: face: float; coupon_rate: float; conversion_ratio: float; years: int; coupon_on_conversion: bool`
  - `@dataclass(frozen=True, slots=True) class ConvertibleMarket: stock_price: float; volatility: float; rate: float; dividend_yield: float`
  - `value_convertible(terms, market) -> ConvertibleValuation` (`value: float`, `nodes: tuple[tuple[ConvertibleNode, ...], ...]`, `ConvertibleNode(stock: float, hold: float, convert: float, value: float, converts: bool)`): annual CRR stock tree, `u = e^{σ}`, `d = 1/u`, `p = (e^{r−q} − d)/(u − d)`; at maturity `max(face + coupon, ratio·S + coupon·[coupon_on_conversion])`; earlier `hold = e^{−r}(p·V_u + (1−p)·V_d) + coupon`, `convert = ratio·S + coupon·[coupon_on_conversion]`, `value = max(hold, convert)`. Today's node uses no coupon, as in the workbook (`I18`, `I19`). Cite Hull, *Options, Futures and Other Derivatives*, ch. on convertible bonds.
- [ ] **Step 1: Failing tests:** `test_workbook_convertible`: face 1000, coupon 10%, ratio 15, 3 years, coupon on conversion yes; S 100, σ 0.2, r 0.05, q 0.09 → value 1504.0110866921 (`Convertible Bond!H4`), today's node hold 1504.0110866921 vs convert 1500 (`I18`, `I19`) → holds; `p` = 0.3527900730 (`C20`). Plus `test_deep_in_the_money_converts_everywhere` (S 1000 → every node converts) and `test_bad_inputs`.
- [ ] **Step 2–5:** fail, implement, green (full suite, mypy, ruff), commit `Convertible bond tree, pinned to the workbook`.

### Task 5: Treasury curve, key-rate hedge

**Files:**
- Create: `bondlab/src/bondlab/data/__init__.py`, `bondlab/src/bondlab/data/treasury.py`, `bondlab/src/bondlab/risk/hedge.py`
- Test: `bondlab/tests/test_treasury.py`, `bondlab/tests/test_hedge.py`, fixture `bondlab/tests/golden/treasury_par_yields_sample.csv` (header and three rows from the 1990 file, three from 2004 with no 30Y column filled, three from 2026 incl. `1.5 Month`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True, slots=True) class Tenor: label: str; months: int; weeks: int` — exactly one of `months`/`weeks` is non-zero; `TENORS: Mapping[str, Tenor]` keyed by CSV header (`"1 Mo"` → months 1, `"1.5 Month"` → weeks 6, `"30 Yr"` → months 360). `Tenor.maturity_from(day: date) -> date` adds the months (end-of-month aware) or weeks. No float tenors anywhere.
  - `parse_treasury_csv(text: str) -> tuple[TreasuryDay, ...]` with `TreasuryDay(day: date, yields: Mapping[Tenor, float])` (decimals, missing tenors absent); raises `ValueError` naming the line when a row is malformed; ignores blank cells.
  - `treasury_curve(day: TreasuryDay, scheme: InterpolationScheme) -> TreasuryCurve` (`curve: DiscountCurve`, `used: tuple[Tenor, ...]`, `missing: tuple[Tenor, ...]`): tenors ≤ 6 months as zero-coupon bills at their bond-equivalent yield, longer tenors as `FixedRateBond.us_treasury` par bonds at 100 from `day`, through the existing `bootstrap()`. Docstring states the CMT methodology assumption and cites the Treasury's "Interest Rate Statistics" methodology page.
  - `dv01_hedge(position_dv01: float, hedge_dv01: float) -> float` (hedge face per 100 face of position, negative = sell) and `key_rate_hedge(position: Sequence[KeyRateDuration], hedges: Sequence[Sequence[KeyRateDuration]], tenors: Sequence[date]) -> KeyRateHedge` (`notionals: tuple[float, ...]`, `residual_dv01: tuple[tuple[date, float], ...]`): solve the square system in DV01 terms at `tenors`; refuse (ValueError with reason) when the matrix is singular or rank-deficient, when `len(tenors) != len(hedges)`, or when a tenor is not a pillar. Cite Tuckman & Serrat, *Fixed Income Securities*, ch. on key-rate exposures.
- [ ] **Step 1: Failing tests:** parser on the fixture (1990 rows have no 1 Mo; 2004 rows lack 30Y; 2026 has 1.5 Month); `test_curve_reprices_par_inputs` (each par bond at 100 to 1e-9 on a 2026 row); `test_missing_tenors_reported` (2004 row → `missing` contains the 30Y tenor; curve still builds); `test_dv01_hedge_sign_and_size` (closed form: `−position/hedge`); `test_key_rate_hedge_zeroes_hedged_tenors` (10y bond hedged with 2y, 5y, 10y, 30y par bonds → residual 0 at those tenors to 1e-9); `test_unsolvable_hedges_refused` (duplicate tenor; more tenors than hedges; hedge with no exposure at a tenor).
- [ ] **Step 2–5:** fail, implement, green, commit `Treasury par-yield parser, Treasury curve and key-rate hedging`.

### Task 6: Service routes in bondlab

**Files:**
- Modify: `bondlab/src/bondlab/ui/service.py` (add handlers to `handle`), Test: `bondlab/tests/test_ui.py` (append)

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces: routes added to the `handle` dispatcher (camelCase JSON as the existing routes use):
  - `treasury-curve` `{day, yields: {label: pct}, scheme}` → `{day, used, missing, grid: [{years, par, zero, forward}], nelsonSiegel: {beta0..decay, grid}}`
  - `bond` `{curve: <treasury-curve input>, bond: {preset: "us_treasury"|"us_corporate", coupon, maturity, frequency}, spreadBp?, yield?, cleanPrice?}` → price, yield, Z-spread, accrued, Macaulay, modified, convexity, DV01, `keyRates: [{tenor, duration, dv01}]`, `cashflows: [{date, yearFraction, amount, discountFactor, presentValue}]`
  - `hedge` `{curve, position: <bond>, faceAmount, mode: "dv01"|"key-rate", hedgeTenors: [label]}` → notionals per hedge, residual DV01s, before/after key-rate profile
  - `basics` `{model: "coupon"|"maturity"|"time-path"|"yield-change"|"duration"|"tangent"|"bootstrap", inputs}` → that model's numbers and chart series
  - `rate-tree` `{volatility, parYields? | lowestRates?, face, coupon, years, right?, strike?}` → valuation and nodes
  - `convertible` `{terms, market}` → valuation and nodes
  The `bond` route must refuse a bond maturing after the curve's last pillar or settling before its reference date: `PayloadError` naming both dates.
- [ ] **Step 1: Failing tests** per route: shape; each workbook golden case through the route equals the engine value exactly; `test_bond_past_the_curve_refused` (a 40-year bond on a 30-year curve → PayloadError naming maturity and last pillar); unknown model name → PayloadError listing the known ones.
- [ ] **Step 2–5:** fail, implement (translation only: no arithmetic beyond unit conversion of percent and basis points), green, commit `Service routes for the fixed income lab`.

### Task 7: API router in quant-core

**Files:**
- Create: `quant-core/api/fixed_income.py`; Modify: `quant-core/api/main.py` (include router), `quant-core/api/Dockerfile` (install bondlab), `quant-core/pyproject.toml` (`api` extra gains `bondlab`)
- Test: `quant-core/tests/test_fixed_income_api.py`

**Interfaces:**
- Consumes: `bondlab.ui.service.handle(route, payload)`, `bondlab.ui.service.PayloadError`.
- Produces: `POST /fixed-income/{route}` for the six routes of Task 6 (allow-list them; anything else 404). Body: a JSON object up to 200 kB (`413` above). `PayloadError` and `ValueError` → 422 with the message. CORS unchanged.
- [ ] **Step 1:** Locally `pip install -e ../bondlab` into quant-core's venv (with `-c ../bondlab/constraints.txt`).
- [ ] **Step 2: Failing tests:** each route 200 on a golden case with the engine's value; unknown route 404; malformed body 422 with the bondlab message; 201 kB body 413; existing 1,950 tests still pass.
- [ ] **Step 3–4:** implement, green.
- [ ] **Step 5:** Dockerfile: `pip install "bondlab @ git+https://github.com/blessedshoh/bondlab@v0.2.0"` (the tag Task 15 creates). Until the repo exists, the deploy step waits (Task 15).
- [ ] **Step 6:** Commit `API: /fixed-income routes served by bondlab`.

### Task 8: Treasury data in D1

**Files:**
- Create: `site/hub/migrations/0002_treasury.sql` (table `treasury_par_yields(day TEXT NOT NULL, tenor TEXT NOT NULL, yield_pct REAL NOT NULL, PRIMARY KEY (day, tenor))` and index on `day`)
- Create: `site/data/` Worker (`wrangler.jsonc`: name `blessedshoh-data`, D1 binding `DB`, crons `["30 23 * * *", "30 3 * * *"]`, no routes, `workers_dev: false`), `site/data/src/worker.ts`, `site/data/src/treasury.ts`
- Create: `site/data/scripts/backfill.mjs` (downloads 1990..current year, writes `backfill.sql`, applied with `wrangler d1 execute blessedshoh-hub --remote --file`)
- Create: `site/kit/treasury-data.ts` (shared reads: `latestDay(db)`, `dayOnOrBefore(db, day)`, `yieldsFor(db, day)`)
- Test: `site/data/tests/treasury.test.mjs`, `site/kit` tests via `site/bonds/tests` (Task 9)

**Interfaces:**
- Produces: `parseTreasuryCsv(text: string): { day: string; yields: Record<string, number> }[]` (throws on HTML or a header without `Date`), `upsertDays(db, days)`; the Worker's `scheduled` handler fetches the current year (and the previous year in the first week of January), upserts only days newer than the latest stored, and writes one `changes` row: `data` / `Treasury curve <day> loaded` or `data` / `Treasury fetch failed: <reason>`. `dayOnOrBefore(db, day)` returns the latest stored day ≤ `day`, or null before 1990-01-02 or for a future date.
- [ ] **Step 1: Failing tests:** parse the Task 5 fixture lines; HTML body → throws "not a CSV"; empty year → zero days, nothing written; `dayOnOrBefore` on a Saturday returns Friday; before 1990 → null.
- [ ] **Step 2–3:** implement, green.
- [ ] **Step 4:** Apply migration locally and remotely (`pnpm --dir site/hub run db:migrate`), run the backfill against local D1, check row counts per year, then remotely.
- [ ] **Step 5:** Deploy the Worker; trigger once with `wrangler dev --test-scheduled` locally and confirm a `changes` row. If treasury.gov refuses Cloudflare's requests, stop and report (fallback in the spec).
- [ ] **Step 6:** Commit `Daily Treasury par yields in D1, with a 1990 backfill`.

### Task 9: Bonds site and the curve area

**Files:**
- Create: `site/bonds/` (copy of `site/options` structure: `astro.config.mjs` port 4323, `wrangler.jsonc` name `blessedshoh-bonds`, custom domain `bonds.blessedshoh.dev`, D1 binding; `src/layouts/Base.astro`, `src/layouts/Doc.astro`, `src/middleware.ts`, `src/pages/hit.ts`, `src/lib/{api,state,track,format,types}.ts`, `src/data/settings.json` with flags `curve, price, hedge, basics` true)
- Create: `src/pages/index.astro` (reads latest or `?day=` curve via kit), `src/pages/curve.json.ts` (GET `?day=YYYY-MM-DD` → stored curve, for the date picker), `src/components/LabApp.svelte` (area tabs), `src/components/CurvePanel.svelte`, `src/components/CurveChart.svelte`
- Modify: `site/pnpm-workspace.yaml` (add `bonds`), `.claude/launch.json` in the session folder (add `bonds`)
- Test: `site/bonds/tests/state.test.mjs`, `site/bonds/tests/curve.test.mjs`

**Interfaces:**
- Consumes: kit `headerLinks`, `toolSettings(db, "bonds")`, `dayOnOrBefore`, `yieldsFor`; API `treasury-curve`.
- Produces: `encodeLab(state: LabState): string` / `decodeLab(text): LabState | null` (`LabState = { area: "curve"|"price"|"hedge"|"basics"; day: string | null; bond?: BondInput; hedge?: HedgeInput; basics?: { model: string; inputs: Record<string, number> } }`), link format `#l=` base64url JSON `v: 1`.
- [ ] **Step 1: Failing tests:** link round trips; bad links refused; `curve.json.ts` handler returns the requested day's nearest earlier data with its real date, 400 before 1990 or in the future.
- [ ] **Step 2–3:** implement; curve chart shows par, zero and forward lines, the scheme switch (monotone convex, linear on zero, natural cubic), the Nelson-Siegel overlay toggle, missing-tenor note, date picker, Treasury attribution, show-your-work panel listing the bootstrapped discount factors.
- [ ] **Step 4:** Browser check, desktop and phone, both themes; a 1995 date and a 2004 date (missing 30Y) render with their notes.
- [ ] **Step 5:** Commit `Fixed income lab: site and today's curve`.

### Task 10: Price a bond

**Files:** Create `site/bonds/src/components/PricePanel.svelte`, `KeyRateChart.svelte`, `CashflowWork.svelte`; Test: `site/bonds/tests/bond.test.mjs`

**Interfaces:** Consumes API `bond`. Produces `BondInput = { preset: "us_treasury" | "us_corporate"; coupon: number; maturity: string; frequency: 1 | 2; mode: "curve" | "yield" | "price"; spreadBp?: number; yield?: number; cleanPrice?: number }` with `isValidBond(input)` (coupon 0–20%, maturity after settlement and ≤ last curve pillar, frequency 1 or 2).
- [ ] **Step 1: Failing tests:** `isValidBond` limits; link round trip with a bond.
- [ ] **Step 2–3:** implement: three modes (price off curve + spread, price from yield, yield and Z-spread from price); strip of price, yield, accrued, durations, convexity, DV01; key-rate bars; show-your-work cashflow table with the sums that make duration and convexity.
- [ ] **Step 4:** Browser check: a 10-year 4% Treasury off today's curve; the API's 422 for a 40-year bond shows as a message.
- [ ] **Step 5:** Commit `Fixed income lab: bond pricer with key-rate durations and show your work`.

### Task 11: Hedge

**Files:** Create `site/bonds/src/components/HedgePanel.svelte`; Test: `site/bonds/tests/hedge.test.mjs`

**Interfaces:** Consumes API `hedge` and `BondInput`. Produces `HedgeInput = { faceAmount: number; mode: "dv01" | "key-rate"; tenors: string[] }` with `isValidHedge` (1–4 distinct tenors for key-rate, exactly one for DV01).
- [ ] **Step 1: Failing tests:** `isValidHedge` limits; round trip.
- [ ] **Step 2–3:** implement: position from the pricer's bond, hedge tenors chosen from the curve's tenors, result table (sell/buy face per hedge), before/after key-rate bars, residual exposure sentence; the API's refusal reason shown as a message.
- [ ] **Step 4:** Browser check; Step 5: commit `Fixed income lab: DV01 and key-rate hedging`.

### Task 12: Basics

**Files:** Create `site/bonds/src/components/BasicsPanel.svelte` and one small component per model (`CouponModel`, `MaturityModel`, `TimePathModel`, `YieldChangeModel`, `DurationModel`, `TangentModel`, `BootstrapModel`, `RateTreeModel`, `ConvertibleModel`); Test: `site/bonds/tests/basics.test.mjs`

**Interfaces:** Consumes API `basics`, `rate-tree`, `convertible`. Each model opens on the workbook's own inputs (the golden rows in Task 2–4) so its first numbers equal the workbook's, and says so ("These are the numbers from my coursework workbook; change any input").
- [ ] **Step 1: Failing test:** every model's default inputs equal the golden rows (shared constants imported from `src/data/basics-defaults.json`, which the bondlab golden CSV is checked against in Task 15).
- [ ] **Step 2–3:** implement; the tree models draw their nodes (rate, value, exercised/converts highlighted); the tangent model draws price, tangent and convexity curves; every model has a show-your-work block with its formula and the inputs substituted (from the route's numbers, no arithmetic in the browser); Black-Scholes links to the options tool.
- [ ] **Step 4:** Browser check each model; Step 5: commit `Fixed income lab: the coursework models`.

### Task 13: Hub, pages and downloads

**Files:**
- Modify: `site/hub/src/lib/validate.ts` (per-tool flag lists: `OPTION_FLAGS` stays, add `BOND_FLAGS`), `site/hub/src/pages/admin/tools.astro` (a "Fixed income lab" section with its switches), `site/hub/src/lib/admin.ts` (save switches for tool `bonds`), `site/hub/src/pages/admin/numbers.astro` (labels for bonds events)
- Create: `site/bonds/src/pages/methodology.astro`, `site/bonds/src/pages/limits.astro`, `site/bonds/public/coursework/` (the two workbooks once the owner provides the cleaned copies)
- [ ] **Step 1:** Hub test (node) that `BOND_FLAGS` keys equal the bonds `settings.json` flags; implement the admin section.
- [ ] **Step 2:** Methodology: curve construction from CMT par yields, schemes, Nelson-Siegel, pricing and conventions, key-rate durations and the hedge solve, the KWF tree, the convertible tree, how each basic model matches the workbook. Limits: CMT yields are fitted constant-maturity yields, not traded bonds; no on-the-run/off-the-run or liquidity effects; corporates priced as Treasury plus a flat spread (no credit curve); key-rate bumps are local; KWF constant volatility and annual steps; the convertible ignores credit risk and call/put features.
- [ ] **Step 3:** Downloads section on the methodology page: "The coursework this was built from", linking the cleaned workbooks; the final without its Bloomberg `Data` sheet, with the note that a later stage rebuilds Q3 on public data.
- [ ] **Step 4:** Commit (site) `Fixed income lab: dashboard switches, methodology, limits, coursework downloads`.

### Task 14: Walkthroughs and exercises (bondlab)

**Files:** Create `bondlab/walkthroughs/01_bond_maths.ipynb`, `02_curves_and_key_rate_risk.ipynb` (built by a scratch script and executed, as in quant-core), `bondlab/exercises/bond_maths.py`, `bondlab/exercises/test_bond_maths.py`, `bondlab/exercises/curves.py`, `bondlab/exercises/test_curves.py`; Modify `bondlab/README.md`

- [ ] **Step 1:** Exercise tests compare the owner's functions with the engine on the workbook cases and random inputs; verified against a scratch solution (all pass) and the blank files (all fail with `NotImplementedError`).
- [ ] **Step 2:** Walkthrough 01: price as discounted cashflows, the coupon and maturity effects, pull to par, duration as a weighted time and a slope, convexity and the tangent picture, the workbook check. Walkthrough 02: bootstrapping from par yields, why the interpolation scheme matters for forwards, Nelson-Siegel, key-rate durations, the key-rate hedge and its residual. Each ends with five check-yourself questions. Read every output and fix any prose that misreads it.
- [ ] **Step 3:** Exercises and notebooks stay outside `src/` and `mypy`'s scope is unchanged; full suite, mypy, ruff green.
- [ ] **Step 4:** Commit `Walkthroughs 01-02 and exercises`.

### Task 15: Publish and deploy

- [ ] **Step 1:** Check `site/bonds/src/data/basics-defaults.json` against `bondlab/tests/golden/workbook_basics.csv` (a quant-core or bondlab test reading both paths, run locally).
- [ ] **Step 2:** Owner creates the public GitHub repo `blessedshoh/bondlab`; add the remote, push `main`, tag `v0.2.0`, push the tag.
- [ ] **Step 3:** Run every suite (bondlab, quant-core, site tests, `svelte-check` for options, bonds, hub).
- [ ] **Step 4:** Deploy the API (`fly deploy --remote-only`), check `POST https://api.blessedshoh.dev/fixed-income/bond` on the 10-year example; deploy `site/bonds`, `site/hub`, `site/data`.
- [ ] **Step 5:** Verify live at `https://bonds.blessedshoh.dev`: today's curve with its date, a 2004 date, the pricer, a key-rate hedge, each basic model's workbook numbers, a shared link, both pages; the owner switches the hub card to live (title "Fixed income lab", address `https://bonds.blessedshoh.dev`).
- [ ] **Step 6:** Push quant-core and site.
