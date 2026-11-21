# Fixed income lab stage 3 (curve PCA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **The owner chose native execution (`superpowers:executing-plans`), followed by one whole-branch review on the most capable model.**

**Goal:** A Factors tab in the bond lab:
- level, slope and curvature from daily Treasury par-yield changes over a window ending on the curve day;
- the pricer bond's factor exposures;
- a level/slope hedge compared with DV01 and key-rate hedges.

**Architecture:**
- **Engine:** a new bondlab module `risk/factors.py` does the PCA, the par-yield DV01s (chain rule through the existing zero-rate key rates), the exposures, the hedge and the historical P&L risk.
- **Service route:** `pca` in `ui/lab_service.py` translates JSON.
- **API:** quant-core serves the route.
- **Site:** a cached `/history.json` endpoint serves the window's stored yields. The browser posts them to the API and draws the result.

**Tech Stack:**
- Python 3.12+ with numpy and scipy (bondlab), FastAPI (quant-core);
- Astro 7 SSR on Cloudflare Workers, Svelte 5 runes, D1, node:test (site).

**Spec:** `quant-core/docs/superpowers/specs/2026-11-21-fixed-income-lab-stage-3-design.md` (read it first; this plan argues from it).

## Global Constraints

- **Branches:** `fixed-income-lab-stage-3` in each of bondlab, quant-core and site. Commit there; never commit to `main`.
- **Approvals:** no push, merge, tag push or deploy without the owner's explicit OK each time.
- **bondlab pushes:** named refs only (`main`, `fixed-income-lab-stage-3`, `v0.3.0`). Never `--all`, `--mirror`, `backup/*` or `refs/original`.
- **Commits:** use each repo's configured GitHub no-reply address. No university email address or university username anywhere.
- **Commit trailer:** end commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **bondlab contract (`bondlab/CLAUDE.md`):**
  - numpy and scipy only; `mypy --strict` with zero `# type: ignore`; `ruff check .` clean (ALL);
  - `@dataclass(frozen=True, slots=True)`; `datetime.date` only; no defaulted conventions;
  - each public docstring states the formula and cites a specific source section;
  - errors name the offending values;
  - expected test values come from a closed form, an independent computation or a cited source, **never the implementation's own output**.
- **No new dependencies:** no npm or pip packages. Charts are hand-drawn SVG; reuse `LineChart.svelte`.
- **D1:** this stage adds **no** new writes. The usage event goes through the existing `/hit` counter only.
- **Candidate tenors:** exactly `3 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr, 20 Yr, 30 Yr`.
- **Window lengths:** exactly 1, 2, 5 or 10 years, default 5.
- **Default hedge tenors:** `2 Yr`, `10 Yr`.
- **Minimum sample:** max(60, 3·m) daily changes.
- **History bounds:** at most **2,700** rows and **10** tenors.
- **Units on the wire:** rates in percent, moves in bp, money in dollars for the position's face.
- **No Python on PATH:** use `bondlab/.venv/Scripts/python` and `quant-core/.venv/Scripts/python`. There is no global `python`.

## Review Focus

1. **A bond longer than the window's longest kept tenor.** Example: a 2030-02-15 bond on curve day 2004-06-30, where the window dropped 30Y so the curve ends at 20Y (2024). Expected: the PCA still renders and the bond section says why it has no exposures, with no 422. Test in Task 4.
2. **A chosen hedge tenor dropped from the window.** Example: 30Y with a 2004 curve day. Expected: the PCA renders and the bond section names the dropped tenor. Test in Task 4.
3. **A curve day near the start of the data with a long window.** Examples: 1990-06-29 with 10 years, or 1990-01-03. Expected: the window starts at the first stored day, the page shows the real start, and too few days gives a clear error rather than a crash. Tests in Tasks 1 and 6.
4. **A real 10-year payload against the API's 200 KB cap.** Expected: it's accepted. Test in Task 5, with a measured size.
5. **Two hedge bonds whose level/slope exposures are nearly collinear.** Example: `1 Yr` and `2 Yr` on a flat curve. Expected: a bond-section message saying the hedges can't be told apart, not a 500 and not absurd notionals. Test in Task 4.

---

### Task 1: PCA of daily par-yield changes (bondlab)

**Files:**
- Create: `bondlab/src/bondlab/risk/factors.py`
- Create: `bondlab/tests/golden/treasury_2019_2023_pca.csv` (fixture)
- Modify: `bondlab/tests/golden/treasury_SOURCE.txt` (provenance)
- Test: `bondlab/tests/test_factors.py`

**Interfaces:**
- Consumes: `Tenor`, `TENORS`, `TreasuryDay` from `bondlab.data.treasury`.
- Produces (all frozen slotted dataclasses, exported in `__all__`):
  - `DroppedTenor(tenor: Tenor, first_missing: date)`
  - `CommonTenors(kept: tuple[Tenor, ...], dropped: tuple[DroppedTenor, ...])`
  - `YieldHistory(days: tuple[date, ...], tenors: tuple[Tenor, ...], yields: tuple[tuple[float, ...], ...])`: decimals, one row per day. `__post_init__` checks:
    - at least 2 days;
    - days strictly increase;
    - each row has `len(tenors)` finite values;
    - tenors are in maturity order with no repeats.
  - `CurvePCA` with fields:
    - `tenors: tuple[Tenor, ...]`
    - `observations: int` (number of daily changes)
    - `eigenvalues: tuple[float, ...]` (bp², descending, all m)
    - `explained: tuple[float, ...]`
    - `loadings: tuple[tuple[float, ...], ...]` (`loadings[k][t]`, all m, first three sign-fixed)
    - `move_days: tuple[date, ...]` (= `days[1:]`)
    - `paths: tuple[tuple[float, ...], ...]` (3 cumulative factor paths in bp, `paths[k][i]`)
    - property `daily_sd -> tuple[float, ...]`, the square roots of the eigenvalues
  - `common_tenors(days: Sequence[TreasuryDay], candidates: Sequence[Tenor]) -> CommonTenors`
  - `yield_history(days: Sequence[TreasuryDay], tenors: Sequence[Tenor]) -> YieldHistory`
  - `curve_pca(history: YieldHistory) -> CurvePCA`
  - `MIN_OBSERVATIONS = 60` (module constant)

- [ ] **Step 1: Make the fixture from the stored Treasury data (read-only D1 query)**

Run from `site/data`:
```bash
npx wrangler d1 execute blessedshoh-hub --remote --json --command "SELECT day, tenor, yield_pct FROM treasury_par_yields WHERE day BETWEEN '2019-01-02' AND '2023-12-29' AND tenor IN ('3 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr') ORDER BY day"
```
- Pivot the output with a node one-liner into the Treasury CSV format: header `Date,"3 Mo",…,"30 Yr"`, dates `MM/DD/YYYY`, values as stored, ascending.
- Write it to `tests/golden/treasury_2019_2023_pca.csv`.
- Add to `treasury_SOURCE.txt`: "treasury_2019_2023_pca.csv: the ten tenors 3M–30Y for 2019-01-02 to 2023-12-29, as stored in the site's D1 table `treasury_par_yields`, which was loaded verbatim from the Treasury's yearly CSVs; public domain."
- Expected: about 1,250 rows and no blank cells. If a blank cell appears, note it and keep the fixture: the test then also covers dropping that tenor.

- [ ] **Step 2: Write the failing tests** in `tests/test_factors.py`

```python
def test_synthetic_changes_recover_known_loadings_and_variances():
    # m=4 tenors, V = a fixed orthonormal 4x4 (e.g. from np.linalg.qr of a fixed matrix),
    # sigma = (12.0, 4.0, 1.5, 0.5) bp. F = n x 4 with orthonormal, centred columns
    # (QR of a centred random matrix, seed 7) scaled by sqrt(N-1), N = number of changes,
    # so X^T X / (N-1) = V diag(sigma^2) V^T exactly. Changes = F diag(sigma) V^T;
    # yields = cumulative sum from a 3% start, /10_000. Build YieldHistory directly.
    pca = curve_pca(history)
    assert pca.eigenvalues == pytest.approx([144.0, 16.0, 2.25, 0.25], rel=1e-10)
    for k in range(4):
        assert abs(np.dot(pca.loadings[k], V[:, k])) == pytest.approx(1.0, abs=1e-10)

def test_matches_svd_of_the_centred_change_matrix_on_real_data():
    # fixture -> parse_treasury_csv -> yield_history(all ten). Independent algorithm:
    # X = centred 10_000 * diff; s = np.linalg.svd(X, full_matrices=False)
    assert pca.eigenvalues == pytest.approx(s**2 / (n_changes - 1), rel=1e-9)
    # right singular vectors equal loadings up to sign (|dot| == 1 to 1e-9) for k < 3

def test_level_slope_curvature_split_is_the_litterman_scheinkman_pattern():
    # Litterman & Scheinkman (1991): level dominates, three factors explain nearly all.
    assert pca.explained[0] > 0.75
    assert sum(pca.explained[:3]) > 0.95

def test_sign_conventions():
    # on the fixture and on the synthetic case built with -V:
    assert sum(pca.loadings[0]) > 0
    assert pca.loadings[1][-1] - pca.loadings[1][0] > 0
    mid = len(pca.tenors) // 2
    assert pca.loadings[2][mid] > (pca.loadings[2][0] + pca.loadings[2][-1]) / 2

def test_factor_paths_telescope_to_the_net_curve_move():
    # paths[k][-1] == 10_000 * (Y_end - Y_start) . loadings[k], to 1e-8
    # and len(paths[k]) == observations == len(move_days)

def test_common_tenors_drops_what_a_window_lacks():
    # hand-built TreasuryDays: 3 Mo missing on 2008-12-10; 20 Yr absent before 1993-10-01;
    # 30 Yr absent 2002-02-18 onward. Expect kept/dropped exactly, with first_missing dates
    # 2008-12-10 / the first day in the list / 2002-02-18.

def test_yield_history_and_pca_refuse_bad_input():
    # yield_history with a kept tenor missing on a day -> ValueError naming day and tenor
    # YieldHistory with non-increasing days -> ValueError naming both days
    # curve_pca with 59 changes (or fewer than 3*m) -> ValueError naming count and minimum
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd bondlab && .venv/Scripts/python -m pytest tests/test_factors.py -q`
Expected: FAIL with `ModuleNotFoundError: bondlab.risk.factors`.

- [ ] **Step 4: Implement `risk/factors.py` (PCA part)**

- Changes: `X = 10_000 * np.diff(Y, axis=0)`, centred by column mean.
- Covariance `S = Xᵀ X / (N - 1)` with N = number of daily changes (sample, ddof=1). This is the spec's `(n−2)` with n = number of days. `np.linalg.eigh`, reordered descending.
- Signs per the spec, step 3:
  - level: sum > 0;
  - slope: last − first > 0;
  - curvature: middle (`m // 2`) above the mean of the ends.
- Paths: `np.cumsum(10_000 * np.diff(Y, axis=0) @ V[:, :3], axis=0)`, from **raw, uncentred** changes.
- Minimum: refuse when `n_changes < max(MIN_OBSERVATIONS, 3 * m)` or `m < 3`.
- Module and function docstrings cite:
  - Litterman, R. and Scheinkman, J. (1991), "Common Factors Affecting Bond Returns", *Journal of Fixed Income* 1(1), 54–61;
  - Jolliffe, I. T., *Principal Component Analysis*, 2nd ed., §1.1 and §2.1 (definition, sign indeterminacy);
  - Tuckman, B. and Serrat, A., *Fixed Income Securities*, 3rd ed., Ch. 6 (empirical hedging and principal components). Check this chapter number against the book's table of contents if available. If you can't check it, cite "the chapter on empirical approaches to risk metrics and hedging" and add the item to the deferred list.

- [ ] **Step 5: Run tests, mypy, ruff**

Run: `.venv/Scripts/python -m pytest tests/test_factors.py -q && .venv/Scripts/python -m mypy && .venv/Scripts/python -m ruff check .`
Expected: all pass; mypy `Success`; ruff `All checks passed!`

- [ ] **Step 6: Commit** in bondlab: `git add src/bondlab/risk/factors.py tests/test_factors.py tests/golden/treasury_2019_2023_pca.csv tests/golden/treasury_SOURCE.txt`, message "Curve PCA: eigen-decomposition of daily par-yield changes".

---

### Task 2: Par-yield DV01s by the chain rule (bondlab)

**Files:**
- Modify: `bondlab/src/bondlab/data/treasury.py` (add `treasury_quote`)
- Modify: `bondlab/src/bondlab/risk/factors.py`
- Test: `bondlab/tests/test_factors.py`, `bondlab/tests/test_treasury.py`

**Interfaces:**
- Consumes:
  - `treasury_curve(day, scheme) -> TreasuryCurve` (`.curve`, `.used`);
  - `key_rate_durations(pricer, curve) -> tuple[KeyRateDuration, ...]`;
  - `bond_curve_pricer`, `CurvePricer`;
  - `DiscountCurve.with_pillar_shifts`.
- Produces:
  - `treasury_quote(curve: DiscountCurve, tenor: Tenor) -> float` in `data/treasury.py`. A bill (≤ 6 months) gives the simple ACT/365 rate `(1/df − 1)/τ`; a longer tenor gives `treasury_par_yield(curve, maturity)`. It's the inverse of how `treasury_curve` reads each quote.
  - `quote_jacobian(curve: TreasuryCurve) -> tuple[tuple[float, ...], ...]`: `J[t][s]`, how much quote t moves per unit shift of zero pillar s. Central difference, 1bp, via `with_pillar_shifts`.
  - `par_yield_dv01s(pricer: CurvePricer, curve: TreasuryCurve, jacobian: Sequence[Sequence[float]]) -> tuple[KeyRateDuration, ...]`. It's one per `curve.used` tenor, with `pillar_date = tenor.maturity_from(curve day)` and `dv01` positive for a price fall. `duration = dv01 / (price · 1e-4)`. The docstring says these are **par-yield** key rates, as opposed to the zero-rate ones in `risk/bump.py`.

- [ ] **Step 1: Write the failing tests**

```python
# test_treasury.py
def test_quotes_read_back_from_the_bootstrapped_curve():
    # for the 2026 fixture day: treasury_quote(built.curve, t) == day.yields[t] to 1e-10 for every used t

# test_factors.py
@pytest.mark.parametrize("maturity", [date(2028, 11, 15), date(2036, 11, 15), date(2051, 11, 15)])
def test_chain_rule_matches_rebuilding_from_bumped_par_yields(maturity):
    # curve day = last fixture day; restrict to the ten tenors; scheme MonotoneConvex
    # independent: for each tenor rebuild treasury_curve with that par yield +/-1bp,
    # reprice the bond, dv01_t = (P_down - P_up)/2
    assert [k.dv01 for k in par_yield_dv01s(...)] == pytest.approx(rebuilt, abs=1e-4)

def test_parallel_par_shift_equals_sum_of_par_dv01s():
    # rebuild with every par yield +/-1bp: (P_down - P_up)/2 == sum(dv01s) to 1e-4
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_factors.py tests/test_treasury.py -q`
Expected: FAIL (`treasury_quote` and `par_yield_dv01s` not defined).

- [ ] **Step 3: Implement**

- `k = key_rate_durations(pricer, curve.curve)` (zero-rate DV01s, positive).
- `J` from `quote_jacobian`.
- Solve `Jᵀ d = k` with `np.linalg.solve`. A singular J raises `ValueError` naming the pillars.
- Docstring formula: `k_s = Σ_t d_t J_ts`. Cite:
  - Tuckman and Serrat, 3rd ed., Ch. 5 (key-rate and par-rate exposures);
  - Hagan and West (2006) for the refit under the same scheme.
- Mark the rebuild test `@pytest.mark.slow` only if the repo already has that marker in `pyproject.toml`. Otherwise keep it unmarked, with three bonds.

- [ ] **Step 4: Run tests, mypy, ruff.** Same commands as Task 1, Step 5. Expected: pass.

- [ ] **Step 5: Commit:** "Par-yield DV01s from zero key rates and the quote Jacobian".

---

### Task 3: Exposures, factor hedge, P&L risk (bondlab)

**Files:**
- Modify: `bondlab/src/bondlab/risk/factors.py`
- Test: `bondlab/tests/test_factors.py`

**Interfaces:**
- Consumes: `CurvePCA`, `par_yield_dv01s` output (`Sequence[KeyRateDuration]` in `pca.tenors` order).
- Produces:
  - `factor_exposures(dv01s: Sequence[KeyRateDuration], pca: CurvePCA) -> tuple[float, float, float]`: price change per 100 face for a +1 standard deviation daily move of each factor, `e_k = −√λ_k Σ_t dv01_t v_kt`.
  - `combined_dv01s(position: Sequence[KeyRateDuration], hedges: Sequence[Sequence[KeyRateDuration]], notionals: Sequence[float]) -> tuple[KeyRateDuration, ...]`: `d_t + Σ_h n_h d_ht`. The duration field is carried as the position's.
  - `FactorHedge(notionals: tuple[float, ...], matrix: tuple[tuple[float, ...], ...], target: tuple[float, ...], residual_exposures: tuple[float, float, float])`.
  - `factor_hedge(position, hedges, pca) -> FactorHedge`. It solves `E n = −e_position` over the first `len(hedges)` factors (1 to 3). It raises `ValueError` like `key_rate_hedge` when rank < len, or when `cond > 1e12` (reuse the same threshold constant, imported or redefined with the same value).
  - `pnl_volatility(dv01s: Sequence[KeyRateDuration], pca: CurvePCA) -> float`: `√(dᵀ S d)` per 100 face per day, with `S = V Λ Vᵀ` from all m components.
  - All three check `len(dv01s) == len(pca.tenors)` and raise naming both counts.

- [ ] **Step 1: Write the failing tests**

```python
def test_level_slope_hedge_leaves_no_level_or_slope_exposure():
    # position 2036 bond; hedges = par 2Y and 10Y Treasuries off the restricted fixture curve
    h = factor_hedge(position, [two, ten], pca)
    assert h.residual_exposures[:2] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert factor_exposures(combined_dv01s(position, [two, ten], h.notionals), pca)[:2] == pytest.approx((0, 0), abs=1e-9)

def test_pnl_volatility_equals_replayed_history():
    # independent: g = -[d.dv01]; daily P&L_i = g . (10_000 * diff(Y))_i ; np.std(..., ddof=1)
    assert pnl_volatility(dv01s, pca) == pytest.approx(replayed, rel=1e-9)

def test_exposure_is_the_dot_product_of_dv01s_and_scaled_loadings():
    # build dv01s by hand (KeyRateDuration with chosen dv01 values) and a synthetic pca
    # (from the Task 1 synthetic case): e_k == -sigma_k * sum(d_t * V[t, k]) to 1e-12

def test_nearly_collinear_hedges_are_refused():
    # two hedges with identical dv01 vectors -> ValueError mentioning "cannot be told apart"

def test_mismatched_lengths_are_refused():
    # 9 dv01s against a 10-tenor pca -> ValueError naming 9 and 10
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL (names not defined).
- [ ] **Step 3: Implement** with the formulas above. Docstrings cite Litterman and Scheinkman (1991) §§ on hedging with factors, and Tuckman and Serrat, 3rd ed., Ch. 6.
- [ ] **Step 4: Run the full bondlab suite, mypy, ruff**

Run: `.venv/Scripts/python -m pytest -q && .venv/Scripts/python -m mypy && .venv/Scripts/python -m ruff check .`
Expected: every test passes, including all existing ones.

- [ ] **Step 5: Commit:** "Factor exposures, the level/slope hedge and historical P&L risk".

---

### Task 4: The `pca` lab route (bondlab service)

**Files:**
- Modify: `bondlab/src/bondlab/ui/lab_service.py` (new section "curve factors", `_MAX_ENTRIES`, `ROUTES`)
- Test: `bondlab/tests/test_lab_service.py`

**Interfaces:**
- Consumes:
  - Tasks 1 to 3;
  - existing `_position_bond`, `_check_within_curve`, `_scheme`, `_rate`, `_num`, `_list`, `_engine`;
  - `dv01_hedge`, `key_rate_hedge`, `residual_key_rates` from `risk/hedge.py`.
- Produces: route `"pca"`.
  - **Payload:** exactly as in the spec. `history.yields` rows hold percent or `null`.
  - **Response keys** (camelCase; tasks 5 to 7 rely on these):
    - `day`, `from`, `observations`;
    - `tenors: [{tenor, years}]` (years = Act/365 from the curve day to the tenor's maturity, via the curve's `time_to`);
    - `dropped: [{tenor, firstMissing}]`;
    - `eigenvalues`, `explained`, `dailySd` (first 3), `loadings` (3 lists of m);
    - `moves: {days, level, slope, curvature}` (cumulative bp);
    - `position`: either `{problem: string}` or the object below.
  - **`position` object:**
    - `settlement`, `faceAmount`;
    - `dv01s: [{tenor, dv01}]` (scaled to face, positive = loses when yields rise);
    - `exposures: [l, s, c]` (dollars for the face);
    - `system: {matrix, target}` (2×2 and 2, per 100 face, for show-your-work);
    - `rows: [{name: "unhedged"|"dv01"|"key-rate"|"level-slope", hedges: [{tenor, face}], exposures: [l, s, c], dailyRisk}]`.
  - **Limits:** `_MAX_ENTRIES` gains `"days": 2700` and `"tenors": 10`. `hedgeTenors` must be exactly 2 distinct entries from the hedge maturities (`1 Yr`…`30 Yr`), else 422. `faceAmount` uses the existing `_LIMITS` entry.
  - **Bond section:** problems in it never fail the route; their message goes in `position.problem`. They are:
    - the bond lies beyond the restricted curve;
    - a hedge tenor was dropped from the window;
    - singular hedges.
  - **DV01 row:** uses the hedge tenor whose maturity is nearest the position's maturity.

- [ ] **Step 1: Write the failing tests** (build `history` from the Task 1 fixture, last 260 rows, in percent):

```python
def test_pca_route_returns_the_engine_numbers():
    out = handle("pca", payload)
    # eigenvalues equal curve_pca(yield_history(...)).eigenvalues exactly; explained sums to 1 (1e-12)
    # moves lists have observations entries; rows names in order unhedged, dv01, key-rate, level-slope
    # the level-slope row's exposures[0:2] are 0 to 1e-6 dollars

def test_dropped_tenors_are_named():   # null out "20 Yr" on one day -> dropped == [{"tenor": "20 Yr", "firstMissing": that day}]
def test_bond_beyond_the_kept_curve_gets_a_problem_not_an_error():
    # null out all "30 Yr" cells, bond maturing 2051 -> 200-equivalent result, position == {"problem": ...mentions "30 Yr" or the last pillar}
def test_dropped_hedge_tenor_gets_a_problem():   # null "10 Yr" on one day, hedgeTenors ["2 Yr","10 Yr"] -> problem names "10 Yr"
def test_collinear_hedges_get_a_problem():       # hedgeTenors ["1 Yr","2 Yr"] on a synthetic flat history where they cannot be separated -> problem mentions "cannot be told apart"
def test_pca_route_bounds():
    # 2701 rows -> PayloadError; 11 tenors -> PayloadError; one hedge tenor -> PayloadError;
    # "1 Mo" as hedge -> PayloadError; a yield of 99 -> PayloadError; position null -> no "position" key or position None
```
For the collinear case, if a real-data pair can't be made singular, build hedges whose DV01 vectors are proportional by using a synthetic history whose first two loadings are zero at the hedge pillars. The point is that the message passes through.

- [ ] **Step 2: Run to verify failure.** Run: `.venv/Scripts/python -m pytest tests/test_lab_service.py -q -k pca`. Expected: FAIL (`unknown route 'pca'`).
- [ ] **Step 3: Implement `_pca_route`**

1. Parse the history into `TreasuryDay`s, skipping nulls, with yields via `_rate`.
2. `common_tenors` over the candidate list in the payload's tenor order.
3. `yield_history`, then `curve_pca`.
4. Curve: `treasury_curve(TreasuryDay(last day, kept tenors only), scheme)`.
5. If a position is given:
   - `_position_bond`, then `_check_within_curve`, catching `PayloadError` into `problem`;
   - the hedge bonds as in `_hedge_route`;
   - `quote_jacobian` once, then `par_yield_dv01s` for the position and both hedges;
   - the four rows via `dv01_hedge`, `key_rate_hedge` (on par DV01s), `factor_hedge`, `combined_dv01s`, `factor_exposures` and `pnl_volatility`;
   - scale per-100 figures by `faceAmount / 100`.

The service computes nothing beyond scaling and unit conversion.

- [ ] **Step 4: Run the whole bondlab suite, mypy, ruff.** Same as Task 3, Step 4. Expected: pass.
- [ ] **Step 5: Measure the time.** Run the route once on a 2,520-row synthetic history with a position, timed with `time.perf_counter`. Expected: under 2 s on this laptop. If it's over, reduce the work (for example, reuse the shifted curves across bonds) before moving on.
- [ ] **Step 6: Bump the version.** `pyproject.toml` version `0.3.0`. Add the stage 3 line to README's feature list if it has one.
- [ ] **Step 7: Commit:** "Lab route pca: factors, exposures and the hedge comparison; version 0.3.0". Do not tag yet: Task 9 tags after the review.

---

### Task 5: API route (quant-core)

**Files:**
- Modify: `quant-core/api/fixed_income.py` (`ROUTES`, and a per-route cap only if needed)
- Modify: `quant-core/api/Dockerfile` (`ARG BONDLAB_VERSION=v0.3.0`)
- Test: `quant-core/tests/test_fixed_income_api.py`

**Interfaces:**
- Consumes: `handle("pca", payload)` from bondlab (installed editable from `../bondlab` in quant-core's `.venv`; check with `pip show bondlab`).
- Produces: `POST /fixed-income/pca`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pca_route_is_served():   # 260-row history built from CURVE-like values with small deterministic noise; 200; body has "eigenvalues" and "position"
def test_pca_ten_year_payload_fits_the_cap():
    # 2,520 rows x 10 tenors, days ISO, yields like 4.27 (two decimals), position + hedgeTenors
    body = json.dumps(payload).encode()
    print(len(body))            # record the size in the commit message
    assert client.post("/fixed-income/pca", content=body, headers={"content-type": "application/json"}).status_code == 200
def test_pca_bad_input_is_422():  # one hedge tenor -> 422 with the bondlab message
```
Also add `"pca"` to the `CASES` dict if the file's existing parametrised tests iterate it.

- [ ] **Step 2: Run to verify failure.** Run: `cd quant-core && .venv/Scripts/python -m pytest tests/test_fixed_income_api.py -q`. Expected: FAIL (404, unknown route).
- [ ] **Step 3: Implement.**
  - Add `"pca"` to `ROUTES`.
  - If the measured 10-year body is over 200,000 bytes, set a per-route cap: `MAX_BODY_BYTES_BY_ROUTE = {"pca": <next 50 KB step above measured>}`, used by `_capped_body`. Other routes stay at 200 KB, with a test that a 250 KB body to `bond` still gets 413.
  - Set the Dockerfile ARG to `v0.3.0`.
- [ ] **Step 4: Run the full quant-core suite.** Run: `.venv/Scripts/python -m pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** in quant-core: "API: /fixed-income/pca, bondlab v0.3.0".

---

### Task 6: Window history endpoint (site)

**Files:**
- Modify: `site/kit/treasury-data.ts` (add `yieldsBetween`)
- Create: `site/bonds/src/lib/history-data.ts`
- Create: `site/bonds/src/pages/history.json.ts`
- Test: `site/bonds/tests/history.test.mjs`

**Interfaces:**
- Consumes: `dayOnOrBefore`, `latestDay`, `FIRST_DAY` (kit); `isIsoDay` (`lib/state.ts`).
- Produces:
  - `yieldsBetween(db, from: string, to: string, tenors: readonly string[]): Promise<{ day: string; tenor: string; yield_pct: number }[]>`: one prepared query, `WHERE day BETWEEN ? AND ? AND tenor IN (?,…)`, `ORDER BY day`.
  - `PCA_TENORS` (the 10 candidates, in maturity order) and `WINDOW_YEARS = [1, 2, 5, 10] as const` in `history-data.ts`.
  - `windowStart(day: string, years: number): string`: same month and day `years` earlier; Feb 29 becomes Feb 28.
  - `StoredHistory { day: string; years: number; from: string; tenors: string[]; days: string[]; yields: (number | null)[][] }`.
  - `historyPayload(db, requestedDay: string | null, years: string | null, today: string): Promise<{status: 200, body: StoredHistory} | {status: 400 | 404 | 503, body: {error: string}}>`. `from` is the first stored day returned, so a window that reaches before 1990 starts at the first stored day.
  - `GET /history.json?day=&years=`:
    - on 200, uses `caches.default` when defined, keyed on `https://bonds.internal/history/<resolvedDay>/<years>`, with `cache-control: public, max-age=86400`;
    - on errors, `no-store`.

- [ ] **Step 1: Write the failing tests** (fake DB in the `curve.test.mjs` style; it records SQL and binds):

```js
test("a five-year window pivots stored rows into one row per day, nulls where unpublished", ...)
  // rows for 2 days, "20 Yr" missing on the first -> yields[0][8] === null; tenors === PCA_TENORS
test("years must be 1, 2, 5 or 10, and the day a real date", ...)   // "3" -> 400; "abc" -> 400; "2026-02-30" -> 400
test("a window reaching before 1990 starts at the first stored day", ...)  // day 1990-06-29, years 10 -> from === first row's day
test("one query, bounded by the window and the ten tenors", ...)    // binds include windowStart and the resolved day and exactly 10 tenors
test("windowStart keeps month and day, with 29 February moving to the 28th", ...)
```

- [ ] **Step 2: Run to verify failure.** Run: `cd site/bonds && pnpm test`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement** the three files to those signatures. In `history.json.ts`, mirror `curve.json.ts`, plus the Cache API: `caches.match(key)`, then compute, then `caches.put(key, response.clone())` on 200. Guard with `typeof caches !== "undefined"`.
- [ ] **Step 4: Run tests.** Run: `pnpm test`. Expected: all pass, the existing ones included.
- [ ] **Step 5: Commit** in site: "Bonds: cached window history for the factors tab".

---

### Task 7: The Factors tab (site)

**Files:**
- Modify: `site/bonds/src/lib/types.ts`, `site/bonds/src/lib/state.ts`, `site/bonds/src/lib/api.ts`, `site/bonds/src/lib/track.ts`, `site/bonds/src/components/LabApp.svelte`, `site/bonds/src/data/settings.json`
- Create: `site/bonds/src/components/FactorsPanel.svelte`, `site/bonds/src/components/VarianceBars.svelte`
- Test: `site/bonds/tests/state.test.mjs` and a new `site/bonds/tests/factors.test.mjs`

**Interfaces:**
- Consumes: `/history.json` (Task 6); `POST /fixed-income/pca` response keys (Task 4); `LineChart.svelte` props `{series, xLabel, yLabel, xUnit?, yDigits?, marker?, label}`; `Work.svelte`.
- Produces:
  - `Area` gains `"factors"`; `AREAS` and `LABELS` gain `factors: "Factors"`; `FLAG.factors = "factors"`.
  - `FactorsInput { years: 1 | 2 | 5 | 10; hedgeTenors: [string, string] }`; `LabState.factors?: FactorsInput`.
  - `isValidFactors(f)`: years in the set; two distinct `HEDGE_TENORS`.
  - Link key `f`. For area `factors` the link carries `b` (bond), `h` (hedge, for its `faceAmount`) and `f`.
  - `pcaRequest(history: StoredHistory, scheme: Scheme, bond: BondInput, faceAmount: number, factors: FactorsInput)` in `state.ts` builds the spec's payload: position `{preset, coupon, maturity}`.
  - `fetchPca` in `api.ts`. `PcaResponse` and `PcaPosition` types mirror the Task 4 keys.
  - `EVENTS` gains `"factors"`, tracked on the first successful PCA.
  - `settings.json` `flags.factors: true`.

- [ ] **Step 1: Write the failing tests**

```js
// state.test.mjs
test("a factors view with bond, face amount, window and hedge tenors round-trips", ...)
test("factors links out of bounds are refused", ...)  // years 3; tenors ["2 Yr","2 Yr"]; ["1 Mo","10 Yr"]; one tenor
// factors.test.mjs
test("pcaRequest sends the history as fetched and only the bond's terms", ...)
  // no mode/spreadBp/yield in position; hedgeTenors copied; faceAmount from argument
```

- [ ] **Step 2: Run to verify failure.** Run: `pnpm test`. Expected: FAIL.
- [ ] **Step 3: Implement the state, types, api and track changes.** Run `pnpm test`; expected PASS.
- [ ] **Step 4: Build `FactorsPanel.svelte`.** Props: `{ apiBase, stored, scheme, bond, hedge, factors (bindable) }`. Layout, top to bottom, from the spec:
  - **Window and hedges:** buttons 1y, 2y, 5y and 10y, plus two hedge-tenor selects from `HEDGE_TENORS`.
  - **Header line:** curve day · n daily changes · tenors used. Dropped tenors are listed with "not published on <date>".
  - **Loadings:** `LineChart` with x = `tenors[].years` and three series: level, slope, curvature.
  - **Explained variance:** `VarianceBars` draws a hand SVG of the first three bars and "rest", with percent labels.
  - **How the factors moved:** `LineChart` with x = decimal year of each day, computed only for drawing, and three series.
  - **Your bond:** a one-line summary of the bond from Price a bond, with a button that switches to that tab. Then the exposures table: factor, a typical day (√λ bp), and P&L for the face.
    - If the response's `position.problem` is set, show it instead.
  - **Hedge comparison table:**
    - columns: Unhedged | DV01 | Key-rate (par) | Level/slope;
    - rows: hedge faces; level, slope and curvature exposure left; daily risk left.
  - **Show your work (`Work.svelte`):**
    - matrix size and dropped tenors;
    - `S = XᵀX/(N−1)`, with N the number of daily changes;
    - λ₁..₃ and √λ;
    - the level exposure as `−√λ₁ · Σ dv01ₜ v₁ₜ` with the actual first three terms written out, then "…", then the total;
    - the 2×2 system `matrix · n = −target` with numbers.

  Fetch flow:
  - On mount, or when the day, years, bond, face, hedge tenors or scheme change, debounce 300 ms;
  - `fetch('/history.json?day=…&years=…')`, then `fetchPca`;
  - abort the previous call with `AbortController`, as the other panels do;
  - loading and error states as in `HedgePanel.svelte`.
- [ ] **Step 5: Wire it into `LabApp.svelte`.**
  - Add the `factors` state, default `{ years: 5, hedgeTenors: ["2 Yr", "10 Yr"] }`.
  - Restore from a link.
  - Encode for area `factors`: bond, hedge and factors.
  - Render `{:else if area === "factors"} <FactorsPanel … bind:factors />`.
- [ ] **Step 6: Check.** Run: `cd site/bonds && pnpm test && pnpm check`, then `cd ../options && pnpm check`. Expected: tests pass, svelte-check `0 errors and 0 warnings` in both.
- [ ] **Step 7: Commit:** "Bonds: Factors tab (loadings, variance, factor paths, exposures, hedge comparison)".

---

### Task 8: Pages and hub switch (site)

**Files:**
- Modify: `site/bonds/src/pages/methodology.astro`, `site/bonds/src/pages/limits.astro`
- Modify: `site/hub/src/lib/validate.ts` (`BOND_FLAGS`), `site/hub/src/lib/labels.ts`
- Test: `site/hub/tests/tools.test.mjs`

**Interfaces:**
- Consumes: `flags.factors` (Task 7), `EVENTS` `"factors"` (Task 7).
- Produces:
  - `BOND_FLAGS` entry `{ key: "factors", label: "Curve factors", help: "Level, slope and curvature from daily Treasury moves, and a factor hedge." }`;
  - label `factors: "Factor analyses run"` under `bonds`.

- [ ] **Step 1: Update the hub test.** Change the `flagsFromForm(form, BOND_FLAGS)` expectation to include `factors: false`. Run `cd site/hub && node --test tests/*.test.mjs` (or the package's test script). Expected: FAIL until validate.ts and labels.ts change.
- [ ] **Step 2: Make the change.** Edit `validate.ts` and `labels.ts`, then run the test again. Expected: PASS.
- [ ] **Step 3: Methodology section "Curve factors (PCA)".** Write the spec's steps 1 to 8 in plain prose, matching the page's existing style (headings, short paragraphs, formulas as the page already writes them). Include:
  - the tenor-set rule and the data facts table (3M December 2008, 20Y from 1993-10-01, the 30Y gap);
  - the chain-rule note;
  - sources: Litterman and Scheinkman (1991), Jolliffe, Tuckman and Serrat.
- [ ] **Step 4: Limits additions:**
  - in-sample;
  - the window changes the answer;
  - exposures are linear (first order);
  - the tenor set changes over time;
  - a weekend or holiday change counts as one day;
  - par yields come from the Treasury's fitted curve, not traded bonds;
  - the hedge comparison is judged on the window it was fitted on;
  - the factors-tab curve uses only the window's tenors, so it can differ slightly from the Price tab.
- [ ] **Step 5: Build the hub.** Run: `cd site/hub && pnpm build`. Expected: build completes with no errors. Also run `cd ../bonds && pnpm build`. Expected: completes.
- [ ] **Step 6: Commit:** "Factors: methodology, limits and the hub switch".

---

### Task 9: Verify everything, browser check, review, release prep

**Files:**
- Modify: `C:\Users\user\Desktop\blessedshoh\.claude\launch.json`. It has no `bonds` entry: add `{ "name": "bonds", "runtimeExecutable": "pnpm", "runtimeArgs": ["--dir", "C:/Users/user/Desktop/blessedshoh/site/bonds", "dev", "--port", "4323"], "port": 4323 }`. Check how `settings.json` `api_base` is overridden for local dev (look at `bonds/src/pages/index.astro` and the env) so the page calls `http://localhost:8000`.
- Notes: `C:\Users\user\AppData\Roaming\Claude\side-session-notes\local_0b999464-ca32-4f52-a770-3b1eab363c23\stage-3-handover.md`

- [ ] **Step 1: Run every suite fresh and keep the output.**
  - bondlab: `pytest -q`, `mypy`, `ruff check .`
  - quant-core: `pytest -q`
  - site/bonds: `pnpm test`, `pnpm check`
  - site/options: `pnpm check`
  - site/hub: tests and `pnpm build`
  - Expected: all green.
- [ ] **Step 2: Browser check.** Start the `api`, `bonds` and `hub` previews. In Factors, check:
  - the default window on the latest day;
  - 10 years on the latest day (no 413, time noted);
  - curve day 2008-12-31 with 1 year (3M dropped);
  - 1992-06-30 with 2 years (20Y dropped);
  - 2004-06-30 with 5 years: a 2020 bond (works), a 2030 bond (problem message: beyond the 20Y end), then the hedge tenor 30Y (problem message);
  - 1990-01-03 (clear error);
  - a share-link round trip;
  - phone width (375) and dark mode.

  The console has no errors. The hub dashboard shows the Curve factors switch. Take screenshots for the owner.
- [ ] **Step 3: Whole-branch review.**
  - Dispatch one fresh reviewer (Agent tool, `model: "opus"`) over all three branches' diffs against `main`, with the spec and this plan.
  - Fix confirmed findings with tests, and re-run Step 1.
- [ ] **Step 4: Prepare the release, without publishing.**
  - Tag `v0.3.0` locally in bondlab: `git tag -a v0.3.0 -m "Curve PCA"`.
  - Write the hand-over note: branches, commits, the deferred list, and the exact push and deploy commands.
- [ ] **Step 5: Ask the owner for OK** to push and deploy. Nothing below happens without it.

### Task 10: Deploy (only after the owner's OK)

- [ ] **Step 1: bondlab.** Push the named refs only: `git push origin fixed-income-lab-stage-3`, then (after merge OK) `git push origin main`, then `git push origin v0.3.0`.
- [ ] **Step 2: quant-core.** Merge to main and push, then `fly deploy` the API. Check:
  - `curl -X POST https://api.blessedshoh.dev/fixed-income/pca` with a small payload gives 200;
  - an existing route still gives 200.
- [ ] **Step 3: site.** Merge to main and push. Deploy bonds, then hub, with `pnpm --dir site/bonds deploy` and `pnpm --dir site/hub deploy`.
- [ ] **Step 4: Verify live** at https://bonds.blessedshoh.dev in the browser: Factors loads, a 10-year window works, and the hub switch is visible.
- [ ] **Step 5: Report** the rulings and deferred items to the owner.
