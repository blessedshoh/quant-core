# IV-crush lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a position carry one earnings event, so the tool shows implied volatility collapsing after the announcement, with an implied-move calculator and walkthrough 03.

**Architecture:** The event lives on `Position` in the engine. One function, `effective_vol`, decides each leg's volatility at any date; `leg_price` uses it, so the grid, Greeks, risk and show-your-work inherit the crush. The API passes an optional `event` through every position route and adds `/options/implied-move`. The front end adds an earnings control, a "day after earnings" curve, a play button and link encoding.

**Tech Stack:** Python 3.14, numpy, scipy, pytest, FastAPI (quant-core); Astro 7, Svelte 5, TypeScript, node:test (site/options); Astro on Cloudflare Workers + D1 (site/hub).

**Spec:** `quant-core/docs/superpowers/specs/2026-11-19-iv-crush-lab-design.md`

## Global Constraints

- No new dependencies in either repository.
- Event limits: `days` in [1, 1825], `move` in [0.005, 0.5] (one-day standard deviation of the log jump).
- With no event, every existing field of every API answer is unchanged; new keys are null.
- The browser does no option maths.
- Links made before this change must still open.
- The event happens before the open on day `e`: at `days_forward >= e` it has happened.
- Volatility scale and shift apply to the base volatility; the event move is never scaled or shifted.
- Grid day axis gains `e − 1e-4` and `e` when `0 < e < first expiry`.
- Feature switch key `event`, on by default, label "Earnings (IV crush)"; usage event `event`; numbers label "Earnings lab used".
- Preset "Straddle into earnings": at-the-money 30-day straddle, 1 contract each side, vol 0.70, event 5 days, move 0.08.

## Review Focus

1. **Event on or after the first expiry** (calendar spread whose short leg expires before earnings): grid has no extra days, chart shows no crush curve, headline says "Earnings after the first expiry: priced only into later legs". Test in Task 1 and Task 6.
2. **Leg volatility too low for the move** (σ²T ≤ m²): API answers 422 naming the leg; the tool shows the message and keeps the last chart. Test in Task 1 and Task 3.
3. **Old links without `e`, and links with a bad `e`**: old ones open unchanged, bad ones are refused like any damaged link. Test in Task 4.
4. **Theta next to the event**: bumped theta at `e − 1e-4` must not step across the event (cap the step at the time left before the event). Test in Task 1.
5. **American position with an event**: grid still returns, with the two extra days. Test in Task 2.

---

### Task 1: Event model in the engine

**Files:**
- Modify: `quant-core/src/quantcore/position.py`
- Modify: `quant-core/src/quantcore/__init__.py` (export `Event`)
- Test: `quant-core/tests/test_event.py` (create)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Event: days: float; move: float` — `__post_init__` raises `ValueError` outside the Global Constraints limits.
  - `Position.event: Event | None = None` (new last field). `Position.__post_init__` calls `base_vol` for every option leg so a bad leg fails at construction.
  - `event_applies(leg: Leg, position: Position) -> bool` — option leg, event set, `leg.days > event.days`.
  - `base_vol(leg: Leg, position: Position) -> float` — `leg.vol` when the event does not apply, else `sqrt((σ²T − m²)/T)`; raises `ValueError(f"leg {i+1}'s {σ:.0%} volatility over {days:g} days cannot contain a ±{m:.0%} earnings move")` when `σ²T ≤ m²` (i = leg index in `position.legs`).
  - `effective_vol(leg, position, days_forward, vol_scale=1.0, vol_shift=0.0) -> np.ndarray` — `b = max(base_vol·scale + shift, MIN_VOL)`; if the event applies and `days_forward < event.days`: `sqrt((b²τ + m²)/τ)` with τ = remaining years (where τ = 0 return `b`); otherwise `b`. With no event this must equal `np.maximum(leg.vol * vol_scale + vol_shift, MIN_VOL)` exactly.
  - `leg_price` uses `effective_vol` in place of its own volatility line.
  - `net_greeks`: legs where `event_applies` go through `_bumped_greeks`. In `_bumped_greeks`, the theta step is `min(0.25, remaining, time_to_event)` where `time_to_event = event.days − days_forward` when positive, else no cap.

- [ ] **Step 1: Write the failing tests** in `tests/test_event.py`:
  - `test_no_event_changes_nothing`: for the five positions in `tests/test_position.py` (import or rebuild them), `profit`, `entry_cost`, `net_greeks` at prices 150, 166, 190 and days 0, 10 are equal (`==`, not approx) with `event=None` and with the field omitted.
  - `test_event_leaves_today_unchanged`: straddle S=166, K=165, 30 days, vol 0.70, r=0.04; with `Event(5, 0.08)`: `entry_cost` and today's `value` equal the no-event ones to 1e-9; `net_greeks` delta and gamma within 1e-4 relative of the no-event closed forms.
  - `test_crush_on_the_day`: same straddle; `value` at day 5 equals `bs.price` at `base_vol` with 25 days left (call + put, ×100) to 1e-9; `value` at day 5 − 1e-4 minus value at day 5 > 0.
  - `test_base_vol_formula`: `base_vol` = `sqrt((0.7**2*30/365 - 0.08**2)/(30/365))` to 1e-12.
  - `test_leg_before_event_unaffected`: leg with days 3 and `Event(5, 0.08)`: `base_vol == leg.vol`, `event_applies` False.
  - `test_too_low_vol_refused`: vol 0.25, days 10, `Event(5, 0.08)` → `ValueError` whose message contains "cannot contain".
  - `test_event_limits`: `Event(0.5, 0.08)`, `Event(5, 0.001)`, `Event(5, 0.6)` raise `ValueError`.
  - `test_monte_carlo_matches_effective_vol`: 1,000,000 paths of diffusion at `base_vol` plus one normal log jump N(−m²/2, m²); discounted call payoff mean within 3 standard errors of `bs.price` at `effective_vol(day 0)`.
  - `test_shift_moves_base_only`: `effective_vol(..., days_forward=0, vol_shift=0.1)` equals `sqrt(((b+0.1)²τ + m²)/τ)`.
  - `test_theta_near_event_is_finite`: `net_greeks(straddle_with_event, days_forward=5 - 1e-4)["theta_per_day"]` has absolute value below 10 × the no-event theta at day 0.

- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/test_event.py -q` — expected: fails on `ImportError: Event`.
- [ ] **Step 3: Implement** the Interfaces above in `position.py`.
- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest -q -p no:warnings` — expected: all pass (1,906 existing plus the new ones).
- [ ] **Step 5: Commit** `git add src/quantcore/position.py src/quantcore/__init__.py tests/test_event.py` and `git commit -m "Engine: one earnings event per position, split volatility, crush after the event"`.

### Task 2: Grid days, implied move, show-your-work

**Files:**
- Modify: `quant-core/src/quantcore/grid.py`
- Create: `quant-core/src/quantcore/earnings.py`
- Modify: `quant-core/src/quantcore/work.py`
- Test: `quant-core/tests/test_event.py` (extend), `quant-core/tests/test_work.py` (extend)

**Interfaces:**
- Consumes: `Event`, `event_applies`, `base_vol`, `effective_vol` (Task 1).
- Produces:
  - `grid.EVENT_EPSILON_DAYS = 1e-4`; `day_axis` appends `e − ε` and `e` (then `np.unique`) when `0 < e < first_expiry`, for European and American axes.
  - `earnings.ImpliedMove` dataclass: `move: float, expected_abs_move: float, base_vol: float, event_variance_share: float` (share = m² / (σ₁²T₁)).
  - `earnings.implied_move(near_days: float, near_vol: float, far_days: float, far_vol: float) -> ImpliedMove` — formulas from the spec's Model section; `expected_abs_move = move * sqrt(2/pi)`; `ValueError` when `near_days >= far_days`, when σ_b² ≤ 0 ("the earlier expiry's volatility is too high for one shared base volatility"), or when m² ≤ 0 ("the later expiry is the more expensive one, so these prices imply no earnings premium").
  - `earnings.event_vols(position) -> list[tuple[int, float, float, float, float]]` — `(leg index, base_vol, vol_before, vol_after, weight)` for legs where the event applies; `vol_before = leg.vol`, `vol_after = base_vol`, `weight = |size| × vega today` (the same weight as `probability.vol_weights`).
  - `work.LegWork` gains `base_vol: float | None`, `vol_after_event: float | None` (None when the event does not apply); `work.Work` gains `event: tuple[float, float] | None` (days, move).

- [ ] **Step 1: Write failing tests:**
  - `test_day_axis_has_the_event_step`: straddle with `Event(5, 0.08)`: `day_axis` contains `5 - 1e-4` and `5`; without event, the axis equals today's 25 cubic points exactly.
  - `test_event_after_first_expiry_adds_no_days`: calendar (short call 3 days, long call 30 days, `Event(5, 0.08)`): `day_axis` has 25 points ending at 3.
  - `test_american_grid_with_event`: American straddle with `Event(5, 0.08)`: `profit_grid` returns; `grid.days` contains 5.
  - `test_implied_move_round_trip`: choose σ_b = 0.35, m = 0.07, T₁ = 7/365, T₂ = 35/365; σ₁ = sqrt(σ_b² + m²/T₁), σ₂ likewise; `implied_move(7, σ₁, 35, σ₂)` recovers move and base_vol to 1e-10; `expected_abs_move == move*sqrt(2/pi)`.
  - `test_implied_move_refusals`: `implied_move(35, 0.4, 7, 0.5)` and `implied_move(7, 0.3, 35, 0.5)` raise `ValueError`.
  - in `test_work.py`, `test_work_shows_the_split`: straddle with event: each leg's `base_vol` equals `base_vol(...)`, `vol_after_event == base_vol`, `Work.event == (5, 0.08)`; without event they are None.
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/test_event.py tests/test_work.py -q` — expected: the new tests fail.
- [ ] **Step 3: Implement** the Interfaces above.
- [ ] **Step 4: Run** the full suite — expected: all pass.
- [ ] **Step 5: Commit** `git commit -m "Engine: event days on the grid, implied-move calculator, split in show your work"`.

### Task 3: API

**Files:**
- Modify: `quant-core/api/main.py`
- Test: `quant-core/tests/test_api.py` (extend)

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces:
  - `class EventIn(BaseModel): days: float = Field(ge=1, le=1825); move: float = Field(ge=0.005, le=0.5)`.
  - `GridRequest.event: EventIn | None = None`; `_position` passes `Event(**event.model_dump())` or None. `ValueError` from construction already maps to 422 in every route.
  - `class EventLegOut(BaseModel): index: int; base_vol: float; vol_before: float; vol_after: float; weight: float`; `class EventSummary(BaseModel): days: float; move: float; legs: list[EventLegOut]`; `Summary.event: EventSummary | None = None`, filled from `earnings.event_vols` (rounded to 6 decimals).
  - `LegWorkOut` gains `base_vol: float | None`, `vol_after_event: float | None`; `WorkResponse` gains `event: EventIn | None`.
  - `POST /options/implied-move`, request `ImpliedMoveRequest {near_days: float(gt 0, le 1825), near_vol: float(ge 0.01, le 5), far_days, far_vol}`, response `ImpliedMoveResponse {move, expected_abs_move, base_vol, event_variance_share}` (6 decimals), 422 with the engine's message.
- [ ] **Step 1: Write failing tests:**
  - `test_no_event_answers_unchanged`: for `LONG_CALL` and a 4-leg condor, the grid, risk, explain and work answers with `"event": None` equal the answers without the key, and `summary["event"] is None`.
  - `test_event_on_every_route`: straddle body with `"event": {"days": 5, "move": 0.08}` returns 200 on grid, risk, explain, work; grid `axes.days` contains 5 and `summary.event.legs` has two entries with `vol_after < vol_before`.
  - `test_event_limits_enforced`: `days` 0.5 and `move` 0.6 → 422.
  - `test_too_low_vol_is_a_422_with_the_reason`: vol 0.25, 10 days, move 0.08 → 422 and `"cannot contain"` in `detail`.
  - `test_implied_move_route`: round-trip values from Task 2 within 1e-5; swapped expiries → 422.
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/test_api.py -q` — expected: new tests fail.
- [ ] **Step 3: Implement** the Interfaces above.
- [ ] **Step 4: Run** the full suite — expected: all pass.
- [ ] **Step 5: Commit** `git commit -m "API: optional earnings event on position routes, implied-move route"`.

### Task 4: Front-end state, types and API client

**Files:**
- Modify: `site/options/src/lib/types.ts`, `site/options/src/lib/state.ts`, `site/options/src/lib/api.ts`
- Test: `site/options/tests/state.test.mjs` (extend)

**Interfaces:**
- Consumes: Task 3 JSON shapes.
- Produces:
  - `types.ts`: `export interface EarningsEvent { days: number; move: number }`; `Position.event?: EarningsEvent`; `Summary.event: { days: number; move: number; legs: { index: number; base_vol: number; vol_before: number; vol_after: number }[] } | null`; `LegWork.base_vol: number | null`, `LegWork.vol_after_event: number | null`; `WorkResponse.event: EarningsEvent | null`; `ImpliedMoveResponse { move; expected_abs_move; base_vol; event_variance_share }`.
  - `state.ts`: `LIMITS.eventDays = [1, 1825]`, `LIMITS.eventMove = [0.005, 0.5]`; `clonePosition` copies `event` when present; `isValidPosition` checks event limits when present; `encodeState` adds `e: [days, move]` only when an event is set (version stays `v: 1`); `decodeState` reads `e` when present.
  - `api.ts`: `positionBody` sends `event` when present; `fetchImpliedMove(apiBase, request: {near_days; near_vol; far_days; far_vol}, signal?) -> Promise<ImpliedMoveResponse>`.
- [ ] **Step 1: Write failing tests** in `state.test.mjs`:
  - `"a link with an earnings event round-trips"`: position with `event {days: 5, move: 0.08}` → encode → decode gives the same event.
  - `"links without an event still open, with no event"`: an encoded string produced by today's `encodeState` (fixture literal) decodes with `event` undefined.
  - `"links with a bad event are refused"`: `e: [0.2, 0.08]` and `e: [5, 0.9]` → `decodeState` returns null.
- [ ] **Step 2: Run** `pnpm --dir site/options test` — expected: new tests fail.
- [ ] **Step 3: Implement** the Interfaces above.
- [ ] **Step 4: Run** `pnpm --dir site/options test` and `pnpm --dir site/options run check` — expected: all pass, 0 errors.
- [ ] **Step 5: Commit** (site repo) `git commit -m "Options tool: earnings event in position state, links and API client"`.

### Task 5: Earnings control, preset and settings plumbing

**Files:**
- Create: `site/options/src/components/EarningsControl.svelte`
- Modify: `site/options/src/lib/templates.ts`, `site/options/src/components/LegsEditor.svelte`, `site/options/src/data/settings.json`, `site/options/src/lib/track.ts`, `site/hub/src/lib/validate.ts`, `site/hub/src/pages/admin/numbers.astro`

**Interfaces:**
- Consumes: Task 4 types and `fetchImpliedMove`.
- Produces:
  - `EarningsControl.svelte` props: `event: EarningsEvent | undefined` (`$bindable`), `apiBase: string`, `maxDays: number` (first expiry), `onenable?: () => void`. A toggle "Earnings before expiry"; when on, fields "Days until earnings" (1 to maxDays) and "Expected move ±%" (0.5 to 50, stored /100); defaults 5 days, 8%. A collapsible "What move is priced in?" card with near/far days and vol % inputs, calling `fetchImpliedMove` and showing move, expected absolute move, base volatility and event share, plus a "Use this move" button that sets `event.move`; engine refusals shown as text.
  - `Template.event?: EarningsEvent`; new template `{ id: "straddle-earnings", name: "Straddle into earnings", event: { days: 5, move: 0.08 } }` building a 1×1 ATM straddle at 30 days and vol 0.70 regardless of the current vol and days. `LegsEditor` sets `position.event = template.event` when a template has one, and clears `position.event` when it does not.
  - `settings.json` `flags.event: true`; `track.ts` adds `"event"`; hub `OPTION_FLAGS` adds `{ key: "event", label: "Earnings (IV crush)", help: "Earnings date and move, the day-after curve and the implied-move calculator." }`; numbers label `event: "Earnings lab used"`.
- [ ] **Step 1: Write the failing test** in `tests/compare.test.mjs` (or a new `tests/templates.test.mjs`): `"the earnings preset is an at-the-money straddle with an event"` — `TEMPLATES.find(t => t.id === "straddle-earnings")`: `build(166, 0.4, 60)` returns a call and a put, quantity 1, same strike `snap(166, strikeStep(166))`, days 30, vol 0.7; `event` deep-equals `{ days: 5, move: 0.08 }`.
- [ ] **Step 2: Run** `pnpm --dir site/options test` — expected: fails.
- [ ] **Step 3: Implement** the Interfaces above (the control is wired into the page in Task 6).
- [ ] **Step 4: Run** tests and `run check` for options and hub — expected: pass, 0 errors.
- [ ] **Step 5: Commit** `git commit -m "Options tool: earnings control, implied-move card, earnings preset, feature switch"`.

### Task 6: Chart, play button, headline and show-your-work

**Files:**
- Modify: `site/options/src/components/OptionsApp.svelte`, `site/options/src/components/PayoffChart.svelte`, `site/options/src/components/WorkPanel.svelte`

**Interfaces:**
- Consumes: Tasks 4–5.
- Produces:
  - `pricingKey` includes `value.event ?? null`.
  - `EarningsControl` rendered under the position editor when `settings.flags.event`, `onenable={() => track("event")}`.
  - `PayoffChart` prop `eventCurve?: Float64Array | null` drawn dashed in `var(--accent)` with legend label "day after earnings".
  - In `OptionsApp`: `eventDay = priced.event && priced.event.days < lastDay ? priced.event.days : null`; `eventCurve = eventDay === null ? null : sliceAt(cube, eventDay, volScale)`; a tick on the date slider at `eventDay` labelled "earnings"; a "Play the crush" button (shown when `eventDay !== null && canSlide`) animating `elapsed` from 0 to `eventDay` over 1.6 s, then holding; a note under the volatility slider when an event is on: "With earnings set, this moves everyday volatility only; the earnings move stays fixed."
  - The date label (`curveName`) reads "after earnings" when `eventDay !== null` and `day` equals `eventDay`.
  - Headline line from `summary.event`: using the leg with the largest `weight` in `summary.event.legs`, text `Earnings in {days} days: volatility {vol_before%} → {vol_after%} after`; when `priced.event` is set but `eventDay` is null: `Earnings after the first expiry: priced only into later legs`.
  - `WorkPanel`: for legs with `base_vol != null`, a step "Earnings split" with `σ_b = √((σ²T − m²)/T) = √(({σ}² × {T} − {m}²)/{T}) = {σ_b}` and "Volatility after earnings = {σ_b}"; when `work.event` is set, an input line `earnings in {days} days, ±{move%}`.
- [ ] **Step 1: Verify in the browser** (local API + `pnpm --dir site/options dev`): choose "Straddle into earnings"; the chart shows three curves; the slider tick sits at day 5; "Play the crush" moves the date to day 5 and the today-curve drops; the headline reads "Earnings in 5 days: volatility 70.0% → …% after"; show-your-work shows the split with numbers; switch the event off and every number returns to its previous value; make a calendar with the short leg at 3 days and event at 5 days: no crush curve, headline "Earnings after the first expiry…"; set vol 25%, 10 days, ±8%: the engine's message appears and the chart keeps the last good position.
- [ ] **Step 2: Run** `pnpm --dir site/options test` and `run check` — expected: pass, 0 errors.
- [ ] **Step 3: Commit** `git commit -m "Options tool: day-after-earnings curve, play the crush, headline, split in show your work"`.

### Task 7: Pages

**Files:**
- Modify: `site/options/src/pages/methodology.astro`, `site/options/src/pages/limits.astro`

- [ ] **Step 1: Add** a methodology section `id="earnings"`, "Earnings volatility", after "American exercise": the split, the valuation rule before/after the event, the event-on-day-`e` convention, the grid's extra dates, and the implied-move formulas with `m·√(2/π)`. Add it to the page's `sections` list.
- [ ] **Step 2: Add** a limits entry under "Prices jump" (or its own section `id="event"`): one fixed move size, normal log jump, one event, slider and scenario shocks move everyday volatility only, and American trees use the effective volatility rather than a jump in the tree.
- [ ] **Step 3: Verify** both pages render locally and the new anchors work.
- [ ] **Step 4: Commit** `git commit -m "Options pages: earnings volatility method and limits"`.

### Task 8: Walkthrough 03 and exercise 3

**Files:**
- Create: `quant-core/walkthroughs/03_earnings_volatility.ipynb` (built and executed by a scratch script, as for 02)
- Create: `quant-core/exercises/earnings.py`, `quant-core/exercises/test_earnings.py`
- Modify: `quant-core/README.md` (module `earnings.py`, walkthrough/exercise row 03; note binomial and arbitrage walkthroughs will be 04 and 05)

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `exercises/earnings.py` with `base_vol(vol, days, move) -> float`, `effective_vol(base, days_left, move, event_ahead: bool) -> float`, `implied_move(near_days, near_vol, far_days, far_vol) -> tuple[float, float]` (move, base_vol), `expected_abs_move(move) -> float`, each raising `NotImplementedError("your turn")`.
- [ ] **Step 1: Write** `exercises/test_earnings.py` comparing each function to the engine on fixed and random inputs (mark refusal cases `@pytest.mark.edge`).
- [ ] **Step 2: Verify** the tests against a scratch solution (all pass), then restore the blank file (tests fail with `NotImplementedError`).
- [ ] **Step 3: Build and execute** the notebook: total variance adds; splitting one volatility; straddle value just before and just after; calendar spread across the event; implied move from two expiries; Monte Carlo check; the P/L explain across the event (large unexplained part); where it is wrong; five check-yourself questions. Read every output and fix any text that misreads it.
- [ ] **Step 4: Commit** `git commit -m "Walkthrough 03: earnings volatility, with exercise 3"`.

### Task 9: Deploy and verify live

- [ ] **Step 1: Run** the full engine suite, options tests and both `run check`s — expected: all pass.
- [ ] **Step 2: Deploy** `fly deploy --remote-only` in quant-core; verify `POST https://api.blessedshoh.dev/options/implied-move` with the round-trip values.
- [ ] **Step 3: Deploy** `pnpm --dir site/options run deploy` and `pnpm --dir site/hub run deploy`.
- [ ] **Step 4: Verify** on https://options.blessedshoh.dev: the earnings preset, the crush animation, the headline, show-your-work, a shared link round-trip, and the methodology anchor.
- [ ] **Step 5: Push** both repositories.
