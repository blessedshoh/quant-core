# IV-crush lab: design

Date: 2026-11-19. Covers both repositories: `quant-core` (engine, API, walkthrough) and `blessedshoh-site` (options tool, dashboard).

## Intent

The options tool at options.blessedshoh.dev is a portfolio piece for buy-side interviewers (funds, asset management, quant). Plain Black-Scholes calculators are common; very few show that an option's implied volatility contains a one-off **earnings component** that vanishes the morning after the announcement ("IV crush"). The lab makes that visible on any position the visitor builds, and lets the owner explain the model on a whiteboard. It is the tool's signature feature in the build plan.

Success: an interviewer types a trade (a straddle into earnings, a calendar spread), sets "earnings in 5 days, ±8%", presses play, and watches the position's value drop when the date crosses the event, with the numbers agreeing everywhere in the tool (chart, Greeks, risk view, show your work, shared link).

## Goals

1. **Event model in the engine.** A position can carry one earnings event: days until it, and the size of the one-day move (standard deviation of the log jump). Every leg that expires after the event has its entered implied volatility split into an everyday ("base") volatility and the event. Before the event, the leg is priced with the full implied volatility; once the date passes the event, only the base volatility remains.
2. **Implied-move calculator.** From two expiries that both fall after the event, back out the event move and the base volatility the market is pricing, and offer "use this move".
3. **In the tool:** an earnings control on the position, a "day after earnings" curve on the chart, an event marker on the date slider, a "play the crush" animation, a headline line showing volatility before and after, an earnings preset, show-your-work lines for the split, and the event carried in shareable links.
4. **Walkthrough 03** (earnings volatility) with exercises and tests, in the same style as walkthroughs 01 and 02.
5. A dashboard switch (`event`) so the owner can hide the lab.

## Non-goals

- Real earnings dates, real option chains or any market data.
- More than one event per position.
- Uncertainty about the size of the move itself (the event move is a fixed input).
- A jump inside the binomial tree. American positions use the tree with the effective volatility (see Model), stated as a limit.

## Anti-goals

- **Any change to numbers when no event is set.** With the event off, every API answer and every number on screen must be identical to today's. Tested.
- A slower tool. Event positions may cost one or two extra grid dates, no more.
- A separate chart or screen that duplicates the main one.

## Constraints

- Existing package rule: no new dependencies (none are needed).
- API input limits stay; the event adds `days` in (0, 1825] and `move` in [0.005, 0.5] (0.5% to 50% one-day standard deviation).
- The browser does no option maths (existing rule): all event pricing happens in the engine.
- Shared links made before this change must still open.

## Approaches considered

| Approach | What it means | Trade-off |
|---|---|---|
| **Inside the main tool (chosen)** | The event becomes part of the position; the existing grid, slider, Greeks, risk view and show-your-work pick it up | Works on any position and keeps every number consistent; less of a stand-alone showcase, which the preset and play button make up for |
| Separate guided screen | A fixed earnings example told step by step, with its own chart | More polished as a demo, but works only on the scripted example and duplicates the chart |
| Front-end only | Shift the volatility slider at the event date | Breaks the rule that the browser does no maths, and Greeks, risk and show-your-work would disagree with the chart |

The owner asked for a recommendation and the first approach was chosen (2026-11-19).

## Model

Notation: entered implied volatility σ for a leg with T years to expiry; event e days from today, with one-day log-move standard deviation m; t = days forward on the chart; τ = years left at t.

**Split at entry.** For a leg that expires after the event (leg days > e), the market's total variance is the everyday variance plus the event variance:

```
σ² T = σ_b² T + m²      so      σ_b = sqrt((σ² T − m²) / T)
```

If σ² T ≤ m², the leg's volatility is too low to contain that move: the request is refused with a plain message ("leg 2's 25% volatility over 10 days cannot contain a ±8% earnings move"). A leg expiring on or before the event day is unaffected (σ_b = σ, no event term).

**Valuation at day t.**

```
event still ahead (t < e):   σ_eff(t) = sqrt((σ_b² τ + m²) / τ)
event passed (t ≥ e):        σ_eff(t) = σ_b
```

At t = 0, σ_eff = σ, so today's prices, entry cost and Greeks equal the no-event values exactly: the split changes nothing today, only how volatility evolves. Prices use Black-Scholes (or the tree, for American positions) with σ_eff. The event is taken to happen before the open on day e: on day e itself it has already happened.

**Volatility slider and shocks** apply to σ_b; the event term m is held fixed. A "+10 points" scenario raises everyday volatility, not the earnings move. (Builder: state this in show-your-work and on the limits page.)

**Greeks** of event positions are measured by bump-and-reprice through the same valuation function (the path the engine already uses for American Greeks), because the closed-form theta assumes a constant volatility, which no longer holds.

**Distribution and chance of profit.** If the event falls before the first expiry, the log price at expiry has variance σ_b² T + m² per leg. The representative volatility today is unchanged (it is built from entered volatilities, which equal σ_eff at t = 0), so chance of profit today needs no change. Distribution and chance at a later horizon are out of scope.

**Implied-move calculator.** Two expiries T₁ < T₂, both after the event, with implied volatilities σ₁, σ₂, assumed to share one base volatility:

```
σ_b² = (σ₂² T₂ − σ₁² T₁) / (T₂ − T₁)
m²   = σ₁² T₁ − σ_b² T₁
expected absolute move = m · sqrt(2/π)      (mean of |jump| for a normal log jump)
```

Refuse when σ_b² ≤ 0 or m² ≤ 0, with the reason (for example, "the later expiry is the more expensive one, so these prices imply no earnings premium").

## Grid

When an event lies strictly inside (0, first expiry), the day axis gains two points: e − ε and e, with ε = 1e-4 days (about nine seconds), so the drop shows as a step rather than being smoothed by interpolation. The front end's square-root-of-time interpolation stays unchanged. A day axis point at e is labelled "after earnings". American grids get the same two points.

## API

- `GridRequest` gains optional `event: {days, move}`. Every route built on it (`grid`, `risk`, `explain`, `work`) accepts it. Absent means no event and identical output to today.
- `GridResponse.summary` gains `event: {days, move, legs: [{base_vol, vol_before, vol_after}]} | null`, for the headline line and the "day after earnings" curve.
- New `POST /options/implied-move`: `{spot, rate, div_yield, days_to_event, near: {days, vol}, far: {days, vol}}` returns `{move, expected_abs_move, base_vol, event_variance_share}` or a 422 with the reason.
- `/options/work` gains per-leg `base_vol` and `vol_after_event` and an event section.

## Front end (options tool)

- **Earnings control** in the position area: a toggle "Earnings before expiry" with two fields, *days until earnings* and *expected move ± %*. Off by default. Validation mirrors the API limits; the "too low" case shows the engine's message.
- **Chart:** a third curve, "day after earnings" (profit at t = e, unchanged price axis), in the accent colour, dashed; the event day marked on the date slider with a small tick and label; a **Play the crush** button that animates the date from today to just after the event and holds.
- **Headline strip:** one line, for example "Earnings in 5 days: volatility 64% → 41% after". With several legs, the line uses the largest-vega leg.
- **Implied-move calculator:** a small card inside the earnings control: near and far expiry days and volatilities, giving the implied move, the expected absolute move, the base volatility and a **Use this move** button.
- **Preset:** "Straddle into earnings" in the templates list, which also switches the event on (5 days, ±8%, at-the-money 30-day straddle at 70%).
- **Show your work:** for event legs, the split line (σ_b from σ, T, m) and σ_eff after the event, with values substituted.
- **Shareable links:** encode the event (`e: [days, move]`); links without it open as before.
- **Usage counting:** event `event` when the toggle is first switched on in a page view.
- **Dashboard:** a feature switch `event` under Tools, on by default, labelled "Earnings (IV crush)"; numbers page label "Earnings lab used".

## Pages

- `/methodology`: a section "Earnings volatility" with the split, the valuation rule and the calculator formulas.
- `/limits`: an entry for the event model: fixed move size, normal log jump, one event, slider shocks only everyday volatility, effective volatility in the tree.

## Walkthrough 03 and exercise 3

`walkthroughs/03_earnings_volatility.ipynb`: total variance and why it adds; splitting one implied volatility; the crush on a straddle (value just before and just after); a calendar spread's behaviour across the event; backing out the implied move from two expiries; checks against simulation (a Monte Carlo with a diffusion plus one normal log jump prices the same as Black-Scholes with σ_eff); where the model is wrong; five check-yourself questions.

`exercises/earnings.py` with `base_vol`, `effective_vol`, `implied_move`, `expected_abs_move`, checked by `exercises/test_earnings.py` against the engine. The binomial and arbitrage walkthroughs become 04 and 05.

## Tests (engine)

- No event: grid, risk, explain and work answers identical to the current ones in every existing field, on fixed requests. The only additions are new keys set to null (`summary.event`, and the work route's event fields).
- At t = 0 the event changes no price, cost or Greek.
- Straddle: value just after the event equals Black-Scholes at σ_b with the remaining time; the drop equals the difference.
- Effective-volatility Black-Scholes matches a Monte Carlo of diffusion plus a normal log jump on the event day, within simulation error.
- Split refused when σ² T ≤ m²; legs expiring on or before the event unaffected.
- Implied-move calculator: round trip (choose σ_b and m, compute σ₁ and σ₂, recover both); refusals.
- The day axis contains e − ε and e; the American path accepts events.
- API: event accepted on all four routes; limits enforced; implied-move route.

## Edge cases (calls made while writing this; check them)

- **Event on or after the first expiry.** The chart stops at the first expiry, so there is no crush curve. The event still lowers the later legs' value at that expiry (a calendar spread whose short leg expires before earnings). The headline reads "Earnings after the first expiry: priced only into later legs".
- **Event days** must be at least 1 day ahead. An event "today" is the same as no event.
- **Risk view crossing the event.** Scenario horizons (7 and 14 days) and the P/L explain's "days pass" that cross the event include the crush automatically. The explain's unexplained part then becomes large, because vega measured today does not foresee the drop. Show-your-work and walkthrough 03 use this as a teaching point; nothing is hidden.
- **Volatility slider** with an event on scales only the everyday part, so at t = 0 it moves total volatility less than before. A note under the slider says so when the event is on.
- **Dashboard default example** does not support an event in this version. The preset covers the demo.

## Left to the builder

Exact placement and styling inside the existing panels, copy wording, the preset's strikes beyond "at the money", and test fixture values. Order of work: engine and tests, API, front end, pages, walkthrough, deploy (API first).
