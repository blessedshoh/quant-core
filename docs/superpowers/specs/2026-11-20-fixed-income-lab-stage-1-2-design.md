# Fixed income lab, stages 1–2: design

Date: 2026-11-20. Touches three repositories: **bondlab** (the fixed income engine, today at `Desktop\githubPortfolio\Bond calculator`, not yet under git), **quant-core** (the Python API on Fly) and **blessedshoh-site** (web).

## Intent

The owner wants interviewers at buy-side firms (funds, asset management, quant, hedge funds) to come away believing two things: **he understands fixed income, and he can build with it**. The section must be practical — a working tool people use for real tasks — not a gallery of class models. It must also contain advanced material that impresses an interviewer. It sits next to the options tool at the same quality bar: the same look, the same depth curve (simple first screen, depth behind panels), show-your-work on every number, methodology and model-limits pages, walkthroughs with exercises.

Primary user: someone with a real bond in front of them (a junior analyst, a self-directed investor, an interviewer probing). Secondary: someone learning fixed income, served by show-your-work and the basics section rather than by the main flow.

Source material: the owner's own Fixed Income Derivatives coursework (two workbooks, entirely his own work; publishing the final exam is permitted by the course):
- `FixedIncomeDeriv_ All_Models.xlsm`: Bootstrapping, Coupon, Maturity, Time path, Yield Change, Duration and Convexity, Tangency, Option pricing, Binomial Tree (Kalotay-Williams-Fabozzi style rate tree, callable/putable), Convertible Bond (CRR stock tree with conversion).
- `FI_final_ShohruhM.xlsx`: Q1 a 5-year NVIDIA callable on a rate tree; Q2 the "Lego" map of products; Q3 Treasury futures CTD on Bloomberg data; Q4 a cheat sheet.

## The whole programme (for context)

1. **Engine, daily data and basics** (this document)
2. **Today's curve, the bond pricer with key-rate durations, the hedge calculator** (this document)
3. Callable bonds with OAS on a Hull-White tree fitted to today's curve (mostly exists in bondlab)
4. Yield-curve PCA on Treasury history
5. Treasury basis: CTD, implied repo, delivery option

Stages 3–5 get their own designs. Uzbekistan is out of scope for now (the owner has other plans for it); bondlab's Uzbek module stays in the library untouched and unexposed.

## Goals (stages 1–2)

1. **bondlab under git and on GitHub** as a public repository, moved next to the other two (`Desktop\blessedshoh\bondlab`), unchanged in behaviour, its contract (`CLAUDE.md`) intact.
2. **New bondlab models for the basics**, each test-pinned to the owner's workbook numbers (golden CSV rows with a provenance column naming sheet and cell):
   - a textbook bond preset (whole-period maturities, settlement on a coupon date, so no accrued) for the coupon, maturity, pull-to-par and yield-shock models;
   - Kalotay-Williams-Fabozzi binomial rate tree (`r_H = r_L·e^{2σ}`), calibrated to a par curve, valuing straight, callable and putable bonds;
   - convertible bond on a CRR stock tree with hold/convert decisions and the coupon-at-conversion toggle from the workbook.
3. **API routes** `/fixed-income/...` in the existing Fly service, built on bondlab's existing JSON↔core translation layer (`bondlab.ui.service`), which computes nothing itself.
4. **Daily Treasury data**: the US Treasury daily par yield curve (14 tenors, 1 month to 30 years; public domain; one CSV per year back to 1990), stored in the existing D1 database by a scheduled Worker, backfilled once to 1990.
5. **A new tool site at `bonds.blessedshoh.dev`**, same stack and look as the options tool, with four areas:
   - **Today's curve** (first screen): par, zero and forward curves from today's Treasury data; scheme switch (monotone convex default, plus linear and cubic options bondlab already has) and a Nelson-Siegel fit overlay; a date picker for any day since 1990; the table of par yields.
   - **Price a bond**: coupon, maturity, frequency and a convention preset (US Treasury, US corporate 30/360); price from today's curve (plus an optional spread) or from a yield, or solve yield/Z-spread from a price; outputs clean and dirty price, accrued, yield, Macaulay and modified duration, convexity, DV01, and **key-rate durations** at the curve tenors.
   - **Hedge**: neutralise a bond position's risk with on-the-run Treasuries priced off today's curve. Two modes: single-bond DV01 hedge, and key-rate hedge (choose up to four hedge tenors; solve the linear system so the hedged position's key-rate durations at those tenors are zero; show the residual exposure elsewhere).
   - **Basics** (the owner's workbook models, interactive): price and yield; bootstrapping; coupon effect; maturity effect; pull to par over time; sudden yield change; duration and convexity with the tangent line; rate tree for callable and putable bonds; convertible bond. Black-Scholes links to the options tool rather than duplicating it.
6. **Show your work** on the pricer (the cashflow table: date, year fraction, cashflow, discount factor, present value; duration and convexity as weighted sums) and on every basic model.
7. **Methodology and "where the model is wrong" pages** for the bonds site.
8. **Downloads**: both original workbooks offered as "the coursework this was built from", after the owner fixes three cells (see Open items) and with the Bloomberg-sourced data handled as decided there.
9. **Walkthroughs** in bondlab, same style as quant-core's: 01 bond maths (price, yield, duration, convexity, the tangent line), 02 curves and key-rate risk (bootstrapping, interpolation choice, forwards, KRDs, the key-rate hedge), each with an exercise file and tests that compare the exercise with the engine.
10. Hub: the existing "Bond calculator — soon" card becomes the live "Fixed income lab" (renamed in the dashboard); the bonds site gets the shared header project switcher and dashboard feature switches, exactly like the options tool.

## Non-goals (stages 1–2)

- OAS, PCA, Treasury basis (stages 3–5).
- Corporate bond reference data, credit curves, ratings. A corporate bond is priced off the Treasury curve plus a spread the user types.
- Real-time prices. The curve is end-of-day, labelled with its date.
- Swaps, caps, floors, swaptions, CDS (the cheat-sheet products; candidates for a later stage).
- Uzbekistan.

## Anti-goals

- Any change to bondlab's existing numbers or contract. Its 432 tests, `mypy --strict` and `ruff` must stay green throughout.
- Pricing logic outside the engine (none in the API layer, none in the browser).
- A first screen that needs a manual before it says anything useful.
- The site depending on treasury.gov being up: pages read stored data only.

## Constraints

- bondlab contract (`CLAUDE.md`) is binding for every bondlab change: numpy/scipy only, `mypy --strict` with no ignores, no silently defaulted conventions (presets are named factories that state their assumptions), frozen slotted dataclasses, `datetime.date` only, docstrings with formula and specific citation, actionable errors, tests with externally sourced expected values.
- Site: no new npm dependencies; two-week minimum package age as before.
- API stays stateless: the browser (or the page server) sends the curve points with each request.
- Treasury data terms: public domain; attribute "U.S. Department of the Treasury" on the page.

## Approaches considered

| Approach | Trade-off |
|---|---|
| **bondlab is the engine, served by the existing Fly API (chosen, 2026-11-20)** | Reuses ~10,000 tested lines (curves, conventions, Hull-White, OAS, spreads); two libraries with their own styles |
| Rebuild fixed income inside quant-core | One style, but weeks of duplicated work and less rigour |
| bondlab as a separate API service | Cleanest separation, but a second server to run with no visible benefit |

The owner also asked for the same style and complexity as the options tool: same look, depth curve, show-your-work, pages and walkthroughs. Engine rigour stays at bondlab's level; the walkthroughs teach the core maths and present conventions as "what a real desk needs".

## Architecture

**bondlab** (moved to `Desktop\blessedshoh\bondlab`, `git init`, public GitHub repo `blessedshoh/bondlab` created by the owner, tagged releases). New modules:
- `instruments/textbook.py` — named preset `textbook_bond(face, coupon_rate, years, frequency)` whose docstring states its assumptions (whole periods, settlement on a coupon date, `Frequency`-based street time).
- `models/kwf_tree.py` — KWF binomial tree: one-year step, rates lognormal with `r_H = r_L e^{2σ}`, calibrated level by level so it reprices the input par curve; backward induction with call (min with call price) and put (max with put price) at each node; reports the tree for drawing. Reference: Fabozzi, *Handbook of Fixed Income Securities*, ch. on valuation of bonds with embedded options (KWF model); the workbook's Fabozzi-style example (3.5% one-year rate, 10% volatility → `r1L ≈ 4.0735%`, `r1H ≈ 4.9755%`, `V0 = 99.8150`) is a golden row.
- `instruments/convertible.py` + `models/convertible_tree.py` — CRR stock tree (`u = e^{σ√Δt}`, `p = (e^{(r−q)Δt} − d)/(u − d)`), value = max(hold, convert) at each node, with the coupon-at-conversion toggle; reference Hull, *Options, Futures and Other Derivatives*, ch. on convertible bonds; golden rows from the workbook's Convertible Bond sheet (`V0 = 1504.01`, decisions per node).
- `risk/hedge.py` — `dv01_hedge(position, hedge, market)` and `key_rate_hedge(position, hedges, market, tenors)` returning hedge notionals and residual KRDs; references Fabozzi (DV01 hedging) and Tuckman & Serrat, *Fixed Income Securities* (key-rate exposures).
- `data/treasury.py` — parse the Treasury par-yield CSV row into `BondQuote`/`DepositQuote` inputs with conventions named explicitly (CMT yields are bond-equivalent semiannual par yields; tenors ≤ 6 months treated as zero-coupon bills on a bond-equivalent basis — state this as an assumption with the Treasury's methodology page as the source). No network code in the core: parsing only.

**quant-core API** gains `api/fixed_income.py` (a FastAPI router) mounted at `/fixed-income`, depending on bondlab installed from its GitHub tag in the Dockerfile and editable (`pip install -e ../bondlab`) locally. Routes (all stateless, the curve points come in the request): `curve` (bootstrap + NS fit → par/zero/forward on a grid), `price`, `risk` (incl. KRDs), `hedge`, `work` (cashflow table), and one route per basic model (`basics/coupon`, `basics/maturity`, `basics/time-path`, `basics/yield-shock`, `basics/tangent`, `basics/bootstrap`, `tree`, `convertible`). Each route translates JSON through `bondlab.ui.service`-style functions (new ones added to bondlab's `ui/service.py`, which already follows the "computes nothing" rule).

**Data**: D1 table `treasury_par_yields(day TEXT, tenor_months INTEGER, yield_pct REAL, PRIMARY KEY (day, tenor_months))` (hub migrations folder). A new Worker `blessedshoh-data` with a cron trigger (daily 23:30 UTC, and a second run at 03:30 UTC to catch late publication) fetches the current year's CSV, upserts new days, and records each run in the hub's `changes` log as "data: Treasury curve <date> loaded" (or the failure reason). A one-off backfill script loads 1990 → today. If treasury.gov refuses requests from Cloudflare, the builder reports it; fallback is a GitHub Actions daily job posting to the same table through a token-protected endpoint (owner sets the secret).

**Site** `site/bonds` (Astro SSR on Cloudflare Workers, Svelte 5, D1 binding, shared kit), deployed to `bonds.blessedshoh.dev`. The page server reads the latest curve (and a requested date) from D1 and sends it with API requests. Shareable links encode the bond, the chosen date and the view. The hub's admin Tools page gains a "Fixed income lab" section with feature switches per area (`curve`, `price`, `hedge`, `basics`), on by default.

## Testing

- bondlab: existing suite stays green; new golden CSVs from the workbooks with provenance (sheet and cell); KWF tree reprices its calibration curve to 1e-10 and reproduces the workbook tree; convertible reproduces the workbook's value and decisions; KRDs sum to the parallel DV01-based duration within bump error; key-rate hedge leaves zero KRD at the hedge tenors (exact) and reports the residual; Treasury CSV parser on a saved fixture file.
- Excel parity: for each basic model, the API answer for the workbook's inputs matches the workbook's cell values (to the cent or to the workbook's displayed precision), stated per model.
- API: shape, limits, 422 on bad conventions, no-curve and stale-date handling.
- Site: link round trips, curve date handling, the stored-data-only rule (no treasury.gov call at request time).
- Browser check at desktop and phone width, both themes.

## Open items for the owner

1. **Bloomberg data in the final's Q3.** The futures sheet's `Data` tab is Bloomberg-sourced; Bloomberg's terms generally forbid republishing. Options: publish the final without the `Data` tab (Q3's numbers then show as broken links), publish Q1/Q2/Q4 only, or keep the final download back until stage 5 rebuilds Q3 on public data. Proposed: publish without the `Data` tab, with a note that stage 5 rebuilds it on public data.
2. **Three cells to fix before publishing**: the final's `Q3!D26` ("Please lets make the model easy"), "Strie Price" (Binomial Tree `B17`, final `Q1!B26`), "Interes rates" (Option pricing `C10`). The owner edits these in Excel; nothing here edits his workbooks.
3. **GitHub**: the owner creates the public repository `blessedshoh/bondlab` (no GitHub CLI here).

## Calls made while writing (check these)

- bondlab moves from `githubPortfolio\Bond calculator` to `blessedshoh\bondlab` so all three repos sit together; its `.venv` is recreated there.
- Site address `bonds.blessedshoh.dev`, name "Fixed income lab".
- First screen is today's curve, not the pricer: it shows real data in one second and leads into pricing.
- The hedge uses on-the-run Treasuries from the curve until stage 5 adds futures.
- Basic models use textbook conventions (stated in the preset), not the full desk conventions, so their numbers match the workbook.
- Walkthroughs live in bondlab; the binomial and arbitrage walkthroughs stay planned in quant-core.

## Gaps found in the builder read, settled (calls; check them)

- **Treasury CSV columns change over the years.** 1990 files have fewer tenors; the 2-month bill appears in 2018, the 4-month in 2022, the 1.5-month (6-week) later; the 30-year is missing from 2002 to early 2006. The parser reads columns by header name, accepts missing tenors per day, and the curve builder bootstraps from whatever tenors that day has.
- **Rate-tree golden cases.** Two workbook cases become golden rows: the Binomial Tree sheet (3.5% one-year rate, 10% volatility, call at the workbook strike → `V0 = 99.8150`, call premium 0.1850) and the final's Q1 NVIDIA 5-year callable (one-year rate 4.858%, volatility 14.16%, call at 985 → `V0 = 982.2212`). The workbooks hold the calibrated lowest-node rates as typed numbers (`r1L = 4.0735%`, NVIDIA `4.4293%`); the golden test feeds the same calibration targets and checks the engine finds those rates to the workbook's precision. Calibration to a full par curve is tested separately (the tree reprices its own inputs).
- **Key-rate tenors** are 1, 2, 3, 5, 7, 10, 20 and 30 years by default (the curve's tenors from one year out), with the bump size reported next to the numbers, as bondlab already does for effective convexity.
- **The hub card** for this tool is switched to live by the owner in the dashboard at launch (title "Fixed income lab", address `https://bonds.blessedshoh.dev`); no migration edits dashboard-managed content.

## Left to the builder

Exact layout inside each area, copy, chart details, golden-row selection, router module layout, and the order of work: bondlab move and git → new models with golden tests → API router → data Worker and backfill → site areas (curve, price, hedge, basics) → pages → walkthroughs → deploy (API first) → hub card.
