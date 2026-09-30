# React dashboard POC on the price-return package (`dev/react_dashboard_poc`)

This is the working plan for the dashboard proof of concept, kept in the repo so work can continue
from any machine or session. Update it as phases finish: mark the phase ✅ with its commit hash in
the table, and add a "Phase N notes" section with anything a later phase needs to know.

## Picking this up

- **Setup:** Python 3.12 or newer: `pip install -e ".[api,stats,dev]"` from the repo root, then
  `pytest`; all tests pass offline. Run everything with `python -m src.api` (after
  `npm run build` in `dashboard/`), or develop with `python scripts/dev.py`. The dashboard needs Node 20 or newer: `npm install` once in
  `dashboard/`, then `npm run dev` (http://localhost:5173) with the API running; `npm test`,
  `npm run typecheck` and `npm run build` are its checks.
- **Data:** develop and test on the demo files (`data_source='demo'`, e.g. SPX; see the revision
  below), which work offline. Yahoo is blocked in the cloud environment this was built in, so
  Yahoo runs happen on the owner's computer.
- **Phases are the unit of work.** Do one phase when asked, then stop; do not start the next one
  unprompted. Commit and push at the end of each phase.
- **Stay within the plan.** Propose new deliverables as changes to this plan.
- **No analysis outside `src/tools/`.** The API wraps and serializes; React displays. If the
  dashboard needs a number the package does not compute, add it to the package, with tests.

## Context

The owner is shaping a workflow for a trading and analytics platform:

1. develop ideas in Jupyter notebooks;
2. promote what proves useful into tested Python packages;
3. build a React dashboard that uses the same packages for interactive analysis;
4. host everything on a server.

Steps 1 and 2 already work in this repo: `price_return_analysis.ipynb` and
`price_return_statistics.ipynb` sit on the tested `src/tools/price_return/` package (203 offline
tests, independent references). This POC proves step 3 on that package, running on the owner's
computer. Docker and hosting (step 4) come afterwards.

Branch `dev/react_dashboard_poc` was cut from `master` at `b44071f`.

## Review of the workflow

**What is sound.** Notebook → package → app is the right shape. It keeps one source of truth for
the analysis, tested once. The promotion step has a real bar here, with independent-reference
tests, mutation checks and a plan per package, and the dashboard should inherit that rather than
bypass it.

**What the four steps leave out:**
- **An API layer between steps 2 and 3.** A browser cannot run Python, so React needs a server that
  calls the package and returns JSON. That server is where the "same Python package" promise is
  kept or broken: it must only wrap and serialize, never re-implement analysis. The same goes for
  the React code.
- **A data layer.** Notebooks, API and a future server all download from Yahoo independently. The
  POC uses an in-memory cache. A shared local store (Parquet or DuckDB), refreshed on a schedule,
  should come before hosting, both for speed and because Yahoo rate-limits.
- **Analytics and trading are different systems.** Order placement needs broker APIs, risk limits,
  audit trails and paper trading first. It should be a separate service that consumes the same
  packages, not a feature of the analytics dashboard. It is out of scope here.
- **Hosting (step 4) adds** authentication, HTTPS, secrets, monitoring and scheduled data
  refresh. Docker is the right first packaging step; the POC is built so that it drops in.

**Alternatives considered.**
- **Plotly Dash:** Python only, and React underneath. Quickest, but it caps UI freedom.
- **Streamlit:** fastest to a first screen, but its rerun model strains under rich interaction
  and multiple users.
- **Voilà:** turns notebooks directly into apps, blurring steps 1 and 3.

The owner chose React + FastAPI. It costs a Node toolchain and two languages, and buys full UI
control and the cleanest path to a hosted platform.

**Owner's decisions (2026-09-29):** React + FastAPI · charts reused from `viz.py` as Plotly
JSON · POC covers overview and rolling stats, streaks and cumulative, the interactive rare-event
table, statistics, and the multi-ticker comparison · code lives in this repo.

**Owner's decision (2026-09-30): demo data.** The repo will be public, and Yahoo is unreachable
where this is built, so development runs on processed Yahoo data rather than on simulation alone:
daily bars for ten tickers with 0.01% seeded noise on every column, labelled as not market data
and for education only. See the revision below.

**Owner's decision (2026-09-30): saved live data.** The dashboard's data choice becomes three
options: demo, live (Yahoo), and saved CSVs of live data, downloaded once and updated on request,
so later runs are offline and fast. Built after Phase 4, as Phase 4a. Uploading your own CSVs,
or reading a folder of arbitrary CSVs, was considered and left out for now.

## Architecture

```
React (dashboard/, Vite + TypeScript) ──/api/*──▶ FastAPI (src/api/) ──▶ src.tools.price_return
   plotly.js draws the figure JSON                  wraps + serializes      (all analysis, all charts)
```

- **Parameters:** the request model is generated from `Params`'s dataclass fields
  (`src/tools/price_return/params.py`), so a new `Params` field appears in the API without
  duplication. Unknown keys are rejected, as `load_ticker_config` already does.
- **Charts:** a chart endpoint calls the existing `viz.plot_*` function and returns
  `fig.to_json()`, so charts are identical to the notebooks and exist once, in Python.
- **Tables:** DataFrames are returned as records, with NaN as `null` and dates as ISO strings.
  Formatting (%, rounding) happens in React, as `format_probability_table` does for notebooks.
- **Data:** `load_price_data` serves three sources: `yahoo`, `demo` (`data/demo`) and
  `simulated`, and gains `local` (saved Yahoo data in `data/local`) in Phase 4a. The dashboard's
  ticker list defaults to the demo set, so it works offline out of the box;
  `/api/tickers?set=yahoo` switches to `configs/tickers.yaml`.
- **Cache:** an in-memory LRU cache of `load_price_data` + `add_rolling_stats`, keyed by the data
  fields of `Params` (ticker, source, dates, simulation settings). `end_date` defaults to today,
  so the cache refreshes daily.
- **Local run:** in development, `uvicorn` runs on :8000 and Vite on :5173, with `/api` proxied,
  so no CORS setup is needed. For everyday use, `npm run build` once, then FastAPI serves the
  built app and the API together on `http://127.0.0.1:8000` (bound to localhost only).

## Phases

| # | Phase | Deliverable |
|---|---|---|
| **0** ✅ | Plan | this plan as `docs/dashboard_plan.md`, linked from the README |
| **1** ✅ | API core | `src/api/` app, parameter schema, serialization, cache; health, tickers and overview endpoints; `api` extra; tests — **done, `2a1e61b`** |
| **1a** ✅ | Revision: demo data | `data/demo` (ten processed files, README, manifest), `demo` data source, `configs/demo_tickers.yaml`, `scripts/make_demo_data.py`; `/api/tickers?set=`; tests; README — **done, `c45b0ca`** |
| **2** ✅ | API complete | streaks, cumulative, rare-event, chart, statistics and multi-ticker endpoints; tests — **done, `6389d05`** |
| **3** ✅ | Dashboard shell | `dashboard/` (Vite, React, TypeScript), parameter panel, tabs, Plotly chart component, Overview tab — **done, `a8af3fa`** |
| **4** ✅ | Dashboard tabs | Streaks & cumulative, Rare events (live filters), Statistics, Multi-ticker — **done, `47e8fe9`** |
| **4a** ✅ | Saved live data | `data/local/` CSV store of Yahoo data (git-ignored), `local` data source, save/update from the dashboard, a refresh script; tests — **done, `a3482d4`** |
| **5** ✅ | One-command local run | FastAPI serves the built app; `python -m src.api`; a dev script for both servers; README "Dashboard" section — **done, `51e9cf0`** |
| **6** ✅ | End-to-end check and docs | Playwright smoke test of every tab; README, changelog, plan statuses; PR — **done, `PHASE6`** |

Working rules, as in the earlier plans: one phase per request, then stop; commit and push at the
end of each phase; tests stay offline (`data_source='demo'` or `'simulated'`, or a stubbed
yfinance); no analysis logic outside `src/tools/`.

## Phase 1: API core (`src/api/`)

- `app.py`: the FastAPI app, with `GET /api/health` and `GET /api/tickers` (from
  `load_ticker_config`, with labels and per-ticker parameters).
- `schemas.py`: `ParamsIn`, generated with pydantic's `create_model` from
  `dataclasses.fields(Params)`, with `extra='forbid'`.
- `serialize.py`: `frame_to_records` (NaN → null, ISO dates) and `figure_to_json`.
- `cache.py`: `prepared_data(params)`, an LRU cache over `load_price_data` + `add_rolling_stats`.
- `POST /api/overview`: `latest_snapshot`, `distribution_summary`, date range and row count.
- `pyproject.toml`: `api = ["fastapi", "uvicorn[standard]"]`, and `httpx` in `dev` for FastAPI's
  `TestClient`.
- **Tests (`tests/test_api.py`):** each endpoint's JSON round-trips to exactly what a direct
  package call returns on the same simulated `Params`. Unknown or invalid parameters give 422. A
  data-source failure (Yahoo) gives a clear error, not a 500 stack trace.

### Phase 1 notes

- `src/api/`: `app.py` (endpoints), `schemas.py` (`ParamsIn`, generated from
  `dataclasses.fields(Params)` with the same types and defaults; `extra='forbid'`),
  `serialize.py` (`clean`, `series_to_dict`, `frame_to_records`, `figure_to_json`), `cache.py`
  (`prices`, `prepared_data`, an LRU cache of 32 downloads).
- **Endpoints:** `GET /api/health`; `GET /api/tickers` (every configured ticker with its label and
  full parameters); `POST /api/overview` (rows, date range, last price, `latest_snapshot`,
  `distribution_summary`). FastAPI's own docs page is at `/docs`.
- **Serialization:** values are converted one by one to Python types rather than through
  `DataFrame.to_json`, which keeps only 10 significant digits by default. Floats round-trip JSON
  exactly (tested on 1,000 values). JSON has no NaN or infinity, so non-finite numbers become
  `null`; that includes an infinite Sortino or Calmar when there is no downside.
- **Errors:** `load_price_data`'s ValueError (unknown ticker, unknown data source) → 422 with its
  message; any other data-source failure (network, Yahoo) → 502 "Data source failed: …"; bad or
  unknown parameters → 422 from the model.
- **Cache key** is the data fields only (`DATA_FIELDS`: source, ticker, dates, simulation
  settings). Changing thresholds, windows or `trade_days` reuses the downloaded prices; the
  rolling statistics are recomputed per request (cheap). `end_date` defaults to today, so a new
  day is a new key.
- **Dependencies:** `api = ["fastapi", "uvicorn[standard]"]`; `dev` gains `httpx2`. Starlette 1.7's
  test client imports `httpx2` first and warns when it falls back to `httpx`. `httpx2` is the
  successor from httpx's author, published under the pydantic organisation.
- **Tests:** `tests/test_api.py`, 14 tests. The overview must equal `latest_snapshot` /
  `distribution_summary` called directly, exactly; also tickers against `load_ticker_config`, the
  model against `Params`, serialization, both error paths, and the cache. Six seeded mutations were
  each caught. Also run under a real uvicorn server with HTTP calls. Suite: 217.

## Revision 1a: demo data (2026-09-30)

- **Files:** `data/demo/{SPX,NDQ,YM,CL,RTY,EURUSD,XAUUSD,AAPL,KO,TSLL}_demo.csv` (Date, Open, High,
  Low, Close, Volume) and `manifest.csv` (Yahoo symbol, rows, dates, noise, seed per file),
  generated by the owner with `scripts/make_demo_data.py`: Yahoo daily bars from 2016, as traded,
  times `1 + N(0, 0.01%)`, seeds 42 to 51, high and low repaired, 6 decimals. `data/demo/README.md`
  states they are processed, not market data, for education only, and lists the quirks: CL's
  negative close on 2020-04-20, XAUUSD built from `GC=F`, zero volumes (all of EURUSD), and the
  shorter RTY (2017) and TSLL (2022) histories.
- **Package:** `load_price_data` gains `data_source='demo'`, reading `data/demo/{ticker}_demo.csv`
  with `end_date` exclusive, as for Yahoo. Both sources now share `_returns_from_closes`, so a
  non-positive close drops its two returns in either. `demo_tickers()` lists the files;
  `DEMO_CONFIG_PATH` points to `configs/demo_tickers.yaml` (all ten, labelled; EURUSD at ±0.1%,
  TSLL at ±1.0%, SPX first).
- **API:** `GET /api/tickers?set=demo|yahoo`, default `demo`; the response names its set and
  source file. Everything else takes the data source from the parameters, as before.
- **Tests:** `tests/test_demo_data.py` (every file against the manifest; the `demo` source
  against the file's closes and returns written out; the date filter's ends; CL; the error
  messages; the config; the README notice) and three new API tests (both ticker sets, demo
  overview parity, an unknown demo ticker → 422). Five of six seeded mutations were caught; the
  sixth (pandas' default float parser instead of `round_trip`) reads these files identically.
  Both price-return notebooks also run clean on demo data. Suite: 240.

## Phase 2: API complete

- `POST /api/streaks`, `/api/cumulative`: `detect_streaks` + `summarize_streaks`,
  `analyze_cumulative` + `summarize_cumulative`.
- `POST /api/rare-events`: `build_historical_analysis` cached per parameters, then
  `low_probability_view(df_his, n_days, p, prob_max, prob_min, change_type)`. This is the server
  half of the `interactive_low_probability` widget.
- `POST /api/charts/{name}`: a registry mapping names to the `viz.plot_*` functions and their
  inputs. It covers the nine analysis charts and five statistics charts.
- `POST /api/statistics/{section}`: `distribution` (`return_moments`, `jarque_bera`,
  `fit_student_t`, `tail_index`, `value_at_risk`), `dependence` (`autocorrelation`, `ljung_box`,
  `variance_ratio`, `arch_lm`), `drawdowns` (`risk_ratios`, `drawdown_table`, `rolling_risk`),
  and `probabilities` (`probability_intervals` + `model_probabilities`, with `n_boot` capped).
- `POST /api/multi-ticker`: `analyze_ticker` for each ticker of a set (`demo` by default, as for
  `/api/tickers`), then `compare_tickers`. Failed tickers are reported alongside results, as
  `rare_case_run` does.
- **Tests:** the same round-trip parity for every endpoint, on demo and simulated data; every
  registered chart builds; the rare-event filters match `low_probability_view` called directly.
  (Done: see Phase 2 notes.)

### Phase 2 notes

- **Endpoints.** The request body is always the `Params` JSON; what to show goes in the query.
  - `POST /api/streaks`, `POST /api/cumulative`: `{summary}`, from `summarize_streaks` and
    `summarize_cumulative`.
  - `POST /api/rare-events?n_days=3&n_days=5&change_type=cumulative`: `low_probability_view` over
    the cached `build_historical_analysis`, filtered by the body's `prob_min` and `prob_max`.
    `n_days` must be in `streak_days` (422 otherwise). The response also gives `total_events`.
  - `GET /api/charts` lists the 14 charts (9 `analysis`, 5 `statistics`) and the options each
    reads. `POST /api/charts/{name}` returns the figure JSON. Options: `window`
    (streak-timeline, rolling-risk), `top` (drawdown), and `n_days`, `n_boot`, `change_type`,
    `change`, `n_years` (event-probabilities). Defaults are the notebooks'.
  - `POST /api/statistics/{distribution|dependence|drawdowns|probabilities}`: the statistics
    notebook's sections. `probabilities` takes `n_days` and `n_boot` (default 1,000, as in the
    notebook, maximum 2,000, about 2 s per 1,000) and returns every event. The dashboard filters
    to the rare ones for display, as the notebook does. `drawdowns` takes `top`. `rolling_risk` is
    served as the `rolling-risk` chart only, since its table is a 2,500-row time series.
  - `GET /api/multi-ticker?set=demo|yahoo&drill_n_days=3`: `rare_case_run`. Per ticker: label,
    rows, dates and the drill table. Also the two `compare_tickers` tables, and `failed` (symbol
    and error); if every ticker fails, the response is a 502. It is a GET: there is no body,
    since the config sets each ticker's parameters. It takes about 9 s for the ten demo tickers
    on the first call and is then cached.
- **Tables** are `{columns, records}`: JavaScript objects put integer-like keys first, so the
  column order is sent explicitly.
- **Caches** (`cache.py`): prices (by data fields); the historical table (by data fields plus
  `trade_days`, `return_thresholds`, `lookback_years`, `streak_days`, so moving `prob_max` or
  `prob_min` only refilters); the event table (plus `n_days`, `n_boot`); `analyze_ticker` results
  (by every field and the label). Failures are not cached.
- **Errors:** a missing `stats` extra (scipy) → 501 with the install hint. Data failures → 422 or
  502 on every endpoint, as in Phase 1 (tested on each).
- **Package changes, both kept out of the API as the rules require:**
  - `stats.event_probability_table(pct_change, n_days, p, n_boot, seed)` joins
    `consecutive_analysis`'s counts, `probability_intervals` and `model_probabilities` row for
    row. The statistics notebook did this in a cell; it now calls the function.
  - `viz.plot_streak_timeline` sets all its streak shapes in one update. `add_vrect` re-validated
    every existing shape, so the chart took 54 s for SPX 2-day streaks and 325 s for CL; it now
    takes 0.8 s and 2.4 s. The figures are identical (JSON compared on nine ticker/window cases),
    and the notebooks get the speed-up too.
- **Tests:** 25 new API tests (45 in all). Every endpoint equals the package called directly in the test,
  on demo and simulated data. All 14 charts equal the notebook's figure, and chart options reach
  the plot. The probability bounds refilter the cached table, and bad requests are rejected. The
  multi-ticker test uses a temporary config with a failing ticker. Two package tests: the event
  table's joins, and the timeline's shapes against the streaks. 11 seeded mutations; the one
  that first survived (an endpoint skipping the data-error mapping) led to the every-endpoint
  error test. Suite: 267.
- **Rendering:** the chart JSON from a live server renders in plotly.js in Chromium. Plotly 6
  sends numeric arrays base64-encoded (`bdata`), which plotly.js 2.28 or newer decodes, so
  Phase 3 needs a recent `plotly.js-dist-min`.
- **For Phase 4, from the renders:** on `event-probabilities`, a range within one decade shows a
  single tick label (the axis is set to decades only), and the two model labels can overlap at
  the right end. Both are in `viz.py`, so both notebooks show them too.

### Addition after Phase 2 (owner's request, 2026-09-30)

- `notebooks/api_examples.ipynb`: example queries to every endpoint over HTTP, with helpers
  (`get`, `post`, `table`, `chart`) that show how a client uses the API. It starts the API
  inside the kernel (a uvicorn thread) when nothing answers at `BASE`, runs on the demo set by
  default (`TICKER_SET = 'yahoo'` for Yahoo), shows each error status, and ends by checking
  `/api/streaks` against `summarize_streaks`. Committed without outputs. Executed here both with
  its own server and against a separately started one, in about 20 s.

## Phase 3: dashboard shell (`dashboard/`)

- Vite + React + TypeScript. `react-plotly.js` with `plotly.js-dist-min` for charts; TanStack
  Query for fetching and caching; plain CSS using the palette tokens `viz.py` already uses, with
  no UI framework.
- `src/api.ts`: a typed client whose types mirror the API responses.
- `components/`: `ParamsPanel` (ticker picker from `/api/tickers`, demo set by default and SPX
  first; dates, thresholds, data source), `PlotlyChart` (renders figure JSON), `DataTable`, `StatTiles`.
- `App.tsx`: the parameters panel plus tabs, with a visible "demo data: processed, not market
  data" note whenever `data_source` is `demo`; the **Overview** tab shows snapshot tiles, price and
  returns, the distribution, and the rolling average and volatility charts.
- `.gitignore` gains `dashboard/node_modules` and `dashboard/dist`.

### Phase 3 notes

- **Stack:** Vite 8, React 19, TypeScript 5.9, TanStack Query 5, `plotly.js-dist-min` pinned to
  **4.1.1**, the plotly.js that Python's plotly 7.1 bundles, so the figure JSON renders exactly
  as in the notebooks. No UI framework; `src/styles.css` holds the palette tokens (viz.py's
  values, with the dark steps of the same hues).
- **Changes from the plan:**
  - `PlotlyChart` calls `Plotly.react` directly instead of using `react-plotly.js`, which is
    unmaintained and untyped for React 19. plotly.js is loaded lazily, on the first chart, as its
    own 4.6 MB chunk; the app itself is 270 kB.
  - There is no separate data-source control: the **Data** control picks the ticker set (demo
    or Yahoo), and each ticker's configured parameters carry the source. A free choice would
    allow combinations that cannot load (demo data for `ES=F`).
- **Files** (`dashboard/src/`):
  - `api.ts`: the typed client for every endpoint. `tests/test_api.py` checks that its `Params`
    type lists exactly the dataclass's fields.
  - `components/ParamsPanel.tsx`: choosing the set or a ticker applies at once, with that
    ticker's config; dates and thresholds apply with **Apply**, which is disabled, with a
    reason, for an invalid range.
  - `components/PlotlyChart.tsx`: `Chart` fetches one chart and `Plot` draws it.
  - `components/DataTable.tsx`, `components/StatTiles.tsx`.
  - `tabs/OverviewTab.tsx`, `App.tsx`: header, theme control, parameters, the demo-data notice,
    and the tabs (the four Phase 4 tabs show a placeholder).
  - `format.ts`, `chartTheme.ts`, `theme.ts`.
- **Overview:** five stat tiles (last price, annualized return and volatility, daily mean,
  number of daily returns), price and returns, the histogram beside the distribution table, and
  the rolling average and volatility charts. The series loads first and the charts after it, so
  a series that fails (Yahoo unreachable) shows one error, not five.
- **Behaviour:** a refetch keeps the previous render, dimmed, until the new one arrives. The
  theme follows the OS unless set. In dark mode `chartTheme.themed` re-steps the figure's colours
  by role (surfaces, grid, axes, near-black ink, the three series colours) and leaves colour
  scales and the data untouched.
- **Checks:** 13 Vitest tests (client, formatting, chart theming; two seeded mutations of the
  theming and two of the field check were each caught), the type check, the production build,
  and 269 Python tests. A Playwright walk-through against the live API and Vite:
  - first load on SPX, a ticker change, applying thresholds, an invalid date range;
  - a placeholder tab and the Yahoo set (one readable error here, where Yahoo is blocked);
  - screenshots in light, in dark, and at 390 px wide (no horizontal scroll).
  Four charts load in about 3 s on a warm server.
- **The Yahoo path, offline:** the API was also run with yfinance replaced by a stub serving each
  demo file under the Yahoo symbol it came from (the manifest's `CL=F`, `GC=F`, `YM=F`, `RTY=F`,
  `EURUSD=X`), in yfinance's own shape, and an empty frame for anything else. The dashboard's
  Yahoo set then drew CL=F and GC=F in full, and ES=F gave one error. The multi-ticker call
  loaded those five and listed the other seven as failed. Via the Yahoo path CL=F equals the
  demo path's CL exactly. That is now `test_yahoo_path_matches_the_demo_path_on_the_same_file`
  (it catches reading `Adj Close` or `auto_adjust=True`), the first offline test of
  `price_return`'s Yahoo branch. Only yfinance's network behaviour remains for the owner's run.
- **Deferred:** a table view for each chart (the dataviz accessibility twin): the charts have
  hover values, and the tables beside them carry the key numbers.
- **For Phase 4, from the renders:** in `return-distribution`, the "Loss thr" and "Win thr"
  labels overlap when the thresholds are close (±0.2% on SPX). This is a `viz.py` issue, like the
  two noted under Phase 2.

## Phase 4: dashboard tabs

- **Streaks & cumulative:** the summary tables with episodes, and the streak and cumulative charts.
- **Rare events:** controls for holding period, change type and probability bounds, re-querying as
  they change; the table shows `count`, `episodes`, `n_windows`, `prob` and `last_occurred`.
- **Statistics:** four sections mirroring the statistics notebook, each with its tables and chart.
  The bootstrap runs on request with a visible progress state.
- **Multi-ticker:** runs over the selected ticker set (demo or Yahoo) on request; cross-ticker streak and
  distribution tables, and any failed tickers listed.

### Phase 4 notes

- **Tabs** (`dashboard/src/tabs/`):
  - `StreaksTab`: the streak and cumulative tables with episodes, the counts and frequency charts,
    and the timeline with a window control. The two-panel cumulative heatmap is full width.
  - `RareEventsTab`: the notebook widget's controls (holding period, event type, and the
    probability bounds as percentages). The bounds are debounced and validated; each change only
    refilters the server's cached table (about 20 ms). An empty result says why and what to
    change.
  - `StatisticsTab`: the notebook's four sections. Key/value tables for the moments, Jarque–Bera,
    Student-t and Hill; VaR and ES pivoted by method (`tables.varTable`, unit-tested); the
    Ljung–Box, variance-ratio and ARCH-LM results; risk-ratio tiles and the deepest drawdowns;
    and the Q-Q, autocorrelation, drawdown and rolling-risk charts. The bootstrap runs on request
    (holding period, 500/1,000/2,000 resamples), with its progress on the button. A "rare only"
    filter matches the notebook's cut, and the two event charts (above/below, longest lookback)
    stack full width.
  - `MultiTickerTab`: runs the selected set on request (about 8 s for the ten demo tickers,
    then cached). Shows failed tickers, the streak and distribution comparisons, and one
    collapsible rare-event table per ticker.
- **Shared pieces:** `QueryState` (loading, error, and the previous result dimmed while
  refetching), `DataTable` column `formats`/`labels`/`columns`, `KeyValues`, `useDebounced`.
- **The three `viz.py` issues from Phases 2 and 3, fixed** (the notebooks get the fixes too):
  - the histogram's threshold labels sit on the outer side of each line;
  - `plot_event_probabilities` labels 1, 2 and 5 in each decade when its range is under two
    decades (`dtick='D2'`), and keeps decade ticks otherwise;
  - its two model labels are pushed a label height apart when their line ends are close, with a
    wider right margin so they are not clipped.
  Three tests, each checked by a seeded mutation.
- **Checks:** 14 Vitest tests, the type check and build, and 272 Python tests. A Playwright
  walk-through of every tab in light, dark and at 390 px:
  - rare-event filters 46 → 21 → 13 events, cumulative only when chosen;
  - invalid bounds blocked with a message;
  - the bootstrap's progress state, and "rare only" at 46 of 104 events;
  - multi-ticker's ten drill tables;
  - no console errors and no horizontal scroll.
  From the screenshots, the cumulative heatmap and the event charts went full width, the
  dependence tables went two across (a p-value column was clipped at three), and the rare-event
  count now says it covers every holding period.

## Phase 4a: saved live data (owner's request, 2026-09-30)

The third data option: live data saved as plain CSV, downloaded once and then updated.

- **Store:** `data/local/{symbol}.csv`, the same layout as the demo files (`Date, Open, High,
  Low, Close, Volume`, as traded: `auto_adjust=False`), plus `data/local/manifest.csv` (symbol,
  rows, first and last date, when it was last updated). `data/local/` is **git-ignored**: it is
  real Yahoo data, and the repo is public. Symbols map to file names safely (`ES=F.csv`,
  `^GSPC.csv`; anything outside letters, digits and `=^._-` is escaped).
- **Package** (`src/tools/price_return/store.py`):
  - `save_local(symbol, start_date='2016-01-01')`: the first download writes the whole history.
    Afterwards it downloads only from a few days before the last saved date to today, and merges.
    The overlap catches Yahoo's revisions of recent bars, which are reported rather than silently
    replacing values.
  - Writes go to a temporary file and are renamed into place, so a failed download never leaves
    a half-written or emptied CSV.
  - `local_tickers()` lists the saved files with their date ranges.
  - `load_price_data(data_source='local')` reads the saved closes through the same cleaning as
    `yahoo` and `demo` (end date exclusive, non-positive closes dropped).
  - The CSV reader for `demo` and `local` becomes one function.
- **API:**
  - `GET /api/local`: the saved symbols, date ranges and last update.
  - `POST /api/local/{symbol}/update`: save or update one symbol. Returns rows added, the new
    range and any revised bars; a Yahoo failure is a 502 and leaves the file as it was.
  - `GET /api/tickers?set=local`: the saved symbols, with each symbol's thresholds from
    `configs/tickers.yaml` when it is there, `data_source` set to `local`.
- **Dashboard:**
  - The Data control offers three choices: **Demo data (offline)**, **Yahoo Finance (live)** and
    **Saved CSV (offline)**.
  - With Yahoo selected, a **Save to CSV** button beside the ticker saves or updates it.
  - With saved CSVs selected, each ticker shows its last saved date, flagged when it is more than
    a few days old, with an **Update** button.
  - While an update runs, its button shows progress; afterwards the charts refetch.
- **Refresh script:** `python scripts/update_local_data.py [symbols…]` updates the given symbols,
  or every symbol in `configs/tickers.yaml`. It is the hook for a scheduled refresh later, which
  hosting will need.
- **Tests** (offline, with the yfinance stub from Phase 3's Yahoo-path test):
  - the first save writes exactly the stub's bars;
  - an update adds only the new rows, and reports a revised overlapping bar;
  - a failed download leaves the file byte-for-byte unchanged;
  - `local` equals the `yahoo` path on the same data;
  - symbols with `=` and `^` round-trip through file names;
  - `data/local/` is ignored by git;
  - the API endpoints match the package.
  Plus a Playwright check of save, update and reload in the dashboard.

### Phase 4a notes

- **Package** (`store.py`), as designed, plus:
  - `config_params(symbol)` in `params.py` gives any symbol its thresholds: its own entry in the
    ticker config, else the config's `defaults`. `local_ticker_config` uses it, so a saved
    symbol outside the config still gets consistent thresholds.
  - `local_tickers` reads the files themselves, so a file added or edited by hand is listed with
    its real range; the manifest supplies only the update time.
  - Saved files get the permissions any new file would. The temporary file is owner-only until
    it is renamed into place.
  - The Yahoo path now drops yfinance's leftover `Price` column-axis name, so `local` and `yahoo`
    return identical frames.
- **API:**
  - `GET /api/local`.
  - `POST /api/local/{symbol}/update?start_date=` (the symbol is URL-encoded, e.g. `%5EGSPC`).
  - `set=local` for `/api/tickers` and `/api/multi-ticker`; an empty saved set gives a 422, not
    "every ticker failed".
  - Every ticker entry, in any set, carries `saved` (range and last update, or null), so the
    Yahoo set shows what is already saved.
  - Cache keys for `local` include the file's modification time, so an update from the dashboard
    or from the refresh script is picked up without a restart.
- **Dashboard:**
  - Data offers Demo (offline), Yahoo Finance (live) and Saved CSV (offline).
  - A saved-data row under the controls shows the saved range, a "⚠ n days old" flag (after
    four calendar days), and **Save to CSV** / **Update CSV** with the result: rows added and any
    revised values.
  - Afterwards every view refetches.
  - The saved set starts with an explanation of how to save.
- **Checks:**
  - 30 store and API tests, offline, on a yfinance stub serving the demo CL file under `CL=F`
    with a movable "today". They cover:
    - first save, update, a revision, and a bar missing from a new download;
    - a failed download, and a write that dies halfway (both leave every file byte for byte);
    - `local` equal to `yahoo`, and symbol file names;
    - git-ignore, file permissions, the endpoints, and cache freshness after an update;
    - the refresh script.
  - 9 seeded mutations, each caught.
  - 3 new Vitest tests (17 in all); 302 Python tests.
  - A Playwright run against the stubbed API with a scratch store: empty state → Save from the
    Yahoo set (flagged 15 days old) → the saved set offline → Yahoo moves on → Update adds 10
    bars (downloading only from ten days before the last bar), the flag clears and every view
    follows → multi-ticker over the saved set.

## Phase 5: one-command local run

- FastAPI mounts `dashboard/dist` as static files when it exists, so
  `python -m src.api` (`src/api/__main__.py`, uvicorn on 127.0.0.1:8000) serves everything.
- `scripts/dev.py` (done as `.py`, not `.sh`, to run on Windows too) runs uvicorn with reload
  plus the Vite dev server, for development.
- README "Dashboard" section: install (`pip install -e ".[api,stats]"`, then `npm install` and
  `npm run build` in `dashboard/`), run, and the architecture sketch.

### Phase 5 notes

- **`src/api/dashboard.py`:** a catch-all route, registered after every API route, so `/api/*`,
  `/docs` and `/openapi.json` always match first.
  - It serves `dashboard/dist`: `index.html` with `no-cache`, and the hashed `assets/` as
    immutable for a year. Any other path without a file extension gets `index.html`.
  - A missing file is a 404, and so is an unknown `/api/...` path: JSON, never the page.
  - Paths are resolved and must stay inside `dist` (encoded `../` is refused; tested).
  - The folder is looked up per request, so `npm run build` needs no restart. Before any build,
    `/` is a 503 page with the three build commands.
  - `/api/health` reports `dashboard_built`. API version 0.3.0.
- **`python -m src.api`** (`src/api/__main__.py`): `--host` (default 127.0.0.1), `--port` (8000),
  `--reload`, `--open`. It prints the URL, says how to build if needed, and warns when listening
  beyond this computer (there is no login).
- **Change from the plan:** `scripts/dev.py` instead of `scripts/dev.sh`, so the one launcher
  works on Windows as well.
  - It installs the npm packages if missing, then runs uvicorn with reload and `npm run dev`.
  - Ctrl+C stops both, and it also stops if either one exits.
  - Each server runs in its own process group, stopped as a group (`killpg`, or `taskkill /T` on
    Windows). The first version stopped only `npm`, which left Vite running on :5173; a test now
    checks that a server's own child process is stopped too.
- **Checks:**
  - 10 tests in `tests/test_serve.py`: page and assets, API precedence, missing files,
    traversal, the not-built page, the runner's defaults and warnings, and the process-group
    stop. 6 seeded mutations were caught.
  - Every tab driven in Chromium against `python -m src.api` on port 8000 (same results as under
    Vite, no errors).
  - The not-built page, shown by moving the build aside while the server ran; restoring it
    worked with no restart.
  - `scripts/dev.py` started, served the API through Vite, and stopped cleanly on SIGINT with no
    processes left.
  - 312 Python tests, 17 Vitest tests.

## Phase 6: end-to-end check and docs

- `dashboard/e2e/smoke.spec.ts` (Playwright, with the preinstalled Chromium): start the API on
  demo data, open each tab, and assert charts render and data arrives. Changing a rare-event
  filter must change the table.
- README (layout, setup, dashboard, changelog); plan statuses; pull request into `master`.

### Phase 6 notes

- **`npm run e2e`** in `dashboard/`: builds, then Playwright (`@playwright/test` pinned to
  1.56.1, the version whose Chromium is preinstalled here) starts `python -m src.api` on port
  8765 (`PYTHON` picks the interpreter) and runs `e2e/smoke.spec.ts`. Ten tests, about 30 s:
  - the overview (tiles, four charts, the demo notice);
  - a ticker change, applied thresholds, and a refused date range;
  - streaks and cumulative (tables, five charts, the timeline window);
  - rare events (a tighter bound shrinks the table; the type filter leaves only that type);
  - statistics (four sections, the bootstrap on request, the two event charts);
  - multi-ticker (ten tickers);
  - the saved-CSV choice (the explanation, or the overview if something is saved);
  - dark theme (page and chart background);
  - phone width (no sideways scroll);
  - the API and its docs beside the page.
  Every test also fails on any browser console error.
- `/docs` is checked over HTTP, not in the browser: FastAPI's page loads Swagger UI from a CDN,
  which this environment blocks.
- Three bugs seeded into the app (the rare-event type filter ignored, dark charts not re-coloured,
  the demo notice hidden) each failed their test.
- The end-to-end files type-check under `tsconfig.node.json` (Node types stay out of the app).
  `npm run typecheck` runs both configs, and Vitest runs only `src/**/*.test.ts`.

## Status (2026-09-30)

All phases are done. What is left is on the owner's computer, where Yahoo is reachable: the first
live run (the Yahoo set, and Save to CSV), and `npx playwright install chromium` before the first
`npm run e2e`. Next steps beyond the POC are under "Deferred".

## Deferred (documented, not built)

- **Docker:** a multi-stage image (Node build → slim Python) plus `docker compose`. This is the
  first step towards hosting.
- **Hosting:** authentication, HTTPS, secrets, monitoring, a shared data store (Phase 4a's CSV
  store is the local first step; Parquet or DuckDB when it outgrows CSV) with scheduled
  refresh, and job queues for long computations.
- **Trading execution:** a separate service, paper trading first.

## Critical files

- **API** (`src/api/`): `app.py` (endpoints), `schemas.py`, `serialize.py`, `cache.py`,
  `charts.py` (chart registry), `dashboard.py` (serves the build), `__main__.py`
  (`python -m src.api`).
- **Dashboard** (`dashboard/`): `src/` (`api.ts` client, `App.tsx`, `components/`, `tabs/`,
  `format.ts`, `chartTheme.ts`, `tables.ts`, `styles.css`), `e2e/smoke.spec.ts`, `vite.config.ts`,
  `playwright.config.ts`, `tsconfig*.json`, `package.json`.
- **Data:** `data/demo/*` and `configs/demo_tickers.yaml`; `data/local/` (git-ignored, created on
  the first save).
- **Scripts:** `make_demo_data.py`, `update_local_data.py`, `dev.py`.
- **Tests:** `test_api.py`, `test_demo_data.py`, `test_local_store.py`, `test_serve.py`, and
  additions to `test_price_return_stats.py`.
- **Package changes**, each with tests:
  - `data.py`: the `demo` and `local` sources;
  - `store.py` (new): saved live data;
  - `params.py`: `DEMO_CONFIG_PATH`, `config_params`;
  - `stats.py`: `event_probability_table`;
  - `viz.py`: the streak timeline 50 to 140 times faster, and three label and axis fixes.
- **Also:** `pyproject.toml` (`api` extra, `httpx2` in dev), `.gitignore`, `README.md`,
  `notebooks/api_examples.ipynb` (new), `notebooks/price_return_statistics.ipynb` (calls
  `event_probability_table`).

## Verification

- `pytest`: 312 tests, all offline (203 before this plan).
- `dashboard/`: `npm test` (17 unit tests), `npm run typecheck`, `npm run build`, and `npm run e2e`
  (10 end-to-end tests of every tab against `python -m src.api` on the demo data).
- Every phase's screens were screenshotted in Chromium and inspected, in light, in dark and at
  phone width. The Yahoo path was run offline through a yfinance stand-in serving the demo files
  under their Yahoo symbols.
- **The owner's run:** `python -m src.api` on the owner's computer, on the demo set and then on
  the Yahoo set for ES=F and the configured tickers, including Save to CSV. This is the first live
  Yahoo run, since Yahoo is blocked in this environment.
