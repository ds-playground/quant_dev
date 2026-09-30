# React dashboard POC on the price-return package (`dev/react_dashboard_poc`)

This is the working plan for the dashboard proof of concept, kept in the repo so work can continue
from any machine or session. Update it as phases finish: mark the phase ✅ with its commit hash in
the table, and add a "Phase N notes" section with anything a later phase needs to know.

## Picking this up

- **Setup:** Python 3.12 or newer: `pip install -e ".[api,stats,dev]"` from the repo root, then
  `pytest`; all tests pass offline. Try the API with `uvicorn src.api.app:app`, then open
  `http://127.0.0.1:8000/docs`. The dashboard needs Node 20 or
  newer (from Phase 3): `npm install` in `dashboard/`.
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

## Architecture

```
React (dashboard/, Vite + TypeScript) ──/api/*──▶ FastAPI (src/api/) ──▶ src.tools.price_return
   react-plotly.js renders figure JSON              wraps + serializes      (all analysis, all charts)
```

- **Parameters:** the request model is generated from `Params`'s dataclass fields
  (`src/tools/price_return/params.py`), so a new `Params` field appears in the API without
  duplication. Unknown keys are rejected, as `load_ticker_config` already does.
- **Charts:** a chart endpoint calls the existing `viz.plot_*` function and returns
  `fig.to_json()`, so charts are identical to the notebooks and exist once, in Python.
- **Tables:** DataFrames are returned as records, with NaN as `null` and dates as ISO strings.
  Formatting (%, rounding) happens in React, as `format_probability_table` does for notebooks.
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
| **2** | API complete | streaks, cumulative, rare-event, chart, statistics and multi-ticker endpoints; tests |
| **3** | Dashboard shell | `dashboard/` (Vite, React, TypeScript), parameter panel, tabs, Plotly chart component, Overview tab |
| **4** | Dashboard tabs | Streaks & cumulative, Rare events (live filters), Statistics, Multi-ticker |
| **5** | One-command local run | FastAPI serves the built app; `python -m src.api`; a dev script for both servers; README "Dashboard" section |
| **6** | End-to-end check and docs | Playwright smoke test of every tab; README, changelog, plan statuses; PR |

Working rules, as in the earlier plans: one phase per request, then stop; commit and push at the
end of each phase; tests stay offline (`data_source='simulated'`, or a stubbed yfinance); no
analysis logic outside `src/tools/`.

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
- `POST /api/multi-ticker`: `analyze_ticker` for each configured ticker, then `compare_tickers`.
  Failed tickers are reported alongside results, as `rare_case_run` does.
- **Tests:** the same round-trip parity for every endpoint; every registered chart builds; the
  rare-event filters match `low_probability_view` called directly.

## Phase 3: dashboard shell (`dashboard/`)

- Vite + React + TypeScript. `react-plotly.js` with `plotly.js-dist-min` for charts; TanStack
  Query for fetching and caching; plain CSS using the palette tokens `viz.py` already uses, with
  no UI framework.
- `src/api.ts`: a typed client whose types mirror the API responses.
- `components/`: `ParamsPanel` (ticker picker from `/api/tickers`, dates, thresholds, data
  source), `PlotlyChart` (renders figure JSON), `DataTable`, `StatTiles`.
- `App.tsx`: the parameters panel plus tabs; the **Overview** tab shows snapshot tiles, price and
  returns, the distribution, and the rolling average and volatility charts.
- `.gitignore` gains `dashboard/node_modules` and `dashboard/dist`.

## Phase 4: dashboard tabs

- **Streaks & cumulative:** the summary tables with episodes, and the streak and cumulative charts.
- **Rare events:** controls for holding period, change type and probability bounds, re-querying as
  they change; the table shows `count`, `episodes`, `n_windows`, `prob` and `last_occurred`.
- **Statistics:** four sections mirroring the statistics notebook, each with its tables and chart.
  The bootstrap runs on request with a visible progress state.
- **Multi-ticker:** runs over `configs/tickers.yaml` on request; cross-ticker streak and
  distribution tables, and any failed tickers listed.

## Phase 5: one-command local run

- FastAPI mounts `dashboard/dist` as static files when it exists, so
  `python -m src.api` (`src/api/__main__.py`, uvicorn on 127.0.0.1:8000) serves everything.
- `scripts/dev.sh` runs uvicorn with reload plus the Vite dev server, for development.
- README "Dashboard" section: install (`pip install -e ".[api,stats]"`, then `npm install` and
  `npm run build` in `dashboard/`), run, and the architecture sketch.

## Phase 6: end-to-end check and docs

- `dashboard/e2e/smoke.spec.ts` (Playwright, with the preinstalled Chromium): start the API on
  simulated data, open each tab, and assert charts render and data arrives. Changing a rare-event
  filter must change the table.
- README (layout, setup, dashboard, changelog); plan statuses; pull request into `master`.

## Deferred (documented, not built)

- **Docker:** a multi-stage image (Node build → slim Python) plus `docker compose`. This is the
  first step towards hosting.
- **Hosting:** authentication, HTTPS, secrets, monitoring, a shared data store with scheduled
  refresh, and job queues for long computations.
- **Trading execution:** a separate service, paper trading first.

## Critical files

- New: `src/api/{__init__,__main__,app,schemas,serialize,cache}.py`, `tests/test_api.py`,
  `dashboard/` (`package.json`, `vite.config.ts`, `src/{main,App,api}.tsx|ts`,
  `src/components/*`, `src/tabs/*`, `e2e/smoke.spec.ts`), `scripts/dev.sh`,
  `docs/dashboard_plan.md`.
- Modified: `pyproject.toml` (`api` extra, `httpx` in dev), `.gitignore`, `README.md`.
- Reused, not changed: `src/tools/price_return/` (`params`, `data`, `analysis`, `pipeline`,
  `viz`, `stats`, `options`).

## Verification

- `pytest`: the existing 203 tests plus the API tests, all offline.
- `npm run build` and `tsc --noEmit` in `dashboard/`.
- Playwright smoke test against the built app on simulated data. Every tab is screenshotted and
  inspected, as the charts were in the last work.
- **The owner's run:** `python -m src.api` on the owner's computer with Yahoo data for ES=F and the
  configured tickers. This is the first real-data run, since Yahoo is blocked in this environment.
