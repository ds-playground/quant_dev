# quant_dev

A work-in-progress research playground for historical price and return analysis, and
the TradingView indicators that go with it.

## Purpose

The project exists to answer one question with evidence rather than intuition: **how
often does a given price move actually happen, and when did it last happen?**

Given a daily return series, the framework measures:

- how often an instrument strings together consecutive up or down days,
- how often a multi-day window compounds past a threshold, regardless of what the
  individual days did,
- and which moves are rare enough to be worth noticing — each carrying the date it
  was last seen, so a "0.08% probability" can be checked against reality rather than
  taken on faith.

Running the same pipeline across instruments is the point rather than a convenience.
A threshold that is unremarkable for natural gas is an extreme event for EUR/USD, and
a cross-instrument comparison only means anything once that is accounted for — which
is why thresholds are configured per ticker.

This is a personal research repo, not a library and not a trading system. Nothing here
places orders. The numerical methods are tested against independent references (hand-counted
series, closed forms, scipy, formulas written out from their papers, and Pine's reference
formulas for the indicators; see [Tests](#tests)). Tests can only cover the cases someone
thought to write, though, so treat the results as research rather than as a basis for trading.

Stage one has three goals:

1. A framework for historical price-return analysis.
2. Basic helper functions for analysis and visualization.
3. Pine scripts for TradingView.

Goals 1 and 2 live in `src/tools/price_return/`; goal 3 lives in `pine_scripts/`.

## Setup

The project uses a conda environment (`quant_env`, Python 3.13). Install the package
once in editable mode from the repo root:

```bash
conda activate quant_env
pip install -e .
```

That puts `src.tools` on the import path permanently, so notebooks and scripts can
`from src.tools.price_return import ...` from any working directory — no `sys.path`
manipulation required.

The technical-analysis libraries are optional extras, because `TA-Lib` needs a C
toolchain and would otherwise block a clean install. `pandas_ta` needs Python 3.12 or
newer, so the project requires 3.12:

```bash
pip install -e ".[dev]"   # pytest, ipywidgets, nbformat, httpx2 (for the API tests)
pip install -e ".[ta]"    # pandas_ta, ta, TA-Lib
pip install -e ".[stats]" # scipy, for price_return.stats
pip install -e ".[api]"   # fastapi, uvicorn, for the dashboard API in src/api
```

`price_return` imports without scipy; only the five statistics functions that need it
(`fit_student_t`, `qq_points(dist='t')`, `ljung_box`, `arch_lm`, `model_probabilities`)
raise an install hint.

`requirements.txt` remains the flat list used by Colab.

## Quickstart

```python
from src.tools.price_return import (
    Params, load_price_data, add_rolling_stats,
    detect_streaks, summarize_streaks, plot_streak_timeline,
)

P  = Params(ticker='ES=F', start_date='2016-01-01', data_source='yahoo')
df = add_rolling_stats(load_price_data(P), P)

streaks = detect_streaks(df, P)
summarize_streaks(df, streaks, P)
plot_streak_timeline(df, streaks, P).show()
```

`Params` carries every tunable value — data source, thresholds, windows, chart
settings — and defaults to a self-contained simulated series, so `Params()` runs
with no network access. Nothing is bound at import time: configure once, pass the
object around, re-run.

`data_source` is `'yahoo'` (downloaded), `'simulated'` (the default), or `'demo'`: the
processed daily files in `data/demo`, which run offline on market-shaped data.

## Demo data

`data/demo/` holds daily bars for ten tickers (SPX, NDQ, YM, CL, RTY, EURUSD, XAUUSD, AAPL,
KO, TSLL), from 2016 to 2026-09. **They are processed data, not market data:** Yahoo Finance
bars with 0.01% random noise added to every open, high, low, close and volume, provided for
education and for running the notebooks, the dashboard and the tests offline. Do not use them
for trading. [`data/demo/README.md`](data/demo/README.md) says how they were made
(`scripts/make_demo_data.py`, seeded), and what to watch for, such as CL's negative close in
April 2020 and XAUUSD being gold futures.

```python
from src.tools.price_return import demo_tickers, load_ticker_config

P  = Params(data_source='demo', ticker='SPX', start_date='2016-01-01')
df = add_rolling_stats(load_price_data(P), P)

demo_tickers()                                     # the tickers with a demo file
load_ticker_config('configs/demo_tickers.yaml')    # all ten, labelled, with thresholds
```

For `'yahoo'` and `'demo'` alike, `end_date` is exclusive and closes are as traded (not
dividend-adjusted).

## Saved live data

Live Yahoo data can be saved as plain CSV in `data/local/` and analysed offline with
`data_source='local'`. The folder is **git-ignored**: it holds real market data, which does not
belong in a public repo. Each symbol is one file (`ES=F.csv`) in the demo files' layout, plus a
`manifest.csv` of date ranges and update times.

```python
from src.tools.price_return import save_local, local_tickers

save_local('ES=F')      # first time: the history since 2016; afterwards: just the new bars
local_tickers()         # what is saved, from when to when
P = Params(data_source='local', ticker='ES=F', start_date='2016-01-01')
```

An update re-downloads the last ten days as well as the new ones. Yahoo sometimes revises
recent bars; a revised value replaces the saved one and is reported. A failed download leaves
the saved file exactly as it was. To update every ticker in `configs/tickers.yaml` (or a list),
run `python scripts/update_local_data.py [symbols...]`; it exits with 1 if any symbol failed,
so it can run on a schedule. In the dashboard, **Save to CSV** and **Update CSV** do the same
for one ticker, and the Data control's **Saved CSV (offline)** lists what is saved.

## Dashboard

A React dashboard over the same package, served with its API by one command
([`docs/dashboard_plan.md`](docs/dashboard_plan.md)). Five tabs: Overview, Streaks &
cumulative, Rare events (live filters), Statistics (the bootstrap on request) and Multi-ticker.
Three data choices: the demo files (offline), Yahoo Finance (live), and live data you have
saved as CSV (offline; see "Saved live data"). With Yahoo Finance, **Other ticker…** at the end
of the ticker list loads any Yahoo symbol (it gets the defaults from `configs/tickers.yaml`), and
keeps it in an "Added" group in this browser. The Multi-ticker tab compares any chosen mix of
the configured and added Yahoo tickers, saved CSVs and demo files; the same symbol live and
saved can sit side by side.

```
browser ──▶ python -m src.api (FastAPI, 127.0.0.1:8000) ──▶ src.tools.price_return
            serves the built React app (dashboard/dist)       all analysis, all charts
            and /api/* (JSON, Plotly figures)
```

**Set up once** (Python 3.12+, Node 20+), from the repo root:

```bash
pip install -e ".[api,stats]"
cd dashboard && npm install && npm run build && cd ..
```

**Run:**

```bash
python -m src.api          # then open http://127.0.0.1:8000  (--open opens it for you)
```

It listens on this computer only. `--port` changes the port; `--host 0.0.0.0` would expose
it to your network, and it has no login, so it warns. The API's own documentation is at
http://127.0.0.1:8000/docs, and `notebooks/api_examples.ipynb` queries it from Python. After
pulling dashboard changes, run `npm install` and `npm run build` in `dashboard/` again
(`npm install` picks up any new packages); no restart is needed. If
the dashboard has never been built, the page at `/` says how.

**Develop:** `python scripts/dev.py` runs the API (restarting on Python changes) and the Vite
dev server (updating the page on every React change) together, on http://localhost:5173, until
Ctrl+C stops both. It works on Windows, macOS and Linux.

**Check it:** in `dashboard/`, `npm test` (unit tests), `npm run typecheck`, and `npm run e2e`,
which builds the dashboard, starts `python -m src.api` on port 8765 and drives every tab in
Chromium on the demo data (about 30 s). The first time, run `npx playwright install chromium`;
set `PYTHON` if your interpreter is not called `python`.

## Methodology

The framework asks a few related questions about a daily return series.

**Price data is unadjusted.** `load_price_data` passes `auto_adjust=False` to
yfinance, so `Close` is the price that actually traded rather than a series
adjusted backwards for dividends and splits. This is deliberate: one use of this
analysis is sizing option strategies, and a strike is set against the traded
price — adjusting the history would misstate where a strike sits relative to
spot.

The cost is that for a dividend payer, an ex-dividend drop registers as a
negative return even though a holder lost nothing. Measured on KO over
2016–2026, the unadjusted series gives a mean daily return of 0.0337% against
0.0464% adjusted, and individual ex-dividend days differ by up to 0.91
percentage points. For the futures and FX pairs in `configs/tickers.yaml` the
two settings are bit-for-bit identical, since neither pays a dividend or splits
— so this choice only bites if you add equities or ETFs. The argument is passed
explicitly because yfinance changed its default, and leaving it implicit meant
the same notebook could produce different numbers on different machines.

**One definition of a multi-day return.** Every `d`-day return in the package is
*compounded*, `(1 + r1)(1 + r2)…(1 + rd) − 1` (`compound_returns`), which is what a
position held over those days earns. Adding up the daily returns instead overstates
losses and understates gains: two −10% days are −19%, not −20%. The gap grows with the
horizon and the volatility. Until 2026-09-29 the rolling columns and the rare-event
"cumulative" rows added returns up while the cumulative-threshold table compounded them.

**Frequencies are shares of complete windows, and overlapping windows are also counted
as episodes.** A `d`-day frequency divides by the `n − d + 1` windows that fit in `n`
days, not by `n`. Rolling windows overlap, so one 6-day winning run is five 2-day
streaks. Each table therefore also gives *episodes*: every unbroken run of qualifying
windows counted once. When episodes are far fewer than the count, the frequency rests
on fewer independent events than it seems to.

**Units.** Percent: `win_threshold`, `loss_threshold`, `cum_thresholds`, `sim_drift`,
`sim_vol`, and the `return_pct` column (0.5 means 0.5%). Decimal:
`return_thresholds`, `prob_max`, `prob_min`, the `PCT Change` columns and
`daily_returns_series` (0.01 means 1%). The mix is historical; renaming would break
`configs/tickers.yaml`.

**Rolling statistics.** For each holding period `d`, the `d`-day compounded return
is computed, then a `trade_days`-long rolling mean and standard deviation over it.
The annualized return is the compounded return over the last `trade_days` days; the
annualized volatility is the daily standard deviation scaled by √`trade_days`. Default
holding periods are 1, 2, 3, 4, 5, 10 and 250 days.

**Win/loss streaks.** A day is a *win* when its return exceeds `win_threshold`
(default +0.5%) and a *loss* when it falls below `loss_threshold` (default −0.5%).
A streak is a rolling window in which *every* day is a win, or every day a loss.
Streaks are reported as counts, as episodes, as a share of complete windows, and as
an average return per streak — then shaded onto a return timeline.

**Cumulative thresholds.** Separately from all-win streaks, this counts rolling
windows whose *compounded* return clears a threshold (default 0.5%, 1%, 2%),
regardless of what the individual days did. A window can clear +2% cumulatively
while containing losing days, so this and the streak view answer different
questions.

**Rare-event probabilities.** For each combination of move size (0.01% up to 5%),
holding period (1 to 30 days) and lookback window (2 or 5 years), this measures how
often a move of at least that size occurred over that horizon: `prob` is the share of
complete windows (`n_windows`), with `count` and `episodes` alongside. Filtering to the low
end — events that did happen but rarely — produces the rare-event table, which is
browsable interactively in `price_return_analysis.ipynb` via `interactive_low_probability`.

**Statistics (`price_return/stats.py`, `notebooks/price_return_statistics.ipynb`).** Four
groups of questions about the same daily returns:

- *Distribution and tails.* Moments and the Jarque–Bera test; a maximum-likelihood
  Student-t fit and Q-Q points against it and a normal; the Hill tail index for each tail
  (smaller is fatter; below 4, kurtosis is not a stable number); and value at risk and
  expected shortfall over 1, 5 and 10-day compounded horizons, by three methods:
  historical, normal, and Cornish–Fisher (the normal quantile corrected for skew and
  kurtosis). Cornish–Fisher overshoots somewhat at 99% on fat tails; historical is the
  one to trust when the history is long. Multi-day VaR uses overlapping windows, so
  longer horizons rest on fewer independent observations than their row counts suggest.
- *Dependence and volatility clustering.* Autocorrelation of returns and of squared
  returns with the 95% band, the Ljung–Box test on both, the Lo–MacKinlay variance ratio
  with its heteroskedasticity-robust z* (above 1: moves persist; below 1: they partly
  reverse), and Engle's ARCH-LM test. All three tests are written out from their papers.
- *Drawdowns and risk-adjusted returns.* The drawdown from the running peak (starting
  capital included), the deepest drawdowns with peak, trough, recovery and durations,
  and Sharpe, Sortino and Calmar ratios, overall and rolling.
- *Uncertainty on the rare-event probabilities.* Each probability gets a 95% interval
  from a stationary block bootstrap: blocks of days are resampled, with a mean length of
  at least twice the holding period, so volatility clustering and the overlap of windows
  survive. Each is also set against an i.i.d. normal and a fitted Student-t (both fitted
  to log returns). Observed well above both means fatter tails or more dependence than an
  i.i.d. model allows. An interval that spans a multiple of the estimate, or a handful of
  `episodes`, means the probability is not precise enough to price on alone.

## Repo layout

```
quant_dev/
├── README.md                              this file
├── pyproject.toml                         packaging; `pip install -e .`
├── requirements.txt                       flat dependency list for Colab
├── config.py                              defaults for the dev/ scratch notebooks only
│
├── configs/
│   ├── tickers.yaml                       ticker set + per-ticker parameters (Yahoo data)
│   └── demo_tickers.yaml                  the same for the ten demo files
│
├── data/
│   ├── demo/                              processed daily bars (not market data) + README, manifest
│   └── local/                             saved live Yahoo data (git-ignored; created on first save)
│
├── scripts/
│   ├── dev.py                             development: the API and the Vite dev server together
│   ├── make_demo_data.py                  regenerates data/demo from Yahoo, with seeded noise
│   └── update_local_data.py               saves or updates live data in data/local
│
├── src/
│   ├── api/                               FastAPI app for the dashboard: wraps price_return, no analysis
│   │   ├── __main__.py                    `python -m src.api`: the dashboard and API on one port
│   │   ├── dashboard.py                   serves the built dashboard (dashboard/dist)
│   │   ├── app.py                         endpoints: health, tickers, overview, streaks, cumulative,
│   │   │                                  rare-events, charts, statistics, multi-ticker
│   │   ├── charts.py                      chart registry: name → viz.plot_* call
│   │   ├── schemas.py                     request model generated from Params
│   │   ├── serialize.py                   JSON conversion: NaN → null, ISO dates, Plotly figures
│   │   └── cache.py                       LRU caches: prices, rare-event tables, ticker runs
│   └── tools/
│       ├── price_return/                  the framework
│       │   ├── __init__.py                re-exports the whole public API
│       │   ├── params.py                  Params + the ticker config that builds it
│       │   ├── data.py                    price loading (Yahoo, demo, local, simulated), rolling statistics
│       │   ├── store.py                   saved live data: save, update, list (data/local)
│       │   ├── analysis.py                streaks, thresholds, rare events
│       │   ├── viz.py                     the Plotly charts (nine analysis, five statistics)
│       │   ├── report.py                  formatting, interactive table, CSV export
│       │   ├── pipeline.py                per-ticker run + cross-ticker comparison
│       │   ├── options.py                 price ranges, move probabilities, expected P&L
│       │   └── stats.py                   distribution, dependence, drawdowns, probability intervals
│       └── ta_tools/                      technical analysis (in progress)
│           ├── backend.py                 the only TA-Lib / pandas_ta import site
│           ├── data.py                    bars: Yahoo (load_bars), CSV (read_bars), seeded (make_bars)
│           ├── overlap.py                 sma, ema, wma (TA-Lib); hma, alma (pandas_ta)
│           ├── volatility.py              stdev, bb (TA-Lib); true_range, atr (Pine-exact)
│           ├── momentum.py                rsi (TA-Lib)
│           ├── pine.py                    Pine primitives no library has (linreg, rma, pivots, ...)
│           └── indicators.py              whole Pine indicators, ported (linreg_candles, trendlines)
│
├── dashboard/                             React dashboard (Vite, TypeScript), talks to src/api
│   ├── src/api.ts                         typed client for every endpoint
│   ├── e2e/smoke.spec.ts                  end-to-end check of every tab (Playwright, npm run e2e)
│   ├── src/components/                    parameter panel, Plotly chart, table, stat tiles
│   └── src/tabs/                          one component per tab
│
├── docs/
│   ├── dashboard_plan.md                  React dashboard POC plan, with status
│   ├── price_return_plan.md               legacy removal + price-return revamp plan, with status
│   └── ta_tools_plan.md                   phased plan for ta_tools, with status
├── notebooks/                             tracked, promoted notebooks
│   ├── api_examples.ipynb                 example queries to every dashboard API endpoint
│   ├── price_return_analysis.ipynb        single-ticker streak and rare-event analysis
│   ├── price_return_statistics.ipynb      statistics of one ticker's returns, option sizing
│   ├── rare_case_run.ipynb                config-driven multi-ticker run
│   ├── ta_package_evaluation.ipynb        TA-Lib vs pandas_ta vs ta
│   ├── ta_tools_primitives.ipynb          how the ta_tools primitives are wrapped
│   ├── ta_tools_exploration.ipynb         every ta_tools indicator on AAPL
│   └── ta_tools_read_data.ipynb           checks every ta_tools data reader
│
├── pine_scripts/                          TradingView indicators
│   ├── *.pine                             the modified indicators
│   └── references/                        their unmodified originals
│
├── tests/
│   ├── test_smoke.py                      offline end-to-end pipeline check
│   ├── test_api.py                        src/api against direct package calls
│   ├── test_demo_data.py                  the demo files and the 'demo' data source
│   ├── test_local_store.py                saving, updating and reading data/local, offline
│   ├── test_serve.py                      serving the dashboard, `python -m src.api`, the dev launcher
│   ├── test_price_return_stats.py         price_return options, methods, statistics, charts
│   └── test_ta_tools.py                   ta_tools, offline
│
└── dev/                                   scratch work — gitignored, never tracked
```

**`price_return` is a package, imported as one module.** It was a single 941-line
file until the sections were split out; `__init__.py` re-exports everything, so
`from src.tools.price_return import ...` works exactly as before and no notebook
needed changing. Import from the submodules directly if you prefer
(`from src.tools.price_return.viz import plot_streak_frequency`).

**`configs/tickers.yaml`** drives `notebooks/rare_case_run.ipynb`, which analyses every
ticker listed there and prints two cross-ticker summary tables. The file has a
`defaults` block merged with per-ticker overrides; keys must be `Params` fields
apart from `label`, and anything else raises rather than being silently ignored.
Delete the file and the notebook falls back to ES=F, NQ=F, YM=F and RTY=F, saying
so as it does.

Per-ticker overrides exist mainly to keep thresholds comparable across asset
classes. FX runs at roughly a third of the equity futures' volatility, so the
default ±0.2% win/loss threshold is about 0.4 standard deviations for a currency
pair against 0.15 for an index future; the FX entries halve it. Comparing streak
frequencies across instruments without that adjustment mostly measures the
threshold, not the market.

**The `dev/` → `notebooks/` convention.** `dev/` is ignored wholesale, as are any
files matching `*_dev*`, `*_tmp*`, `*_wip*`, `*_old*`, `*_bak*` and `*copy*`. Work
happens in `dev/`; copying a notebook into `notebooks/` is what promotes it to
tracked. Expect the two copies to drift — `notebooks/` is the published one.

**Notebooks.**

| Notebook | Status |
|---|---|
| `api_examples.ipynb` | Current, committed without outputs. Example calls to every endpoint of the dashboard API (`src/api`) over HTTP, with the answers as tables and charts; runs on the demo data by default, and starts the API inside the kernel if none is running. Needs the `api` and `stats` extras; not for Colab. |
| `price_return_analysis.ipynb` | Current. Single-ticker streak, threshold and rare-event analysis, built on `src/tools/price_return/`. |
| `price_return_statistics.ipynb` | Current, committed without outputs. One ticker (default `ES=F`, thresholds from `configs/tickers.yaml`): distribution and tails, dependence, drawdowns and risk, the rare-event probabilities with bootstrap intervals and model comparisons, and option sizing with the four P&L grids from the retired `test_es`. Needs the `stats` extra. |
| `rare_case_run.ipynb` | Current. Config-driven; runs every ticker in `configs/tickers.yaml` and emits two cross-ticker summary tables. |
| `ta_package_evaluation.ipynb` | Committed with outputs. Compares TA-Lib, pandas_ta and ta; the basis for choosing TA-Lib. |
| `ta_tools_primitives.ipynb` | Committed with outputs. How each `ta_tools` primitive is wrapped, and how it compares with Pine. |
| `ta_tools_exploration.ipynb` | Current. AAPL since January 2023: moving averages, Bollinger Bands, both Pine ports, RSI and ATR. |
| `ta_tools_read_data.ipynb` | Current, committed without outputs. Runs `make_bars`, `load_bars` (daily and intraday, with Yahoo's history limits) and `read_bars` (TradingView exports, other layouts, your own file), checking each against the shared output contract; ends with a pass/fail count. |

**Two config mechanisms, deliberately.** `config.py` serves the `dev/` scratch
notebooks. Everything under `src/` uses the `Params` dataclass instead, which is
authoritative for the package. They overlap; that is intentional, so scratch work
can be retuned without touching the package.

## API

`src/tools/price_return/`, grouped as it is in `__all__`:

| Function | Purpose |
|---|---|
| `Params` | Every tunable value for a run |
| `load_price_data` | A date / price / return_pct frame, from Yahoo Finance, the demo files or simulation |
| `add_rolling_stats` | Adds `PCT Change {d}` / `{d} Av` / `{d} STD` columns plus the annualized pair |
| `latest_snapshot`, `show_latest_snapshot` | Latest annualized and daily mean/vol |
| `daily_returns_series` | Date-indexed decimal daily returns |
| `compound_returns` | The compounded `n`-day return ending on each day: the package's one definition |
| `demo_tickers` | The tickers with a file in `data/demo` |
| `save_local`, `read_local` | Save or update a symbol's live data in `data/local`; read the saved bars |
| `local_tickers`, `local_ticker_config` | What is saved; `{symbol: Params}` for it, thresholds from the ticker config |
| `detect_streaks` | Every rolling window where all days are wins, or all losses |
| `summarize_streaks` | One row per window: counts, frequencies, average returns |
| `analyze_cumulative` | Rolling windows whose compounded return clears each threshold |
| `summarize_cumulative` | The above as a count/frequency table |
| `consecutive_analysis` | Probability of a ± move over `n_days`, within the last `n_years` |
| `build_historical_analysis` | The above across every threshold × holding period × lookback |
| `filter_low_probability`, `low_probability_view` | The rare-but-observed rows |
| `format_probability_table` | Display copy with threshold and prob as percentages |
| `interactive_low_probability` | Live-filtered rare-event table (ipywidgets) |
| `plot_rolling_average`, `plot_rolling_volatility` | Rolling return and volatility per holding period |
| `plot_price_and_returns` | Price above, daily returns below, with threshold lines |
| `plot_return_distribution` | Return histogram coloured by sign |
| `plot_streak_counts`, `plot_streak_frequency` | Streaks as counts and as a share of trading days |
| `plot_streak_timeline` | Returns with win/loss streak periods shaded |
| `plot_cumulative_heatmap`, `plot_cumulative_counts` | Threshold-clearing counts and frequencies |
| `export_tables` | Write a `{filename: DataFrame}` mapping to CSV |
| `load_ticker_config` | `{ticker: Params}` from `configs/tickers.yaml` (or another config, such as `configs/demo_tickers.yaml`), with a built-in fallback |
| `config_params` | One symbol's `Params` from the ticker config, or from its defaults if not listed |
| `analyze_ticker`, `compare_tickers` | The whole pipeline for one ticker; cross-ticker streak and distribution tables |
| `distribution_summary` | One row of return-distribution statistics, using the ticker's thresholds |
| `price_range` | One-standard-deviation price band some days ahead, or over the hours left in a session |
| `move_probabilities` | How often the 1/5/10-day return moved above, below or beyond a scaled, actual or fixed threshold |
| `expected_pnl` | Expected P&L per contract from those probabilities and a `{horizon: {direction: [win, loss]}}` grid |
| `return_moments`, `jarque_bera` | Moments, and the Jarque–Bera normality test |
| `fit_student_t`, `qq_points` | Student-t fit; sorted returns against normal or t quantiles |
| `tail_index` | Hill tail index for losses and gains |
| `value_at_risk` | VaR and expected shortfall: historical, normal, Cornish–Fisher, over compounded horizons |
| `autocorrelation`, `ljung_box` | Autocorrelation of returns, squared and absolute returns; Ljung–Box on returns and squares |
| `variance_ratio`, `arch_lm` | Lo–MacKinlay variance ratio with robust z*; Engle's ARCH-LM test |
| `drawdown_series`, `drawdown_table`, `max_drawdown` | Drawdown from the running peak; the deepest drawdowns with dates and durations |
| `risk_ratios`, `rolling_risk` | Annualized return and volatility, Sharpe, Sortino, Calmar; rolling versions |
| `stationary_bootstrap`, `bootstrap_interval` | Politis–Romano resampling indices; a percentile interval for any statistic |
| `probability_intervals`, `model_probabilities` | Rare-event probabilities with bootstrap intervals; what i.i.d. normal and Student-t models predict |
| `event_probability_table` | Both of those joined with the observed counts and episodes: the table `plot_event_probabilities` draws |
| `plot_qq`, `plot_autocorrelation` | Q-Q against normal and t; autocorrelation panels with the 95% band |
| `plot_drawdown`, `plot_rolling_risk` | Underwater chart; rolling volatility and Sharpe panels |
| `plot_event_probabilities` | Observed probabilities with intervals against both models, log scale |

## Technical analysis

`src/tools/ta_tools/` wraps TA-Lib, with pandas_ta for breadth, behind a consistent
API, and is where the Pine indicators below are being ported to Python. Its phased
plan, with the status of each phase, is `docs/ta_tools_plan.md`.

**Which library, and why.** `notebooks/ta_package_evaluation.ipynb` compares
`pandas_ta`, `ta` and `TA-Lib` on a seeded offline OHLC fixture, judged on coverage
and API shape:

| | indicators (ex-patterns) | Pine primitives | DataFrame accessor | risk |
|---|---|---|---|---|
| **TA-Lib 0.6.8** | 97 | **7/10** | no | stable; C extension already builds here |
| pandas_ta 0.4.71b0 | 151 | 6/10 | yes | beta, not validated against pandas 3 |
| ta 0.11.0 | 80 | 3/10 | no | pure python, stable |

**TA-Lib wins.** Coverage is close once TA-Lib's 61 candlestick pattern recognisers
are set aside, and all three return a Series with the index preserved. What decides
it is the Pine primitives: `ta` has no linear regression at all, which rules it out
given LinReg Candles calls it four times and ZLSMA nests it. TA-Lib alone exposes
`LINEARREG_SLOPE` and `LINEARREG_INTERCEPT` alongside `LINEARREG`, which is what
makes Pine's `offset` semantics reconstructible as
`LINEARREG - LINEARREG_SLOPE * offset`. `pandas_ta` also takes an `offset`, but it
means a post-shift of the output series — a different operation, and an easy trap.

Two smaller findings from the same run: `ta` returns ATR with **no** warm-up NaNs
where the other two return 14, so it seeds differently; and TA-Lib's C extension is
already built in this environment, contrary to the assumption behind keeping these
libraries in the optional `[ta]` extra.

**The package is part wrapper, part original code.** No library supplies pivot
high/low with publication delay, or a harness for Pine `var` series that depend on
their own previous bar. Those are implemented here regardless of which library is
wrapped.

**Two backends, one import site.** TA-Lib is the primary; pandas_ta is used only
for indicators TA-Lib has no equivalent for — there are 106 once naming aliases
are discounted, including `supertrend`, `donchian`, `kc`, `vwap` and `zigzag`. On
any overlap TA-Lib wins, so the pandas_ta beta is never on the critical path.
`backend.py` is the only module that imports either library, and loads pandas_ta
lazily so importing `ta_tools` never touches the beta. A `CAPABILITIES` map records
each primitive's source as `talib`, `pandas_ta`, `derived` or `custom`.

Two pandas_ta functions are deliberately *not* used. `rma` is a genuine Pine
primitive TA-Lib lacks, but it is a few lines on the one primitive where
controlling warm-up matters most, so it is implemented here. And `pivots` computes
support/resistance levels, not Pine's `pivothigh`/`pivotlow` swing detection.

**Ported indicators** live in `indicators.py` and are built only from the primitives
above, so they inherit their Pine parity. Arguments keep the Pine input names and
defaults, and chart-only inputs (colours, line widths, visibility toggles) are dropped:
an indicator returns data, not a drawing. `linreg_candles` ports
`Linear_Regression_Candles_and_Slope.pine` and returns `lrc_open/high/low/close`,
`lrc_signal`, `lrc_slope` and `lrc_bull`:

```python
from src.tools import ta_tools
bars = ta_tools.make_bars()
lrc = ta_tools.linreg_candles(bars['open'], bars['high'], bars['low'], bars['close'])
tl = ta_tools.trendlines(bars['high'], bars['low'], bars['close'])   # realtime by default
```

`trendlines` ports `Trendlines_with_Breaks_Style_Options.pine` (LuxAlgo). Its
`backpaint` defaults to **False**, the realtime series, where every value uses only
the bars up to it. Pine defaults to `True`, which draws each line from the pivot bar
itself; that needs `length` bars of future data, so the line columns shift `length`
bars into the past and are NaN for the last `length` rows. Breakout signals never
move, in either mode. Use `backpaint=True` to reproduce the TradingView chart, never
in a backtest.

**Where bars come from.** Three loaders return the same frame: `open/high/low/close`,
then `volume` when the source has one, float64, indexed by `date` oldest first.
`make_bars` simulates them offline. `load_bars(ticker, start, end=None, interval='1d')`
fetches Yahoo Finance bars as traded (not dividend-adjusted); intraday intervals from
`'1m'` to `'1h'` keep exchange time, and a request older than Yahoo keeps (30 days for
1m, 60 for 5m–30m, 730 for 1h) raises instead of coming back empty.
`read_bars(path, columns=None, daily=False, tz=None)` reads a CSV, including a
TradingView *Export chart data* file as it comes, with its plotted indicator columns
kept after the prices, since those are what a port is compared against. Every loader
rejects repeated or unsorted timestamps and a price missing after its first value,
because TA-Lib turns everything after such a gap into NaN.

```python
bars = ta_tools.load_bars('AAPL', '2023-01-01')
tv = ta_tools.read_bars('export.csv', daily=True, tz='America/New_York')
```

TradingView bars can also come from the TradingView connector, but only inside a Claude
session: it cannot be called from Python, a notebook or Colab. Ask Claude to fetch the
bars and write them as a CSV in the export layout (`time` in Unix seconds, lower-case
prices, `Volume`), then read that with `read_bars`. For AAPL, the connector's daily
closes matched Yahoo's exactly. Each bar is written out by Claude, so this suits small,
one-off pulls, not a routine feed.

Unlike the other notebooks here, the evaluation notebook is committed **with its
outputs**. The decision is the deliverable, and the previous comparison notebook was
useless precisely because it saved none.

## Pine scripts

Each script in `pine_scripts/` is a modification of a published TradingView
indicator. The unmodified originals are kept alongside in `pine_scripts/references/`,
and each script's header records its author, license, and what was changed.

| Script | Upstream author | License |
|---|---|---|
| `ZLSMA_Zero_Lag_LSMA_and_Slope.pine` | veryfid | MPL-2.0 |
| `Machine_Learning_Lorentzian_Classification_Label_Size.pine` | jdehorty | MPL-2.0 |
| `Trendlines_with_Breaks_Style_Options.pine` | LuxAlgo | CC BY-NC-SA 4.0 |
| `Linear_Regression_Candles_and_Slope.pine` | ugurvu, emiliolb | none stated (TradingView House Rules) |

Note that the LuxAlgo script's CC BY-NC-SA 4.0 is **non-commercial and share-alike**,
which constrains how that one file may be reused or redistributed.

## Tests

```bash
pytest
```

All tests run offline. `tests/test_smoke.py` runs the analysis pipeline end to end
against simulated data and checks that every public name resolves.
`tests/test_price_return_stats.py` checks the numerical methods against independent
references, never against the code under test: hand-counted and hand-built series,
closed forms, scipy's own implementations, formulas written out in the test, and
seeded simulations with known answers (Pareto tails, AR(1) and GARCH paths, bootstrap
coverage). `tests/test_ta_tools.py` does the same for `ta_tools`. `tests/test_api.py`
checks each API response against a direct package call, value for value, and
`tests/test_demo_data.py` checks the demo files against their manifest and the `demo` data
source against the files. New test groups are also checked by seeding deliberate bugs and
confirming a test fails on each. The dashboard has its own unit tests (`npm test` in
`dashboard/`, for the client, formatting and chart theming), a type check (`npm run
typecheck`), and an end-to-end check (`npm run e2e`) that drives every tab of the built
dashboard in Chromium against the real API; see "Dashboard".

## Changelog

Commit dates, newest first. This is a research repo, so there are no version tags.

### 2026-09-30
- Dashboard: **Other ticker…** loads any Yahoo symbol, not only the configured ones, and the
  Multi-ticker tab has a menu to choose tickers from the default list, the saved CSVs and the demo
  files. New `GET /api/ticker` and `POST /api/multi-ticker`. Saving live data now stops at
  yesterday: it had also stored today's unfinished bar.
- The dashboard POC is complete (plan Phase 6): `npm run e2e` in `dashboard/` checks every tab
  end to end, in Chromium, against `python -m src.api` on the demo data.
- One command runs the dashboard (plan Phase 5): `python -m src.api` serves the built React app
  and the API together on http://127.0.0.1:8000, and `python scripts/dev.py` runs both
  development servers. See "Dashboard".
- Saved live data (plan Phase 4a): `data_source='local'` reads Yahoo data saved as CSV in
  `data/local/` (git-ignored). `save_local` downloads once and then adds only new bars,
  reporting any bar Yahoo has revised, and never leaves a half-written file. Also a refresh
  script, `scripts/update_local_data.py`, API endpoints, and in the dashboard a third data
  choice, **Saved CSV (offline)**, with Save and Update buttons and a flag on stale data.
- Demo data: `data/demo/` holds processed daily bars for ten tickers, Yahoo Finance data with
  0.01% seeded noise, labelled as not market data and for education only (see its README).
  `load_price_data` reads them with `data_source='demo'`, `configs/demo_tickers.yaml` lists
  them, and `scripts/make_demo_data.py` regenerates them. Demo loading reuses the Yahoo cleaning step,
  including dropping the returns around a non-positive close.
- Started the React dashboard POC (`docs/dashboard_plan.md`). `src/api/` is a FastAPI app
  over `price_return`, with health, ticker-list and overview endpoints; its request model is
  generated from `Params`, and it computes nothing itself. The ticker list defaults to the
  demo set, so the dashboard runs offline. New `api` extra; `httpx2` joins `dev`.
- `configs/tickers.yaml`'s header named a notebook that no longer exists; fixed.
- The dashboard API is complete (plan Phase 2): streak and cumulative summaries, the rare-event
  table with live bounds, all 14 charts as Plotly JSON, the four statistics sections and the
  multi-ticker comparison, each equal to the package called directly.
- The dashboard's four remaining tabs (plan Phase 4): Streaks & cumulative, Rare events with
  live filters, Statistics (the statistics notebook's four sections, the bootstrap on request)
  and Multi-ticker. Three chart fixes in `viz.py`, which the notebooks get too: the histogram's
  threshold labels no longer overlap, the event-probability chart labels 1, 2 and 5 within each
  decade when its range is narrow, and its two model labels are kept apart.
- The dashboard itself (plan Phase 3): `dashboard/`, a Vite + React + TypeScript app with the
  parameter panel, tabs and the Overview tab (stat tiles, the four overview charts from
  `viz.py`, the distribution table), in light and dark themes. It runs on the demo data by
  default. See "Dashboard" for how to start it.
- New `notebooks/api_examples.ipynb`: example queries to every API endpoint, with the answers
  shown as tables and charts, ending with one answer checked against the package directly.
- New `event_probability_table` in `price_return.stats`, which the statistics notebook now
  calls instead of joining three tables in a cell.
- `plot_streak_timeline` is 50 to 140 times faster, with an identical figure: it added one
  shape at a time, which took 54 s for SPX's 2-day streaks and over 5 minutes for CL's.

### 2026-09-29
- New `price_return/stats.py` and `notebooks/price_return_statistics.ipynb`: distribution
  and tails (Jarque–Bera, Student-t fit, Hill tail index, VaR and expected shortfall three
  ways), dependence (autocorrelation, Ljung–Box, variance ratio, ARCH-LM), drawdowns and
  risk-adjusted returns, and bootstrap intervals and model comparisons for the rare-event
  probabilities. Five matching charts in `viz.py`. `scipy` is a new optional extra,
  `stats`. The notebook ends with option sizing, running the four P&L grids from the
  retired `test_es` unchanged.
- `price_return_analysis_v0.5.ipynb` is renamed `price_return_analysis.ipynb`, the one
  version kept now that v0.1 is gone. It and `rare_case_run` link to the package, and the
  wording matches compounded moves, complete-window frequencies and episodes.
- Fixed three inconsistencies in `price_return`, so results in
  `price_return_analysis` and `rare_case_run` shift (explained under Methodology). Multi-day returns are now
  compounded everywhere; they were added up in the rolling columns, the annualized
  return and the rare-event "cumulative" rows. Every frequency now divides by the
  complete windows, not the days (streak frequencies and rare-event `prob` rise
  slightly, most for long holding periods). New `episodes` counts, and `n_windows` in
  the rare-event table. On simulated series like the configured tickers, the annualized
  return rises by 1.3 to 13 points, depending on volatility, and 3-day tables barely move.
- Removed the notebook-era code, now that everything in it has a successor
  (`docs/price_return_plan.md` maps each piece): `src/tools/basic.py`,
  `notebooks/test_es.ipynb`, `test_ko.ipynb`, `test_ta_packages.ipynb` and the frozen
  `price_return_analysis_v0.1.ipynb`. The four `basic.py` functions that had no
  successor were first rebuilt in `price_return/options.py` as `price_range`,
  `move_probabilities` and `expected_pnl`, which return tables instead of printing and
  match the originals exactly on `test_es`'s P&L grids. The rest is covered by the
  `price_return` package, `price_return_analysis`, `rare_case_run` and
  `ta_package_evaluation`. `matplotlib` is no longer a dependency; only the removed
  files used it.

### 2026-09-28
- The project now requires Python 3.12 or newer: pandas_ta, in the `ta` extra, publishes
  nothing for 3.11, so `pip install -e ".[ta]"` failed there.
- `ta_tools` tests now check that every indicator moves with its input: shifting and
  scaling prices shifts and scales price-level outputs, only scales spreads such as ATR,
  and leaves RSI, breaks and latches unchanged.
- Data sources: `load_bars` takes an `interval` for intraday Yahoo bars and refuses
  requests older than Yahoo keeps; new `read_bars` reads CSV files, including
  TradingView chart exports with their indicator columns. All loaders share one output
  contract and reject unsorted or repeated timestamps and gaps in prices. `load_bars`
  still drops Yahoo's occasional incomplete row, as before.
- `notebooks/ta_tools_read_data.ipynb` runs every data reader and checks its output. It
  found that `read_bars` could read a number one unit off in its last digit (pandas'
  default CSV parser); it now reads back exactly the floats written.
- `read_bars` kept a time with no offset, such as the plain dates in TradingView's daily
  export, as UTC, so with `tz='America/New_York'` every daily bar landed a day early. Such
  times are now read as already local to `tz`. Found with two real TradingView exports,
  which also gave the first TradingView parity check: `linreg_candles`' signal line, `rsi`,
  `bb` and `ema` match the chart to within 1e-10.

### 2026-09-27
- `src/tools/ta_tools/` package skeleton: `backend.py` as the single import site
  for TA-Lib (primary) and pandas_ta (secondary, loaded lazily), a `CAPABILITIES`
  map, and a seeded OHLC bar simulator whose bars are always valid.
- First primitives: `sma`, `ema`, `wma`, `stdev`, `atr`, `bb`, `rsi` from TA-Lib, `hma` and
  `alma` from pandas_ta. Each is checked against an independent implementation of Pine's
  definition. `ema` and `stdev` match exactly; `atr` starts one bar later than Pine's
  and differs by ~4% at first, converging within a few hundred bars; `alma` equals
  Pine's `floor=true` variant, not its default. TA-Lib rejects length 1 for SMA, EMA
  and STDDEV (and WMA, BBANDS, RSI), so those return Pine's answer directly. `wma`, `bb` and
  `rsi` match Pine exactly, except that on a perfectly flat window TA-Lib's RSI is 0 where
  TradingView's built-in RSI script gives 100. Multi-output indicators return a DataFrame
  with named columns (`bb` gives `bb_mid_20`, `bb_upper_20`, `bb_lower_20`).
- Pine primitives that no library supplies with Pine's semantics, in `pine.py`: `linreg`
  with Pine's `offset` (built from TA-Lib's `LINEARREG` and `LINEARREG_SLOPE`), `rma`,
  `pivot_high`/`pivot_low` published `right` bars after the pivot, `change`, `crossover`,
  `crossunder`, `barssince`, `nz`, and `recurse` for Pine `var` state that depends on its
  own previous bar. Pine does not document how pivots break ties; here a flat top of
  equal highs yields one pivot, at its last bar. A no-look-ahead test now covers every
  indicator: removing the bars after any point never changes the values before it.
- `atr` rebuilt as `rma(true_range)`, matching Pine's `ta.atr` exactly, with a new
  `true_range` (Pine's `ta.tr`). TA-Lib's ATR is no longer used: it has no true range
  for the first bar, so it started a bar later and was 1-4% off Pine early on.
- First ported indicator: `linreg_candles` in `indicators.py`, from
  `Linear_Regression_Candles_and_Slope.pine`. It matches a line-by-line rewrite of the
  script (built on `numpy.polyfit`) for both the SMA and EMA signal. Because the four
  prices are fitted separately, about 2% of LinReg candles have a high below the body
  or a low above it; Pine draws these as they are, and so does the port.
- `load_bars(ticker, start, end=None)` fetches daily OHLCV from Yahoo Finance in the same
  shape as `make_bars`, unadjusted like `price_return`. `notebooks/ta_tools_exploration.ipynb`
  uses it to chart every `ta_tools` indicator on AAPL since January 2023.
- Second ported indicator: `trendlines`, from `Trendlines_with_Breaks_Style_Options.pine`,
  with the ATR, Stdev and Linreg slope methods. The lines and breakout latches follow
  Pine's bar-by-bar `var` logic through `recurse`, checked against a closed-form rewrite
  of the script. `backpaint` defaults to False (realtime); `True` reproduces Pine's chart
  and is tested to look ahead. Pine's `Linreg` slope is half the absolute least-squares
  slope and, unlike the other two, is not divided by `length`; the port keeps this.

### 2026-09-24
- Evaluated `pandas_ta`, `ta` and `TA-Lib` on coverage and API shape; chose TA-Lib to
  wrap, on the strength of its linear-regression primitives. See Technical analysis
  above and `notebooks/ta_package_evaluation.ipynb`.

### 2026-09-19
- `price_return` became a package: `params`, `data`, `analysis`, `viz`, `report`,
  `pipeline`. The public API is unchanged — `__init__.py` re-exports it, so no
  notebook or test needed editing. Both notebooks were run before and after and
  produce byte-identical output.
- Colab bootstrap fixed: `pip install -e` registers its import finder through a
  `.pth` file that Python only reads at interpreter startup, so installing in one
  cell and importing in the next never worked. The repo now goes on `sys.path`.
- Price data is explicitly unadjusted (`auto_adjust=False`) — see Methodology.

### 2026-09-17
- Price data is now explicitly unadjusted (`auto_adjust=False`), so `Close` is the
  traded price. Option strikes reference the traded price, and yfinance had changed
  this default — see Methodology for the size of the difference.
- The rare-case analysis is driven by `configs/tickers.yaml`: one run covers every
  ticker listed and emits two cross-ticker summary tables. Deleting the config falls
  back to ES=F, NQ=F, YM=F and RTY=F, saying so as it does.
- `load_price_data` guards against two failures found while running across twelve
  instruments — an unknown symbol, which previously surfaced as a `ZeroDivisionError`
  inside a plotting function several cells later, and non-positive prices, after
  CL=F settled at -$37.63 on 2020-04-20 and produced a -306% "return" that flipped
  crude's mean drift negative.
- Notebook CSV exports are gitignored.

### 2026-09-16
- Removed the `samplemod` template this repo was forked from: `sample/`, `docs/`,
  `README.rst`, `MANIFEST.in`, `Makefile`, `setup.py`, and a `LICENSE` carrying a
  third party's copyright.
- `pyproject.toml` replaces `setup.py`, so `pip install -e .` makes `src.tools`
  importable from any working directory.
- Dropped the `sys.path` bootstrap from every notebook — it made the working
  directory load-bearing, so notebooks only imported correctly when run from
  `notebooks/`. Colab now does an editable install instead of `os.chdir`.
- Pine scripts gained descriptions and change notes;
  `Linear_Regression_Candles_and_Slope.pine` had no attribution at all despite
  deriving from two community scripts.
- First real `README.md`, and a smoke test replacing three placeholder tests that
  asserted `True`.

### 2026-09-15
- Extracted the analysis out of the notebooks into `src/tools/price_return.py`, with
  every tunable value carried on a `Params` dataclass; added the v0.5 notebook built
  on it.

### 2026-09-11
- First `price_return_analysis` notebook (v0.1), a self-contained monolith — since
  frozen as a reference implementation.

### 2026-06 to 2026-09
- Pine script work: linear-regression candles with slope colouring, ZLSMA, the
  LuxAlgo trendline indicator and jdehorty's Lorentzian classifier, each kept
  alongside its unmodified original under `pine_scripts/references/`.

### 2026-04-30
- Initial commit, from the `samplemod` project template.

## License

No license is granted. This is a personal work-in-progress repository and all
rights are reserved.

This does **not** extend to the Pine scripts, which carry their own upstream
licenses as listed above; those terms govern those files and are not overridden
by this notice.
