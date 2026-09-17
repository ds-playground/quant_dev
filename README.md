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
places orders, and the numerical methods have not been checked against an independent
reference implementation — the test suite is a smoke test, not a correctness proof.

Stage one has three goals:

1. A framework for historical price-return analysis.
2. Basic helper functions for analysis and visualization.
3. Pine scripts for TradingView.

Goals 1 and 2 live in `src/tools/price_return.py`; goal 3 lives in `pine_scripts/`.

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
toolchain and would otherwise block a clean install:

```bash
pip install -e ".[dev]"   # pytest, ipywidgets, nbformat, matplotlib
pip install -e ".[ta]"    # pandas_ta, ta, TA-Lib
```

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

**Rolling statistics.** For each holding period `d`, the `d`-day cumulative return
is computed, then a `trade_days`-long rolling mean and standard deviation over it.
Annualized figures are derived from the same pair. Default holding periods are
1, 2, 3, 4, 5, 10 and 250 days.

**Win/loss streaks.** A day is a *win* when its return exceeds `win_threshold`
(default +0.5%) and a *loss* when it falls below `loss_threshold` (default −0.5%).
A streak is a rolling window in which *every* day is a win, or every day a loss.
Streaks are reported as counts, as a share of all trading days, and as an average
return per streak — then shaded onto a return timeline.

**Cumulative thresholds.** Separately from all-win streaks, this counts rolling
windows whose *compounded* return clears a threshold (default 0.5%, 1%, 2%),
regardless of what the individual days did. A window can clear +2% cumulatively
while containing losing days, so this and the streak view answer different
questions.

**Rare-event probabilities.** For each combination of move size (0.01% up to 5%),
holding period (1 to 30 days) and lookback window (2 or 5 years), this measures how
often a move of at least that size occurred over that horizon. Filtering to the low
end — events that did happen but rarely — produces the rare-event table, which is
browsable interactively in the v0.5 notebook via `interactive_low_probability`.

## Repo layout

```
quant_dev/
├── README.md                              this file
├── pyproject.toml                         packaging; `pip install -e .`
├── requirements.txt                       flat dependency list for Colab
├── config.py                              defaults for the dev/ scratch notebooks only
│
├── configs/
│   └── tickers.yaml                       ticker set + per-ticker parameters
│
├── src/
│   └── tools/
│       ├── price_return.py                the framework: data, analysis, charts, config
│       └── basic.py                       superseded notebook-era draft (see below)
│
├── notebooks/                             tracked, promoted notebooks
│   ├── price_return_analysis_v0.1.ipynb   frozen reference monolith
│   ├── price_return_analysis_v0.5.ipynb   current single-ticker analysis
│   ├── rare_case_run.ipynb                config-driven multi-ticker run
│   └── test_es / test_ko / test_ta_packages.ipynb
│
├── pine_scripts/                          TradingView indicators
│   ├── *.pine                             the modified indicators
│   └── references/                        their unmodified originals
│
├── tests/
│   └── test_smoke.py                      offline end-to-end pipeline check
│
└── dev/                                   scratch work — gitignored, never tracked
```

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
| `price_return_analysis_v0.1.ipynb` | Frozen. Self-contained monolith, no package imports; kept as a reference implementation. |
| `price_return_analysis_v0.5.ipynb` | Current. The same analysis built on `src/tools/price_return.py`. |
| `rare_case_run.ipynb` | Current. Config-driven; runs every ticker in `configs/tickers.yaml` and emits two cross-ticker summary tables. |
| `test_es.ipynb`, `test_ko.ipynb`, `test_ta_packages.ipynb` | Exploratory, built on the older `basic.py`. |

**Two config mechanisms, deliberately.** `config.py` serves the `dev/` scratch
notebooks. Everything under `src/` uses the `Params` dataclass instead, which is
authoritative for the package. They overlap; that is intentional, so scratch work
can be retuned without touching the package.

**`src/tools/basic.py` is superseded.** It is the earlier notebook-era draft, kept
because the three `test_*` notebooks still import it. It has no docstrings and its
own older `consecutive_analysis` with a different signature from the one in
`price_return.py`. Four of its functions — `profit_estimate`, `sd_and_cond`,
`accepted_min_max`, `projected_min_max` — have no successor yet, so it cannot
simply be deleted.

## API

`src/tools/price_return.py`, grouped as it is in `__all__`:

| Function | Purpose |
|---|---|
| `Params` | Every tunable value for a run |
| `load_price_data` | A date / price / return_pct frame, from Yahoo Finance or simulation |
| `add_rolling_stats` | Adds `PCT Change {d}` / `{d} Av` / `{d} STD` columns plus the annualized pair |
| `latest_snapshot`, `show_latest_snapshot` | Latest annualized and daily mean/vol |
| `daily_returns_series` | Date-indexed decimal daily returns |
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

`tests/test_smoke.py` runs the analysis pipeline end to end against simulated data
(no network) and checks that every name the v0.5 notebook imports still resolves.
It is a smoke test, not a correctness suite — the numerical methods are not yet
verified against independent reference values.

## Changelog

Commit dates, newest first. This is a research repo, so there are no version tags.

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
