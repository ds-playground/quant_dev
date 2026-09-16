# quant_dev

A work-in-progress playground for historical price and return analysis, plus the
TradingView indicators that go with it.

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

P  = Params(ticker='KO', start_date='2016-01-01', data_source='yahoo')
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
├── config.py           defaults for the dev/ scratch notebooks only
├── configs/            ticker sets read by the analysis notebooks
├── pyproject.toml      packaging; `pip install -e .`
├── requirements.txt    flat dependency list for Colab
├── dev/                scratch work — gitignored, never tracked
├── notebooks/          promoted, tracked notebooks
├── pine_scripts/       TradingView indicators (+ references/ for the originals)
├── src/tools/          the package
└── tests/              smoke test
```

**`configs/tickers.yaml`** drives `dev/rare_case_run.ipynb`, which analyses every
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

## License

No license is granted. This is a personal work-in-progress repository and all
rights are reserved.

This does **not** extend to the Pine scripts, which carry their own upstream
licenses as listed above; those terms govern those files and are not overridden
by this notice.
