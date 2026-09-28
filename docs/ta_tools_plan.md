# Build a technical-analysis package (`dev/ta_tools`)

This is the working plan for `src/tools/ta_tools/`, kept in the repo so work can continue from
any machine or session. Update it as phases finish: mark the phase ✅ with its commit hash in
the table, and add a "Phase N notes" section with anything a later phase needs to know.

## Picking this up

- **Setup:** Python 3.12 or newer, since pandas_ta publishes nothing for 3.11 (the owner's
  `quant_env` is 3.13). `pip install -e ".[ta,dev]"` from the repo root, then `pytest`; all tests
  should pass offline. Notebooks that fetch prices need network access to Yahoo Finance.
- **Versions are not pinned.** The Phase 1 and Phase 3 figures were measured on TA-Lib 0.6.8
  and pandas_ta 0.4.71b0; a fresh install may pull newer versions (TA-Lib 0.8.1 as of
  2026-09-28). The tests are the check that still holds; the quoted counts may drift.
- **Phases are the unit of work.** Do one phase when asked, then stop; do not start the next
  one unprompted. Commit and push only when asked.
- **Stay within the plan.** Don't offer or add deliverables the plan doesn't list, such as
  extending a notebook to cover a later phase. Propose them as a change to this plan instead.
- **Keep changes to existing working code small.** New capability gets built properly; code
  that already works is changed only as far as the task needs.
- **Notebooks with saved outputs** are executed with
  `PLOTLY_RENDERER="plotly_mimetype+notebook_connected"`. The `json` renderer saves figures that
  don't display.
- **Pine parity** is checked against independent implementations written out from Pine's
  definitions, never against the code under test.

## Context

The repo's technical-analysis story is currently three overlapping libraries declared in an
optional extra (`pandas_ta`, `ta`, `TA-Lib`), one throwaway notebook that compares them on a
single indicator, and four TradingView Pine scripts whose logic exists only as Pine. The goal is
a Python package of the owner's own: evaluate the three libraries, wrap the winner behind a
consistent API, and port the Pine indicators to Python.

Branch `dev/ta_tools` is cut from `master` at `b48164d`. **This branch is purely additive** —
`basic.py` and the old notebooks are not touched.

Decisions already made (do not revisit):

- **Wrap one library** rather than implement everything from scratch.
- Package lives at **`src/tools/ta_tools/`**, sibling to `src/tools/price_return/`.
- Evaluation is delivered as a **tracked notebook plus a README section**.
- **Primitives first**, before whole indicators.
- Evaluation is judged on **coverage and API shape**.
- **Lorentzian Classification is out of scope**, documented as a future project.
- **`basic.py` and the old notebooks are deferred** — no deletions here.
- **TA-Lib is the primary backend, pandas_ta the secondary** (decided after Phase 1). Rule:
  use TA-Lib for everything it has; use pandas_ta only where TA-Lib has **no** equivalent;
  implement it ourselves wherever Pine semantics must be exact. On any overlap TA-Lib wins, so
  the pandas_ta beta is never on the critical path.

## Phases at a glance

| # | Phase | Tasks | Deliverable | Depends on |
|---|---|---|---|---|
| **1** ✅ | Evaluate the three libraries | 1.1 enumerate coverage programmatically · 1.2 probe API shape on one series · 1.3 score against what the Pine ports need · 1.4 write the decision up | `notebooks/ta_package_evaluation.ipynb` + README section — **done, `17b2e89`** | — |
| **2** ✅ | Package skeleton | 2.1 create `ta_tools/` with `__init__.py`, `__all__` · 2.2 `backend.py` — the single import site for **TA-Lib (primary) and pandas_ta (secondary)**, plus a `CAPABILITIES` map recording each primitive's source · 2.3 seeded **OHLC** bar simulator · 2.4 declare the dependency in `pyproject.toml` | importable package + offline fixture — **done, `b46dd0a`** | 1 |
| **3** ✅ | Primitives — wrapped | 3.1 `sma`, `ema`, `stdev`, `atr` via TA-Lib · 3.2 breadth via pandas_ta where TA-Lib has nothing (`hma`, `alma`) · 3.3 uniform Series-in/Series-out + NaN warm-up contract · **3.4 `wma`, `bb`, `rsi` via TA-Lib** (added) | `overlap.py`, `volatility.py`, `momentum.py` — **done, `3a20a79`** (+ notebook `6162601`) | 2 |
| **4** ✅ | Primitives — Pine gaps | 4.1 `linreg(series, length, offset)` · 4.2 `rma` (custom, for Pine seeding) · 4.3 `pivot_high/low(left, right)` with publication delay · 4.4 `change/crossover/crossunder/barssince/nz` · 4.5 stateful-recursion harness · 4.6 `true_range` + Pine-exact `atr` on `rma` (added) | `ta_tools/pine.py` — **done, `d03c717`** (+ notebook `af67f54`) | 3 |
| **5** ✅ | Port: vectorisable indicator | 5.1 LinReg Candles + Slope | `ta_tools/indicators.py` — **done, `7b24c0f`** | 4 |
| **6** ✅ | Port: stateful indicator | 6.1 slope methods (atr/stdev/linreg) · 6.2 recursive rails · 6.3 breakout latches · 6.4 backpaint vs realtime modes | Trendlines with Breaks — **done, `18572ce`** | 4, 5 |
| **7** ✅ | Close the test gaps | Most of the original 7.1–7.4 shipped with Phases 3–6 (see Phase 7 below). 7.1 equivariance beyond `linreg` · 7.2 no-look-ahead for `trendlines` `stdev` method | `tests/test_ta_tools.py` — **done, `d770d17`** | 3–6 |
| **8** ✅ | Data sources | 8.1 shared `_normalise(frame)` · 8.2 `load_bars(..., interval=)` for intraday Yahoo bars · 8.3 `read_bars(path, ...)` for files, incl. TradingView exports · 8.4 TradingView connector → CSV snapshot workflow · 8.5 further API adapters only when a real one is needed | `ta_tools/data.py`, tests — **done, `89e61df`** (8.5 stays deferred) | 2 |
| **—** | *Deferred* | ZLSMA + Slope (from Phase 5) · Lorentzian Classification · `basic.py` removal · old-notebook removal | documented only | — |

## Phase 1 — Evaluate the three libraries

The existing `notebooks/test_ta_packages.ipynb` computes SMA-20 on ^GSPC in each library and
draws three near-identical charts, with no saved outputs and no conclusion. Each cell rebinds the
name `ta`, so the three cannot coexist in one kernel run. **Replace it with a new notebook**
rather than extending it; leave the old one in place (deferred, per above).

Import all three under distinct aliases in one cell (`import talib`, `import ta as ta_lib`,
`import pandas_ta`) so every later cell can compare them side by side.

I already measured the two criteria; the notebook should reproduce this rather than assert it:

**1.1 Coverage** — enumerable directly: `talib.get_function_groups()`, `pandas_ta.Category`, and
module introspection for `ta`.

| | functions | groups | note |
|---|---|---|---|
| TA-Lib 0.6.8 | 158 | 10 | **61 are candlestick patterns**; 17 overlap, 30 momentum |
| pandas_ta 0.4.71b0 | 151 | 9 | 43 momentum, 36 overlap |
| ta 0.11.0 | 80 | 5 | 28 trend, 21 volatility |

Headline counts flatter TA-Lib — strip the 61 pattern functions and it is comparable to the
others. The notebook should say so rather than rank on the raw number.

**1.2 API shape** — same series, same indicator, tabulate return type, index preservation and
warm-up NaNs. All three return a `pandas.Series` with the index preserved for SMA. The
interesting divergence is ATR(14): TA-Lib and pandas_ta emit 14 warm-up NaNs, **`ta` emits none**,
so it is seeding differently — a real behavioural difference to note under "API shape". Also
demonstrate `pandas_ta`'s `df.ta.sma(length=20)` accessor, which is its distinguishing feature
and works on pandas 3.0.2.

**1.3 Fitness for the Pine ports** — the decisive section, because coverage counts do not capture
it:

| need | TA-Lib | pandas_ta | ta |
|---|---|---|---|
| linear regression | `LINEARREG`, `LINEARREG_SLOPE`, `LINEARREG_INTERCEPT` | `linreg` | **absent** |
| Pine `offset` semantics | reconstructible | ✗ (see trap below) | ✗ |
| RMA (Wilder) exposed | internal only | `rma` | ✗ |
| pivot high/low + delay | ✗ | ✗ (`pivots` is S/R pivot points, unrelated) | ✗ |

`ta` has **no linear regression at all** (`AttributeError: module 'ta.trend' has no attribute
'linreg'`), which on this project's own terms rules it out — LinReg Candles calls it four times
and ZLSMA nests it.

**Trap to document:** `pandas_ta.linreg(close, length, offset=...)` takes an `offset`, but it is
pandas_ta's generic *post-shift of the output series*, not Pine's "evaluate the fitted line
`offset` bars back". They are different operations. Pine's version reconstructs cleanly from
TA-Lib as `LINEARREG - LINEARREG_SLOPE * offset` (verified).

**1.4** End on one summary table and a written recommendation, copied into the README.

## Phase 2 — Package skeleton

Follow `price_return/` conventions: package directory, `__init__.py` re-exporting everything with
an `__all__` (the smoke test asserts every `__all__` name resolves).

**Do not** copy the `Params` dataclass pattern. `Params` suits a pipeline where one config threads
through twenty functions; indicators take two or three arguments each and are composed ad hoc.
Plain keyword arguments with defaults matching the Pine originals are the better fit — flagging
this as a deliberate divergence from the sibling package.

All calls into either library go through **one `backend.py`**, so swapping libraries later
touches one file. Nothing else imports `talib`/`pandas_ta` directly. `backend.py` also exposes a
`CAPABILITIES` map recording where each primitive comes from, in one of four values:

| value | meaning | failure exposure |
|---|---|---|
| `talib` | TA-Lib supplies it directly | a primary-backend swap |
| `pandas_ta` | only pandas_ta has it | pandas_ta beta breakage |
| `derived` | composed from backend calls (e.g. Pine `linreg` offset) | a swap that drops a building block |
| `custom` | implemented here | nothing external — needs the heaviest tests |

Two backend-specific rules. TA-Lib is imported eagerly, since the package is built on it.
pandas_ta is imported **lazily**, on first use of a pandas_ta-backed indicator, so importing
`ta_tools` never pays for — or risks — the beta unless you actually use it.

Two things deliberately **not** taken from pandas_ta despite looking like wins, both measured in
the Phase 1 follow-up:

- **`rma`** — a real Pine primitive, and TA-Lib only has it inside ATR/RSI. But it is
  `ewm(alpha=1/length, adjust=False)` with Pine's SMA-seeded warm-up — a few lines, on the one
  primitive where controlling warm-up matters most for Pine parity. Implement it; mark `custom`.
- **`pivots`** — computes support/resistance *levels*, not Pine `pivothigh`/`pivotlow` swing
  detection. Stays `custom`.

Measured backing for the pandas_ta role: 106 genuinely additional indicators once naming aliases
are removed (`stdev`/`STDDEV`, `uo`/`ULTOSC` etc.), including `supertrend`, `donchian`, `kc`,
`vwap`, `zigzag`, `ha`, `hma`, `alma`, `squeeze` — none with a TA-Lib equivalent. All 21 tested
run cleanly on pandas 3.0.2.

**`ta_tools` needs its own OHLC fixture.** `price_return/data.py` simulates a close-only series
(`date`, `price`, `return_pct`), but every indicator here needs open/high/low/close. Add a seeded
bar simulator — GBM close with O/H/L derived from it — so the tests and the evaluation notebook
run offline. Do not try to reuse the `price_return` loader.

## Phases 3–4 — Primitives

Phase 3 wraps what the library already does well. Phase 4 supplies what **no** library provides:

- **`linreg(series, length, offset=0)`** — must accept an arbitrary Series, since ZLSMA feeds a
  linreg its own output.
- **`pivot_high/pivot_low(series, left, right)`** — extreme of `left` bars before and `right`
  after, **published `right` bars later**. That delay is what "backpainting" means and is the
  single most misunderstood part of the Trendlines script.
- **Pine helpers** — `change`, `crossover`, `crossunder`, `barssince`, `nz`.
- **A recursion harness** for Pine `var` series whose value depends on their own previous bar.
  Not vectorisable; an explicit loop is fine at this data size — reach for numba only if profiling
  says so.

### Phase 3 findings that bear on Phase 4 and 6

Measured against independent implementations of Pine's definitions:

| primitive | vs Pine |
|---|---|
| `ema` | exact (≤ 6e-14) |
| `stdev` | exact — population, as Pine's `biased=true` default; pandas' `ddof=1` would be wrong |
| `atr` | TA-Lib's **starts one bar later** (Pine's bar-0 true range is `high - low`; TA-Lib has none) — 4.2% off at the first value, identical by ~bar 250. **Resolved in Phase 4:** `atr` is now `rma(true_range)`, exact. |
| `alma` (pandas_ta) | equals Pine's `floor=true` form, not its `floor=false` default |
| `wma`, `bb` | exact (≤ 2e-11); `bb` uses population std like `stdev` |
| `rsi` | exact, **except a perfectly flat window**: TA-Lib gives 0, TradingView's built-in RSI script 100 — TA-Lib's kept, documented |
| length 1 | TA-Lib rejects it for SMA/EMA/WMA/STDDEV/BBANDS/RSI; Pine accepts it — wrappers return Pine's answer |

**Decided in Phase 4:** `atr` became Pine-exact `rma(true_range)`, before Trendlines' `Atr`
slope method needed it (see Phase 4 notes).

### Phase 4 notes

- `linreg` is `derived`: `LINEARREG - LINEARREG_SLOPE * offset`, matching `numpy.polyfit` to 1e-10
  and exact on a straight line at every offset. It takes its own output (warm-up 10, then 20).
- `pivot_high`/`pivot_low`: Pine does not document tie-breaking. Chosen rule: `>=` every bar on
  the left, `>` every bar on the right, so a flat top yields **one** pivot at its last bar.
  Verified against a brute-force oracle on data with forced ties. **Unverified against Pine** —
  Tier 3 (TradingView export) is what would settle it; matters for Trendlines on tick-sized data.
- `recurse` is the `var` harness; it copies state per bar, so a step function that mutates its
  argument cannot corrupt earlier rows. It reproduces `rma` exactly.
- The no-look-ahead test covers every Series primitive from Phases 3 and 4 (14, counting
  `true_range`); Phases 5 and 6 added the indicators.
- **`atr` rebuilt (owner's decision):** `atr = rma(true_range)`, with `true_range` = Pine's `ta.tr(true)`.
  Exact to 1e-12, warm-up `length-1` like Pine; now `custom`. `rma` stays ours — pandas_ta's has
  no SMA seed (values from bar 0, 0.7% off at bar 13), and TA-Lib does not expose one.

## Phases 5–6 — The Pine ports

### Phase 5 notes

- `linreg_candles(open, high, low, close, linreg_length=11, signal_length=11, sma_signal=True,
  lin_reg=True)` — Pine input names and defaults kept; chart-only inputs (colours, widths,
  `show_candles`, `show_slope_color`) dropped. Registered `derived` in `CAPABILITIES`.
- Returns one DataFrame: `lrc_open/high/low/close`, `lrc_signal`, `lrc_slope`, `lrc_bull`. Column
  names carry no parameters (unlike `bb_mid_20`) — seven long names otherwise; revisit if two
  instances ever need joining onto the same frame.
- Warm-up: candles `L-1`, signal `L+S-2`, slope `L+S-1`. TA-Lib's SMA/EMA skip leading NaNs, so
  the EMA seeds from the first valid LinReg close, as Pine's does. `lrc_bull` is False in the
  warm-up (Pine's `na < na`).
- Verified against a line-by-line rewrite on `numpy.polyfit` (SMA and EMA signal), a straight-line
  analytic case, `lin_reg=False`, length 1, and no-look-ahead.
- The four fits are independent, so ~2% of LinReg candles have a wick inside the body. Pine plots
  them regardless; documented, not "fixed".

**LinReg Candles + Slope** — `linreg` on each of O/H/L/C, then
`signal = sma_or_ema(linreg_close, signal_length)`, `slope = signal.diff()`. Fully vectorised.

**ZLSMA + Slope** *(deferred — see Deferred, with reasons)* — `lsma = linreg(src, n, off)`,
`lsma2 = linreg(lsma, n, off)`, `zlsma = 2*lsma - lsma2`. Watch NaN propagation across the first
`2*length` bars.

**Trendlines with Breaks** — the hard one, and the reason it gets its own phase. Pivot detection,
then a per-bar slope from one of three methods, then genuinely recursive rails
(`upper := ph ? ph : upper - slope_ph`) and breakout latches that depend on their own previous
values. Expose backpaint and realtime as an explicit mode rather than hard-coding Pine's default.
The drawn `line.new` objects reduce to `(anchor_bar, anchor_price, slope)` tuples — return data,
not chart objects.

### Phase 6 notes

- `trendlines(high, low, close, length=14, mult=1.0, calc_method='atr', backpaint=False)` in
  `indicators.py`, registered `derived`. Pine's `calcMethod` options in lower case; style inputs
  dropped.
- Returns one DataFrame: `tl_upper`, `tl_lower`, `tl_upper_slope`, `tl_lower_slope`,
  `tl_pivot_high`, `tl_pivot_low`, `tl_upos`, `tl_dnos` (0/1), `tl_upper_break`,
  `tl_lower_break`. Pine's extended `line.new` objects are the rows with a pivot: anchor bar, price
  `tl_upper`/`tl_lower`, slope — no separate structure needed.
- Rails and latches follow Pine's `var` logic bar by bar through `recurse`. Pivots use `high`/`low`
  (Pine's `ta.pivothigh/low` defaults); a pivot of exactly 0 counts as none, as Pine's float truth
  test does.
- Realtime plots Pine's `upper - slope_ph*length` (the line at the confirming bar); backpaint plots
  `upper` itself shifted `-length`. Breaks and latches are identical in both modes.
- Before the first pivot, Pine's `var upper = 0.` plots at zero; here the rails are NaN. The latch
  still follows Pine: `close > 0` sets `upos` on bar 0, without a break event.
- Pine's `Linreg` slope is `|cov(src, n)| / var(n) / 2`: half the absolute least-squares slope, not
  divided by `length` like the others. Kept as written; built as `|linreg(0) - linreg(1)| / 2`.
- Verified against a closed-form rewrite (anchor-at-last-pivot, latch = cummax within a pivot
  segment) for all three methods; slope formulas written literally with `bar_index`; backpaint vs
  realtime shift identity; no-look-ahead in realtime; backpaint shown to look ahead. Three seeded
  mutations of the implementation were each caught.

## Phase 7 — Close the test gaps

The original Phase 7 (property tests, pivot oracle, polyfit reference, README section, changelog)
was built alongside Phases 3–6 and is **already in place**: no-look-ahead over the 14 Series
primitives, both `linreg_candles` signals and the `atr`/`linreg` `trendlines` methods; the
brute-force pivot oracle with ties; `linreg` vs `numpy.polyfit`; warm-up counts; the
backpaint-equals-shifted-realtime identity; `__all__` and `CAPABILITIES` coverage; the README
"Technical analysis" section; changelog entries through 2026-09-27. What is left:

- **7.1 Equivariance beyond `linreg`** — adding `c` / scaling by `k` for the price-level
  primitives (`sma`, `ema`, `wma`, `hma`, `alma`, `rma`, `bb`) and for `linreg_candles`;
  `trendlines` rails likewise. The spread measures differ: `atr`, `stdev` and `true_range` are
  unchanged by `+c` and scale by `k`; `rsi` is unchanged by both.
- **7.2 No-look-ahead for `trendlines(calc_method='stdev')`**, the one slope method not in the set.

Tier 2 needs nothing further: it covers `sma` and `ema`, and `atr` is no longer on its list (see
Verification).

### Phase 7 notes

- Equivariance uses one map, `x -> 3x + 7` on open/high/low/close (volume untouched), and sorts
  each output into one of three kinds: a **level** maps the same way; a **spread** (`stdev`,
  `true_range`, `atr`, `lrc_slope`, the trendline slopes) only scales by 3; a **signal** (`rsi`,
  `lrc_bull`, breaks, latches) is unchanged. `k > 0` keeps every price comparison, so pivots land
  on the same bars (their prices are levels) and breaks and latches must match exactly.
- Tolerance `rtol=1e-9, atol=1e-9`; the worst case on the primitives is `stdev`, 6e-13 relative.
  The `atol` is there for outputs that cross zero (the slopes).
- Covered: 13 primitives (`linreg` already had its own test), `linreg_candles` with both signals,
  and `trendlines` for all three methods in both backpaint and realtime modes.
- Seeded mutations, each caught: a constant added to the `stdev` slope; `lrc_slope` expressed as a
  percentage of the signal; `atr` wrongly labelled a level in the table; the new `stdev`
  no-look-ahead case switched to `backpaint=True`.
- Suite: 130 tests (108 before + 21 equivariance + 1 no-look-ahead).

## Phase 8 — Data sources

**Done ahead of this phase:** `load_bars(ticker, start, end=None)` was added with the AAPL
exploration notebook (`notebooks/ta_tools_exploration.ipynb`), because `price_return`'s loader
returns the close only. Daily Yahoo bars in `make_bars`' shape, `auto_adjust=False`, two offline
tests with a stubbed yfinance.

**Design decision:** one reader per source, not one `load_bars` with a `source=` switch. Each
source has its own arguments (a symbol and dates vs a path and column layout), dependencies,
authentication and limits; a single function would collect mode-specific arguments and `if`
branches. What keeps the sources interchangeable is a shared output contract, enforced in one place.

- **8.1 `_normalise(frame)`** — every loader ends with it: lower-case `open/high/low/close`, plus
  `volume` **when the source has it** (index symbols such as SPX have none; the column is then
  absent, not NaN), index named `date`, sorted oldest first, duplicate timestamps rejected,
  float64, and interior NaN in the price columns rejected (TA-Lib turns everything after an
  interior NaN into NaN; leading NaN is fine). Other columns pass through (see 8.3).
  `load_bars` is refactored onto it **without changing its behaviour**: it keeps dropping
  Yahoo's incomplete rows before calling `_normalise`, as its `.dropna()` does today, so a real
  ticker never starts raising. The rejection is for files, where a gap is a data problem to
  surface.
- **8.2 Intraday** — `load_bars(ticker, start, end=None, interval='1d')`, passed to yfinance.
  Yahoo's history limits: 1m ≈ 7 days, 5m–30m ≈ 60 days, 1h ≈ 730 days; raise or warn when the
  request exceeds them rather than silently returning fewer bars. Intraday index is tz-aware
  (exchange time); daily stays date-only.
- **8.3 `read_bars(path, ...)`** — CSV into the same frame, with a column mapping and time parsing.
  Must read a TradingView *Export chart data* CSV directly, since that is the Tier 3 parity format:
  lower-case prices, optional `Volume`, and `time` in Unix seconds or ISO 8601 (accept both;
  confirm which the export dialog offers when making the first fixture). The export also carries
  **every plotted indicator series as extra columns** — the values Tier 3 compares against — so
  `read_bars` keeps them, after the price columns, rather than cutting the frame to OHLCV. They may
  have interior NaN (a backpainted line is `na` between segments), so the NaN check stays on prices.
- **8.4 TradingView connector → CSV** — tested 2026-09-28: 20 daily `NASDAQ:AAPL` bars matched
  Yahoo's closes exactly on all 20 days, volumes within Yahoo's rounding to 100 shares (the latest,
  still-updating bar differed by 52k). Bars are split-adjusted only (same basis as
  `auto_adjust=False`), delayed 15+ min, regular session only, max 5,000 per call, 1m to monthly.
  The connector is reachable only from a Claude session, not from Python, a notebook or Colab, so
  the workflow is: fetch in session → write a CSV → `read_bars`. Every bar has to be transcribed
  by hand into the file, so it suits **small, one-off pulls and Tier 3 fixtures** (validate against
  `load_bars`), not a routine feed or multi-thousand-bar downloads.
- **8.5 Other APIs** (Alpaca, Polygon, IBKR, …) — deferred until one is actually needed; each would
  be a thin loader ending in `_normalise`, with its dependency optional. TradingView has no official
  data API; unofficial websocket scrapers are fragile and sit badly with its terms, so not used.

Tests: `_normalise` rejections (unsorted, duplicates, interior NaN in prices), a frame without
volume, `load_bars` still dropping an incomplete Yahoo row, the `interval` pass-through and limit
check with stubbed yfinance, and `read_bars` on a small TradingView-format fixture with an
indicator column, in both time formats.

### Phase 8 notes

- `_normalise` also fixes the index **resolution** to microseconds (pandas 3's default, as in
  `make_bars`): Unix seconds otherwise parse to `datetime64[s]` and ISO strings to `[us]`, and
  frames from two sources would not compare equal. Missing price columns raise `ValueError`, a
  non-datetime index `TypeError`, and an all-NaN price column counts as a gap. Trailing NaN is
  rejected like interior NaN.
- Yahoo limits, refined from the plan's "1m ≈ 7 days": Yahoo keeps **1m for 30 days** and serves
  it **7 days per request**; 2m–90m 60 days; 60m/1h 730 days, counted back from today. Checked
  before calling yfinance, which otherwise only logs an error and returns an empty frame (which
  `load_bars` would misreport as an unknown ticker). Unknown intervals are refused too.
- **Verified against live Yahoo by the owner (2026-09-28)**, by running
  `notebooks/ta_tools_read_data.ipynb`, with all checks passing: the interval pass-through,
  exchange-time index and bar spacing for 1m/5m/15m/1h, and one request just inside each history
  limit served in full. (The unit tests still stub yfinance, since this session's network blocks
  Yahoo.)
- `read_bars(path, columns=None, daily=False, tz=None)`: times without an offset are taken as UTC;
  `tz` converts; `daily=True` keeps the calendar date in `tz`, which matters outside US hours (a
  Tokyo bar at local midnight is the previous day in UTC — tested). Intraday data squeezed to
  dates is caught as repeated timestamps. Non-price columns pass through untouched.
- **Export time formats, settled by two real exports** (the owner's `NASDAQ:AAPL` 1D and 30m, 2026-09-28):
  the daily export writes `time` as a **plain date** (`2007-06-22`), the intraday one as **Unix
  seconds**. A third export (15m) wrote **ISO times with the owner's local offset**
  (`2026-05-14T16:30:00+01:00`), so the format varies between exports; `read_bars` takes all three. The 30m export includes extended hours (04:00–19:30 New York, 32 bars a day), unlike
  Yahoo's regular session. Repeated plot titles arrive as `Plot`, `Plot.1`, ... and untitled ones as
  `Unnamed: 7`; they pass through as extra columns.
- **Bug they exposed, fixed:** a time without an offset was read as UTC, so a plain date with
  `tz='America/New_York'` became 20:00 the evening before, and `daily=True` put every bar a day
  early. Such times are now wall-clock times already in `tz` (UTC if none) and are never shifted;
  Unix seconds and times with an offset are still converted. Two tests fail on the old code, and
  the notebook now checks the plain-date form as well.
- 8.4 run on 2026-09-28: 10 daily `NASDAQ:AAPL` bars from the connector, written as an
  export-layout CSV and read with `read_bars(daily=True, tz='America/New_York')`. The connector's
  `t` is Unix seconds at the 09:30 New York open, so the file needs no conversion. The frame
  matched the connector's own summary (volume total, range, last close) and gave the ten business
  days 14–25 September. Not committed (open judgment call 3); not compared with `load_bars`,
  as Yahoo is blocked here.
- Seeded mutations, each caught: no `dropna` in `load_bars`; no gap check; no Yahoo limit check;
  `tz` ignored in `read_bars`; no index resolution fix.
- Suite: 153 tests (130 before + 23).
- **Added after the phase, at the owner's request:** `notebooks/ta_tools_read_data.ipynb`, a
  hands-on check of every reader against the contract (✓/✗ per check, total at the end),
  committed without outputs for the owner to run. It covers the two items above that could
  not be verified here: live intraday Yahoo, including one request just inside each history
  limit, and a real TradingView export (section 7 prints its header and first row, which
  settles the time-format question). Run under a stubbed yfinance here: 88 of 88 checks pass.
  Its round-trip check found `read_bars` parsing CSV floats up to one unit off in the last
  digit; it now uses `float_precision='round_trip'` (154 tests).

## Verification

The Pine originals run on TradingView and cannot be executed locally, so there is **no reference
implementation to diff against**. Verification therefore runs in three tiers.

**Tier 1 — property tests** in `tests/test_ta_tools.py`, offline and seeded, following the
existing suite's conventions (`tests/test_smoke.py` stays untouched). All in place; the ZLSMA
items wait on ZLSMA itself (deferred):

- *No look-ahead* — the highest-value test. Truncate the input at bar `i`, recompute, and assert
  every value at bars ≤ `i` is unchanged. Run it against every indicator in `backpaint=False`
  mode. An indicator that silently peeks is the failure mode that matters most here.
- *Analytic exactness* — `linreg` of a perfect straight line returns that line; of a constant,
  the constant with slope 0; *(deferred)* ZLSMA of a linear series is the series.
- *Independent reference* — `linreg` at a chosen bar against `numpy.polyfit` over the same window,
  which is a genuine second implementation rather than a restatement.
- *Equivariance* — adding `c` shifts the output by `c`; scaling by `k` scales it by `k` (spreads
  only scale; oscillators and signals are unchanged; see Phase 7 notes).
- *Pivots* — a brute-force `O(n·length)` oracle versus the production implementation, and assert
  the published index is exactly `pivot index + right`.
- *Trendlines* — rails reset to the pivot price at a pivot and decay monotonically between;
  `upos`/`dnos` ∈ {0,1}; and `backpaint=True` output equals realtime output shifted by `length`.
- *Warm-up contract* — exact leading-NaN counts (`length-1`; *(deferred)* ~`2*length-2` for ZLSMA), since
  warm-up is precisely where the three libraries disagree.
- *API* — every `__all__` name resolves; `CAPABILITIES` covers every primitive.

**Tier 2 — cross-library agreement** for the standard primitives only (`sma`, `ema`, against
pandas_ta's own code), where a second library is a free second opinion. Not `atr`: it is now
Pine-exact and differs from TA-Lib's by design, which a test already shows.

**Tier 3 — TradingView parity**, the only true check of a port. Add the indicator to a chart on
one fixed symbol, timeframe and date range, use *Export chart data* to get a CSV of the plotted
series, commit a trimmed copy under `tests/fixtures/`, and compare with a stated tolerance. Static
fixtures keep the suite offline. Until that exists, parity rests on a manual visual overlay —
which is the owner's to do, and should be stated as such rather than implied.

**First Tier 3 run (2026-09-28), not yet a committed fixture:** on the owner's two `NASDAQ:AAPL`
exports (1D, 4,847 bars from 2007; 30m, 7,169 bars), computed from the exported OHLC after a
300-bar warm-up:

| Series on the chart | `ta_tools` | max relative difference |
|---|---|---|
| LinReg Candles *Signal Line* | `linreg_candles(..., linreg_length=14, signal_length=3)`, SMA signal | 1.6e-13 (1D), 3.1e-13 (30m) |
| RSI | `rsi(close, 14)` | 3.3e-11 |
| Bollinger Bands basis / upper | `bb(close, 200, 2.0)` | 3.5e-15 / 1.6e-13 |
| an EMA | `ema(close, 15)` | 0 |

The chart runs LinReg Candles at 14/3, not the script's 11/11 defaults; a search over lengths
2–60 and both signal types found no other setting within 0.4%. The candle columns are empty
(the chart hides the candles), so `lrc_open/high/low/close` and `lrc_bull` are still unchecked,
as is `trendlines`, which is not on the chart. The files stay out of the repo (open call 3).

**Second Tier 3 run (2026-09-28)**, a 15m `NASDAQ:AAPL` export (5,977 bars, 14 May – 28 Sep) with
the owner's stated settings:

| Series | Settings | `ta_tools` | Result |
|---|---|---|---|
| EMA | 9 | `ema(close, 9)` | identical |
| Plot.1 | EMA 15 | `ema(close, 15)` | identical |
| ZLSMA | 14 | `2*linreg(close, 14) - linreg(linreg(close, 14), 14)` | 3.6e-13 |
| Signal Line | LinReg 14/3 | `linreg_candles(..., 14, 3)` | 3.0e-13 |
| Basis / Upper / Lower | BB, EMA basis, 1 SD | `ema(close, 100)` ± `stdev(close, 100)` | exact after bar 795 (EMA(100) warm-up; TradingView had earlier history) |
| Plot | HMA 15 | `hma(close, 15)` | **no match**: 2.5% max, correlation 0.999 |
| ParabolicSAR | 0.02, 0.02, 0.2 | TA-Lib `SAR(0.02, 0.2)` (not wrapped) | **differs** on 1,048 bars, to the end of the file |

- **ZLSMA** needs nothing new: composed from `linreg` it already matches Pine, so the deferred
  port is a thin wrapper.
- **`bb` has no EMA basis**; the match above is composed by hand. An `ma=` option on `bb` would be
  a change to this plan (TradingView's BB offers SMA, EMA, SMMA/RMA, WMA and VWMA).
- **HMA 15 unresolved.** Tried: both roundings of `length/2` and `sqrt(length)`; lengths 4–60 on
  close, hl2, hlc3, ohlc4 and open; shifts of ±5 bars; EHMA and THMA (the "Hull Suite" variants).
  Nothing within 0.5%. Next step is to confirm which indicator draws that plot (built-in *Hull
  Moving Average*, or a community script) and its source. Until then `hma` is unverified against
  TradingView, and its Phase 3 check (against the WMA definition) is the only one.
- **SAR** is not a `ta_tools` primitive. TA-Lib's differs from TradingView's, so a SAR, if wanted,
  would have to be built Pine-exact, like `atr`.

`pytest` stays green and `import src.tools.ta_tools` must work from outside the repo, as
`price_return` does.

## Open judgment calls

These are genuinely the owner's to decide, and the plan does not presume them:

1. ~~**Which library wins**~~ — **resolved**: TA-Lib primary, pandas_ta secondary (see Decisions).
2. ~~**Default `backpaint`**~~ — **resolved**: `False` (realtime) by default; `True` reproduces
   Pine's chart and is opt-in.
3. **Committing TradingView CSVs** — the only true parity check, against licensing and repo size.
   Phase 8.4 makes producing the bar data cheap for small fixtures; the indicator values still
   need TradingView's chart export.
4. ~~**numba**~~ — **not needed**: `trendlines` through `recurse` takes 0.07 s for 5,000 bars
   and 0.34 s for 50,000.
5. ~~**Return shape**~~ — **resolved**: multi-output indicators return one DataFrame with
   named columns (e.g. `bb_mid_20`, `bb_upper_20`, `bb_lower_20`), matching what pandas_ta's
   multi-output indicators already return. Single-output ones stay a named Series.

## Deferred, with reasons

- **ZLSMA + Slope** — deferred from Phase 5 by the owner. Nothing else depends on it; it
  needs only `linreg` (Phase 4), so it can be picked up any time after that.

- **Lorentzian Classification** — 577 Pine lines importing `MLExtensions` and `KernelFunctions`,
  neither vendored here. Needs normalised RSI/WaveTrend/CCI/ADX, regime and volatility filters,
  and two kernel regressions reimplemented before the classifier itself, whose approximate-NN
  search is `O(bars × maxBarsBack)` and inherently loop-based. A project of its own.
- **`basic.py` removal** — deferred by the owner. Note for when it resurfaces: `sd_and_cond`,
  `profit_estimate`, `accepted_min_max` and `projected_min_max` have no successor anywhere, and
  `profit_estimate` is the repo's only expected-value/position-sizing code — directly relevant to
  the option-strategy use case.
- **Old notebooks** (`test_es`, `test_ko`, `test_ta_packages`) — deferred. `test_es.ipynb` cells
  33–36 hold hand-tuned ES option P&L scenario grids that exist nowhere else.
