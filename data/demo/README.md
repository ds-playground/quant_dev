# Demo data: processed, not market data

**These files are not market data.** They are processed from daily bars downloaded from Yahoo
Finance, with random noise added, and are provided **for education and demonstration only**:
running the notebooks, the dashboard and the tests without a network connection. Do not use them
for trading or investment decisions, and do not redistribute them as market data. If you use
Yahoo Finance data yourself, check Yahoo's terms first.

## How they were made

`scripts/make_demo_data.py`, run on 2026-09-30:

1. Daily bars from 2016-01-01 via `yfinance`, **as traded** (`auto_adjust=False`: not adjusted for
   dividends), matching how `price_return` loads Yahoo data.
2. Every open, high, low, close and volume multiplied by `1 + e`, with `e` drawn from a normal
   distribution with standard deviation **0.01%** (`NOISE = 0.0001`), from a separate seeded random
   stream per file (seed below).
3. High and low reset to `max(open, high, close)` and `min(open, low, close)`, since independent
   noise could otherwise put a close above its high.
4. Rounded to 6 decimals and written as `Date, Open, High, Low, Close, Volume`.

`manifest.csv` records, for each file, the Yahoo symbol, row count, date range, noise level, seed
and generation date.

## Files

| File | Underlying series | Yahoo symbol | Rows | Dates | Seed |
|---|---|---|---|---|---|
| `SPX_demo.csv` | S&P 500 index | `^GSPC` | 2,700 | 2016-01-04 to 2026-09-29 | 42 |
| `NDQ_demo.csv` | Nasdaq 100 index | `^NDX` | 2,700 | 2016-01-04 to 2026-09-29 | 43 |
| `YM_demo.csv` | E-mini Dow futures | `YM=F` | 2,701 | 2016-01-04 to 2026-09-29 | 44 |
| `CL_demo.csv` | WTI crude oil futures | `CL=F` | 2,700 | 2016-01-04 to 2026-09-29 | 45 |
| `RTY_demo.csv` | E-mini Russell 2000 futures | `RTY=F` | 2,322 | 2017-07-10 to 2026-09-29 | 46 |
| `EURUSD_demo.csv` | EUR/USD | `EURUSD=X` | 2,796 | 2016-01-01 to 2026-09-30 | 47 |
| `XAUUSD_demo.csv` | COMEX gold futures (no spot gold series on Yahoo) | `GC=F` | 2,699 | 2016-01-04 to 2026-09-29 | 48 |
| `AAPL_demo.csv` | Apple | `AAPL` | 2,699 | 2016-01-04 to 2026-09-28 | 49 |
| `KO_demo.csv` | Coca-Cola | `KO` | 2,699 | 2016-01-04 to 2026-09-28 | 50 |
| `TSLL_demo.csv` | Direxion Daily TSLA Bull 2X | `TSLL` | 1,038 | 2022-08-09 to 2026-09-28 | 51 |

## What to know

- **CL has one negative close** (2020-04-20), when WTI futures settled below zero.
  `load_price_data` drops the two returns spanning it, with a warning.
- **XAUUSD is gold futures,** not spot gold: Yahoo had no spot series, so the script fell back to
  `GC=F`. The file keeps the XAUUSD name.
- **Futures** (YM, CL, RTY, XAUUSD) are Yahoo's continuous front-month series, which jump at
  contract rolls.
- **Volume** is zero on every EURUSD row (Yahoo reports none for FX), and on a few days elsewhere:
  XAUUSD 36, RTY 9, CL 7, YM 7, SPX 1.
- **Shorter histories:** RTY starts in July 2017 (Yahoo's series begins there) and TSLL in August
  2022 (its launch).

## Using them

```python
from src.tools.price_return import Params, load_price_data, load_ticker_config, demo_tickers

demo_tickers()                                   # ['AAPL', 'CL', ..., 'YM']
df = load_price_data(Params(data_source='demo', ticker='SPX', start_date='2016-01-01'))
config = load_ticker_config('configs/demo_tickers.yaml')   # all ten, with labels and thresholds
```

`ta_tools.read_bars('data/demo/SPX_demo.csv', columns={'Date': 'time'}, daily=True)` reads the
full OHLCV bars. The dashboard's ticker list uses `configs/demo_tickers.yaml` by default.

To regenerate (needs network access to Yahoo): `python scripts/make_demo_data.py` from the repo
root. The noise is seeded, but Yahoo occasionally revises history, so a later run may differ.
