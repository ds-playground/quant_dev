"""Regenerate data/demo: processed daily bars for demos and offline work. Not market data.

Downloads daily bars from Yahoo Finance (as traded, `auto_adjust=False`), multiplies every
open/high/low/close and volume by (1 + e), with e ~ Normal(0, 0.01%), seeded per file,
restores low <= open, close <= high, and writes data/demo/{NAME}_demo.csv plus manifest.csv.
See data/demo/README.md. Run from the repo root: `python scripts/make_demo_data.py`.
"""
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

START = '2016-01-01'
NOISE = 0.0001            # 0.01% relative noise on each price column and volume
SEED = 42
OUT = Path('data/demo')   # run from the repo root

# Demo name -> Yahoo symbols to try, in order.
TICKERS = {
    'SPX':    ['^GSPC'],               # S&P 500 index
    'NDQ':    ['^NDX'],                # Nasdaq 100 index
    'YM':     ['YM=F'],                # E-mini Dow futures
    'CL':     ['CL=F'],                # WTI crude oil futures
    'RTY':    ['RTY=F'],               # E-mini Russell 2000 futures
    'EURUSD': ['EURUSD=X'],
    'XAUUSD': ['XAUUSD=X', 'GC=F'],    # spot gold; gold futures if Yahoo has no spot series
    'AAPL':   ['AAPL'],
    'KO':     ['KO'],
    'TSLL':   ['TSLL'],                # 2x TSLA ETF, trading since Aug 2022
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (name, symbols) in enumerate(TICKERS.items()):
        for symbol in symbols:
            raw = yf.download(symbol, start=START, auto_adjust=False, progress=False,
                              multi_level_index=False)
            if not raw.empty:
                break
        else:
            print(f'{name}: no data for any of {symbols}; skipped')
            continue

        rng = np.random.default_rng(SEED + i)       # one stream per file, reproducible
        df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(
            subset=['Open', 'High', 'Low', 'Close']).copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] *= 1 + rng.normal(0, NOISE, len(df))
        # Noise can break low <= open, close <= high; restore it.
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        df['Volume'] = (df['Volume'].fillna(0) * (1 + rng.normal(0, NOISE, len(df)))).round()
        df.index.name = 'Date'
        df.round(6).to_csv(OUT / f'{name}_demo.csv')

        manifest.append({'file': f'{name}_demo.csv', 'yahoo_symbol': symbol, 'rows': len(df),
                         'first': df.index[0].date(), 'last': df.index[-1].date(),
                         'noise': NOISE, 'seed': SEED + i, 'generated': dt.date.today()})
        print(f'{name}: {symbol}, {len(df)} rows')

    pd.DataFrame(manifest).to_csv(OUT / 'manifest.csv', index=False)


if __name__ == '__main__':
    main()
