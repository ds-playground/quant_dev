"""Loading a price series and deriving rolling statistics from it."""

import numpy as np
import pandas as pd

from .params import _params


def load_price_data(p=None, verbose=True):
    """Return a date / price / return_pct frame, from Yahoo Finance or simulation.

    `return_pct` is in percent (0.5 == +0.5%). The simulated series spans the same
    start..end business-day range as the real one, so both paths are comparable.
    """
    p = _params(p)

    if p.data_source == 'yahoo':
        # pip install yfinance
        import yfinance as yf
        # auto_adjust=False keeps `Close` as the actually-traded price rather than
        # a dividend-adjusted series. Option strikes are set against the traded
        # price, so adjusting it would misstate where a strike sits relative to
        # spot. Passed explicitly because yfinance flipped this default, and the
        # two settings give different returns for anything paying a dividend.
        raw = yf.download(p.ticker, start=p.start_date, end=p.end_date,
                          auto_adjust=False)
        if raw.empty:
            raise ValueError(
                f"No price data for ticker {p.ticker!r} between {p.start_date} and "
                f"{p.end_date}. Yahoo Finance returns an empty frame for unknown "
                f"symbols, so check the ticker first.")
        if isinstance(raw.columns, pd.MultiIndex):     # yfinance can return a (Price, Ticker) MultiIndex
            raw.columns = raw.columns.get_level_values(0)
        df = raw[['Close']].rename(columns={'Close': 'price'}).reset_index()
        df = df.rename(columns={'Date': 'date'})
        df['return_pct'] = df['price'].pct_change() * 100

        # A percent change spanning a non-positive price is meaningless - WTI
        # settled at -$37.63 on 2020-04-20, which yields a -306% "return" and a
        # -127% one the next day. Drop those rather than let them distort every
        # downstream statistic; a change between two positive prices is kept
        # however large.
        spans_nonpositive = (df['price'] <= 0) | (df['price'].shift(1) <= 0)
        if spans_nonpositive.any():
            flagged = df.loc[df['price'] <= 0, 'date'].dt.strftime('%Y-%m-%d').tolist()
            print(f'Warning: {p.ticker} has {len(flagged)} non-positive price(s) '
                  f'({", ".join(flagged)}); dropping '
                  f'{int(spans_nonpositive.sum())} affected return(s).')
            df.loc[spans_nonpositive, 'return_pct'] = np.nan

        df = df.dropna().reset_index(drop=True)

    elif p.data_source == 'simulated':
        np.random.seed(p.random_seed)
        dates = pd.bdate_range(start=p.start_date, end=p.end_date)
        daily_returns = np.random.normal(loc=p.sim_drift, scale=p.sim_vol, size=len(dates))   # % per day
        prices = p.sim_start_price * np.cumprod(1 + daily_returns / 100)
        df = pd.DataFrame({
            'date':        dates,
            'price':       prices,
            'return_pct':  daily_returns
        })

    else:
        raise ValueError(f"data_source must be 'simulated' or 'yahoo', got {p.data_source!r}")

    df['date'] = pd.to_datetime(df['date'])
    if verbose:
        print(f'Data source: {p.data_source}   |   Loaded {len(df)} trading days   |   '
              f'Avg return: {df.return_pct.mean():.3f}%   |   '
              f'Std: {df.return_pct.std():.3f}%')
    return df

def add_rolling_stats(df, p=None):
    """Add `PCT Change {d}` / `{d} Av` / `{d} STD` columns plus the annualized pair.

    For each holding period `d`, the d-day cumulative return is computed, then its
    rolling mean and std over `trade_days` days. Returns are decimal fractions here
    (0.0004), converted from the percent-scale `return_pct` column.
    """
    p = _params(p)
    trade_d = p.trade_days
    df = df.copy()

    # Base 1-day series and the annualized pair everything else builds on.
    df['PCT Change 1']     = df['return_pct'] / 100
    df['PCT Change 1 Av']  = df['PCT Change 1'].rolling(trade_d).mean()
    df['PCT Change 1 STD'] = df['PCT Change 1'].rolling(trade_d).std()
    df['PCT Change Annualized']     = df['PCT Change 1'].rolling(trade_d).sum()
    df['PCT Change Annualized STD'] = df['PCT Change 1 STD'] * trade_d ** 0.5

    for d in p.roll_windows:
        if d == 1:
            continue    # already computed above
        df[f'PCT Change {d}']     = df['PCT Change 1'].rolling(d).sum()
        df[f'PCT Change {d} Av']  = df[f'PCT Change {d}'].rolling(trade_d).mean()
        df[f'PCT Change {d} STD'] = df[f'PCT Change {d}'].rolling(trade_d).std()

    return df

def latest_snapshot(df, p=None):
    """Latest annualized and daily mean/vol, as a Series of decimal fractions."""
    return pd.Series({
        'annualized_return':     df['PCT Change Annualized'].iloc[-1],
        'annualized_volatility': df['PCT Change Annualized STD'].iloc[-1],
        'daily_avg_return':      df['PCT Change 1 Av'].iloc[-1],
        'daily_std_dev':         df['PCT Change 1 STD'].iloc[-1],
    })

def show_latest_snapshot(df, p=None):
    """Print the latest rolling vol/mean snapshot and return it."""
    p = _params(p)
    s = latest_snapshot(df, p)
    print(f'=== Annualized (latest {p.trade_days}d) ===')
    print(f"Return:     {s['annualized_return']:.2%}")
    print(f"Volatility: {s['annualized_volatility']:.2%}")
    print()
    print(f'=== Daily (latest {p.trade_days}d rolling) ===')
    print(f"Avg return: {s['daily_avg_return']:.4%}")
    print(f"Std dev:    {s['daily_std_dev']:.4%}")
    return s

def daily_returns_series(df):
    """Date-indexed decimal daily returns, the input the historical helpers expect."""
    return df.set_index('date')['PCT Change 1']
