"""Daily OHLCV bars: from Yahoo Finance, or seeded so tests and notebooks run without the network."""
import numpy as np
import pandas as pd


def make_bars(n=500, seed=0, start='2023-01-02', drift=0.0004, vol=0.011):
    """Daily OHLCV bars from a geometric random walk; the same seed gives the same bars."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, vol, n))
    prev_close = np.r_[close[0], close[:-1]]
    open_ = prev_close * (1 + rng.normal(0, vol / 4, n))

    # Extend from the body, not from close alone, so low <= open, close <= high always holds.
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 3, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 3, n)))

    return pd.DataFrame(
        {
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': rng.integers(1_000, 10_000, n).astype(float),
        },
        index=pd.bdate_range(start, periods=n, name='date'),
    )


def load_bars(ticker, start, end=None):
    """Daily OHLCV bars from Yahoo Finance, in the same shape as make_bars."""
    import yfinance as yf
    # auto_adjust=False, as in price_return: prices stay as traded rather than
    # dividend-adjusted, because option strikes are set against the traded price.
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f'No price data for ticker {ticker!r} from {start}. Yahoo Finance '
                         f'returns an empty frame for unknown symbols, so check the ticker first.')
    if isinstance(raw.columns, pd.MultiIndex):     # yfinance can return a (Price, Ticker) MultiIndex
        raw.columns = raw.columns.get_level_values(0)
    bars = raw[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
    bars.index = pd.DatetimeIndex(bars.index, name='date')
    bars.columns.name = None
    return bars.astype('float64').dropna()
