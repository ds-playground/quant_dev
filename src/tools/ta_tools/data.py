"""Seeded OHLCV bars, so tests and notebooks run without the network."""
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
