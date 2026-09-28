"""Momentum oscillators."""
import numpy as np
import pandas as pd

from .backend import as_float_series, provides, talib


@provides('talib')
def rsi(series, length):
    """Wilder's RSI, as Pine's ta.rsi; the first `length` values are NaN."""
    series = as_float_series(series)
    if length == 1:  # TA-Lib rejects length 1; RSI then reduces to the direction of each change.
        change = series.diff()
        out = pd.Series(np.where(change > 0, 100.0, 0.0), index=series.index).where(change.notna())
        return out.rename(f'rsi_{length}')
    # A perfectly flat window gives 0, TA-Lib's convention; TradingView's built-in RSI script gives 100.
    return talib.RSI(series, timeperiod=length).rename(f'rsi_{length}')
