"""Volatility measures."""
import pandas as pd

from .backend import as_float_series, provides, talib


@provides('talib')
def stdev(series, length):
    """Rolling population standard deviation, as Pine's ta.stdev computes by default."""
    series = as_float_series(series)
    if length == 1:  # TA-Lib rejects length 1; one value has zero spread.
        return (series * 0.0).rename(f'stdev_{length}')
    return talib.STDDEV(series, timeperiod=length, nbdev=1).rename(f'stdev_{length}')


@provides('talib')
def atr(high, low, close, length):
    """Wilder-smoothed average true range; the first `length` values are NaN."""
    # Starts one bar later than Pine's ta.atr, which counts bar 0's true range as high - low.
    # The two differ by ~4% at the first value and converge (under 0.01% by bar 100 at length 14).
    out = talib.ATR(as_float_series(high), as_float_series(low), as_float_series(close),
                    timeperiod=length)
    return out.rename(f'atr_{length}')


@provides('talib')
def bb(series, length, mult=2.0):
    """Bollinger Bands as in Pine's ta.bb: SMA basis +/- mult population standard deviations."""
    series = as_float_series(series)
    if length == 1:  # TA-Lib rejects length 1; with zero spread all three bands are the input.
        upper = mid = lower = series
    else:
        upper, mid, lower = talib.BBANDS(series, timeperiod=length,
                                         nbdevup=mult, nbdevdn=mult, matype=0)
    return pd.DataFrame({f'bb_mid_{length}': mid,
                         f'bb_upper_{length}': upper,
                         f'bb_lower_{length}': lower})
