"""Volatility measures."""
import pandas as pd

from .backend import as_float_series, provides, talib
from .pine import rma


@provides('talib')
def stdev(series, length):
    """Rolling population standard deviation, as Pine's ta.stdev computes by default."""
    series = as_float_series(series)
    if length == 1:  # TA-Lib rejects length 1; one value has zero spread.
        return (series * 0.0).rename(f'stdev_{length}')
    return talib.STDDEV(series, timeperiod=length, nbdev=1).rename(f'stdev_{length}')


@provides('custom')
def true_range(high, low, close):
    """Pine's ta.tr(true): the largest of high - low and the gaps from the previous close."""
    high, low, close = as_float_series(high), as_float_series(low), as_float_series(close)
    prev = close.shift(1)
    out = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return out.rename('true_range')   # bar 0 has no previous close, so it is high - low, as in Pine


@provides('custom')
def atr(high, low, close, length):
    """Pine's ta.atr: rma of the true range; the first length-1 values are NaN."""
    # Built here rather than on TA-Lib's ATR, which has no bar-0 true range and so starts a bar
    # later, seeded from a different window: 1-4% off Pine at first, converging over ~250 bars.
    return rma(true_range(high, low, close), length).rename(f'atr_{length}')


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
