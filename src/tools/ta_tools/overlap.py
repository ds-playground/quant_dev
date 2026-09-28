"""Moving averages: TA-Lib where it has them, pandas_ta where it does not."""
from .backend import as_float_series, pandas_ta, provides, talib


@provides('talib')
def sma(series, length):
    """Simple moving average; the first length-1 values are NaN."""
    series = as_float_series(series)
    if length == 1:  # TA-Lib rejects length 1; Pine returns the input unchanged.
        return series.rename(f'sma_{length}')
    return talib.SMA(series, timeperiod=length).rename(f'sma_{length}')


@provides('talib')
def ema(series, length):
    """Exponential moving average, SMA-seeded as in Pine's ta.ema; the first length-1 values are NaN."""
    series = as_float_series(series)
    if length == 1:  # TA-Lib rejects length 1; with alpha = 1 Pine returns the input unchanged.
        return series.rename(f'ema_{length}')
    return talib.EMA(series, timeperiod=length).rename(f'ema_{length}')


@provides('talib')
def wma(series, length):
    """Linearly weighted moving average, newest bar heaviest; the first length-1 values are NaN."""
    series = as_float_series(series)
    if length == 1:  # TA-Lib rejects length 1; one bar's weighted average is itself.
        return series.rename(f'wma_{length}')
    return talib.WMA(series, timeperiod=length).rename(f'wma_{length}')


@provides('pandas_ta')
def hma(series, length):
    """Hull moving average; TA-Lib has no equivalent."""
    return pandas_ta().hma(as_float_series(series), length=length).rename(f'hma_{length}')


@provides('pandas_ta')
def alma(series, length, offset=0.85, sigma=6.0):
    """Arnaud Legoux moving average, equal to Pine's ta.alma with floor=true."""
    # pandas_ta's own `offset` argument shifts the output; ALMA's offset is its `dist_offset`.
    out = pandas_ta().alma(as_float_series(series), length=length,
                           sigma=float(sigma), dist_offset=float(offset))
    return out.rename(f'alma_{length}')
