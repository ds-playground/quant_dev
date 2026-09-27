"""Pine Script primitives that neither TA-Lib nor pandas_ta provides with Pine's semantics."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .backend import as_float_series, provides, talib


@provides('derived')
def linreg(series, length, offset=0):
    """Pine's ta.linreg: the least-squares line over `length` bars, evaluated `offset` bars back."""
    series = as_float_series(series)
    name = f'linreg_{length}_{offset}'
    if length == 1:  # TA-Lib rejects length 1; a line through one point is that point.
        return series.rename(name)
    # Pine: intercept + slope * (length - 1 - offset). LINEARREG is the offset-0 value.
    value = talib.LINEARREG(series, timeperiod=length)
    slope = talib.LINEARREG_SLOPE(series, timeperiod=length)
    return (value - slope * offset).rename(name)


@provides('custom')
def rma(series, length):
    """Pine's ta.rma (Wilder smoothing): seeded with the SMA of the first `length` values."""
    series = as_float_series(series)
    valid = np.flatnonzero(series.notna().to_numpy())
    out = pd.Series(np.nan, index=series.index, name=f'rma_{length}')
    if len(valid) < length:
        return out
    seed_at = valid[0] + length - 1
    seeded = series.copy()
    seeded.iloc[:seed_at] = np.nan
    seeded.iloc[seed_at] = series.iloc[valid[0]:seed_at + 1].mean()
    out[:] = seeded.ewm(alpha=1 / length, adjust=False).mean().to_numpy()
    out.iloc[:seed_at] = np.nan
    return out


def _pivot(values, left, right):
    """Pivot highs of `values`, each published `right` bars after the pivot bar."""
    out = np.full(len(values), np.nan)
    width = left + right + 1
    if len(values) < width:
        return out
    windows = sliding_window_view(values, width)   # row k ends at bar k + width - 1: no look-ahead
    centre = windows[:, left]
    with np.errstate(invalid='ignore'):
        # Pine does not document its tie-breaking. >= on the left and > on the right means a flat
        # top of equal highs yields exactly one pivot, at its last bar.
        is_pivot = ((centre >= windows[:, :left].max(axis=1, initial=-np.inf))
                    & (centre > windows[:, left + 1:].max(axis=1, initial=-np.inf))
                    & ~np.isnan(windows).any(axis=1))
    out[width - 1:] = np.where(is_pivot, centre, np.nan)
    return out


@provides('custom')
def pivot_high(series, left, right):
    """Pine's ta.pivothigh: the pivot's value, published `right` bars after the pivot; NaN elsewhere."""
    series = as_float_series(series)
    return pd.Series(_pivot(series.to_numpy(), left, right), index=series.index,
                     name=f'pivot_high_{left}_{right}')


@provides('custom')
def pivot_low(series, left, right):
    """Pine's ta.pivotlow: the pivot's value, published `right` bars after the pivot; NaN elsewhere."""
    series = as_float_series(series)
    return pd.Series(-_pivot(-series.to_numpy(), left, right), index=series.index,
                     name=f'pivot_low_{left}_{right}')


@provides('custom')
def change(series, length=1):
    """Pine's ta.change: the difference from `length` bars ago."""
    series = as_float_series(series)
    return (series - series.shift(length)).rename(f'change_{length}')


def _operand(value, index):
    return as_float_series(value) if isinstance(value, pd.Series) else pd.Series(float(value), index=index)


@provides('custom')
def crossover(a, b):
    """Pine's ta.crossover: `a` is above `b` now and was at or below it on the previous bar."""
    a = as_float_series(a)
    b = _operand(b, a.index)
    return ((a > b) & (a.shift(1) <= b.shift(1))).rename('crossover')


@provides('custom')
def crossunder(a, b):
    """Pine's ta.crossunder: `a` is below `b` now and was at or above it on the previous bar."""
    a = as_float_series(a)
    b = _operand(b, a.index)
    return ((a < b) & (a.shift(1) >= b.shift(1))).rename('crossunder')


@provides('custom')
def barssince(condition):
    """Pine's ta.barssince: bars since `condition` was last true; NaN until it first is."""
    condition = condition.fillna(False).astype(bool)
    position = pd.Series(np.arange(len(condition), dtype=float), index=condition.index)
    last_true = position.where(condition).ffill()
    return (position - last_true).rename('barssince')


@provides('custom')
def nz(series, replacement=0.0):
    """Pine's nz: replace NaN with `replacement`."""
    return series.fillna(replacement)


@provides('custom')
def recurse(inputs, step, init):
    """Pine `var` state, bar by bar: state = step(copy of state, row as namedtuple); returns each bar's state."""
    state = dict(init)
    rows = []
    for bar in inputs.itertuples(index=False):
        state = step(dict(state), bar)
        rows.append(dict(state))
    return pd.DataFrame(rows, index=inputs.index, columns=list(init))
