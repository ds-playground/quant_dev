"""The feature frame: what a model may know at the close of bar t.

Every column at t is computed from bar t and the bars before it, never later, which the
no-look-ahead test checks column by column. Every column is also free of the price scale
(returns, ratios, an oscillator, the weekday), so one model can be compared across tickers. pandas
and numpy only: no TA-Lib, so the `ml` extra does not need the `ta` one.
"""
import math

import numpy as np
import pandas as pd

from .data import bar_returns

EWMA_LAMBDA = 0.94          # RiskMetrics' daily decay

FEATURES = [
    'ret_0', 'ret_1', 'ret_2', 'ret_3', 'ret_4',     # the return of bar t, and four lags
    'abs_ret',                                         # its size
    'vol_20', 'vol_60', 'vol_ewma',                    # close-to-close volatility
    'vol_parkinson', 'vol_gk',                         # 20-bar range volatility
    'tr_close',                                        # true range over the close
    'dist_sma20', 'dist_sma50',                        # the close against its averages
    'rsi_14',                                          # Wilder's RSI
    'dow',                                             # day of the week, Monday 0
]


def ewma_vol(returns, lam=EWMA_LAMBDA, min_periods=20):
    """EWMA volatility known at each bar, `sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_t^2`,
    seeded with the first squared return. NaN for the first `min_periods - 1` returns."""
    return np.sqrt((returns ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=min_periods).mean())


def wilder_rsi(close, length=14):
    """Wilder's RSI on price changes: each average is seeded with the mean of the first `length`
    changes, then smoothed by 1/`length`, as Pine's `ta.rsi` and TA-Lib's RSI. The first `length`
    values are NaN; a window with no losses is 100."""
    change = close.astype(float).diff().to_numpy()
    out = np.full(len(change), np.nan)
    if len(change) <= length:
        return pd.Series(out, index=close.index)
    gain, loss = np.clip(change, 0, None), np.clip(-change, 0, None)
    avg_gain, avg_loss = gain[1:length + 1].mean(), loss[1:length + 1].mean()
    for t in range(length, len(change)):
        if t > length:
            avg_gain = (avg_gain * (length - 1) + gain[t]) / length
            avg_loss = (avg_loss * (length - 1) + loss[t]) / length
        out[t] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return pd.Series(out, index=close.index)


def features(bars):
    """The 16 `FEATURES` for every bar, as a Date-indexed frame (NaN during each warm-up)."""
    o, h, l, c = (bars[k].astype(float) for k in ('Open', 'High', 'Low', 'Close'))
    r = bar_returns(bars)
    positive = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    log_hl = np.log((h / l).where(positive))
    log_co = np.log((c / o).where(positive))
    prev_c = c.shift(1)
    true_range = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    f = pd.DataFrame(index=bars.index)
    for lag in range(5):
        f[f'ret_{lag}'] = r.shift(lag)
    f['abs_ret'] = r.abs()
    f['vol_20'] = r.rolling(20).std()
    f['vol_60'] = r.rolling(60).std()
    f['vol_ewma'] = ewma_vol(r)
    f['vol_parkinson'] = np.sqrt((log_hl ** 2).rolling(20).mean() / (4 * math.log(2)))
    f['vol_gk'] = np.sqrt((0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2).rolling(20).mean())
    f['tr_close'] = (true_range / c).where(positive & (prev_c > 0))
    f['dist_sma20'] = (c / c.rolling(20).mean() - 1).where(positive)
    f['dist_sma50'] = (c / c.rolling(50).mean() - 1).where(positive)
    f['rsi_14'] = wilder_rsi(c, 14)
    f['dow'] = bars.index.dayofweek.astype(float)
    return f[FEATURES]
