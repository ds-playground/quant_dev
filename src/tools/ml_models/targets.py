"""The labels, which alone may look past bar t, and the modelling frame that joins them to the
features.

The direction label is binary by the owner's choice (`docs/ml_plan.md`); a flat band scaled by
volatility is available as an option, never a fixed one. The range label is tomorrow's return,
whose quantiles `TAUS` the range models forecast: a 90% and a 68% interval and the median.
"""
import numpy as np
import pandas as pd

from .data import bar_returns
from .features import features

TAUS = (0.05, 0.1587, 0.5, 0.8413, 0.95)


def next_return(bars):
    """`r_{t+1}`, tomorrow's close-to-close return, at bar t (NaN where it is undefined)."""
    return bar_returns(bars).shift(-1).rename('y_ret')


def next_direction(bars, flat=None, sigma=None):
    """Tomorrow's direction at bar t: 1.0 up, 0.0 down (or unchanged), NaN where unknown.

    With `flat=k` and `sigma` (a volatility known at t, such as `features(bars)['vol_ewma']`),
    three classes instead: 1.0 above `k * sigma`, -1.0 below `-k * sigma`, 0.0 in between.
    """
    r = next_return(bars)
    if flat is None:
        return (r > 0).astype(float).where(r.notna()).rename('y_up')
    if sigma is None:
        raise ValueError('a flat band needs sigma, the volatility known at each bar')
    band = flat * sigma.reindex(r.index)
    out = pd.Series(np.select([r > band, r < -band], [1.0, -1.0], 0.0), index=r.index)
    return out.where(r.notna() & band.notna()).rename('y_dir')


def dataset(bars):
    """The modelling frame: the `FEATURES` and the labels `y_ret` and `y_up`, on the rows where
    all are defined, plus `vol_250`, the trailing 250-bar volatility `price_range` is fed. That is
    a baseline input, not a feature, so it is NaN in its warm-up instead of removing rows, and it
    tolerates a few undefined returns in its window (SYN-OIL's negative settle drops two). The
    last bar, with no tomorrow, is never included."""
    frame = features(bars).assign(y_ret=next_return(bars), y_up=next_direction(bars)).dropna()
    vol_250 = bar_returns(bars).rolling(250, min_periods=240).std()
    return frame.assign(vol_250=vol_250.reindex(frame.index))

