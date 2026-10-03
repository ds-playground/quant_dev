"""The labels, which alone may look past bar t, and the modelling frame that joins them to the
features.

Two labels, one per model (the owner's choice, 2026-10-03):

- `y_ret`, tomorrow's return, whose quantiles `TAUS` the return model forecasts: the median as
  the point forecast, and a 68% and a 90% interval;
- `y_wdl`, tomorrow's win (1), draw (0) or loss (-1) against a fixed threshold, 0.2% by default.

The binary up/down label of the first plan is gone: win/draw/loss replaces it.
"""

from .data import bar_returns
from .features import WDL_THRESHOLD, features, win_draw_loss

TAUS = (0.05, 0.1587, 0.5, 0.8413, 0.95)


def next_return(bars):
    """`r_{t+1}`, tomorrow's close-to-close return, at bar t (NaN where it is undefined)."""
    return bar_returns(bars).shift(-1).rename('y_ret')


def next_wdl(bars, threshold=WDL_THRESHOLD):
    """Tomorrow's outcome at bar t: 1 (win), 0 (draw), -1 (loss) against `threshold`."""
    return win_draw_loss(bar_returns(bars), threshold).shift(-1).rename('y_wdl')


def dataset(bars, wdl_threshold=WDL_THRESHOLD):
    """The modelling frame: the `FEATURES` and the labels `y_ret` and `y_wdl` (tomorrow's
    win/draw/loss against `wdl_threshold`), on the rows where all are defined, plus `vol_250`,
    the trailing 250-bar volatility `price_range` is fed. That is a baseline input, not a
    feature, so it is NaN in its warm-up instead of removing rows, and it tolerates a few
    undefined returns in its window (SYN-OIL's negative settle drops two). The last bar, with no
    tomorrow, is never included."""
    frame = features(bars, wdl_threshold).assign(
        y_ret=next_return(bars), y_wdl=next_wdl(bars, wdl_threshold)).dropna()
    vol_250 = bar_returns(bars).rolling(250, min_periods=240).std()
    frame = frame.assign(vol_250=vol_250.reindex(frame.index))
    frame.attrs['wdl_threshold'] = wdl_threshold
    return frame

