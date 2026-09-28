"""Whole indicators ported from the Pine scripts in pine_scripts/, built only from ta_tools primitives."""
import numpy as np
import pandas as pd

from .backend import as_float_series, provides
from .overlap import ema, sma
from .pine import change, linreg, pivot_high, pivot_low, recurse
from .volatility import atr, stdev


@provides('derived')
def linreg_candles(open, high, low, close, linreg_length=11, signal_length=11,
                   sma_signal=True, lin_reg=True):
    """Linear Regression Candles and Slope (pine_scripts/Linear_Regression_Candles_and_Slope.pine).

    Each of open, high, low and close is replaced by its ta.linreg(length, 0); the signal line is
    the SMA (or EMA) of the LinReg close, and the slope is its one-bar change. Arguments keep the
    Pine input names and defaults. Returns a DataFrame of lrc_open/high/low/close, lrc_signal,
    lrc_slope and lrc_bull (LinReg open below LinReg close; False in the warm-up, as na < na is
    false in Pine). The candles are NaN for linreg_length-1 bars, the signal for a further
    signal_length-1, the slope for one more.

    The four fits are independent, so a LinReg high can fall below the body (about 2% of bars on
    make_bars data); Pine plots such candles as they are, and so does this.
    """
    ohlc = {'open': open, 'high': high, 'low': low, 'close': close}
    if lin_reg:
        ohlc = {k: linreg(v, linreg_length) for k, v in ohlc.items()}
    else:  # Pine's lin_reg = false passes the raw candles through.
        ohlc = {k: as_float_series(v) for k, v in ohlc.items()}
    signal = (sma if sma_signal else ema)(ohlc['close'], signal_length)
    return pd.DataFrame({'lrc_open': ohlc['open'],
                         'lrc_high': ohlc['high'],
                         'lrc_low': ohlc['low'],
                         'lrc_close': ohlc['close'],
                         'lrc_signal': signal,
                         'lrc_slope': change(signal),
                         'lrc_bull': ohlc['open'] < ohlc['close']})


SLOPE_METHODS = ('atr', 'stdev', 'linreg')


def _trendline_slope(high, low, close, length, mult, calc_method):
    """Per-bar slope of a new trendline, by Pine's three calcMethod options."""
    if calc_method == 'atr':
        return atr(high, low, close, length) / length * mult
    if calc_method == 'stdev':
        return stdev(close, length) / length * mult
    # Pine: |sma(src*n) - sma(src)*sma(n)| / variance(n) / 2, i.e. half the absolute least-squares
    # slope over `length` bars; unlike the others it is not divided by length.
    if length == 1:  # Pine's variance of one bar is 0, and 0/0 is na.
        return pd.Series(np.nan, index=close.index)
    return (linreg(close, length, 0) - linreg(close, length, 1)).abs() / 2 * mult


@provides('derived')
def trendlines(high, low, close, length=14, mult=1.0, calc_method='atr', backpaint=False):
    """Trendlines with Breaks (pine_scripts/Trendlines_with_Breaks_Style_Options.pine, LuxAlgo).

    At each pivot high the upper trendline restarts from the pivot and then falls by a fixed
    slope per bar; the lower trendline does the same, rising, from each pivot low. A close above
    the upper line is an upper break, a close below the lower line a lower break; each fires once
    until the next pivot re-arms it. Arguments keep the Pine inputs (calcMethod in lower case);
    style inputs are dropped.

    backpaint=False (the default) is the realtime series: every value uses only bars up to its
    own. backpaint=True reproduces Pine's default chart, which draws the lines from the pivot bar
    itself: the line columns move `length` bars into the past, so they use `length` bars of
    future data and the last `length` rows are NaN. Breaks and latches never move, in either mode.

    Returns a DataFrame:
      tl_upper, tl_lower              the trendlines; NaN until the first pivot (Pine's start at 0)
      tl_upper_slope, tl_lower_slope  the per-bar slope each line is moving by
      tl_pivot_high, tl_pivot_low     the pivot's price, on the bar its line restarts; else NaN
      tl_upos, tl_dnos                Pine's upos/dnos break latches, 0 or 1
      tl_upper_break, tl_lower_break  True on the bar a latch goes from 0 to 1
    Pine's extended lines are the rows with a pivot: anchored at that bar and price, extended by
    the slope.
    """
    if calc_method not in SLOPE_METHODS:
        raise ValueError(f'calc_method must be one of {SLOPE_METHODS}, got {calc_method!r}')
    high, low, close = as_float_series(high), as_float_series(low), as_float_series(close)
    ph, pl = pivot_high(high, length, length), pivot_low(low, length, length)
    # Pine tests `ph ?`, and a float is false when na or 0, so a pivot at exactly 0 is no pivot.
    inputs = pd.DataFrame({'ph': ph.where(ph != 0),
                           'pl': pl.where(pl != 0),
                           'slope': _trendline_slope(high, low, close, length, mult, calc_method),
                           'close': close})

    def step(s, bar):
        ph, pl = bar.ph == bar.ph, bar.pl == bar.pl     # not NaN
        s['slope_ph'] = bar.slope if ph else s['slope_ph']
        s['slope_pl'] = bar.slope if pl else s['slope_pl']
        s['upper'] = bar.ph if ph else s['upper'] - s['slope_ph']
        s['lower'] = bar.pl if pl else s['lower'] + s['slope_pl']
        s['upos'] = 0 if ph else 1 if bar.close > s['upper'] - s['slope_ph'] * length else s['upos']
        s['dnos'] = 0 if pl else 1 if bar.close < s['lower'] + s['slope_pl'] * length else s['dnos']
        return s

    state = recurse(inputs, step, {'upper': 0.0, 'lower': 0.0, 'slope_ph': 0.0, 'slope_pl': 0.0,
                                   'upos': 0, 'dnos': 0})

    if backpaint:
        upper, lower = state['upper'], state['lower']
    else:  # the line `length` bars on from the pivot, where it is confirmed: Pine's realtime plot
        upper = state['upper'] - state['slope_ph'] * length
        lower = state['lower'] + state['slope_pl'] * length
    # Pine's var lines start at 0 and would plot at zero until the first pivot; NaN here instead.
    seen_ph = inputs['ph'].notna().cummax()
    seen_pl = inputs['pl'].notna().cummax()
    lines = pd.DataFrame({'tl_upper': upper.where(seen_ph),
                          'tl_lower': lower.where(seen_pl),
                          'tl_upper_slope': state['slope_ph'].where(seen_ph),
                          'tl_lower_slope': state['slope_pl'].where(seen_pl),
                          'tl_pivot_high': inputs['ph'],
                          'tl_pivot_low': inputs['pl']})
    if backpaint:  # Pine's plot offset = -length: each line is drawn from the pivot bar itself
        lines = lines.shift(-length)
    latches = state[['upos', 'dnos']].astype('int64')
    return lines.assign(tl_upos=latches['upos'], tl_dnos=latches['dnos'],
                        tl_upper_break=latches['upos'].diff() > 0,
                        tl_lower_break=latches['dnos'].diff() > 0)
