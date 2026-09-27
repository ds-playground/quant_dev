"""Whole indicators ported from the Pine scripts in pine_scripts/, built only from ta_tools primitives."""
import pandas as pd

from .backend import as_float_series, provides
from .overlap import ema, sma
from .pine import change, linreg


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
