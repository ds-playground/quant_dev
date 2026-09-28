"""OHLCV bars from Yahoo Finance or a CSV file, or seeded so tests and notebooks run offline.

Every loader ends with _normalise, so bars from any source have the same shape as make_bars.
"""
import numpy as np
import pandas as pd

PRICES = ['open', 'high', 'low', 'close']


def make_bars(n=500, seed=0, start='2023-01-02', drift=0.0004, vol=0.011):
    """Daily OHLCV bars from a geometric random walk; the same seed gives the same bars."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, vol, n))
    prev_close = np.r_[close[0], close[:-1]]
    open_ = prev_close * (1 + rng.normal(0, vol / 4, n))

    # Extend from the body, not from close alone, so low <= open, close <= high always holds.
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 3, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 3, n)))

    return pd.DataFrame(
        {
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': rng.integers(1_000, 10_000, n).astype(float),
        },
        index=pd.bdate_range(start, periods=n, name='date'),
    )


def _normalise(frame):
    """The output contract every loader ends with, so bars from any source are interchangeable.

    Columns open/high/low/close, then volume when the source has one, then any other columns as
    they came (a TradingView export's indicator plots). Prices and volume are float64; the index
    is a microsecond DatetimeIndex named 'date', oldest first, with no repeated timestamp. A
    price that is NaN after its first value is rejected, because TA-Lib turns everything after
    such a gap into NaN; leading NaN is fine.
    """
    missing = [column for column in PRICES if column not in frame.columns]
    if missing:
        raise ValueError(f'bars need columns {PRICES}; missing {missing}')
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f'bars need a DatetimeIndex, got {type(frame.index).__name__}')
    if frame.index.has_duplicates:
        repeated = frame.index[frame.index.duplicated()]
        raise ValueError(f'{len(repeated)} repeated timestamps, first {repeated[0]}')
    if not frame.index.is_monotonic_increasing:
        raise ValueError('bars must be sorted oldest first')

    core = PRICES + (['volume'] if 'volume' in frame.columns else [])
    bars = frame[core + [column for column in frame.columns if column not in core]].copy()
    bars[core] = bars[core].astype('float64')
    for column in PRICES:
        first = bars[column].first_valid_index()
        if first is None:
            raise ValueError(f'{column} has no values')
        gaps = bars.loc[first:, column].isna()
        if gaps.any():
            raise ValueError(f'{column} is NaN at {gaps.sum()} bars after its first value, '
                             f'first at {gaps.idxmax()}')
    # One resolution for every source (pandas' default, as make_bars has), so frames compare
    # and join cleanly: Unix seconds would otherwise parse to seconds, ISO strings to microseconds.
    bars.index = pd.DatetimeIndex(bars.index, name='date').as_unit('us')
    bars.columns.name = None
    return bars


# How many days back from today Yahoo serves each intraday interval. Older requests come back
# empty (yfinance only logs the error), so load_bars refuses them instead.
YAHOO_LOOKBACK_DAYS = {'1m': 30, '2m': 60, '5m': 60, '15m': 60, '30m': 60, '90m': 60,
                       '60m': 730, '1h': 730}
YAHOO_1M_SPAN_DAYS = 7    # 1m bars also come at most 7 days per request
YAHOO_DAILY_OR_LONGER = ('1d', '5d', '1wk', '1mo', '3mo')


def _naive(when):
    when = pd.Timestamp(when)
    return when.tz_localize(None) if when.tz is not None else when


def _check_yahoo_limits(interval, start, end):
    """Refuse an interval Yahoo does not have, or intraday history older than it keeps."""
    if interval in YAHOO_DAILY_OR_LONGER:
        return
    if interval not in YAHOO_LOOKBACK_DAYS:
        raise ValueError(f'interval must be one of '
                         f'{list(YAHOO_LOOKBACK_DAYS) + list(YAHOO_DAILY_OR_LONGER)}, '
                         f'got {interval!r}')
    today = pd.Timestamp.today().normalize()
    start = _naive(start)
    end = today if end is None else _naive(end)
    days = YAHOO_LOOKBACK_DAYS[interval]
    oldest = today - pd.Timedelta(days=days)
    if start < oldest:
        raise ValueError(f'Yahoo serves {interval} bars for the last {days} days only: start '
                         f'{start.date()} is before {oldest.date()}')
    if interval == '1m' and end - start > pd.Timedelta(days=YAHOO_1M_SPAN_DAYS):
        raise ValueError(f'Yahoo serves 1m bars {YAHOO_1M_SPAN_DAYS} days per request at most: '
                         f'{start.date()} to {end.date()} is longer; split the request')


def load_bars(ticker, start, end=None, interval='1d'):
    """OHLCV bars from Yahoo Finance, in the same shape as make_bars.

    interval is yfinance's: '1m' to '90m' or '1h' intraday, '1d' and longer otherwise. Intraday
    bars keep yfinance's timezone-aware index in exchange time; daily bars are dates. A request
    for intraday history older than Yahoo keeps (YAHOO_LOOKBACK_DAYS) raises rather than
    returning nothing.
    """
    _check_yahoo_limits(interval, start, end)
    import yfinance as yf
    # auto_adjust=False, as in price_return: prices stay as traded rather than
    # dividend-adjusted, because option strikes are set against the traded price.
    raw = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False,
                      progress=False)
    if raw.empty:
        raise ValueError(f'No price data for ticker {ticker!r} from {start}. Yahoo Finance '
                         f'returns an empty frame for unknown symbols, so check the ticker first.')
    if isinstance(raw.columns, pd.MultiIndex):     # yfinance can return a (Price, Ticker) MultiIndex
        raw.columns = raw.columns.get_level_values(0)
    bars = raw[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
    bars.index = pd.DatetimeIndex(bars.index)
    # Yahoo occasionally returns a row with a field missing. Drop it, as load_bars always has,
    # so a real ticker never trips _normalise's gap check.
    return _normalise(bars.astype('float64').dropna())


def read_bars(path, columns=None, daily=False, tz=None):
    """OHLCV bars from a CSV file, in the same shape as load_bars.

    Reads a TradingView "Export chart data" file as it comes: a `time` column in Unix seconds or
    ISO 8601, lower-case prices, an optional `Volume`, and one column per plotted indicator
    series, kept after the prices. Other layouts pass `columns`, a mapping from their names to
    time/open/high/low/close/volume; those names are otherwise matched ignoring case.

    Times without an offset are taken as UTC, and `tz` converts them. `daily=True` keeps only
    each bar's calendar date in `tz`, the date index load_bars gives daily bars.
    """
    # round_trip parses each number to the exact float that was written; pandas' default parser
    # can be one unit off in the last digit, which would blur a parity check against the file.
    frame = pd.read_csv(path, float_precision='round_trip')
    if columns:
        frame = frame.rename(columns=columns)
    known = ['time'] + PRICES + ['volume']
    frame = frame.rename(columns={name: name.lower() for name in frame.columns
                                  if isinstance(name, str) and name.lower() in known})
    if 'time' not in frame.columns:
        raise ValueError(f'{path} has no time column; map one with columns={{...: "time"}}')

    times = frame.pop('time')
    if pd.api.types.is_numeric_dtype(times):
        times = pd.to_datetime(times, unit='s', utc=True)
    else:
        times = pd.to_datetime(times, utc=True, format='ISO8601')
    if tz is not None:
        times = times.dt.tz_convert(tz)
    if daily:
        times = times.dt.tz_localize(None).dt.normalize()
    frame.index = pd.DatetimeIndex(times)
    return _normalise(frame)
