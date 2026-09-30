"""In-memory caches, so a dashboard session does not refetch Yahoo or recompute slow tables."""
import dataclasses
from functools import lru_cache

from src.tools.price_return import (Params, add_rolling_stats, analyze_ticker,
                                    build_historical_analysis, daily_returns_series,
                                    event_probability_table, load_price_data)

# The Params fields that decide which prices load_price_data returns. The analysis fields
# (thresholds, windows, trade_days) do not, so changing them reuses the cached prices.
DATA_FIELDS = ('data_source', 'ticker', 'start_date', 'end_date', 'random_seed', 'sim_drift',
               'sim_vol', 'sim_start_price')
# The further fields the rare-event tables read. prob_max and prob_min only filter them, so
# moving those bounds reuses the table.
HISTORY_FIELDS = DATA_FIELDS + ('trade_days', 'return_thresholds', 'lookback_years',
                                'streak_days')


def _key(p, fields):
    """A hashable key from some Params fields; lists become tuples."""
    return tuple((name, tuple(value) if isinstance(value, list) else value)
                 for name, value in ((name, getattr(p, name)) for name in fields))


def _params(key):
    return Params(**{name: list(value) if isinstance(value, tuple) else value
                     for name, value in key})


@lru_cache(maxsize=32)
def _prices(key):
    return load_price_data(_params(key), verbose=False)


def prices(p):
    """`load_price_data(p)`, cached by the data fields. `end_date` defaults to today, so a new
    day is a new key. Treat the result as read-only; `add_rolling_stats` copies it."""
    return _prices(_key(p, DATA_FIELDS))


def prepared_data(p):
    """Prices plus rolling statistics, as the notebooks compute them."""
    return add_rolling_stats(prices(p), p)


def returns(p):
    """Date-indexed decimal daily returns, the input of the historical and statistics helpers."""
    return daily_returns_series(prepared_data(p))


@lru_cache(maxsize=16)
def _history(key):
    p = _params(key)
    return build_historical_analysis(returns(p), p)


def history(p):
    """`build_historical_analysis`, cached by the fields it reads."""
    return _history(_key(p, HISTORY_FIELDS))


@lru_cache(maxsize=16)
def _events(key, n_days, n_boot):
    p = _params(key)
    return event_probability_table(returns(p), n_days, p, n_boot=n_boot)


def events(p, n_days, n_boot):
    """`event_probability_table` (bootstrap intervals and models), cached: it takes seconds."""
    return _events(_key(p, HISTORY_FIELDS), n_days, n_boot)


@lru_cache(maxsize=64)
def _ticker_result(key, label, drill_n_days):
    p = _params(key)
    p.label = label                 # a display name set by load_ticker_config, not a field
    return analyze_ticker(p, drill_n_days=drill_n_days)


def ticker_result(p, drill_n_days):
    """`analyze_ticker(p)`, cached by every Params field (a config can set any of them) and the
    label. Failures are not cached, so a ticker that failed is retried on the next request."""
    fields = [f.name for f in dataclasses.fields(Params)]
    return _ticker_result(_key(p, fields), getattr(p, 'label', p.ticker), drill_n_days)


_CACHES = (_prices, _history, _events, _ticker_result)


def cache_info():
    return {f.__name__.lstrip('_'): f.cache_info() for f in _CACHES}


def cache_clear():
    for f in _CACHES:
        f.cache_clear()
