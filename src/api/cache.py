"""An in-memory cache of downloaded prices, so a dashboard session does not refetch Yahoo."""
from functools import lru_cache

from src.tools.price_return import Params, add_rolling_stats, load_price_data

# The Params fields that decide which prices load_price_data returns. The analysis fields
# (thresholds, windows, trade_days) do not, so changing them reuses the cached prices.
DATA_FIELDS = ('data_source', 'ticker', 'start_date', 'end_date', 'random_seed', 'sim_drift',
               'sim_vol', 'sim_start_price')


@lru_cache(maxsize=32)
def _prices(key):
    return load_price_data(Params(**dict(key)), verbose=False)


def prices(p):
    """`load_price_data(p)`, cached by the data fields. `end_date` defaults to today, so a new
    day is a new key. Treat the result as read-only; `add_rolling_stats` copies it."""
    return _prices(tuple((name, getattr(p, name)) for name in DATA_FIELDS))


def prepared_data(p):
    """Prices plus rolling statistics, as the notebooks compute them."""
    return add_rolling_stats(prices(p), p)


cache_info = _prices.cache_info
cache_clear = _prices.cache_clear
