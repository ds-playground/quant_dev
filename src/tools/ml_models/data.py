"""Daily bars for the ML models, and their close-to-close returns.

Bars come from two places only: the synthetic tickers, generated offline (`synthetic_bars`), and
live Yahoo data the owner has saved in the git-ignored `data/local` (`save_local`, `save_all`,
or the dashboard's Download all). Nothing here downloads: a symbol that is not saved raises, with
the command that saves it.
"""
import pandas as pd

from src.tools.price_return.store import local_path, read_local
from src.tools.price_return.synthetic import synthetic_bars, synthetic_tickers

PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close']


def ticker_bars(ticker, source=None, directory=None):
    """`ticker`'s daily Open/High/Low/Close(/Volume) bars, Date-indexed, oldest first.

    `source` is 'synthetic' (generated, offline) or 'local' (saved in `data/local`, or in
    `directory`). Left as None, a synthetic ticker's name picks 'synthetic' and anything else
    'local'.
    """
    if source is None:
        source = 'synthetic' if ticker in synthetic_tickers() else 'local'
    if source == 'synthetic':
        bars = synthetic_bars(ticker)
    elif source == 'local':
        if not local_path(ticker, directory).is_file():
            raise FileNotFoundError(
                f'{ticker!r} is not saved in {local_path(ticker, directory).parent}. Save it first '
                f'with save_local({ticker!r}) from src.tools.price_return, or the dashboard\'s '
                f'Download all.')
        bars = read_local(ticker, directory)
    else:
        raise ValueError(f"source must be 'synthetic' or 'local', got {source!r}")
    missing = [c for c in PRICE_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f'{ticker!r} bars have no {", ".join(missing)} column')
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError(f'{ticker!r} bars are not in date order without repeats')
    return bars


def bar_returns(bars):
    """Close-to-close decimal returns, NaN for the first bar and for any return that spans a
    non-positive close (SYN-OIL's 2020-04-20, WTI's April 2020), as `load_price_data` drops them."""
    close = bars['Close'].astype(float)
    positive = (close > 0) & (close.shift(1) > 0)
    return close.pct_change().where(positive).rename('return')
