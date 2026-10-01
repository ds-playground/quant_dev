"""Synthetic daily bars: market-shaped price series generated from code, for offline work.

Each ticker is a seeded GARCH(1,1) process with Student-t shocks, so its returns have what real
ones have and an i.i.d. normal series lacks: fat tails, and volatility clustering (calm and wild
spells). A scheduled sell-off gives each series a drawdown worth charting, and SYN-OIL settles
below zero for one day, the case `load_price_data` has to handle for WTI crude in April 2020.

Nothing here is market data, and nothing is stored: the bars are generated on first use, from a
seed derived from the ticker name, so every machine gets the same series. The tests, the
end-to-end checks and the documentation's screenshots all use them. (NumPy does not promise the
same random stream across major versions, so a NumPy upgrade may change the numbers, never their
properties.)
"""
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

END = '2026-09-30'          # the last bar of every series, so results do not drift with the clock


@dataclass(frozen=True)
class Profile:
    label: str
    drift: float                # mean daily return, %
    vol: float                  # long-run daily volatility, %
    tail_df: float              # Student-t degrees of freedom: lower is fatter-tailed
    alpha: float = 0.16         # GARCH: how strongly yesterday's shock raises today's variance
    beta: float = 0.82          # GARCH: how much of yesterday's variance carries over
    start_price: float = 100.0
    start: str = '2016-01-04'
    # (first day, last day, volatility multiple, mean daily return % while it lasts): a sell-off.
    stress: tuple = (('2020-02-20', '2020-03-23', 3.0, -1.0),)
    max_move: float = 0.15      # the largest daily move either way, as a fraction
    negative_close: str | None = None   # one day settling below zero


PROFILES = {
    'SYN-INDEX': Profile('Synthetic equity index', drift=0.045, vol=1.05, tail_df=4.5,
                         start_price=2000),
    'SYN-TECH':  Profile('Synthetic tech index', drift=0.065, vol=1.45, tail_df=4.5,
                         start_price=4500),
    'SYN-GOLD':  Profile('Synthetic gold', drift=0.03, vol=0.9, tail_df=5, alpha=0.10, beta=0.88,
                         start_price=1100, stress=(('2020-03-09', '2020-03-19', 2.0, -0.8),)),
    'SYN-FX':    Profile('Synthetic currency pair', drift=0.0, vol=0.45, tail_df=6, alpha=0.08,
                         beta=0.90, start_price=1.10,
                         stress=(('2022-03-01', '2022-09-28', 1.6, -0.08),)),
    'SYN-OIL':   Profile('Synthetic crude oil', drift=0.02, vol=2.3, tail_df=3.5, start_price=40,
                         negative_close='2020-04-20', max_move=0.35,
                         stress=(('2020-03-02', '2020-04-17', 2.5, -1.2),)),
    'SYN-LEV':   Profile('Synthetic 2x leveraged fund', drift=0.06, vol=3.6, tail_df=4,
                         start_price=20, start='2022-08-10', stress=(), max_move=0.30),
}


def synthetic_tickers():
    """The synthetic tickers, in the order of the synthetic ticker config."""
    return list(PROFILES)


def synthetic_bars(ticker):
    """`ticker`'s daily bars: a Date-indexed Open/High/Low/Close/Volume frame, business days from
    its start to END, in the layout of a saved data/local file."""
    if ticker not in PROFILES:
        raise ValueError(f'No synthetic data for ticker {ticker!r}. Synthetic tickers: '
                         f'{", ".join(PROFILES)}.')
    return _bars(ticker).copy()


@lru_cache(maxsize=None)
def _bars(ticker):
    pr = PROFILES[ticker]
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    dates = pd.bdate_range(pr.start, END)
    n = len(dates)

    # Student-t shocks scaled to unit variance, through GARCH(1,1) with a long-run variance of vol².
    z = rng.standard_t(pr.tail_df, n) / np.sqrt(pr.tail_df / (pr.tail_df - 2))
    long_run = (pr.vol / 100) ** 2
    omega = long_run * (1 - pr.alpha - pr.beta)
    multiple, mean = np.ones(n), np.full(n, pr.drift / 100)
    for first, last, vol_multiple, daily_mean in pr.stress:
        span = (dates >= first) & (dates <= last)
        multiple[span], mean[span] = vol_multiple, daily_mean / 100
    returns, variance = np.empty(n), long_run
    for t in range(n):
        returns[t] = mean[t] + np.sqrt(variance) * multiple[t] * z[t]
        # The sell-off's extra volatility is imposed, so it does not feed the GARCH recursion.
        shock = (returns[t] - mean[t]) / multiple[t]
        variance = omega + pr.alpha * shock ** 2 + pr.beta * variance
    returns = np.clip(returns, -pr.max_move, pr.max_move)

    close = pr.start_price * np.cumprod(1 + returns)
    close[0] = pr.start_price
    open_ = np.r_[pr.start_price, close[:-1]] * (1 + rng.normal(0, 0.25, n) * np.abs(returns))
    wicks = np.abs(rng.normal(0, 0.5, (2, n))) * (pr.vol / 100)
    high = np.maximum(open_, close) * (1 + wicks[0])
    low = np.minimum(open_, close) * (1 - wicks[1])
    if pr.negative_close:
        i = dates.get_loc(pd.Timestamp(pr.negative_close))
        close[i] = -0.9 * abs(close[i - 1])                       # one settle below zero, then back
        high[i], low[i] = max(open_[i], close[i - 1]), close[i] * 1.05
        open_[i + 1] = 0.98 * close[i + 1]
        low[i + 1] = min(low[i + 1], open_[i + 1], close[i + 1])
    volume = np.round(1e6 * np.exp(rng.normal(0, 0.3, n)) * (1 + 20 * np.abs(returns)))
    bars = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close,
                         'Volume': volume.astype('int64')}, index=pd.Index(dates, name='Date'))
    bars[['Open', 'High', 'Low', 'Close']] = bars[['Open', 'High', 'Low', 'Close']].round(6)
    bars.index.freq = None                       # as a saved file reads back: just dates
    return bars
