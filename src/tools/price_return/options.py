"""Option-sizing helpers: price ranges from volatility, move probabilities, expected P&L.

Ported from the notebook-era `basic.py` (`projected_min_max`, `accepted_min_max`, `sd_and_cond`,
`profit_estimate`). These return tables instead of printing. Returns and volatilities are decimal
fractions (0.01 == 1%), like the `PCT Change` columns from `add_rolling_stats`.
"""

import math

import numpy as np
import pandas as pd

from .params import _params

DIRECTIONS = ('above', 'below', 'exceeding')


def _round_out(value, tick, up):
    """Round away from the centre to a multiple of `tick`, as basic.py's ceil/floor(x * 10) / 10."""
    scale = 1 / tick
    if abs(scale - round(scale)) < 1e-9:    # 0.1, 0.25, 0.01: multiply by a whole number, exactly
        scale = round(scale)
    return (math.ceil(value * scale) if up else math.floor(value * scale)) / scale


def price_range(price, daily_vol, days=None, hours=None, session_hours=6.5, tick=None):
    """The one-standard-deviation price band `days` ahead, or over the `hours` left in a session.

    Volatility scales with the square root of time: `daily_vol * sqrt(days)`, or
    `daily_vol * sqrt(hours / session_hours)` within a trading day. The band is lognormal,
    `price * exp(±vol)`. `tick` rounds the band outwards to a price increment (`tick=0.1` gives
    basic.py's rounding). `daily_vol` is a decimal, e.g. `latest_snapshot(df)['daily_std_dev']`.
    """
    if (days is None) == (hours is None):
        raise ValueError('give exactly one of days or hours')
    if days is not None:
        if days < 0:
            raise ValueError(f'days must be >= 0, got {days}')
        vol, horizon = daily_vol * math.sqrt(days), f'{days} days'
    else:
        if not 0 <= hours <= session_hours:
            raise ValueError(f'hours must be between 0 and session_hours ({session_hours}), '
                             f'got {hours}')
        vol, horizon = daily_vol * math.sqrt(hours / session_hours), f'{hours} hours'

    low, high = price * math.exp(-vol), price * math.exp(vol)
    if tick is not None:
        low, high = _round_out(low, tick, up=False), _round_out(high, tick, up=True)
    return pd.Series({'price': price, 'daily_vol': daily_vol, 'horizon': horizon, 'vol': vol,
                      'low': low, 'high': high})


def move_probabilities(df, p=None, horizons=(1, 5, 10), scaled=None, fixed=None):
    """How often the n-day return moved beyond a threshold, over the trailing `trade_days` windows.

    `df` comes from `add_rolling_stats`, which must include each horizon in `roll_windows`. The
    threshold for each horizon comes from up to three methods:

      scaled - a daily threshold grown with the square root of time: `scaled * sqrt(n)`
      actual - the latest rolling standard deviation of the n-day return (`PCT Change n STD`)
      fixed  - the same constant for every horizon

    `actual` is always included; `scaled` and `fixed` only when given. For each threshold, a
    window counts as `above` (return > threshold), `below` (< -threshold) or `exceeding`
    (|return| > threshold). `prob` is `count / n_obs`, where `n_obs` is the number of complete
    windows among the trailing `trade_days` rows.
    """
    p = _params(p)
    missing = [n for n in horizons if f'PCT Change {n}' not in df.columns]
    if missing:
        raise KeyError(f'no PCT Change column for horizon(s) {missing}; add them to '
                       f'Params.roll_windows ({p.roll_windows}) and rerun add_rolling_stats')

    rows = []
    for n in horizons:
        moves = df[f'PCT Change {n}'].iloc[-p.trade_days:]
        complete = moves.dropna()
        thresholds = {'actual': df[f'PCT Change {n} STD'].iloc[-1]}
        if scaled is not None:
            thresholds['scaled'] = scaled * math.sqrt(n)
        if fixed is not None:
            thresholds['fixed'] = fixed
        for method in ('scaled', 'actual', 'fixed'):
            if method not in thresholds:
                continue
            threshold = thresholds[method]
            hits = {'above': complete > threshold,
                    'below': complete < -threshold,
                    'exceeding': complete.abs() > threshold}
            for direction in DIRECTIONS:
                count = int(hits[direction].sum())
                rows.append({'horizon': n, 'method': method, 'threshold': threshold,
                             'direction': direction, 'count': count, 'n_obs': len(complete),
                             'prob': count / len(complete) if len(complete) else np.nan})
    return pd.DataFrame(rows)


def expected_pnl(probabilities, pnl, contract_size=1):
    """Expected P&L per contract: `(1 - prob) * win + prob * loss`, times `contract_size`.

    `probabilities` is a `move_probabilities` table, where `prob` is the chance of the move that
    loses. `pnl` maps horizon -> direction -> [win, loss], the shape the old notebook grids used:

        {1: {'above': [0.15, -0.3], 'below': [0.15, -0.3], 'exceeding': [0.2, -0.3]}, ...}

    Rows whose horizon or direction is not in `pnl` are dropped.
    """
    grid = pd.DataFrame([{'horizon': n, 'direction': direction, 'win': win, 'loss': loss}
                         for n, by_direction in pnl.items()
                         for direction, (win, loss) in by_direction.items()])
    out = probabilities.merge(grid, on=['horizon', 'direction'], how='inner')
    out['expected_pnl'] = ((1 - out['prob']) * out['win'] + out['prob'] * out['loss']) * contract_size
    return out
