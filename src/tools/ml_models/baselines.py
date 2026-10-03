"""What a model has to beat, forecast out of sample fold by fold.

Each baseline uses only the fold's training rows and values known at the test row's close:

- `constant`: the training window's empirical return quantiles, the same width every day;
- `price_range`: `options.price_range`'s lognormal band, `exp(z * sigma) - 1`, with sigma the
  trailing 250-bar volatility, as `latest_snapshot` feeds it;
- `vol_20`: the training mean plus z times the 20-bar volatility;
- `ewma_std_q`: the EWMA volatility at t times the training quantiles of `r_{t+1} / sigma_t`.
  On fat-tailed returns it covers as it claims, where the normal-quantile bands over-cover
  (`docs/ml_plan.md`, *Evaluation*). It is the band the models are compared with.

For tomorrow's win, draw or loss, two baselines:

- `frequencies`: the training window's shares of wins, draws and losses;
- `volatility`: the same shares, but of the training window's standardized returns scaled by the
  EWMA volatility at t: the chance that tomorrow clears the threshold given how volatile it is
  now. With a fixed threshold, draws are commoner in calm spells, so this baseline uses
  volatility and nothing about direction.
"""
from statistics import NormalDist

import numpy as np
import pandas as pd

from .targets import TAUS

BANDS = ('constant', 'price_range', 'vol_20', 'ewma_std_q')
WDL_CLASSES = (-1, 0, 1)                     # loss, draw, win: the column order of every forecast
WDL_BASELINES = ('frequencies', 'volatility')


def range_baselines(d, folds, taus=TAUS):
    """Every baseline's return quantile for every test row: columns `(band, tau)` for `BANDS`."""
    z = {tau: NormalDist().inv_cdf(tau) for tau in taus}
    parts = []
    for train, test in folds:
        tr, te = d.iloc[train], d.iloc[test]
        standardized = tr['y_ret'] / tr['vol_ewma']
        cols = {}
        for tau in taus:
            cols[('constant', tau)] = np.full(len(te), tr['y_ret'].quantile(tau))
            cols[('price_range', tau)] = np.exp(z[tau] * te['vol_250']).to_numpy() - 1
            cols[('vol_20', tau)] = (tr['y_ret'].mean() + z[tau] * te['vol_20']).to_numpy()
            cols[('ewma_std_q', tau)] = (te['vol_ewma'] * standardized.quantile(tau)).to_numpy()
        parts.append(pd.DataFrame(cols, index=te.index))
    out = pd.concat(parts)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=['band', 'tau'])
    return out[[(band, tau) for band in BANDS for tau in taus]]


def _shares(counts, n):
    """Class shares with half a count added to each, so no class gets probability 0."""
    return (np.asarray(counts, dtype=float) + 0.5) / (n + 1.5)


def wdl_baselines(d, folds, threshold=None):
    """Both win/draw/loss baselines for every test row: columns `(baseline, class)`, each row's
    three probabilities summing to 1. `threshold` defaults to the one `dataset` was built with."""
    threshold = d.attrs.get('wdl_threshold') if threshold is None else threshold
    if threshold is None:
        raise ValueError('no win/draw/loss threshold: pass one, or build d with dataset()')
    parts = []
    for train, test in folds:
        tr, te = d.iloc[train], d.iloc[test]
        y = tr['y_wdl'].to_numpy()
        freq = _shares([np.sum(y == c) for c in WDL_CLASSES], len(y))
        z = np.sort((tr['y_ret'] / tr['vol_ewma']).to_numpy())
        cut = threshold / te['vol_ewma'].to_numpy()          # the threshold in standardized units
        wins = len(z) - np.searchsorted(z, cut, side='right')    # z > cut
        losses = np.searchsorted(z, -cut, side='left')           # z < -cut
        draws = len(z) - wins - losses
        vol = np.column_stack([_shares([l, dr, w], len(z)) for l, dr, w in zip(losses, draws, wins)]).T
        cols = {('frequencies', c): np.full(len(te), freq[i]) for i, c in enumerate(WDL_CLASSES)}
        cols.update({('volatility', c): vol[:, i] for i, c in enumerate(WDL_CLASSES)})
        parts.append(pd.DataFrame(cols, index=te.index))
    out = pd.concat(parts)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=['baseline', 'class'])
    return out[[(b, c) for b in WDL_BASELINES for c in WDL_CLASSES]]
