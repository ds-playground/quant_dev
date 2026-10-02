"""What a model has to beat, forecast out of sample fold by fold.

Each baseline uses only the fold's training rows and values known at the test row's close:

- direction: the base rate, the up-share of the training labels;
- `constant`: the training window's empirical return quantiles, the same width every day;
- `price_range`: `options.price_range`'s lognormal band, `exp(z * sigma) - 1`, with sigma the
  trailing 250-bar volatility, as `latest_snapshot` feeds it;
- `vol_20`: the training mean plus z times the 20-bar volatility;
- `ewma_std_q`: the EWMA volatility at t times the training quantiles of `r_{t+1} / sigma_t`.
  On fat-tailed returns it covers as it claims, where the normal-quantile bands over-cover
  (`docs/ml_plan.md`, *Evaluation*). It is the band the models are compared with.
"""
from statistics import NormalDist

import numpy as np
import pandas as pd

from .targets import TAUS

BANDS = ('constant', 'price_range', 'vol_20', 'ewma_std_q')


def direction_baseline(d, folds):
    """The base rate for every test row of `folds`: a Series named `p_base`."""
    parts = [pd.Series(d['y_up'].iloc[train].mean(), index=d.index[test]) for train, test in folds]
    return pd.concat(parts).rename('p_base')


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
