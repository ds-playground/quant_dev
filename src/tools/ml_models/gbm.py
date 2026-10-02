"""LightGBM models for the next day: one `binary` model for the direction, one `quantile` model
per tau for the range.

The range models fit the standardized return `r_{t+1} / sigma_t` (sigma_t the EWMA volatility at
t) and scale back by sigma_t: fitted to raw returns they under-covered, spending their splits on
rediscovering the volatility scale (`docs/ml_plan.md`, Phase 1 notes). Each fold early-stops on
the purged last 20% of its training window, then refits on the whole window with the trees that
found. `deterministic=True` on one thread, so a rerun gives the same forecasts to the bit.

LightGBM is the `ml` extra, imported only when a model is fitted.
"""
import numpy as np

from .features import FEATURES
from .split import purged_tail

GBM_PARAMS = dict(num_leaves=15, min_data_in_leaf=50, learning_rate=0.03, feature_fraction=0.8,
                  lambda_l2=1.0, deterministic=True, force_row_wise=True, num_threads=1, seed=0,
                  verbose=-1)


def _lightgbm():
    try:
        import lightgbm
    except ImportError as exc:
        raise ImportError('This needs LightGBM; from the repo root run: pip install -e ".[ml]"') from exc
    return lightgbm


class LightGBMModel:
    """The LightGBM direction and range models, fitted fold by fold by `walk_forward_forecasts`.

    `direction(d, train, test)` returns up-probabilities for the test rows; `quantiles(d, train,
    test, taus)` returns a (test rows, taus) array of return quantiles, sorted along each row so
    no two cross. Both also return a dict of what the fold learnt: the trees kept, and the
    direction model's gain importance per feature.
    """

    name = 'LightGBM'

    def __init__(self, features=FEATURES, params=None, max_trees=300, patience=30, tail=0.2,
                 horizon=1):
        self.features = list(features)
        self.params = dict(GBM_PARAMS, **(params or {}))
        self.max_trees, self.patience, self.tail, self.horizon = max_trees, patience, tail, horizon

    def _fit_predict(self, X, y, train, test, **objective):
        lgb = _lightgbm()
        params = dict(self.params, **objective)
        fit, val = purged_tail(train, self.horizon, self.tail)
        probe = lgb.train(params, lgb.Dataset(X[fit], y[fit]), num_boost_round=self.max_trees,
                          valid_sets=[lgb.Dataset(X[val], y[val])],
                          callbacks=[lgb.early_stopping(self.patience, verbose=False)])
        trees = max(probe.best_iteration, 1)
        model = lgb.train(params, lgb.Dataset(X[train], y[train]), num_boost_round=trees)
        return model.predict(X[test]), trees, model

    def direction(self, d, train, test):
        X = d[self.features].to_numpy()
        p, trees, model = self._fit_predict(X, d['y_up'].to_numpy(), train, test, objective='binary')
        importance = dict(zip(self.features, model.feature_importance('gain')))
        return p, {'trees': trees, 'importance': importance}

    def quantiles(self, d, train, test, taus):
        X = d[self.features].to_numpy()
        sigma = d['vol_ewma'].to_numpy()
        y = d['y_ret'].to_numpy() / sigma
        columns, trees = [], []
        for tau in taus:
            q, n, _ = self._fit_predict(X, y, train, test, objective='quantile', alpha=tau)
            columns.append(q)
            trees.append(n)
        q = np.sort(np.column_stack(columns), axis=1) * sigma[test][:, None]
        return q, {'trees': trees}
