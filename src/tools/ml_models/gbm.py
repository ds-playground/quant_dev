"""LightGBM models for the next day, two of them (the owner's choice, 2026-10-03):

- `LightGBMReturnModel` forecasts tomorrow's return: one `quantile` model per tau, whose median is
  the point forecast and whose outer quantiles are the 68% and 90% bands;
- `LightGBMWDLModel` forecasts tomorrow's win, draw or loss: one `multiclass` model.

The return model fits the standardized return `r_{t+1} / sigma_t` (sigma_t the EWMA volatility at
t) and scales back by sigma_t: fitted to raw returns it under-covered, spending its splits on
rediscovering the volatility scale (`docs/ml_plan.md`, Phase 1 notes). Each fit early-stops on the
purged last 20% of the training window, then refits on the whole window with the trees that
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


class _LightGBM:
    """What both LightGBM models share: the settings and one early-stopped, refitted fit."""

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

    def _importance(self, model):
        return dict(zip(self.features, model.feature_importance('gain')))


class LightGBMReturnModel(_LightGBM):
    """Tomorrow's return, fitted fold by fold by `walk_forward_forecasts`.

    `quantiles(d, train, test, taus)` returns a (test rows, taus) array of return quantiles,
    sorted along each row so no two cross; the median is the point forecast. It also returns the
    trees each quantile kept and the median model's gain importance per feature.
    """

    name = 'LightGBM'
    target = 'return'

    def quantiles(self, d, train, test, taus):
        X = d[self.features].to_numpy()
        sigma = d['vol_ewma'].to_numpy()
        y = d['y_ret'].to_numpy() / sigma
        columns, trees, importance = [], [], {}
        for tau in taus:
            q, n, model = self._fit_predict(X, y, train, test, objective='quantile', alpha=tau)
            columns.append(q)
            trees.append(n)
            if tau == 0.5:
                importance = self._importance(model)
        q = np.sort(np.column_stack(columns), axis=1) * sigma[test][:, None]
        return q, {'trees': trees, 'importance': importance}


class LightGBMWDLModel(_LightGBM):
    """Tomorrow's win, draw or loss, fitted fold by fold by `walk_forward_forecasts`.

    `wdl(d, train, test)` returns a (test rows, 3) array of probabilities of a loss, a draw and a
    win, in that order (`baselines.WDL_CLASSES`), with the trees kept and the gain importance.
    """

    name = 'LightGBM'
    target = 'wdl'

    def wdl(self, d, train, test):
        X = d[self.features].to_numpy()
        y = d['y_wdl'].to_numpy() + 1                       # -1, 0, 1 -> classes 0, 1, 2
        p, trees, model = self._fit_predict(X, y, train, test, objective='multiclass', num_class=3)
        return np.asarray(p).reshape(len(test), 3), {'trees': trees, 'importance': self._importance(model)}
