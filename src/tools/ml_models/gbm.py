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
from .targets import TAUS

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
    """What both LightGBM models share: the settings, and one early-stopped, refitted fit.

    `fit(d, train)` trains on the training rows and returns the trained state (the boosters, the
    trees each kept, the gain importance); `predict(state, d, rows)` forecasts any rows of a frame
    with the model's feature columns. A walk-forward fold is the two together, and so is the final
    model saved by `registry.finalize`, so both are made the same way.
    """

    def __init__(self, features=FEATURES, params=None, max_trees=300, patience=30, tail=0.2,
                 horizon=1):
        self.features = list(features)
        self.params = dict(GBM_PARAMS, **(params or {}))
        self.max_trees, self.patience, self.tail, self.horizon = max_trees, patience, tail, horizon

    def settings(self):
        """The keyword arguments that rebuild this model: `type(model)(**model.settings())`."""
        return dict(features=list(self.features), params=dict(self.params),
                    max_trees=self.max_trees, patience=self.patience, tail=self.tail,
                    horizon=self.horizon)

    def _fit(self, X, y, train, **objective):
        lgb = _lightgbm()
        params = dict(self.params, **objective)
        fit, val = purged_tail(train, self.horizon, self.tail)
        probe = lgb.train(params, lgb.Dataset(X[fit], y[fit]), num_boost_round=self.max_trees,
                          valid_sets=[lgb.Dataset(X[val], y[val])],
                          callbacks=[lgb.early_stopping(self.patience, verbose=False)])
        trees = max(probe.best_iteration, 1)
        return lgb.train(params, lgb.Dataset(X[train], y[train]), num_boost_round=trees), trees

    def _importance(self, model):
        return dict(zip(self.features, model.feature_importance('gain')))

    @staticmethod
    def pack(state):
        """`state` with each booster as LightGBM's own text, which reloads to the same forecasts."""
        return {**state, 'boosters': [b.model_to_string() for b in state['boosters']]}

    @staticmethod
    def unpack(packed):
        lgb = _lightgbm()
        return {**packed, 'boosters': [lgb.Booster(model_str=b) for b in packed['boosters']]}


class LightGBMReturnModel(_LightGBM):
    """Tomorrow's return, fitted fold by fold by `walk_forward_forecasts`.

    `quantiles(d, train, test, taus)` returns a (test rows, taus) array of return quantiles,
    sorted along each row so no two cross; the median is the point forecast. It also returns the
    trees each quantile kept and the median model's gain importance per feature.
    """

    name = 'LightGBM'
    target = 'return'

    def fit(self, d, train, taus=TAUS):
        """One quantile model per tau on the training rows: `{'taus', 'boosters', 'trees',
        'importance'}`."""
        X = d[self.features].to_numpy()
        y = d['y_ret'].to_numpy() / d['vol_ewma'].to_numpy()
        boosters, trees, importance = [], [], {}
        for tau in taus:
            model, n = self._fit(X, y, train, objective='quantile', alpha=tau)
            boosters.append(model)
            trees.append(n)
            if tau == 0.5:
                importance = self._importance(model)
        return {'taus': tuple(taus), 'boosters': boosters, 'trees': trees, 'importance': importance}

    def predict(self, state, d, rows):
        """The return quantiles for `rows` of `d`, a (rows, taus) array."""
        X = d[self.features].to_numpy()[rows]
        q = np.column_stack([b.predict(X) for b in state['boosters']])
        return np.sort(q, axis=1) * d['vol_ewma'].to_numpy()[rows][:, None]

    def quantiles(self, d, train, test, taus):
        state = self.fit(d, train, taus)
        return self.predict(state, d, test), {'trees': state['trees'], 'importance': state['importance']}


class LightGBMWDLModel(_LightGBM):
    """Tomorrow's win, draw or loss, fitted fold by fold by `walk_forward_forecasts`.

    `wdl(d, train, test)` returns a (test rows, 3) array of probabilities of a loss, a draw and a
    win, in that order (`baselines.WDL_CLASSES`), with the trees kept and the gain importance.
    """

    name = 'LightGBM'
    target = 'wdl'

    def fit(self, d, train):
        """One multiclass model on the training rows: `{'boosters', 'trees', 'importance'}`."""
        X = d[self.features].to_numpy()
        y = d['y_wdl'].to_numpy() + 1                       # -1, 0, 1 -> classes 0, 1, 2
        model, trees = self._fit(X, y, train, objective='multiclass', num_class=3)
        return {'boosters': [model], 'trees': trees, 'importance': self._importance(model)}

    def predict(self, state, d, rows):
        """The loss, draw and win probabilities for `rows` of `d`, a (rows, 3) array."""
        X = d[self.features].to_numpy()[rows]
        return np.asarray(state['boosters'][0].predict(X)).reshape(len(X), 3)

    def wdl(self, d, train, test):
        state = self.fit(d, train)
        return self.predict(state, d, test), {'trees': state['trees'], 'importance': state['importance']}
