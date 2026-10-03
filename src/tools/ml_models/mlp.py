"""scikit-learn multilayer perceptrons for the next day, two of them, as for LightGBM and the GRU:

- `MLPReturnModel` forecasts tomorrow's return: an `MLPRegressor` on the standardized return
  `r_{t+1} / sigma_t`, whose forecast is the centre; the quantiles add the network's own errors on
  the held-out validation tail, so the median is the point forecast and the outer quantiles are
  the 68% and 90% bands;
- `MLPWDLModel` forecasts tomorrow's win, draw or loss: an `MLPClassifier` (a regressor gives no
  probabilities), with the same settings.

The owner's starting point (2026-10-03): two hidden layers, each with as many neurons as there are
features, ReLU, Adam, and scikit-learn's defaults for everything else. Two departures, which the
plan's hard rules need:

- `random_state` is set, from `seed` and the fold's first test row as for the GRU, so a rerun
  gives the same forecasts (the default, None, draws a new seed each time);
- the features are standardized with the training rows' means and standard deviations
  (scikit-learn's MLP does not scale its inputs, and a scaler fitted on more rows would look
  ahead).

The return network trains on the training rows less the purged last 20% (`tail`); its errors on
that tail set each tau's offset, as the GRU's recalibration does. The W/D/L network trains on all
the training rows. Forecasts are computed here from the trained weights, one row at a time, so a
day's forecast does not depend on which other days are forecast with it, and a saved model needs
only its weights, not a pickled scikit-learn object.

scikit-learn is in the `ml` extra, imported only when a model is fitted.
"""
import warnings

import numpy as np

from .features import FEATURES
from .sequence import quantile_shift
from .split import purged_tail
from .targets import TAUS


def _sklearn():
    try:
        import sklearn.neural_network as nn
    except ImportError as exc:
        raise ImportError('This needs scikit-learn; from the repo root run: pip install -e ".[ml]"') from exc
    return nn


_ACTIVATIONS = {'identity': lambda a: a, 'relu': lambda a: np.maximum(a, 0),
                'tanh': np.tanh, 'logistic': lambda a: 1 / (1 + np.exp(-a))}


class _MLP:
    """What both MLPs share: the settings, the fit, and the forward pass.

    `hidden_layer_sizes` defaults to two layers as wide as the feature list. `params` passes any
    other `MLPRegressor` / `MLPClassifier` keyword argument; left empty, scikit-learn's defaults.
    """

    name = 'MLP'

    def __init__(self, features=FEATURES, hidden_layer_sizes=None, activation='relu',
                 solver='adam', params=None, seed=0, tail=0.2, horizon=1):
        self.features = list(features)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes or (len(self.features),) * 2)
        self.activation, self.solver = activation, solver
        self.params = dict(params or {})
        self.seed, self.tail, self.horizon = seed, tail, horizon

    def settings(self):
        """The keyword arguments that rebuild this model: `type(model)(**model.settings())`."""
        return dict(features=list(self.features), hidden_layer_sizes=list(self.hidden_layer_sizes),
                    activation=self.activation, solver=self.solver, params=dict(self.params),
                    seed=self.seed, tail=self.tail, horizon=self.horizon)

    def _estimator(self, cls, fold_start):
        return cls(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
                   solver=self.solver, random_state=self.seed + int(fold_start), **self.params)

    def _fit(self, cls, d, y, train, fit_rows, fold_start):
        """Standardize on the training rows, fit on `fit_rows`: the weights and what it took."""
        X = d[self.features].to_numpy(dtype=float)
        train = np.asarray(train)
        mu, sd = X[train].mean(axis=0), X[train].std(axis=0) + 1e-12
        net = self._estimator(cls, fold_start)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            net.fit((X[fit_rows] - mu) / sd, y[fit_rows])
        converged = not any('converge' in str(w.message).lower() for w in caught)
        return {'mu': mu, 'sd': sd, 'coefs': [w.copy() for w in net.coefs_],
                'intercepts': [b.copy() for b in net.intercepts_], 'activation': net.activation,
                'out_activation': net.out_activation_, 'n_iter': int(net.n_iter_),
                'converged': converged}, net

    def _forward(self, state, d, rows):
        """The network's output for `rows` of `d`, one row at a time, as scikit-learn computes it."""
        X = (d[self.features].to_numpy(dtype=float)[np.asarray(rows)] - state['mu']) / state['sd']
        act = _ACTIVATIONS[state['activation']]
        last = len(state['coefs']) - 1
        out = []
        for x in X:
            a = x[None, :]
            for i, (w, b) in enumerate(zip(state['coefs'], state['intercepts'])):
                a = a @ w + b
                if i < last:
                    a = act(a)
            if state['out_activation'] == 'softmax':
                a = np.exp(a - a.max(axis=1)[:, None])
                a = a / a.sum(axis=1)[:, None]
            out.append(a)
        return np.vstack(out) if out else np.empty((0, len(state['intercepts'][-1])))

    def _info(self, state):
        return {'n_iter': state['n_iter'], 'converged': state['converged']}

    @staticmethod
    def pack(state):
        """The state as it is saved: already plain numpy arrays and numbers."""
        return state

    @staticmethod
    def unpack(packed):
        return packed


class MLPReturnModel(_MLP):
    """Tomorrow's return, fitted fold by fold by `walk_forward_forecasts`.

    `quantiles(d, train, test, taus)` returns a (test rows, taus) array of return quantiles: the
    network's forecast of `r_{t+1} / sigma_t` plus the tau-quantile of its errors on the
    validation tail (`shift`), scaled back by sigma_t; the median is the point forecast.
    `fit(d, train)` and `predict(state, d, rows)` are the two halves.
    """

    target = 'return'

    def fit(self, d, train, taus=TAUS, fold_start=None):
        nn = _sklearn()
        train = np.asarray(train)
        fold_start = train[-1] + self.horizon if fold_start is None else fold_start
        y = d['y_ret'].to_numpy() / d['vol_ewma'].to_numpy()
        fit_rows, val = purged_tail(train, self.horizon, self.tail)
        state, _ = self._fit(nn.MLPRegressor, d, y, train, fit_rows, fold_start)
        centre = self._forward(state, d, val)[:, 0]
        state['shift'] = quantile_shift(np.repeat(centre[:, None], len(taus), axis=1), y[val], taus)
        state['taus'] = tuple(taus)
        return state

    def predict(self, state, d, rows):
        """The return quantiles for `rows` of `d`, a (rows, taus) array."""
        centre = self._forward(state, d, rows)[:, :1]
        q = np.sort(centre + state['shift'], axis=1)
        return q * d['vol_ewma'].to_numpy()[np.asarray(rows)][:, None]

    def quantiles(self, d, train, test, taus):
        state = self.fit(d, train, taus, fold_start=test[0])
        return self.predict(state, d, test), {**self._info(state), 'shift': state['shift']}


class MLPWDLModel(_MLP):
    """Tomorrow's win, draw or loss, fitted fold by fold by `walk_forward_forecasts`.

    `wdl(d, train, test)` returns a (test rows, 3) array of probabilities of a loss, a draw and a
    win, in that order (`baselines.WDL_CLASSES`, which is the classifier's sorted `classes_`).
    `fit(d, train)` and `predict(state, d, rows)` are the two halves.
    """

    target = 'wdl'

    def fit(self, d, train, fold_start=None):
        nn = _sklearn()
        train = np.asarray(train)
        fold_start = train[-1] + self.horizon if fold_start is None else fold_start
        y = d['y_wdl'].to_numpy().astype(int)
        state, net = self._fit(nn.MLPClassifier, d, y, train, train, fold_start)
        if list(net.classes_) != [-1, 0, 1]:
            raise ValueError(f'the training rows hold the outcomes {list(net.classes_)}, not all of '
                             'loss, draw and win')
        return state

    def predict(self, state, d, rows):
        """The loss, draw and win probabilities for `rows` of `d`, a (rows, 3) array."""
        return self._forward(state, d, rows)

    def wdl(self, d, train, test):
        state = self.fit(d, train, fold_start=test[0])
        return self.predict(state, d, test), self._info(state)
