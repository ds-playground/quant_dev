"""GRU sequence models for the next day, two of them (the owner's choice, 2026-10-03), each reading
the last `window` rows of its inputs:

- `GRUReturnModel` forecasts tomorrow's return: one output per tau, whose median is the point
  forecast and whose outer quantiles are the 68% and 90% bands;
- `GRUWDLModel` forecasts tomorrow's win, draw or loss: three logits.

The owner chose a GRU over an LSTM or a temporal CNN (`docs/ml_plan.md`, *Models*). Per fold, the
features are standardized with the training window's means and standard deviations; the range
network fits the standardized return `r_{t+1} / sigma_t`, as LightGBM's does, and is scaled back.
Training: Adam with a small weight decay, shuffled batches, early stopping on the purged last 20%
of the training window, keeping the best epoch. Each fold is seeded from its first test row and
runs on CPU with deterministic algorithms, so a rerun gives the same forecasts; each day is
forecast on its own, so its forecast does not depend on which other days are forecast with it.

The return model's quantiles are recalibrated by default (the owner's choice, `docs/ml_plan.md`, Phase 4):
each tau's forecast is shifted by the tau-quantile of the network's own errors on the validation
tail, the rows early stopping holds out inside the training window. Uncorrected, the network's
intervals were too narrow, and much more so when retrained less often.

PyTorch is the `ml` extra, imported only when a model is fitted.
"""
from contextlib import contextmanager

import numpy as np

from .features import FEATURES

# What the GRU reads each day. The owner keeps all three until a model is trained on real data
# (docs/ml_plan.md, Phase 4); 'returns_range' is the provisional default.
GRU_INPUTS = {
    'returns_range': ['ret_0', 'tr_close'],      # each day's return and its true range
    'returns': ['ret_0'],                        # each day's return
    'features': list(FEATURES),                  # the features LightGBM reads
}
from .split import purged_tail
from .targets import TAUS


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError('This needs PyTorch; from the repo root run: pip install -e ".[ml]" (for a '
                          'CPU-only build, first pip install torch --index-url '
                          'https://download.pytorch.org/whl/cpu)') from exc
    return torch


def windows(X, rows, window):
    """The `window` rows of `X` ending at each of `rows`: shape (len(rows), window, columns).
    Row t's window is rows t - window + 1 to t, never later."""
    rows = np.asarray(rows)
    if len(rows) and rows.min() < window - 1:
        raise ValueError(f'row {rows.min()} has fewer than {window} rows behind it')
    return X[rows[:, None] + np.arange(-window + 1, 1)[None, :]]


def quantile_shift(q, y, taus):
    """Per tau, the shift that makes the forecasts `q[:, i]` cover their share of `y`: the
    tau-quantile of `y - q[:, i]`. Added to the forecasts, a share of about tau of `y` lies below."""
    q, y = np.asarray(q, dtype=float), np.asarray(y, dtype=float)
    return np.array([np.quantile(y - q[:, i], tau) for i, tau in enumerate(taus)])


class _GRU:
    """What both GRU models share: the inputs, the network and its training.

    `features` names one of `GRU_INPUTS` ('returns_range', 'returns', 'features') or lists the
    dataset columns to read; each is read over the last `window` rows. Each forecast also returns
    the epochs trained and the best one, whose weights are kept. `threads` sets PyTorch's CPU
    threads for the fit (None leaves PyTorch's default); give 1 when running tickers in parallel
    processes.
    """

    def __init__(self, features='returns_range', window=60, hidden=32, batch=128, max_epochs=40,
                 patience=5, learning_rate=1e-3, weight_decay=1e-4, tail=0.2, horizon=1, seed=0,
                 threads=None):
        self.inputs = features if isinstance(features, str) else 'custom'
        self.features = list(GRU_INPUTS[features] if isinstance(features, str) else features)
        self.name = f'GRU ({self.inputs})'
        self.window, self.hidden, self.batch = window, hidden, batch
        self.max_epochs, self.patience = max_epochs, patience
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.tail, self.horizon, self.seed, self.threads = tail, horizon, seed, threads

    def _net(self, torch, n_in, n_out):
        class GRUNet(torch.nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.gru = torch.nn.GRU(n_in, hidden, batch_first=True)
                self.head = torch.nn.Linear(hidden, n_out)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1])

        return GRUNet(self.hidden)

    def settings(self):
        """The keyword arguments that rebuild this model: `type(model)(**model.settings())`."""
        return dict(features=self.inputs if self.inputs != 'custom' else list(self.features),
                    window=self.window, hidden=self.hidden, batch=self.batch,
                    max_epochs=self.max_epochs, patience=self.patience,
                    learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                    tail=self.tail, horizon=self.horizon, seed=self.seed, threads=self.threads)

    @contextmanager
    def _torch_session(self):
        """PyTorch with deterministic algorithms and this model's threads, restored afterwards."""
        torch = _torch()
        deterministic = torch.are_deterministic_algorithms_enabled()
        threads = torch.get_num_threads()
        torch.use_deterministic_algorithms(True)
        if self.threads:
            torch.set_num_threads(self.threads)
        try:
            yield torch
        finally:
            torch.use_deterministic_algorithms(deterministic)
            torch.set_num_threads(threads)

    def _fit(self, d, y, train, n_out, loss_fn, fold_start):
        """Train on the training rows: the best epoch's weights (as numpy arrays) and the
        standardization, which is all `_raw` needs to forecast. Seeded from `fold_start`, the
        fold's first test row, so this fold trains the same whenever it is fitted, whichever rows
        are then forecast."""
        with self._torch_session() as torch:
            seed = self.seed + int(fold_start)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            X = d[self.features].to_numpy()
            train = np.asarray(train)
            train = train[train >= self.window - 1]       # rows with a full window behind them
            mu, sd = X[train].mean(axis=0), X[train].std(axis=0) + 1e-12
            Xs = ((X - mu) / sd).astype(np.float32)
            y = np.asarray(y, dtype=np.float32)
            fit, val = purged_tail(train, self.horizon, self.tail)
            net = self._net(torch, X.shape[1], n_out)
            opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate,
                                   weight_decay=self.weight_decay)
            xv = torch.from_numpy(windows(Xs, val, self.window))
            yv = torch.from_numpy(y[val])
            best, best_state, best_epoch, wait, epochs = np.inf, None, 0, 0, 0
            for epochs in range(1, self.max_epochs + 1):
                net.train()
                for batch in np.array_split(rng.permutation(fit), max(1, len(fit) // self.batch)):
                    opt.zero_grad()
                    loss = loss_fn(net(torch.from_numpy(windows(Xs, batch, self.window))),
                                   torch.from_numpy(y[batch]))
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = loss_fn(net(xv), yv).item()
                if v < best - 1e-6:
                    best, best_epoch, wait = v, epochs, 0
                    best_state = {k: x.detach().numpy().copy() for k, x in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        return {'n_in': X.shape[1], 'n_out': n_out, 'mu': mu, 'sd': sd, 'weights': best_state,
                'epochs': epochs, 'best_epoch': best_epoch}

    def _raw(self, state, d, rows):
        """The network's raw outputs for `rows` of `d`, from a trained `state`."""
        with self._torch_session() as torch:
            net = self._net(torch, state['n_in'], state['n_out'])
            net.load_state_dict({k: torch.from_numpy(v) for k, v in state['weights'].items()})
            net.eval()
            Xs = ((d[self.features].to_numpy() - state['mu']) / state['sd']).astype(np.float32)
            with torch.no_grad():
                # One row at a time: batched, the kernels round differently with the batch's
                # size, so a day's forecast would depend on which other days were forecast with it.
                out = [net(torch.from_numpy(windows(Xs, [row], self.window))).numpy()
                       for row in np.asarray(rows)]
        return np.vstack(out).astype(float) if out else np.empty((0, state['n_out']))

    def _info(self, state):
        return {'epochs': state['epochs'], 'best_epoch': state['best_epoch']}

    @staticmethod
    def pack(state):
        """The state as it is saved: already plain numpy arrays and numbers."""
        return state

    @staticmethod
    def unpack(packed):
        return packed


class GRUReturnModel(_GRU):
    """Tomorrow's return, fitted fold by fold by `walk_forward_forecasts`.

    `quantiles(d, train, test, taus)` returns a (test rows, taus) array of return quantiles,
    sorted along each row; the median is the point forecast. With `recalibrate` (the default)
    each tau is shifted by the network's own errors on the validation tail, reported as `shift`.
    `fit(d, train)` and `predict(state, d, rows)` are the two halves, as for LightGBM.
    """

    target = 'return'

    def __init__(self, *args, recalibrate=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.recalibrate = recalibrate

    def settings(self):
        return {**super().settings(), 'recalibrate': self.recalibrate}

    def fit(self, d, train, taus=TAUS, fold_start=None):
        """Train on the training rows, seeded from `fold_start` (default: the first row a fold
        on these training rows would test), and with `recalibrate`, find the shift."""
        torch = _torch()
        t = torch.tensor(taus, dtype=torch.float32)

        def pinball(p, y):
            u = y.unsqueeze(1) - p
            return torch.maximum(t * u, (t - 1) * u).mean()

        train = np.asarray(train)
        fold_start = train[-1] + self.horizon if fold_start is None else fold_start
        y = d['y_ret'].to_numpy() / d['vol_ewma'].to_numpy()
        state = {**self._fit(d, y, train, len(taus), pinball, fold_start), 'taus': tuple(taus)}
        if self.recalibrate:
            _, val = purged_tail(train[train >= self.window - 1], self.horizon, self.tail)
            state['shift'] = quantile_shift(self._raw(state, d, val), y[val], taus)
        return state

    def predict(self, state, d, rows):
        """The return quantiles for `rows` of `d`, a (rows, taus) array."""
        out = self._raw(state, d, rows)
        if 'shift' in state:
            out = out + state['shift']
        return np.sort(out, axis=1) * d['vol_ewma'].to_numpy()[np.asarray(rows)][:, None]

    def quantiles(self, d, train, test, taus):
        state = self.fit(d, train, taus, fold_start=test[0])
        info = self._info(state)
        if 'shift' in state:
            info['shift'] = state['shift']
        return self.predict(state, d, test), info


class GRUWDLModel(_GRU):
    """Tomorrow's win, draw or loss, fitted fold by fold by `walk_forward_forecasts`.

    `wdl(d, train, test)` returns a (test rows, 3) array of probabilities of a loss, a draw and a
    win, in that order (`baselines.WDL_CLASSES`): three logits, cross-entropy, softmax.
    `fit(d, train)` and `predict(state, d, rows)` are the two halves, as for LightGBM.
    """

    target = 'wdl'

    def fit(self, d, train, fold_start=None):
        """Train on the training rows, seeded from `fold_start` (default: the first row a fold
        on these training rows would test)."""
        torch = _torch()

        def cross_entropy(p, t):
            return torch.nn.functional.cross_entropy(p, t.long())

        train = np.asarray(train)
        fold_start = train[-1] + self.horizon if fold_start is None else fold_start
        y = d['y_wdl'].to_numpy() + 1                       # -1, 0, 1 -> classes 0, 1, 2
        return self._fit(d, y, train, 3, cross_entropy, fold_start)

    def predict(self, state, d, rows):
        """The loss, draw and win probabilities for `rows` of `d`, a (rows, 3) array."""
        out = self._raw(state, d, rows)
        p = np.exp(out - out.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    def wdl(self, d, train, test):
        state = self.fit(d, train, fold_start=test[0])
        return self.predict(state, d, test), self._info(state)
