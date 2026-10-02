"""A GRU sequence model for the next day: a direction network (one logit) and a range network
(one output per tau), each reading the last `window` rows of the features.

The owner chose a GRU over an LSTM or a temporal CNN (`docs/ml_plan.md`, *Models*). Per fold, the
features are standardized with the training window's means and standard deviations; the range
network fits the standardized return `r_{t+1} / sigma_t`, as LightGBM's does, and is scaled back.
Training: Adam with a small weight decay, shuffled batches, early stopping on the purged last 20%
of the training window, keeping the best epoch. Each fold is seeded from its first test row and
runs on CPU with deterministic algorithms, so a rerun gives the same forecasts; each day is
forecast on its own, so its forecast does not depend on which other days are forecast with it.

The range forecasts are recalibrated by default (the owner's choice, `docs/ml_plan.md`, Phase 4):
each tau's forecast is shifted by the tau-quantile of the network's own errors on the validation
tail, the rows early stopping holds out inside the training window. Uncorrected, the network's
intervals were too narrow, and much more so when retrained less often.

PyTorch is the `ml` extra, imported only when a model is fitted.
"""
import numpy as np

from .features import FEATURES

# What the GRU reads each day. The owner keeps all three until a model is trained on real data
# (docs/ml_plan.md, Phase 4); 'returns_range' is the provisional default. With the 16 features
# (960 inputs per example against about 1,600 training rows), the GRU found no planted signal.
GRU_INPUTS = {
    'returns_range': ['ret_0', 'tr_close'],      # each day's return and its true range
    'returns': ['ret_0'],                        # each day's return
    'features': list(FEATURES),                  # the 16 features LightGBM reads
}
from .split import purged_tail


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


class GRUModel:
    """The GRU direction and range models, fitted fold by fold by `walk_forward_forecasts`.

    `features` names one of `GRU_INPUTS` ('returns_range', 'returns', 'features') or lists the
    dataset columns to read; each is read over the last `window` rows.

    Same interface as `gbm.LightGBMModel`: `direction(d, train, test)` returns up-probabilities;
    `quantiles(d, train, test, taus)` returns a (test rows, taus) array of return quantiles, sorted
    along each row. Both also return the epochs trained and the best one, whose weights are kept;
    the range model also its recalibration shift per tau (`recalibrate=False` turns it off).
    `threads` sets PyTorch's CPU threads for the fit (None leaves PyTorch's default); give 1 when
    running tickers in parallel processes.
    """

    name = 'GRU'

    def __init__(self, features='returns_range', window=60, hidden=32, batch=128, max_epochs=40,
                 patience=5, learning_rate=1e-3, weight_decay=1e-4, tail=0.2, horizon=1, seed=0,
                 threads=None, recalibrate=True):
        self.inputs = features if isinstance(features, str) else 'custom'
        self.features = list(GRU_INPUTS[features] if isinstance(features, str) else features)
        self.name = f'GRU ({self.inputs})'
        self.window, self.hidden, self.batch = window, hidden, batch
        self.max_epochs, self.patience = max_epochs, patience
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.tail, self.horizon, self.seed, self.threads = tail, horizon, seed, threads
        self.recalibrate = recalibrate

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

    def _fit_predict(self, d, y, train, rows, n_out, loss_fn, fold_start):
        torch = _torch()
        deterministic = torch.are_deterministic_algorithms_enabled()
        threads = torch.get_num_threads()
        torch.use_deterministic_algorithms(True)
        if self.threads:
            torch.set_num_threads(self.threads)
        try:
            return self._train(torch, d, y, train, rows, n_out, loss_fn, fold_start)
        finally:
            torch.use_deterministic_algorithms(deterministic)
            torch.set_num_threads(threads)

    def _train(self, torch, d, y, train, rows, n_out, loss_fn, fold_start):
        # Seeded from the fold's first test row, so this fold trains the same whenever it is fitted,
        # whichever rows are then forecast.
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
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
                best_state = {k: x.clone() for k, x in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            # One row at a time: batched, the kernels round differently with the batch's size, so a
            # day's forecast would depend on which other days were forecast with it.
            out = np.vstack([net(torch.from_numpy(windows(Xs, [row], self.window))).numpy()
                             for row in np.asarray(rows)])
        return out, {'epochs': epochs, 'best_epoch': best_epoch}

    def direction(self, d, train, test):
        torch = _torch()

        def bce(p, t):
            return torch.nn.functional.binary_cross_entropy_with_logits(p[:, 0], t)

        out, info = self._fit_predict(d, d['y_up'].to_numpy(), train, test, 1, bce, test[0])
        return 1 / (1 + np.exp(-out[:, 0].astype(float))), info

    def quantiles(self, d, train, test, taus):
        torch = _torch()
        t = torch.tensor(taus, dtype=torch.float32)

        def pinball(p, y):
            u = y.unsqueeze(1) - p
            return torch.maximum(t * u, (t - 1) * u).mean()

        sigma = d['vol_ewma'].to_numpy()
        y = d['y_ret'].to_numpy() / sigma
        test = np.asarray(test)
        if not self.recalibrate:
            out, info = self._fit_predict(d, y, train, test, len(taus), pinball, test[0])
            return np.sort(out.astype(float), axis=1) * sigma[test][:, None], info
        rows = np.asarray(train)
        _, val = purged_tail(rows[rows >= self.window - 1], self.horizon, self.tail)
        out, info = self._fit_predict(d, y, train, np.r_[val, test], len(taus), pinball, test[0])
        out = out.astype(float)
        shift = quantile_shift(out[:len(val)], y[val], taus)
        q = np.sort(out[len(val):] + shift, axis=1) * sigma[test][:, None]
        return q, {**info, 'shift': shift}
