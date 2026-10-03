"""Out-of-sample forecasts fold by fold, their scores against the baselines, and the sanity checks.

Two kinds of model (the owner's choice, 2026-10-03), told apart by their `target`:

- a return model (`target = 'return'`, e.g. `gbm.LightGBMReturnModel`) has
  `quantiles(d, train, test, taus)`: tomorrow's return quantiles, whose median is the point
  forecast and whose outer quantiles are the 68% and 90% bands;
- a win/draw/loss model (`target = 'wdl'`, e.g. `gbm.LightGBMWDLModel`) has `wdl(d, train, test)`:
  the probabilities of tomorrow's loss, draw and win.

Each fold, a model sees only the training rows' labels and forecasts the test rows.
`walk_forward_forecasts` runs it through the folds beside the baselines, so model and baselines
are scored on the same days.

The checks (`docs/ml_plan.md`, *The sanity checks, as tests*), on series whose direction cannot be
forecast but whose volatility can:

- win/draw/loss: no better than the volatility baseline (`wdl_scores`), but better than the
  plain class frequencies, pooled over tickers (`wdl_check`);
- return, the range: a lower pinball loss than the constant band on every ticker, misses that
  cluster less, and a pooled gain whose interval lies wholly above zero (`range_check`);
- return, the point forecast: no better than the EWMA band's median (`point_scores`).

`planted_signal_bars` builds the positive control: a series whose direction can be forecast.
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import BANDS, WDL_BASELINES, WDL_CLASSES, range_baselines, wdl_baselines
from .data import ticker_bars
from .features import WDL_THRESHOLD
from .metrics import (bootstrap_means, mean_interval, multiclass_brier, multiclass_log_loss,
                      range_scores)
from .split import walk_forward
from .targets import TAUS, dataset


@dataclass
class Forecasts:
    """One model's out-of-sample forecasts for one ticker, beside the baselines'.

    For a return model: `bands`, columns `(band, tau)` for the baselines and `'model'`, and
    `y_ret`, the realized returns. For a win/draw/loss model: `wdl`, columns `(source, class)`
    with the loss, draw and win probabilities of both baselines and the model, and `y_wdl`, what
    happened. `folds`: one row per fold, with what each fit reported.
    """
    ticker: str
    model: str
    target: str
    folds: pd.DataFrame
    bands: pd.DataFrame = None
    y_ret: pd.Series = None
    taus: tuple = TAUS
    wdl: pd.DataFrame = None
    y_wdl: pd.Series = None
    wdl_threshold: float = WDL_THRESHOLD
    data: pd.DataFrame = field(default=None, repr=False)

    def quantiles(self, band='model'):
        """`{tau: forecast}` for one band, as `range_scores` takes it."""
        return {tau: self.bands[(band, tau)] for tau in self.taus}

    def point(self, band='model'):
        """The point forecast of tomorrow's return: `band`'s median."""
        return self.bands[(band, 0.5)]


def walk_forward_forecasts(d, model, folds=None, taus=TAUS, ticker='', first_train=756, step=63):
    """`model`'s forecasts for every test row of `folds` (by default `walk_forward(len(d),
    first_train, step)`), beside the baselines for its target. A larger `step` retrains less
    often over the same test days."""
    if getattr(model, 'target', None) not in ('return', 'wdl'):
        raise ValueError(f"model.target must be 'return' or 'wdl', got {getattr(model, 'target', None)!r}")
    folds = walk_forward(len(d), first_train, step) if folds is None else folds
    forecasts, rows = [], []
    for k, (train, test) in enumerate(folds):
        row = {'fold': k, 'train_rows': len(train), 'train_from': d.index[train[0]],
               'train_to': d.index[train[-1]], 'test_from': d.index[test[0]],
               'test_to': d.index[test[-1]]}
        if model.target == 'return':
            f, info = model.quantiles(d, train, test, taus)
        else:
            f, info = model.wdl(d, train, test)
        forecasts.append(np.asarray(f, dtype=float))
        rows.append({**row, **info})
    test_index = d.index[np.concatenate([test for _, test in folds])]
    f = np.vstack(forecasts)
    common = dict(ticker=ticker, model=getattr(model, 'name', type(model).__name__),
                  target=model.target, folds=pd.DataFrame(rows).set_index('fold'), data=d)
    if model.target == 'return':
        bands = range_baselines(d, folds, taus)
        for i, tau in enumerate(taus):
            bands[('model', tau)] = f[:, i]
        return Forecasts(**common, bands=bands, y_ret=d.loc[test_index, 'y_ret'], taus=tuple(taus))
    threshold = d.attrs.get('wdl_threshold', WDL_THRESHOLD)
    wdl = wdl_baselines(d, folds, threshold)
    for i, c in enumerate(WDL_CLASSES):
        wdl[('model', c)] = f[:, i]
    return Forecasts(**common, wdl=wdl, y_wdl=d.loc[test_index, 'y_wdl'], wdl_threshold=threshold)


def evaluate(ticker, model, source=None, directory=None, wdl_threshold=WDL_THRESHOLD, **kwargs):
    """`walk_forward_forecasts` for a ticker's bars (synthetic, or saved in `data/local`), with
    wins and losses counted against `wdl_threshold`."""
    d = dataset(ticker_bars(ticker, source, directory), wdl_threshold)
    return walk_forward_forecasts(d, model, ticker=ticker, **kwargs)


def evaluate_many(tickers, model, n_jobs=None, **kwargs):
    """`evaluate` for each ticker, in up to `n_jobs` processes at once (default: one per CPU, at
    most one per ticker). Each ticker is fitted exactly as alone, so the forecasts are the same as
    with `n_jobs=1`; only the wall time changes."""
    tickers = list(tickers)
    n_jobs = min(n_jobs or os.cpu_count() or 1, len(tickers))
    if n_jobs <= 1:
        return [evaluate(t, model, **kwargs) for t in tickers]
    # Spawned, not forked: forking after PyTorch or OpenMP has started its threads can hang.
    with ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(evaluate, t, model, **kwargs) for t in tickers]
        return [f.result() for f in futures]


def _needs(fc, target):
    if fc.target != target:
        raise ValueError(f'{fc.ticker}: these are {fc.target} forecasts, not {target} ones')


def wdl_scores(fc, level=0.99, margin=0.015, mean_block=20, n_boot=1000, seed=0):
    """Tomorrow's win/draw/loss forecasts against both baselines: the shares that happened, the
    log loss and Brier score of each source, and the per-day log-loss gains with `level` bootstrap
    intervals.

    `no_better_than_volatility` is the check for series whose direction cannot be forecast: the
    model may use how volatile tomorrow will be (which makes draws likelier or rarer), but no more,
    so its gain over the volatility baseline must not lie wholly above zero and must be at most
    `margin`. The volatility baseline (EWMA volatility) is not the best a model can do: on the
    synthetic tickers the generator's own GARCH volatility beats it by up to 0.014 (SYN-INDEX), so
    a better volatility forecast may land above it, and the margin sits just past that.
    `beats_volatility` is the interval lying wholly above zero, which a planted signal must show.
    """
    _needs(fc, 'wdl')
    y = fc.y_wdl.to_numpy()
    out = {'days': len(y)}
    out.update({f'share_{name}': float(np.mean(y == c))
                for name, c in zip(('loss', 'draw', 'win'), WDL_CLASSES)})
    losses = {}
    for source in (*WDL_BASELINES, 'model'):
        p = fc.wdl[source].to_numpy()
        losses[source] = multiclass_log_loss(p, y, WDL_CLASSES)
        out[f'log_loss_{source}'] = float(losses[source].mean())
        out[f'brier_{source}'] = float(multiclass_brier(p, y, WDL_CLASSES).mean())
    for baseline in WDL_BASELINES:
        ci = mean_interval(losses[baseline] - losses['model'], level, mean_block, n_boot, seed)
        out.update({f'gain_vs_{baseline}': ci['estimate'], f'lower_vs_{baseline}': ci['lower'],
                    f'upper_vs_{baseline}': ci['upper']})
    out['no_better_than_volatility'] = bool(out['lower_vs_volatility'] <= 0
                                            and out['gain_vs_volatility'] <= margin)
    out['beats_volatility'] = bool(out['lower_vs_volatility'] > 0)
    return pd.Series(out, name=fc.ticker)


def wdl_check(forecasts, baseline='frequencies', source='model', level=0.95, mean_block=20,
              n_boot=1000, seed=0):
    """The pooled win/draw/loss gain of `source` over `baseline` across several tickers'
    `Forecasts`, each ticker weighing the same: (per-ticker gains, pooled gain with its `level`
    interval, whether that interval lies wholly above zero)."""
    gains, draws = {}, []
    for fc in forecasts:
        _needs(fc, 'wdl')
        y = fc.y_wdl.to_numpy()
        g = (multiclass_log_loss(fc.wdl[baseline].to_numpy(), y, WDL_CLASSES)
             - multiclass_log_loss(fc.wdl[source].to_numpy(), y, WDL_CLASSES))
        gains[fc.ticker] = float(g.mean())
        draws.append(bootstrap_means(g, mean_block, n_boot, seed))
    pooled_draws = np.mean(draws, axis=0)
    a = (1 - level) / 2
    pooled = pd.Series({'estimate': float(pooled_draws.mean()),
                        'lower': float(np.quantile(pooled_draws, a)),
                        'upper': float(np.quantile(pooled_draws, 1 - a))})
    return pd.Series(gains), pooled, bool(pooled['lower'] > 0)


def point_scores(fc, level=0.99, margin=0.01, mean_block=20, n_boot=1000, seed=0):
    """The point forecast (the model's median) against the baselines' medians: mean absolute
    errors, and the per-day gain over the EWMA band's median with its `level` interval.

    `no_better_than_ewma` is the check for series whose direction cannot be forecast: the gain's
    interval does not lie wholly above zero and the gain is at most `margin` (a share of the EWMA
    median's mean absolute error).
    """
    _needs(fc, 'return')
    y = fc.y_ret.to_numpy()
    errors = {b: np.abs(y - fc.point(b).to_numpy()) for b in (*BANDS, 'model')}
    out = {f'mae_{b}': float(e.mean()) for b, e in errors.items()}
    ci = mean_interval(errors['ewma_std_q'] - errors['model'], level, mean_block, n_boot, seed)
    out.update({'gain_vs_ewma': ci['estimate'], 'lower_vs_ewma': ci['lower'],
                'upper_vs_ewma': ci['upper']})
    out['no_better_than_ewma'] = bool(ci['lower'] <= 0
                                      and ci['estimate'] <= margin * out['mae_ewma_std_q'])
    return pd.Series(out, name=fc.ticker)


def range_table(fc):
    """`range_scores` for every baseline and the model, one row per band."""
    _needs(fc, 'return')
    bands = [b for b in (*BANDS, 'model') if (b, fc.taus[0]) in fc.bands.columns]
    return pd.DataFrame({b: range_scores(fc.quantiles(b), fc.y_ret, fc.taus)[0] for b in bands}).T


def range_check(forecasts, level=0.95, mean_block=20, n_boot=1000, seed=0):
    """The plan's range check over several tickers' return `Forecasts`.

    Returns (per-ticker table, pooled gain over the constant band with its `level` interval,
    passes). The pooled gain averages each ticker's bootstrap draws of the mean per-day pinball
    gain, as a share of the constant band's mean loss, so every ticker weighs the same.
    """
    rows, draws = {}, []
    for fc in forecasts:
        _needs(fc, 'return')
        const, const_daily = range_scores(fc.quantiles('constant'), fc.y_ret, fc.taus)
        model, model_daily = range_scores(fc.quantiles('model'), fc.y_ret, fc.taus)
        rel = (const_daily - model_daily) / const_daily.mean()
        draws.append(bootstrap_means(rel, mean_block, n_boot, seed))
        rows[fc.ticker] = {'gain_vs_constant': float(rel.mean()),
                           'lower_pinball': model['pinball'] < const['pinball'],
                           'less_clustering': model['cluster_ratio_68'] < const['cluster_ratio_68']}
    table = pd.DataFrame(rows).T
    pooled_draws = np.mean(draws, axis=0)
    a = (1 - level) / 2
    pooled = pd.Series({'estimate': float(pooled_draws.mean()),
                        'lower': float(np.quantile(pooled_draws, a)),
                        'upper': float(np.quantile(pooled_draws, 1 - a))})
    passes = bool(table['lower_pinball'].all() and table['less_clustering'].all() and pooled['lower'] > 0)
    return table, pooled, passes


def feature_importance(fc):
    """The model's gain importance per feature, as a share of the total, averaged over the folds
    (the W/D/L model, or the return model's median). Empty when the model reports none."""
    if 'importance' not in fc.folds:
        return pd.Series(dtype=float)
    shares = [pd.Series(imp) / max(sum(imp.values()), 1e-12) for imp in fc.folds['importance']]
    return pd.concat(shares, axis=1).mean(axis=1).sort_values(ascending=False)


def planted_signal_bars(ticker='SYN-INDEX', repeat=0.65, seed=1):
    """`ticker`'s synthetic bars with each day's sign re-drawn to repeat the previous day's sign
    `repeat` of the time. Sizes, volatility and the bars' shapes are kept, so only the direction
    becomes predictable: the positive control for the win/draw/loss check. At 65% the W/D/L
    model finds it clearly; at 60%, diluted by the draws, not reliably."""
    from src.tools.price_return.synthetic import synthetic_bars
    bars = synthetic_bars(ticker)
    r = bars['Close'].pct_change().fillna(0).to_numpy()
    rng = np.random.default_rng(seed)
    sign = np.empty(len(r))
    sign[0] = 1.0
    for t in range(1, len(r)):
        sign[t] = sign[t - 1] if rng.random() < repeat else -sign[t - 1]
    close = bars['Close'].iloc[0] * np.cumprod(1 + np.abs(r) * sign)
    factor = close / bars['Close'].to_numpy()
    out = bars.copy()
    for col in ('Open', 'High', 'Low', 'Close'):
        out[col] = bars[col] * factor
    return out
