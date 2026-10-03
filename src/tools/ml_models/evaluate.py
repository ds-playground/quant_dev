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
from .metrics import (bootstrap_means, classification_scores, confusion_matrix, mean_interval,
                      multiclass_brier, multiclass_log_loss, pinball, range_scores,
                      regression_scores)
from .split import walk_forward
from .targets import TAUS, dataset


@dataclass
class Forecasts:
    """One model's out-of-sample forecasts for one ticker, beside the baselines'.

    For a return model: `bands`, columns `(band, tau)` for the baselines and `'model'`, and
    `y_ret`, the realized returns. For a win/draw/loss model: `wdl`, columns `(source, class)`
    with the loss, draw and win probabilities of both baselines and the model, and `y_wdl`, what
    happened. `folds`: one row per fold, with what each fit reported. `settings`: the model's
    settings, and `walk_forward`: the `first_train` and `step` the folds came from (None for
    folds given by hand); `registry.finalize` checks both before saving the forecasts with a
    model.
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
    settings: dict = None
    walk_forward: dict = None
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
    plan = dict(first_train=first_train, step=step) if folds is None else None
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
                  target=model.target, folds=pd.DataFrame(rows).set_index('fold'), data=d,
                  settings=model.settings() if hasattr(model, 'settings') else None,
                  walk_forward=plan)
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


WDL_NAMES = ('loss', 'draw', 'win')        # the names of WDL_CLASSES, in order


def _wdl_days(forecasts):
    """Every test day of one or several tickers' win/draw/loss `Forecasts`: {source: (days, 3)
    probabilities} for both baselines and the model, and what happened."""
    forecasts = [forecasts] if isinstance(forecasts, Forecasts) else list(forecasts)
    for fc in forecasts:
        _needs(fc, 'wdl')
    y = np.concatenate([fc.y_wdl.to_numpy() for fc in forecasts])
    p = {s: np.vstack([fc.wdl[s].to_numpy() for fc in forecasts]) for s in (*WDL_BASELINES, 'model')}
    return p, y


def wdl_confusion(forecasts, source='model', soft=False):
    """The win/draw/loss confusion matrix of `source` ('model', 'volatility' or 'frequencies')
    over one or several tickers' `Forecasts`: rows what happened, columns the forecast, each in
    the order loss, draw, win.

    Hard (the default): each day counted under its most probable outcome. Soft: each day's three
    probabilities added under the three outcomes, so a row sums to the days of its outcome, and
    divided by them it is the average probability given to each outcome on such days. The soft
    matrix reads the probabilities themselves; the hard one keeps only their largest, which is
    nearly always the commonest outcome.
    """
    p, y = _wdl_days(forecasts)
    p = p[source]
    if soft:
        m = np.vstack([p[y == c].sum(axis=0) for c in WDL_CLASSES])
    else:
        m = confusion_matrix(y, np.asarray(WDL_CLASSES)[p.argmax(axis=1)], WDL_CLASSES)
    return pd.DataFrame(m, index=pd.Index(WDL_NAMES, name='actual'),
                        columns=pd.Index(WDL_NAMES, name='forecast'))


def wdl_metrics(forecasts, level=0.95, mean_block=20, n_boot=1000, seed=0):
    """The usual classification scores of the win/draw/loss forecasts, over one or several
    tickers' `Forecasts` (pooled), for both baselines and the model.

    Returns `(table, gains)`. `table`, one row per source:
    - read as calls (the most probable outcome each day): accuracy, balanced accuracy, macro and
      weighted F1, MCC (`metrics.classification_scores`), and the share of days called a loss,
      a draw or a win;
    - read as probabilities, what the models are trained on: log loss and Brier score.

    `gains`: the model's gain over each baseline in balanced accuracy and in log loss (the
    baseline's minus the model's), with `level` bootstrap intervals. With several tickers the
    bootstrap blocks run across the joins between them.

    A call per day throws most of the probabilities away. On series where draws are rarer than
    wins, the most probable outcome is nearly always "win", so accuracy says little; balanced
    accuracy, macro F1 and MCC do not reward always calling the commonest outcome.
    """
    p, y = _wdl_days(forecasts)
    classes = np.asarray(WDL_CLASSES)
    counts = {c: int((y == c).sum()) for c in WDL_CLASSES}
    present = [c for c in WDL_CLASSES if counts[c]]
    # each day's share of the balanced accuracy, so that the mean over days is that accuracy
    weight = np.array([len(y) / (len(present) * counts[c]) for c in y])
    rows, balanced, losses = {}, {}, {}
    for source, probs in p.items():
        call = classes[probs.argmax(axis=1)]
        summary, per_class = classification_scores(y, call, WDL_CLASSES)
        balanced[source] = (call == y) * weight
        losses[source] = multiclass_log_loss(probs, y, WDL_CLASSES)
        rows[source] = {'days': len(y), **summary,
                        **{f'called_{name}': per_class['called'].iloc[i]
                           for i, name in enumerate(WDL_NAMES)},
                        'log_loss': float(losses[source].mean()),
                        'brier': float(multiclass_brier(probs, y, WDL_CLASSES).mean())}
    table = pd.DataFrame(rows).T
    table['days'] = table['days'].astype(int)
    gains = {}
    for b in WDL_BASELINES:
        gains[('balanced_accuracy', b)] = mean_interval(balanced['model'] - balanced[b], level,
                                                        mean_block, n_boot, seed)
        gains[('log_loss', b)] = mean_interval(losses[b] - losses['model'], level, mean_block,
                                               n_boot, seed)
    gains = pd.DataFrame(gains).T
    gains.index.names = ['metric', 'over']
    return table, gains


def wdl_class_report(forecasts, source='model'):
    """Per outcome (loss, draw, win) of `source`'s calls, over one or several tickers'
    `Forecasts`: precision, recall, F1, support and the share of days called that way."""
    p, y = _wdl_days(forecasts)
    call = np.asarray(WDL_CLASSES)[p[source].argmax(axis=1)]
    _, per_class = classification_scores(y, call, WDL_CLASSES)
    per_class.index = pd.Index(WDL_NAMES, name='outcome')
    return per_class


def _return_days(forecasts):
    forecasts = [forecasts] if isinstance(forecasts, Forecasts) else list(forecasts)
    for fc in forecasts:
        _needs(fc, 'return')
    taus = tuple(forecasts[0].taus)
    if any(tuple(fc.taus) != taus for fc in forecasts):
        raise ValueError('the forecasts have different taus')
    y = np.concatenate([fc.y_ret.to_numpy() for fc in forecasts])
    bands = [b for b in (*BANDS, 'model') if all((b, taus[0]) in fc.bands.columns for fc in forecasts)]
    q = {b: {tau: np.concatenate([fc.bands[(b, tau)].to_numpy() for fc in forecasts]) for tau in taus}
         for b in bands}
    return q, y, taus


def return_metrics(forecasts):
    """The usual regression scores of the return forecasts, over one or several tickers'
    `Forecasts` (pooled), one row for the model, one for each baseline band, and one for "no
    change" (a forecast of zero).

    The point forecast is each band's median: R2, R2 against no change, MAE, RMSE, bias,
    correlation, IC (rank correlation), hit rate and the share forecast up
    (`metrics.regression_scores`). Errors are in basis points of return (`_bp`). The range adds
    68% and 90% coverage, the 68% band's mean width and the mean pinball loss.

    All on returns, not prices: tomorrow's close is so close to today's that a price-level R2 is
    near 1 for any forecast, the no-change one included. For daily returns an R2 near zero is
    normal; below zero means worse than the benchmark.
    """
    q, y, taus = _return_days(forecasts)
    rows = {}
    for band, quantiles in q.items():
        s = regression_scores(y, quantiles[0.5])
        r, _ = range_scores(quantiles, y, taus)
        rows['model' if band == 'model' else band] = {
            'days': len(y), 'r2': s['r2'], 'r2_vs_no_change': s['r2_vs_zero'],
            'mae_bp': s['mae'] * 1e4, 'rmse_bp': s['rmse'] * 1e4, 'bias_bp': s['bias'] * 1e4,
            'corr': s['corr'], 'ic': s['ic'], 'hit_rate': s['hit_rate'],
            'forecast_up': s['forecast_up'], 'cover_68': r.get('cover_68'),
            'cover_90': r.get('cover_90'), 'width_68_bp': r.get('width_68', np.nan) * 1e4,
            'pinball_bp': r['pinball'] * 1e4}
    s = regression_scores(y, np.zeros_like(y))
    rows['no change'] = {'days': len(y), 'r2': s['r2'], 'r2_vs_no_change': 0.0,
                         'mae_bp': s['mae'] * 1e4, 'rmse_bp': s['rmse'] * 1e4,
                         'bias_bp': s['bias'] * 1e4, 'corr': np.nan, 'ic': np.nan,
                         'hit_rate': np.nan, 'forecast_up': 0.0}
    order = ['model', 'no change', *[b for b in BANDS if b in rows]]
    table = pd.DataFrame(rows).T.loc[order]
    table['days'] = table['days'].astype(int)
    return table


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


def daily_losses(fc):
    """`fc`'s model loss per test day: the multiclass log loss for win/draw/loss forecasts, the
    pinball loss averaged over the taus for return forecasts."""
    if fc.target == 'wdl':
        loss = multiclass_log_loss(fc.wdl['model'].to_numpy(), fc.y_wdl.to_numpy(), WDL_CLASSES)
        return pd.Series(loss, index=fc.y_wdl.index)
    y = fc.y_ret.to_numpy()
    loss = np.mean([pinball(fc.bands[('model', tau)].to_numpy(), y, tau) for tau in fc.taus], axis=0)
    return pd.Series(loss, index=fc.y_ret.index)


def compare_forecasts(a, b, level=0.95, mean_block=20, n_boot=1000, seed=0):
    """Two models' out-of-sample forecasts of the same thing, on the days both forecast: how much
    `b` gains over `a` (a's loss minus b's, per day: log loss for win/draw/loss, mean pinball for
    the return), with a `level` bootstrap interval. `b_better` is that interval lying wholly
    above zero, `a_better` wholly below. For return forecasts, the point forecasts' mean absolute
    errors too.

    This is how a new model is compared with a saved one (`registry.load_model(...).forecasts`)
    without retraining it. Both must forecast the same ticker's same target with the same taus,
    and agree on what happened on the shared days.
    """
    if a.target != b.target:
        raise ValueError(f'cannot compare {a.target} forecasts with {b.target} ones')
    if a.target == 'return' and tuple(a.taus) != tuple(b.taus):
        raise ValueError(f'different taus: {a.taus} and {b.taus}')
    if a.target == 'wdl' and a.wdl_threshold != b.wdl_threshold:
        raise ValueError(f'different win/draw/loss thresholds: {a.wdl_threshold} and {b.wdl_threshold}')
    ya, yb = (a.y_wdl, b.y_wdl) if a.target == 'wdl' else (a.y_ret, b.y_ret)
    days = ya.index.intersection(yb.index)
    if len(days) == 0:
        raise ValueError('the two forecasts share no days')
    if not np.allclose(ya.loc[days].to_numpy(), yb.loc[days].to_numpy(), rtol=0, atol=1e-12):
        raise ValueError('the two forecasts disagree on what happened: not the same data')
    la, lb = daily_losses(a).loc[days], daily_losses(b).loc[days]
    ci = mean_interval((la - lb).to_numpy(), level, mean_block, n_boot, seed)
    out = {'model_a': a.model, 'model_b': b.model, 'target': a.target, 'days': len(days),
           'first': days[0], 'last': days[-1], 'loss_a': float(la.mean()), 'loss_b': float(lb.mean()),
           'gain_b_over_a': ci['estimate'], 'lower': ci['lower'], 'upper': ci['upper'],
           'b_better': bool(ci['lower'] > 0), 'a_better': bool(ci['upper'] < 0)}
    if a.target == 'return':
        y = ya.loc[days]
        out['mae_a'] = float((y - a.point().loc[days]).abs().mean())
        out['mae_b'] = float((y - b.point().loc[days]).abs().mean())
    return pd.Series(out, name=a.ticker)


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
