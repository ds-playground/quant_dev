"""Out-of-sample forecasts fold by fold, their scores against the baselines, and the sanity checks.

A model is any object with `direction(d, train, test)` and `quantiles(d, train, test, taus)`, as
`gbm.LightGBMModel`: each fold, it sees only the training rows' labels and returns forecasts for
the test rows. `walk_forward_forecasts` runs it through the folds beside the baselines, so model
and baselines are scored on the same days.

The checks are the plan's (`docs/ml_plan.md`, *The sanity checks, as tests*):

- direction: no better than chance, i.e. the `level` interval of `log loss(base rate) - log
  loss(model)` does not lie wholly above zero, and its mean is at most `margin`;
- range: a lower pinball loss than the constant band on every ticker, misses that cluster less,
  and a pooled gain whose interval lies wholly above zero.

`planted_signal_bars` builds the positive control: a series the direction model must find.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import BANDS, direction_baseline, range_baselines
from .data import ticker_bars
from .metrics import bootstrap_means, brier, log_loss, mean_interval, range_scores
from .split import walk_forward
from .targets import TAUS, dataset


@dataclass
class Forecasts:
    """One model's out-of-sample forecasts for one ticker, beside the baselines'.

    `direction`: `y_up`, `p_base`, `p_model` per test day. `bands`: columns `(band, tau)` for the
    baselines and `'model'`. `y_ret`: the realized next-day returns. `folds`: one row per fold.
    """
    ticker: str
    model: str
    direction: pd.DataFrame
    bands: pd.DataFrame
    y_ret: pd.Series
    folds: pd.DataFrame
    taus: tuple = TAUS
    data: pd.DataFrame = field(default=None, repr=False)

    def quantiles(self, band='model'):
        """`{tau: forecast}` for one band, as `range_scores` takes it."""
        return {tau: self.bands[(band, tau)] for tau in self.taus}


def walk_forward_forecasts(d, model, folds=None, taus=TAUS, ticker='', direction=True, ranges=True):
    """`model`'s forecasts for every test row of `folds` (by default `walk_forward(len(d))`),
    beside the baselines'. `direction=False` or `ranges=False` skips that half."""
    folds = walk_forward(len(d)) if folds is None else folds
    p_model, q_model, rows = [], [], []
    for k, (train, test) in enumerate(folds):
        row = {'fold': k, 'train_rows': len(train), 'train_from': d.index[train[0]],
               'train_to': d.index[train[-1]], 'test_from': d.index[test[0]],
               'test_to': d.index[test[-1]]}
        if direction:
            p, info = model.direction(d, train, test)
            p_model.append(np.asarray(p, dtype=float))
            row.update({f'direction_{key}': v for key, v in info.items()})
        if ranges:
            q, info = model.quantiles(d, train, test, taus)
            q_model.append(np.asarray(q, dtype=float))
            row.update({f'range_{key}': v for key, v in info.items()})
        rows.append(row)
    test_index = d.index[np.concatenate([test for _, test in folds])]

    direction_frame = pd.DataFrame({'y_up': d.loc[test_index, 'y_up']})
    if direction:
        direction_frame['p_base'] = direction_baseline(d, folds)
        direction_frame['p_model'] = np.concatenate(p_model)
    bands = range_baselines(d, folds, taus)
    if ranges:
        q = np.vstack(q_model)
        for i, tau in enumerate(taus):
            bands[('model', tau)] = q[:, i]
    return Forecasts(ticker=ticker, model=getattr(model, 'name', type(model).__name__),
                     direction=direction_frame, bands=bands, y_ret=d.loc[test_index, 'y_ret'],
                     folds=pd.DataFrame(rows).set_index('fold'), taus=tuple(taus), data=d)


def evaluate(ticker, model, source=None, directory=None, **kwargs):
    """`walk_forward_forecasts` for a ticker's bars (synthetic, or saved in `data/local`)."""
    d = dataset(ticker_bars(ticker, source, directory))
    return walk_forward_forecasts(d, model, ticker=ticker, **kwargs)


def evaluate_many(tickers, model, n_jobs=None, **kwargs):
    """`evaluate` for each ticker, in up to `n_jobs` processes at once (default: one per CPU, at
    most one per ticker). Each ticker is fitted exactly as alone, so the forecasts are the same as
    with `n_jobs=1`; only the wall time changes."""
    tickers = list(tickers)
    n_jobs = min(n_jobs or os.cpu_count() or 1, len(tickers))
    if n_jobs <= 1:
        return [evaluate(t, model, **kwargs) for t in tickers]
    with ProcessPoolExecutor(n_jobs) as pool:
        futures = [pool.submit(evaluate, t, model, **kwargs) for t in tickers]
        return [f.result() for f in futures]


def direction_scores(fc, level=0.99, margin=0.002, mean_block=20, n_boot=1000, seed=0):
    """The direction model against the base rate: log loss and Brier of each, the per-day log-loss
    gain `base - model` with its `level` bootstrap interval, and the two verdicts.

    `no_better_than_chance` is the plan's check for unpredictable series: the interval does not
    lie wholly above zero and the gain is at most `margin`. `beats_base_rate` is the interval
    lying wholly above zero, which the positive control must show.
    """
    f = fc.direction
    y = f['y_up'].to_numpy()
    gain = log_loss(f['p_base'], y) - log_loss(f['p_model'], y)
    ci = mean_interval(gain, level, mean_block, n_boot, seed)
    return pd.Series({
        'days': len(y),
        'log_loss_base': float(log_loss(f['p_base'], y).mean()),
        'log_loss_model': float(log_loss(f['p_model'], y).mean()),
        'brier_base': float(brier(f['p_base'], y).mean()),
        'brier_model': float(brier(f['p_model'], y).mean()),
        'gain': ci['estimate'], 'lower': ci['lower'], 'upper': ci['upper'],
        'forecast_spread': float(f['p_model'].std()),
        'no_better_than_chance': bool(ci['lower'] <= 0 and ci['estimate'] <= margin),
        'beats_base_rate': bool(ci['lower'] > 0),
    }, name=fc.ticker)


def range_table(fc):
    """`range_scores` for every baseline and the model, one row per band."""
    bands = [b for b in (*BANDS, 'model') if (b, fc.taus[0]) in fc.bands.columns]
    return pd.DataFrame({b: range_scores(fc.quantiles(b), fc.y_ret, fc.taus)[0] for b in bands}).T


def range_check(forecasts, level=0.95, mean_block=20, n_boot=1000, seed=0):
    """The plan's range check over several tickers' `Forecasts`.

    Returns (per-ticker table, pooled gain over the constant band with its `level` interval,
    passes). The pooled gain averages each ticker's bootstrap draws of the mean per-day pinball
    gain, as a share of the constant band's mean loss, so every ticker weighs the same.
    """
    rows, draws = {}, []
    for fc in forecasts:
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
    """The direction model's gain importance per feature, as a share of the total, averaged over
    the folds. Empty when the model reports none."""
    if 'direction_importance' not in fc.folds:
        return pd.Series(dtype=float)
    shares = [pd.Series(imp) / max(sum(imp.values()), 1e-12) for imp in fc.folds['direction_importance']]
    return pd.concat(shares, axis=1).mean(axis=1).sort_values(ascending=False)


def planted_signal_bars(ticker='SYN-INDEX', repeat=0.6, seed=1):
    """`ticker`'s synthetic bars with each day's sign re-drawn to repeat the previous day's sign
    `repeat` of the time. Sizes, volatility and the bars' shapes are kept, so only the direction
    becomes predictable: the positive control for the direction check."""
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
