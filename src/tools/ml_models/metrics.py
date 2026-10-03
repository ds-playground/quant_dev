"""Scores for the forecasts, and the bootstrap that compares them.

Direction: log loss, the Brier score with Murphy's decomposition, and a reliability table.
Range: interval coverage with Kupiec's test (is the miss rate right?) and Christoffersen's
independence test (do misses cluster?), and pinball loss. Per-day losses come back as arrays, so
two forecasts can be compared day by day with `mean_interval`, a stationary block bootstrap
(`price_return.stats.stationary_bootstrap`) that keeps the clustering of the losses.

scipy supplies only the chi-squared p-values, imported where needed (`pip install -e ".[stats]"`).
"""
import math

import numpy as np
import pandas as pd

from src.tools.price_return.stats import stationary_bootstrap

from .targets import TAUS


def _chi2_sf(x, df=1):
    try:
        from scipy import stats
    except ImportError as exc:
        raise ImportError('This needs scipy; from the repo root run: pip install -e ".[stats]"') from exc
    return float(stats.chi2.sf(x, df))


def _xlogy(x, y):
    """x * log(y), and 0 where x is 0 (so an empty cell adds nothing to a log-likelihood)."""
    return 0.0 if x == 0 else x * math.log(y)


def _arrays(*xs):
    out = [np.asarray(x, dtype=float) for x in xs]
    if len({len(x) for x in out}) > 1:
        raise ValueError('inputs differ in length')
    return out


def log_loss(p, y, eps=1e-12):
    """Per-day log loss of up-probabilities `p` for outcomes `y` (1 up, 0 down)."""
    p, y = _arrays(p, y)
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p, y):
    """Per-day Brier score, `(p - y)^2`."""
    p, y = _arrays(p, y)
    return (p - y) ** 2


def _bins(p, bins):
    return pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)


def reliability_table(p, y, bins=10):
    """Per forecast bin of equal width: the mean forecast, the observed up-frequency, the days."""
    p, y = _arrays(p, y)
    g = pd.DataFrame({'forecast': p, 'observed': y}).groupby(_bins(p, bins), observed=True)
    out = g.agg(forecast=('forecast', 'mean'), observed=('observed', 'mean'), days=('observed', 'size'))
    out.index.name = 'bin'
    return out


def brier_decomposition(p, y, bins=10):
    """Murphy (1973): Brier = reliability - resolution + uncertainty, over forecast bins.

    Exact when the forecasts within each bin are equal (as for a base rate); with spread inside
    the bins, the identity is off by the within-bin variance of the forecasts.
    """
    p, y = _arrays(p, y)
    table = reliability_table(p, y, bins)
    n, base = len(y), y.mean()
    w = table['days'] / n
    return {'brier': float(brier(p, y).mean()),
            'reliability': float((w * (table['forecast'] - table['observed']) ** 2).sum()),
            'resolution': float((w * (table['observed'] - base) ** 2).sum()),
            'uncertainty': float(base * (1 - base))}


def multiclass_log_loss(p, y, classes, eps=1e-12):
    """Per-day log loss of class probabilities `p` (days x classes, in the order of `classes`) for
    the outcomes `y`: minus the log of the probability given to what happened."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    index = np.searchsorted(np.asarray(classes), y)
    if not np.array_equal(np.asarray(classes)[np.clip(index, 0, len(classes) - 1)], y):
        raise ValueError(f'outcomes outside the classes {tuple(classes)}')
    return -np.log(np.clip(p[np.arange(len(y)), index], eps, 1))


def multiclass_brier(p, y, classes):
    """Per-day Brier score over several classes: the squared distance between the probabilities
    and the one-hot outcome (0 is perfect, 2 the worst)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    onehot = (y[:, None] == np.asarray(classes)[None, :]).astype(float)
    return ((p - onehot) ** 2).sum(axis=1)


def pinball(q, y, tau):
    """Per-day pinball (quantile) loss of the `tau`-quantile forecast `q`."""
    q, y = _arrays(q, y)
    u = y - q
    return np.maximum(tau * u, (tau - 1) * u)


def interval_misses(lower, upper, y):
    """True on the days `y` fell outside [`lower`, `upper`]."""
    lower, upper, y = _arrays(lower, upper, y)
    return (y < lower) | (y > upper)


def kupiec(misses, rate):
    """Kupiec's unconditional-coverage test: is the miss share consistent with `rate`?
    Returns (likelihood ratio, p-value), chi-squared with 1 degree of freedom."""
    m = np.asarray(misses, dtype=bool)
    n, x = len(m), int(m.sum())
    lr = -2 * (_xlogy(n - x, 1 - rate) + _xlogy(x, rate) - _xlogy(n - x, (n - x) / n) - _xlogy(x, x / n))
    return lr, _chi2_sf(lr)


def christoffersen(misses):
    """Christoffersen's independence test: does a miss make the next day's miss likelier?

    Returns (likelihood ratio, p-value, miss rate after a miss / miss rate after a hit). A ratio
    near 1 means misses do not cluster.
    """
    m = np.asarray(misses, dtype=int)
    a, b = m[:-1], m[1:]
    n00, n01 = int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    n10, n11 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1)))
    p01 = n01 / (n00 + n01) if n00 + n01 else math.nan
    p11 = n11 / (n10 + n11) if n10 + n11 else math.nan
    p = (n01 + n11) / len(a)
    restricted = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    free = (_xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
            if n00 + n01 and n10 + n11 else restricted)
    lr = -2 * (restricted - free)
    ratio = p11 / p01 if p01 else math.nan
    return lr, _chi2_sf(lr), ratio


def _central_intervals(taus):
    """(nominal coverage in %, lower tau, upper tau) for each symmetric pair in `taus`."""
    out = []
    for lo in sorted(t for t in taus if t < 0.5):
        hi = next((t for t in taus if math.isclose(t, 1 - lo, abs_tol=1e-9)), None)
        if hi is not None:
            out.append((round((1 - 2 * lo) * 100), lo, hi))
    return out


def range_scores(quantiles, y, taus=TAUS):
    """Scores of a quantile forecast, `quantiles[tau]` per day, against the returns `y`.

    For each symmetric interval in `taus` (68% and 90% by default): coverage, Kupiec's p-value,
    Christoffersen's p-value, the clustering ratio and the mean width; and the pinball loss
    averaged over `taus`. Returns (scores, per-day pinball loss averaged over `taus`).
    """
    y = np.asarray(y, dtype=float)
    scores = {}
    for pct, lo, hi in _central_intervals(taus):
        miss = interval_misses(quantiles[lo], quantiles[hi], y)
        scores[f'cover_{pct}'] = 1 - miss.mean()
        scores[f'kupiec_p_{pct}'] = kupiec(miss, 1 - (hi - lo))[1]
        _, p_ind, ratio = christoffersen(miss)
        scores[f'christoffersen_p_{pct}'] = p_ind
        scores[f'cluster_ratio_{pct}'] = ratio
        scores[f'width_{pct}'] = float(np.mean(np.asarray(quantiles[hi]) - np.asarray(quantiles[lo])))
    daily = np.mean([pinball(quantiles[tau], y, tau) for tau in taus], axis=0)
    scores['pinball'] = float(daily.mean())
    return scores, daily


def bootstrap_means(x, mean_block=20, n_boot=1000, seed=0):
    """Stationary-bootstrap draws of the mean of a daily series (blocks of mean length
    `mean_block` keep its clustering)."""
    x = np.asarray(x, dtype=float)
    return x[stationary_bootstrap(len(x), mean_block, n_boot, seed)].mean(axis=1)


def mean_interval(x, level=0.95, mean_block=20, n_boot=1000, seed=0):
    """The mean of a daily series, such as one forecast's loss minus another's, with a
    stationary-bootstrap percentile interval."""
    draws = bootstrap_means(x, mean_block, n_boot, seed)
    a = (1 - level) / 2
    return pd.Series({'estimate': float(np.mean(x)), 'lower': float(np.quantile(draws, a)),
                      'upper': float(np.quantile(draws, 1 - a))})
