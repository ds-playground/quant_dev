"""Statistics of a daily return series.

Four groups: distribution and tails, dependence and volatility clustering, drawdowns and
risk-adjusted returns, and the uncertainty of the package's event probabilities. Inputs are
decimal daily returns, e.g. `daily_returns_series(df)`; multi-day returns are compounded
(`compound_returns`), as everywhere in the package.

The Ljung-Box, variance-ratio and ARCH-LM tests are written out from their papers. scipy
supplies the chi-squared distribution and the Student-t fit, imported only where needed; install
it with `pip install -e ".[stats]"`.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from .analysis import consecutive_analysis
from .data import compound_returns
from .params import _params

_NORMAL = NormalDist()


def _scipy_stats():
    try:
        from scipy import stats
    except ImportError as exc:
        raise ImportError('This needs scipy; from the repo root run: pip install -e ".[stats]"') from exc
    return stats


def _clean(returns):
    """A float Series without NaN, whatever array-like came in."""
    return pd.Series(returns, dtype=float).dropna() if not isinstance(returns, pd.Series) \
        else returns.astype(float).dropna()


def _moments(x):
    """Mean, sample std, and the moment (biased) skewness and excess kurtosis."""
    d = x - x.mean()
    m2 = np.mean(d ** 2)
    return x.mean(), x.std(ddof=1), np.mean(d ** 3) / m2 ** 1.5, np.mean(d ** 4) / m2 ** 2 - 3


# ── Distribution and tails ──────────────────────────────────────────────────
def return_moments(returns):
    """Count, mean, std, skew, excess kurtosis, min and max (skew and kurtosis bias-corrected)."""
    r = _clean(returns)
    return pd.Series({'n': len(r), 'mean': r.mean(), 'std': r.std(), 'skew': r.skew(),
                      'excess_kurtosis': r.kurt(), 'min': r.min(), 'max': r.max()})


def jarque_bera(returns):
    """Jarque-Bera normality test: JB = n/6 * (S^2 + K^2/4), chi-squared with 2 df under normality.

    S and K are the moment skewness and excess kurtosis, as in the original test. A small p-value
    says the returns are not normal; for daily market returns it is almost always tiny, driven
    by the excess kurtosis (fat tails).
    """
    x = _clean(returns).to_numpy()
    _, _, s, k = _moments(x)
    stat = len(x) / 6 * (s ** 2 + k ** 2 / 4)
    return pd.Series({'statistic': stat, 'p_value': math.exp(-stat / 2),   # chi2(2) survival
                      'skew': s, 'excess_kurtosis': k, 'n': len(x)})


def fit_student_t(returns):
    """Maximum-likelihood Student-t fit: degrees of freedom, location and scale."""
    df, loc, scale = _scipy_stats().t.fit(_clean(returns).to_numpy())
    return pd.Series({'df': df, 'loc': loc, 'scale': scale})


def qq_points(returns, dist='normal'):
    """Sorted returns against a fitted distribution's quantiles, for a Q-Q plot.

    `dist` is 'normal' (sample mean and std) or 't' (the `fit_student_t` fit). Points on the
    45-degree line mean the distribution describes the data; tails bending away mean it does not.
    """
    x = np.sort(_clean(returns).to_numpy())
    probs = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    if dist == 'normal':
        theoretical = x.mean() + x.std(ddof=1) * np.array([_NORMAL.inv_cdf(q) for q in probs])
    elif dist == 't':
        stats = _scipy_stats()
        theoretical = stats.t.ppf(probs, *stats.t.fit(x))
    else:
        raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")
    return pd.DataFrame({'probability': probs, 'theoretical': theoretical, 'sample': x})


def tail_index(returns, k=None):
    """Hill estimates of the tail index alpha, for losses (left) and gains (right).

    Uses the `k` largest moves in each tail (default 5% of the sample, at least 10). A smaller
    alpha means a fatter tail: moments above alpha do not exist, so alpha < 4 means the kurtosis
    is not a stable number. A normal tail has no finite alpha; its estimate keeps growing with k.
    """
    r = _clean(returns).to_numpy()
    k = int(k or max(10, int(0.05 * len(r))))
    out = {}
    for tail, x in (('left', -r), ('right', r)):
        x = np.sort(x[x > 0])[::-1]
        out[tail] = k / np.sum(np.log(x[:k] / x[k])) if len(x) > k else np.nan
    out['k'] = k
    return pd.Series(out)


def value_at_risk(returns, horizons=(1, 5, 10), levels=(0.95, 0.99)):
    """Value at risk and expected shortfall of the h-day compounded return, as positive losses.

    historical      - the empirical quantile, and the mean of the returns at or below it
    normal          - from the mean and std of the h-day returns
    cornish_fisher  - the normal quantile corrected for skewness and excess kurtosis; its
                      shortfall averages the corrected quantiles over the tail
    h-day returns come from overlapping windows, so longer horizons rest on fewer independent
    observations than their row counts suggest.
    """
    r = _clean(returns)
    rows = []
    for h in horizons:
        x = compound_returns(r, h).dropna().to_numpy()
        mu, sd, s, k = _moments(x)

        def cornish_fisher(z):
            return (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
                    - (2 * z ** 3 - 5 * z) * s ** 2 / 36)

        for level in levels:
            a = 1 - level
            q = np.quantile(x, a)
            z = _NORMAL.inv_cdf(a)
            tail_z = np.array([_NORMAL.inv_cdf(u) for u in (np.arange(1000) + 0.5) / 1000 * a])
            for method, var, es in (
                    ('historical', -q, -x[x <= q].mean()),
                    ('normal', -(mu + sd * z), -(mu - sd * _NORMAL.pdf(z) / a)),
                    ('cornish_fisher', -(mu + sd * cornish_fisher(z)),
                     -(mu + sd * cornish_fisher(tail_z).mean()))):
                rows.append({'horizon': h, 'level': level, 'method': method, 'var': var,
                             'es': es, 'n_windows': len(x)})
    return pd.DataFrame(rows)


# ── Dependence and volatility clustering ────────────────────────────────────
def _acf(x, lags):
    d = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.sum(d * d)
    return np.array([np.sum(d[k:] * d[:-k]) / denom for k in range(1, lags + 1)])


def autocorrelation(returns, lags=20):
    """Autocorrelation of returns, squared returns and absolute returns, by lag.

    `band` is the approximate 95% band for no autocorrelation, 1.96/sqrt(n). Returns usually sit
    inside it; squared and absolute returns sitting well above it for many lags is volatility
    clustering: big moves follow big moves, of either sign.
    """
    r = _clean(returns)
    return pd.DataFrame({'lag': np.arange(1, lags + 1), 'returns': _acf(r, lags),
                         'squared': _acf(r ** 2, lags), 'absolute': _acf(r.abs(), lags),
                         'band': 1.96 / math.sqrt(len(r))})


def ljung_box(returns, lags=(5, 10, 20)):
    """Ljung-Box Q = n(n+2) * sum_k rho_k^2 / (n-k), chi-squared with m df if there is no memory.

    Run on returns (is there memory in direction?) and on squared returns (is there memory in
    size, i.e. volatility clustering?).
    """
    chi2 = _scipy_stats().chi2
    r = _clean(returns)
    n = len(r)
    rows = []
    for series, x in (('returns', r), ('squared', r ** 2)):
        rho = _acf(x, max(lags))
        for m in lags:
            q = n * (n + 2) * np.sum(rho[:m] ** 2 / (n - np.arange(1, m + 1)))
            rows.append({'series': series, 'lags': m, 'statistic': q, 'p_value': chi2.sf(q, m)})
    return pd.DataFrame(rows)


def variance_ratio(returns, periods=(2, 5, 10, 20)):
    """Lo-MacKinlay (1988) variance ratio of log returns, with the heteroskedasticity-robust z*.

    VR(q) compares the variance of overlapping q-day sums with q times the 1-day variance. It is
    1 for a random walk; above 1, moves persist (trend); below 1, they partly reverse (mean
    reversion). `z` assumes constant volatility; `z_robust` (and its two-sided `p_value`) does not,
    and is the one to read for market data.
    """
    x = np.log1p(_clean(returns).to_numpy())
    n = len(x)
    mu = x.mean()
    d = x - mu
    var_1 = np.sum(d ** 2) / (n - 1)
    rows = []
    for q in periods:
        sums = np.convolve(x, np.ones(q), 'valid')                # the n - q + 1 overlapping sums
        var_q = np.sum((sums - q * mu) ** 2) / (q * (n - q + 1) * (1 - q / n))
        vr = var_q / var_1
        phi = 2 * (2 * q - 1) * (q - 1) / (3 * q * n)
        # Lo-MacKinlay's delta(j) carries a factor n that their z* removes with sqrt(n); both are
        # left out here, so theta, like phi, is the variance of VR itself.
        delta = np.array([np.sum(d[j:] ** 2 * d[:-j] ** 2) / np.sum(d ** 2) ** 2
                          for j in range(1, q)])
        theta = np.sum((2 * (q - np.arange(1, q)) / q) ** 2 * delta)
        z_robust = (vr - 1) / math.sqrt(theta)
        rows.append({'q': q, 'variance_ratio': vr, 'z': (vr - 1) / math.sqrt(phi),
                     'z_robust': z_robust, 'p_value': 2 * (1 - _NORMAL.cdf(abs(z_robust)))})
    return pd.DataFrame(rows)


def arch_lm(returns, lags=5):
    """Engle's (1982) ARCH-LM test for volatility clustering.

    Squared demeaned returns are regressed on `lags` of their own lags; n * R^2 is chi-squared with
    `lags` df when volatility does not cluster. A small p-value means it does.
    """
    r = _clean(returns).to_numpy()
    e2 = (r - r.mean()) ** 2
    y = e2[lags:]
    X = np.column_stack([np.ones(len(y))] + [e2[lags - j:len(e2) - j] for j in range(1, lags + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    stat = len(y) * r2
    return pd.Series({'lags': lags, 'statistic': stat,
                      'p_value': _scipy_stats().chi2.sf(stat, lags), 'r_squared': r2})


# ── Drawdowns and risk-adjusted returns ─────────────────────────────────────
def drawdown_series(returns):
    """Wealth from 1, its running peak (starting capital included), and the drawdown from it."""
    r = _clean(returns)
    wealth = (1 + r).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return pd.DataFrame({'wealth': wealth, 'peak': peak, 'drawdown': wealth / peak - 1})


def drawdown_table(returns, top=5):
    """The deepest `top` drawdowns, each from its peak through its trough to recovery.

    `peak` is the last day at the old high (NaT when it was the starting capital); `recovery` is
    the first day back at it (NaT if not yet). Durations are in trading days.
    """
    dd = drawdown_series(returns)['drawdown']
    under = (dd < -1e-12).to_numpy()
    dates = dd.index
    rows = []
    starts = np.flatnonzero(under & ~np.r_[False, under[:-1]])
    for s in starts:
        e = s
        while e + 1 < len(under) and under[e + 1]:
            e += 1
        trough = s + int(np.argmin(dd.iloc[s:e + 1].to_numpy()))
        recovered = e + 1 < len(under)
        rows.append({'depth': dd.iloc[trough],
                     'peak': dates[s - 1] if s > 0 else pd.NaT,
                     'trough': dates[trough],
                     'recovery': dates[e + 1] if recovered else pd.NaT,
                     'days_to_trough': trough - s + 1,
                     'days_to_recover': (e + 1 - trough) if recovered else np.nan})
    table = pd.DataFrame(rows, columns=['depth', 'peak', 'trough', 'recovery', 'days_to_trough',
                                        'days_to_recover'])
    return table.sort_values('depth').head(top).reset_index(drop=True)


def max_drawdown(returns):
    """The deepest drawdown as one row of `drawdown_table`; depth 0 if there was none."""
    table = drawdown_table(returns, top=1)
    if table.empty:
        return pd.Series({'depth': 0.0, 'peak': pd.NaT, 'trough': pd.NaT, 'recovery': pd.NaT,
                          'days_to_trough': 0, 'days_to_recover': 0})
    return table.iloc[0]


def risk_ratios(returns, trade_days=250, risk_free=0.0):
    """Annualized return and volatility, Sharpe, Sortino and Calmar ratios.

    `risk_free` is an annual decimal rate, converted to a daily one. Sharpe and Sortino use the
    mean daily excess return, scaled by sqrt(trade_days); Sortino divides by the downside
    deviation sqrt(mean(min(excess, 0)^2)). `annual_return` is compounded; Calmar divides it by
    the absolute maximum drawdown.
    """
    r = _clean(returns)
    excess = r - ((1 + risk_free) ** (1 / trade_days) - 1)
    annual_return = (1 + r).prod() ** (trade_days / len(r)) - 1
    downside = math.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    depth = max_drawdown(r)['depth']
    return pd.Series({
        'annual_return': annual_return,
        'annual_volatility': r.std() * math.sqrt(trade_days),
        'sharpe': excess.mean() / r.std() * math.sqrt(trade_days),
        'sortino': excess.mean() / downside * math.sqrt(trade_days) if downside else np.inf,
        'calmar': annual_return / abs(depth) if depth else np.inf,
        'max_drawdown': depth,
    })


def rolling_risk(returns, window=250, trade_days=250, risk_free=0.0):
    """Rolling annualized volatility, Sharpe and Sortino over the trailing `window` days."""
    r = _clean(returns)
    excess = r - ((1 + risk_free) ** (1 / trade_days) - 1)
    mean = excess.rolling(window).mean()
    std = r.rolling(window).std()
    downside = np.sqrt((np.minimum(excess, 0) ** 2).rolling(window).mean())
    root = math.sqrt(trade_days)
    return pd.DataFrame({'volatility': std * root, 'sharpe': mean / std * root,
                         'sortino': mean / downside * root})


# ── Uncertainty on probabilities ────────────────────────────────────────────
def stationary_bootstrap(n, mean_block, n_boot, seed=0):
    """Politis-Romano (1994) stationary bootstrap: an `n_boot` x `n` array of resampling indices.

    Blocks start at random positions, run on (wrapping at the end) and stop with probability
    1/`mean_block` each day, so block lengths are geometric with that mean. Keeping days together
    preserves volatility clustering and the overlap of multi-day windows, which resampling single
    days would destroy.
    """
    rng = np.random.default_rng(seed)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    restart = rng.random((n_boot, n)) < 1 / mean_block
    jump = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], jump[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_interval(returns, statistic, n_boot=1000, mean_block=10, level=0.95, seed=0):
    """Estimate and percentile interval of `statistic(array)` under the stationary bootstrap."""
    x = _clean(returns).to_numpy()
    idx = stationary_bootstrap(len(x), mean_block, n_boot, seed)
    draws = np.array([statistic(x[i]) for i in idx])
    a = (1 - level) / 2
    return pd.Series({'estimate': statistic(x), 'lower': np.quantile(draws, a),
                      'upper': np.quantile(draws, 1 - a), 'std_error': draws.std(ddof=1)})


def _event_probabilities(x, thresholds, n_days):
    """The four `consecutive_analysis` probabilities for every row of `x` (samples x days).

    Returns an array (samples, thresholds, 4) ordered consecutive above/below, cumulative
    above/below, each the share of the complete n_days windows.
    """
    x = np.atleast_2d(x)
    zero = np.zeros((x.shape[0], 1))
    log_sum = np.hstack([zero, np.cumsum(np.log1p(x), axis=1)])
    cum = np.expm1(log_sum[:, n_days:] - log_sum[:, :-n_days])
    out = np.empty((x.shape[0], len(thresholds), 4))

    def all_days(mask):
        count = np.hstack([zero, np.cumsum(mask, axis=1)])
        return (count[:, n_days:] - count[:, :-n_days]) == n_days

    for i, thr in enumerate(thresholds):
        out[:, i, 0] = all_days(x >= thr).mean(axis=1)
        out[:, i, 1] = all_days(x <= -thr).mean(axis=1)
        out[:, i, 2] = (cum >= thr).mean(axis=1)
        out[:, i, 3] = (cum <= -thr).mean(axis=1)
    return out


_EVENTS = (('consecutive', 'above'), ('consecutive', 'below'),
           ('cumulative', 'above'), ('cumulative', 'below'))


def _lookback(pct_change, n_years, p):
    return pct_change.dropna().iloc[-n_years * p.trade_days:].to_numpy(dtype=float)


def probability_intervals(pct_change, n_days, p=None, thresholds=None, lookback_years=None,
                          n_boot=500, level=0.95, mean_block=None, seed=0):
    """`consecutive_analysis` probabilities with stationary-bootstrap confidence intervals.

    One row per threshold x lookback x event type for a single holding period `n_days`. `prob`
    equals `consecutive_analysis`'s; `lower` / `upper` show how much it could move on another
    sample of the same market. Blocks average `mean_block` days (default max(2 * n_days, 10)),
    so most windows are resampled whole.
    """
    p = _params(p)
    thresholds = list(p.return_thresholds if thresholds is None else thresholds)
    mean_block = mean_block or max(2 * n_days, 10)
    a = (1 - level) / 2
    rows = []
    for n_years in (p.lookback_years if lookback_years is None else lookback_years):
        x = _lookback(pct_change, n_years, p)
        point = _event_probabilities(x, thresholds, n_days)[0]
        draws = _event_probabilities(x[stationary_bootstrap(len(x), mean_block, n_boot, seed)],
                                     thresholds, n_days)
        lower, upper = np.quantile(draws, a, axis=0), np.quantile(draws, 1 - a, axis=0)
        for i, thr in enumerate(thresholds):
            for j, (change_type, change) in enumerate(_EVENTS):
                rows.append({'change_type': change_type, 'change': change, 'threshold': thr,
                             'n_days': n_days, 'n_years': n_years, 'prob': point[i, j],
                             'lower': lower[i, j], 'upper': upper[i, j]})
    return pd.DataFrame(rows)


def model_probabilities(pct_change, n_days, p=None, thresholds=None, lookback_years=None,
                        n_sims=100_000, seed=0):
    """What an i.i.d. normal and a fitted Student-t would predict for the same events.

    Both are fitted to the log returns of each lookback. `prob_normal` is exact (log returns add
    up over n_days); `prob_t` is exact for consecutive events and simulated (`n_sims` paths) for
    cumulative ones. Where the observed probability sits well above both, the tails are fatter or
    the days more dependent than an i.i.d. model allows.
    """
    stats = _scipy_stats()
    p = _params(p)
    thresholds = list(p.return_thresholds if thresholds is None else thresholds)
    rng = np.random.default_rng(seed)
    rows = []
    for n_years in (p.lookback_years if lookback_years is None else lookback_years):
        y = np.log1p(_lookback(pct_change, n_years, p))
        mu, sd = y.mean(), y.std(ddof=1)
        t_df, t_loc, t_scale = stats.t.fit(y)
        sums = stats.t.rvs(t_df, t_loc, t_scale, size=(n_sims, n_days), random_state=rng).sum(axis=1)
        root = math.sqrt(n_days)
        for thr in thresholds:
            up, down = math.log1p(thr), math.log1p(-thr)
            normal = (
                (1 - _NORMAL.cdf((up - mu) / sd)) ** n_days,
                _NORMAL.cdf((down - mu) / sd) ** n_days,
                1 - _NORMAL.cdf((up - n_days * mu) / (sd * root)),
                _NORMAL.cdf((down - n_days * mu) / (sd * root)),
            )
            student = (
                stats.t.sf(up, t_df, t_loc, t_scale) ** n_days,
                stats.t.cdf(down, t_df, t_loc, t_scale) ** n_days,
                np.mean(sums >= up),
                np.mean(sums <= down),
            )
            for j, (change_type, change) in enumerate(_EVENTS):
                rows.append({'change_type': change_type, 'change': change, 'threshold': thr,
                             'n_days': n_days, 'n_years': n_years,
                             'prob_normal': normal[j], 'prob_t': student[j]})
    return pd.DataFrame(rows)


def event_probability_table(pct_change, n_days, p=None, n_boot=500, seed=0):
    """Every rare-event probability for one holding period, with its uncertainty and the models.

    One row per threshold x lookback x event type: `consecutive_analysis`'s `count` and
    `episodes`, `probability_intervals`' `prob`, `lower` and `upper`, and `model_probabilities`'
    `prob_normal` and `prob_t`. This is the table `plot_event_probabilities` draws.
    """
    p = _params(p)
    keys = ['change_type', 'change', 'threshold', 'n_days', 'n_years']
    observed = pd.concat([consecutive_analysis(pct_change, thr, n_days, n_years, p)
                          for thr in p.return_thresholds for n_years in p.lookback_years])
    return (probability_intervals(pct_change, n_days, p, n_boot=n_boot, seed=seed)
            .merge(model_probabilities(pct_change, n_days, p, seed=seed), on=keys)
            .merge(observed[keys + ['count', 'episodes']], on=keys))
