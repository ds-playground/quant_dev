"""Streaks, cumulative thresholds, rare-event probabilities, distributions."""

import numpy as np
import pandas as pd

from .params import _params


def detect_streaks(df, p=None):
    """Find every rolling window where *all* days are wins (or all are losses)."""
    p = _params(p)
    win_thr, loss_thr = p.win_threshold, p.loss_threshold

    results = {}
    rets = df['return_pct'].values
    dates = df['date'].values

    for w in p.windows:
        wins, losses = [], []
        for i in range(w - 1, len(rets)):
            window_rets = rets[i - w + 1 : i + 1]
            avg = window_rets.mean()
            entry = {
                'start':      pd.Timestamp(dates[i - w + 1]),
                'end':        pd.Timestamp(dates[i]),
                'returns':    window_rets.tolist(),
                'avg_return': round(float(avg), 4),
                'min_return': round(float(window_rets.min()), 4),
                'max_return': round(float(window_rets.max()), 4),
            }
            if all(r >= win_thr for r in window_rets):
                wins.append(entry)
            if all(r <= loss_thr for r in window_rets):
                losses.append(entry)
        results[w] = {'wins': wins, 'losses': losses}
    return results

def summarize_streaks(df, streaks, p=None):
    """One row per window: streak counts, frequencies and average returns."""
    p = _params(p)
    total_windows = len(df)
    rows = []
    for w in p.windows:
        nw = len(streaks[w]['wins'])
        nl = len(streaks[w]['losses'])
        rows.append({
            'Window': f'{w}d',
            'Win Streaks': nw,
            'Win Freq %': round(nw / total_windows * 100, 2),
            'Loss Streaks': nl,
            'Loss Freq %': round(nl / total_windows * 100, 2),
            'Avg Win Ret %': round(np.mean([s['avg_return'] for s in streaks[w]['wins']]) if nw else 0, 3),
            'Avg Loss Ret %': round(np.mean([s['avg_return'] for s in streaks[w]['losses']]) if nl else 0, 3),
        })
    return pd.DataFrame(rows)

def analyze_cumulative(df, p=None):
    """Count rolling windows whose *compounded* return clears each threshold."""
    p = _params(p)
    rets = df['return_pct'].values
    dates = df['date'].values
    results = {}

    for w in p.windows:
        results[w] = {}
        for thr in p.cum_thresholds:
            hits = []
            for i in range(w - 1, len(rets)):
                window_rets = rets[i - w + 1 : i + 1]
                cum_ret = (np.prod(1 + window_rets / 100) - 1) * 100
                if cum_ret >= thr:
                    hits.append({
                        'end_date': pd.Timestamp(dates[i]),
                        'cum_return': round(float(cum_ret), 4)
                    })
            n_windows = len(rets) - w + 1
            results[w][thr] = {
                'count': len(hits),
                'frequency': round(len(hits) / n_windows * 100, 2),
                'hits': hits
            }
    return results

def summarize_cumulative(cum_analysis, p=None):
    """Pivot the cumulative-threshold results into a count/frequency table."""
    p = _params(p)
    rows = []
    for w in p.windows:
        row = {'Window': f'{w}d'}
        for thr in p.cum_thresholds:
            d = cum_analysis[w][thr]
            row[f'≥{thr}% Count']  = d['count']
            row[f'≥{thr}% Freq %'] = d['frequency']
        rows.append(row)
    return pd.DataFrame(rows)

def consecutive_analysis(pct_change, return_threshold, n_days, n_years, p=None):
    """Probability of a +/- `return_threshold` move over `n_days`, within the last `n_years`.

    Two event types are measured on the trailing window:
      consecutive - every one of the n_days moves beyond the threshold
      cumulative  - the n_days returns *summed* move beyond the threshold
    each in both directions ('above' = winning, 'below' = losing).

    Returns a 4-row DataFrame: one row per event type x direction.
    """
    p = _params(p)
    window = pct_change.dropna().iloc[-n_years * p.trade_days:]
    cum_sum = window.rolling(n_days).sum()

    masks = {
        ('consecutive', 'above'): (window >=  return_threshold).rolling(n_days).sum() == n_days,
        ('consecutive', 'below'): (window <= -return_threshold).rolling(n_days).sum() == n_days,
        ('cumulative',  'above'): cum_sum >=  return_threshold,
        ('cumulative',  'below'): cum_sum <= -return_threshold,
    }

    rows = []
    for (change_type, change), mask in masks.items():
        hits = window.index[mask.to_numpy()]
        rows.append({
            'change_type':   change_type,
            'change':        change,
            'threshold':     return_threshold,
            'n_days':        n_days,
            'n_years':       n_years,
            'n_obs':         len(window),          # trading days actually available
            'count':         int(mask.sum()),
            'prob':          mask.sum() / len(window),
            'last_occurred': hits[-1].date() if len(hits) else pd.NaT,
        })
    return pd.DataFrame(rows)

def build_historical_analysis(pct_change, p=None):
    """Run consecutive_analysis across every threshold x holding period x lookback."""
    p = _params(p)
    frames = [
        consecutive_analysis(pct_change, return_threshold, n_days, n_years, p)
        for return_threshold in p.return_thresholds
        for n_years          in p.lookback_years
        for n_days           in p.streak_days
    ]
    return pd.concat(frames, ignore_index=True)

def filter_low_probability(df_his, n_days, p=None, prob_max=None, prob_min=None):
    """Keep the rare-but-observed events for a single holding period.

    `n_days` is required: probabilities are only comparable within one holding
    period, so every filtered view belongs to a single value from `streak_days`.
    Events rarer than `prob_max` are kept; those below `prob_min` are dropped
    as never-observed in the lookback window. Both default to the `Params` values.
    """
    p = _params(p)
    prob_max = p.prob_max if prob_max is None else prob_max
    prob_min = p.prob_min if prob_min is None else prob_min

    out = df_his[(df_his['n_days'] == n_days)
                 & (df_his['prob'] < prob_max)
                 & (df_his['prob'] > prob_min)]
    return out.sort_values(['n_years', 'change_type', 'change', 'threshold'],
                           ascending=[True, True, True, False])

def low_probability_view(df_his, n_days, p=None, prob_max=None, prob_min=None,
                         change_type=None):
    """Rare-event rows for one or several holding periods, most likely first.

    `n_days` accepts a single holding period or any iterable of them.
    `change_type` filters to 'consecutive' or 'cumulative' (or a list of both);
    None keeps every event type.
    """
    p = _params(p)
    days = [n_days] if isinstance(n_days, (int, np.integer)) else list(n_days)
    frames = [filter_low_probability(df_his, d, p, prob_max, prob_min) for d in days]
    out = pd.concat(frames, ignore_index=True) if frames else df_his.iloc[:0]

    if change_type is not None:
        types = [change_type] if isinstance(change_type, str) else list(change_type)
        out = out[out['change_type'].isin(types)]

    return out.sort_values('prob', ascending=False).reset_index(drop=True)

def distribution_summary(df, p=None, label=None):
    """One-row frame of return-distribution stats, using this ticker's thresholds.

    Streaks count *days clearing a threshold*, which tracks the median rather
    than the mean - the two diverge under skew, so both are reported here.
    """
    p = _params(p)
    r = df['return_pct']
    up = int((r > p.win_threshold).sum())
    down = int((r < p.loss_threshold).sum())

    return pd.DataFrame([{
        'ticker':       label or getattr(p, 'label', None) or p.ticker,
        'drift (mean)': round(r.mean(), 4),
        'median':       round(r.median(), 4),
        'skew':         round(r.skew(), 3),
        'days > +thr':  round(up / len(r) * 100, 2),
        'days < -thr':  round(down / len(r) * 100, 2),
        'up:down':      round(up / down, 3) if down else np.nan,
    }])
