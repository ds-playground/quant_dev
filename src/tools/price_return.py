"""Historical price & return analysis.

Helpers for the `price_return_analysis` notebooks: loading a price series
(simulated or Yahoo Finance), deriving rolling statistics, detecting win/loss
streaks, measuring cumulative-threshold and rare-event probabilities, and
drawing the Plotly charts for each.

Every tunable value lives on `Params`, so a notebook configures once and passes
that object around::

    from src.tools.price_return import Params, load_price_data, add_rolling_stats

    P  = Params(ticker='KO', start_date='2016-01-01')
    df = add_rolling_stats(load_price_data(P), P)
"""

import datetime as dt
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    'Params',
    # data
    'load_price_data', 'add_rolling_stats', 'latest_snapshot', 'show_latest_snapshot',
    'daily_returns_series',
    # streak & cumulative analysis
    'detect_streaks', 'summarize_streaks', 'analyze_cumulative', 'summarize_cumulative',
    # rare-event probabilities
    'consecutive_analysis', 'build_historical_analysis', 'filter_low_probability',
    'low_probability_view', 'format_probability_table', 'interactive_low_probability',
    # charts
    'plot_rolling_average', 'plot_rolling_volatility', 'plot_price_and_returns',
    'plot_return_distribution', 'plot_streak_counts', 'plot_streak_frequency',
    'plot_streak_timeline', 'plot_cumulative_heatmap', 'plot_cumulative_counts',
    # export
    'export_tables',
]


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
def _today():
    return dt.date.today().strftime('%Y-%m-%d')


@dataclass
class Params:
    """Every tunable value for a price/return analysis run.

    Defaults reproduce the notebook's out-of-the-box configuration, so
    ``Params()`` is a valid starting point and individual fields can be
    overridden by keyword.
    """

    # ── Price data ───────────────────────────────────────────────────────
    data_source: str = 'simulated'      # 'simulated' or 'yahoo'
    ticker: str = 'AAPL'
    start_date: str = '2020-01-01'
    end_date: str = field(default_factory=_today)

    # Simulation-only knobs (ignored when data_source == 'yahoo')
    random_seed: int = 42
    sim_drift: float = 0.04             # mean daily return (%)
    sim_vol: float = 1.2                # daily return std dev (%)
    sim_start_price: float = 150.0

    # ── Rolling / streak analysis ────────────────────────────────────────
    trade_days: int = 250               # trading days per year
    win_threshold: float = 0.5          # min daily return (%) counted as a 'win'
    loss_threshold: float = -0.5        # max daily return (%) counted as a 'loss'
    windows: list = field(default_factory=lambda: [2, 3, 5])
    cum_thresholds: list = field(default_factory=lambda: [0.5, 1.0, 2.0])
    roll_windows: list = None           # defaults to [1, 2, 3, 4, 5, 10, trade_days]

    # ── Charts ───────────────────────────────────────────────────────────
    chart_windows: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 10])
    timeline_window: int = 2

    # ── Historical (rare-event) analysis ─────────────────────────────────
    # size of move tested, decimal (0.01 = a 1% return)
    return_thresholds: list = field(default_factory=lambda: [
        0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02,
        0.015, 0.01, 0.005, 0.002, 0.001, 0.0001])
    lookback_years: list = field(default_factory=lambda: [2, 5])
    streak_days: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 10, 20, 30])
    prob_max: float = 0.10              # "low probability" cutoff
    prob_min: float = 1e-4              # drop events never seen in the lookback window

    def __post_init__(self):
        if self.roll_windows is None:
            self.roll_windows = [1, 2, 3, 4, 5, 10, self.trade_days]


def _params(p):
    """Fall back to stock defaults when a caller passes nothing."""
    return Params() if p is None else p


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_price_data(p=None, verbose=True):
    """Return a date / price / return_pct frame, from Yahoo Finance or simulation.

    `return_pct` is in percent (0.5 == +0.5%). The simulated series spans the same
    start..end business-day range as the real one, so both paths are comparable.
    """
    p = _params(p)

    if p.data_source == 'yahoo':
        # pip install yfinance
        import yfinance as yf
        raw = yf.download(p.ticker, start=p.start_date, end=p.end_date)
        if isinstance(raw.columns, pd.MultiIndex):     # yfinance can return a (Price, Ticker) MultiIndex
            raw.columns = raw.columns.get_level_values(0)
        df = raw[['Close']].rename(columns={'Close': 'price'}).reset_index()
        df = df.rename(columns={'Date': 'date'})
        df['return_pct'] = df['price'].pct_change() * 100
        df = df.dropna().reset_index(drop=True)

    elif p.data_source == 'simulated':
        np.random.seed(p.random_seed)
        dates = pd.bdate_range(start=p.start_date, end=p.end_date)
        daily_returns = np.random.normal(loc=p.sim_drift, scale=p.sim_vol, size=len(dates))   # % per day
        prices = p.sim_start_price * np.cumprod(1 + daily_returns / 100)
        df = pd.DataFrame({
            'date':        dates,
            'price':       prices,
            'return_pct':  daily_returns
        })

    else:
        raise ValueError(f"data_source must be 'simulated' or 'yahoo', got {p.data_source!r}")

    df['date'] = pd.to_datetime(df['date'])
    if verbose:
        print(f'Data source: {p.data_source}   |   Loaded {len(df)} trading days   |   '
              f'Avg return: {df.return_pct.mean():.3f}%   |   '
              f'Std: {df.return_pct.std():.3f}%')
    return df


def add_rolling_stats(df, p=None):
    """Add `PCT Change {d}` / `{d} Av` / `{d} STD` columns plus the annualized pair.

    For each holding period `d`, the d-day cumulative return is computed, then its
    rolling mean and std over `trade_days` days. Returns are decimal fractions here
    (0.0004), converted from the percent-scale `return_pct` column.
    """
    p = _params(p)
    trade_d = p.trade_days
    df = df.copy()

    # Base 1-day series and the annualized pair everything else builds on.
    df['PCT Change 1']     = df['return_pct'] / 100
    df['PCT Change 1 Av']  = df['PCT Change 1'].rolling(trade_d).mean()
    df['PCT Change 1 STD'] = df['PCT Change 1'].rolling(trade_d).std()
    df['PCT Change Annualized']     = df['PCT Change 1'].rolling(trade_d).sum()
    df['PCT Change Annualized STD'] = df['PCT Change 1 STD'] * trade_d ** 0.5

    for d in p.roll_windows:
        if d == 1:
            continue    # already computed above
        df[f'PCT Change {d}']     = df['PCT Change 1'].rolling(d).sum()
        df[f'PCT Change {d} Av']  = df[f'PCT Change {d}'].rolling(trade_d).mean()
        df[f'PCT Change {d} STD'] = df[f'PCT Change {d}'].rolling(trade_d).std()

    return df


def latest_snapshot(df, p=None):
    """Latest annualized and daily mean/vol, as a Series of decimal fractions."""
    return pd.Series({
        'annualized_return':     df['PCT Change Annualized'].iloc[-1],
        'annualized_volatility': df['PCT Change Annualized STD'].iloc[-1],
        'daily_avg_return':      df['PCT Change 1 Av'].iloc[-1],
        'daily_std_dev':         df['PCT Change 1 STD'].iloc[-1],
    })


def show_latest_snapshot(df, p=None):
    """Print the latest rolling vol/mean snapshot and return it."""
    p = _params(p)
    s = latest_snapshot(df, p)
    print(f'=== Annualized (latest {p.trade_days}d) ===')
    print(f"Return:     {s['annualized_return']:.2%}")
    print(f"Volatility: {s['annualized_volatility']:.2%}")
    print()
    print(f'=== Daily (latest {p.trade_days}d rolling) ===')
    print(f"Avg return: {s['daily_avg_return']:.4%}")
    print(f"Std dev:    {s['daily_std_dev']:.4%}")
    return s


def daily_returns_series(df):
    """Date-indexed decimal daily returns, the input the historical helpers expect."""
    return df.set_index('date')['PCT Change 1']


# ─────────────────────────────────────────────────────────────────────────────
# Streak & cumulative analysis
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Historical (rare-event) probabilities
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Charts — each builds and returns a Plotly figure; call .show() on the result
# ─────────────────────────────────────────────────────────────────────────────
def plot_rolling_average(df, p=None):
    """Rolling average return, one line per holding period."""
    p = _params(p)
    fig = go.Figure()
    for d in p.chart_windows:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df[f'PCT Change {d} Av'] * 100,
            mode='lines', name=f'{d}d'
        ))
    fig.update_layout(
        title=f'{p.trade_days}-Day Rolling Average Return by Holding Period',
        xaxis_title='Date', yaxis_title='Rolling Avg Return (%)',
        height=400, plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=50, r=20, t=80, b=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def plot_rolling_volatility(df, p=None):
    """Rolling std dev per holding period, with the annualized figure overlaid."""
    p = _params(p)
    fig = go.Figure()
    for d in p.chart_windows:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df[f'PCT Change {d} STD'] * 100,
            mode='lines', name=f'{d}d'
        ))
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['PCT Change Annualized STD'] * 100,
        mode='lines', name='Annualized', line=dict(color='black', dash='dash')
    ))
    fig.update_layout(
        title=f'{p.trade_days}-Day Rolling Volatility (Std Dev) by Holding Period',
        xaxis_title='Date', yaxis_title='Rolling Std Dev (%)',
        height=400, plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=50, r=20, t=80, b=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def plot_price_and_returns(df, p=None):
    """Price on top, daily returns below, with win/loss threshold lines."""
    p = _params(p)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4],
        subplot_titles=(f'{p.ticker} Price', 'Daily Return (%)')
    )

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['price'],
        name='Price', line=dict(color='#378ADD', width=1.5)
    ), row=1, col=1)

    colors_ret = ['#1D9E75' if r >= 0 else '#D85A30' for r in df['return_pct']]
    fig.add_trace(go.Bar(
        x=df['date'], y=df['return_pct'],
        name='Daily Return %', marker_color=colors_ret, opacity=0.8
    ), row=2, col=1)

    fig.add_hline(y=p.win_threshold,  line_dash='dash', line_color='#1D9E75', opacity=0.6, row=2, col=1)
    fig.add_hline(y=p.loss_threshold, line_dash='dash', line_color='#D85A30', opacity=0.6, row=2, col=1)

    fig.update_layout(
        height=500, title_text=f'{p.ticker} — Historical Price & Daily Returns',
        showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=50, r=20, t=60, b=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def plot_return_distribution(df, p=None):
    """Histogram of daily returns, coloured by sign, with threshold markers."""
    p = _params(p)
    bins = np.arange(-5, 5.5, 0.5)
    hist_vals, bin_edges = np.histogram(df['return_pct'], bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_colors = ['#1D9E75' if c >= 0 else '#D85A30' for c in bin_centers]

    fig = go.Figure(go.Bar(
        x=bin_centers, y=hist_vals,
        marker_color=bar_colors, opacity=0.85,
        hovertemplate='Return: %{x:.1f}%<br>Days: %{y}<extra></extra>'
    ))
    fig.add_vline(x=p.win_threshold,  line_dash='dash', line_color='#1D9E75', annotation_text='Win thr')
    fig.add_vline(x=p.loss_threshold, line_dash='dash', line_color='#D85A30', annotation_text='Loss thr')
    fig.update_layout(
        title='Daily Return Distribution',
        xaxis_title='Return (%)', yaxis_title='Frequency (days)',
        height=350, plot_bgcolor='white', paper_bgcolor='white',
        bargap=0.05, margin=dict(l=50, r=20, t=60, b=40)
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def plot_streak_counts(streaks, p=None):
    """Grouped bars: how many win vs loss streaks occurred per window."""
    p = _params(p)
    windows_labels = [f'{w}d' for w in p.windows]
    win_counts  = [len(streaks[w]['wins'])   for w in p.windows]
    loss_counts = [len(streaks[w]['losses']) for w in p.windows]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f'Win streaks (each day ≥{p.win_threshold}%)',
        x=windows_labels, y=win_counts,
        marker_color='#1D9E75', text=win_counts, textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name=f'Loss streaks (each day ≤{p.loss_threshold}%)',
        x=windows_labels, y=loss_counts,
        marker_color='#D85A30', text=loss_counts, textposition='outside'
    ))
    fig.update_layout(
        title='Consecutive Daily Win & Loss Streaks by Window',
        xaxis_title='Window', yaxis_title='Count',
        barmode='group', height=400,
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=50, r=20, t=80, b=40)
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def plot_streak_frequency(df, streaks, p=None):
    """The same streaks as a share of all trading days."""
    p = _params(p)
    windows_labels = [f'{w}d' for w in p.windows]
    win_freq  = [round(len(streaks[w]['wins'])   / len(df) * 100, 2) for w in p.windows]
    loss_freq = [round(len(streaks[w]['losses']) / len(df) * 100, 2) for w in p.windows]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Win freq %', x=windows_labels, y=win_freq,
        marker_color='#1D9E75',
        text=[f'{v}%' for v in win_freq], textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Loss freq %', x=windows_labels, y=loss_freq,
        marker_color='#D85A30',
        text=[f'{v}%' for v in loss_freq], textposition='outside'
    ))
    fig.update_layout(
        title='Streak Frequency as % of All Trading Days',
        xaxis_title='Window', yaxis_title='Frequency (%)',
        barmode='group', height=400,
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=50, r=20, t=80, b=40)
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def plot_streak_timeline(df, streaks, p=None, window=None):
    """Daily returns with win (green) and loss (red) streak periods shaded."""
    p = _params(p)
    window = p.timeline_window if window is None else window

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['return_pct'],
        mode='lines', name='Daily return',
        line=dict(color='#B4B2A9', width=1), opacity=0.7
    ))

    for s in streaks[window]['wins']:
        fig.add_vrect(
            x0=s['start'], x1=s['end'],
            fillcolor='#1D9E75', opacity=0.15, line_width=0
        )

    for s in streaks[window]['losses']:
        fig.add_vrect(
            x0=s['start'], x1=s['end'],
            fillcolor='#D85A30', opacity=0.15, line_width=0
        )

    fig.add_hline(y=p.win_threshold,  line_dash='dot', line_color='#1D9E75', opacity=0.8)
    fig.add_hline(y=p.loss_threshold, line_dash='dot', line_color='#D85A30', opacity=0.8)

    fig.update_layout(
        title=f'{window}-day streak timeline (green=win, red=loss)',
        xaxis_title='Date', yaxis_title='Daily Return (%)',
        height=380, plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False, margin=dict(l=50, r=20, t=60, b=40)
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def plot_cumulative_heatmap(cum_analysis, p=None):
    """Side-by-side heatmaps of threshold-clearing counts and frequencies."""
    p = _params(p)
    z_count = [[cum_analysis[w][t]['count'] for t in p.cum_thresholds] for w in p.windows]
    z_freq  = [[cum_analysis[w][t]['frequency'] for t in p.cum_thresholds] for w in p.windows]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Count of windows exceeding threshold', 'Frequency (%) of windows exceeding threshold')
    )

    fig.add_trace(go.Heatmap(
        z=z_count,
        x=[f'≥{t}%' for t in p.cum_thresholds],
        y=[f'{w}d' for w in p.windows],
        colorscale='Blues', text=z_count, texttemplate='%{text}',
        showscale=True, colorbar=dict(x=0.45, len=0.8)
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=z_freq,
        x=[f'≥{t}%' for t in p.cum_thresholds],
        y=[f'{w}d' for w in p.windows],
        colorscale='Purples', text=z_freq, texttemplate='%{text}%',
        showscale=True, colorbar=dict(x=1.02, len=0.8)
    ), row=1, col=2)

    windows_label = ' / '.join(f'{w}d' for w in p.windows)
    fig.update_layout(
        title=f'Cumulative Return Threshold Analysis ({windows_label} windows)',
        height=320, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=60, r=60, t=80, b=40)
    )
    return fig


def plot_cumulative_counts(cum_analysis, p=None):
    """Grouped bars of how many windows cleared each cumulative threshold."""
    p = _params(p)
    windows_labels = [f'{w}d' for w in p.windows]
    palette = ['#378ADD', '#7F77DD', '#D4537E']

    fig = go.Figure()
    for thr, color in zip(p.cum_thresholds, palette):
        counts = [cum_analysis[w][thr]['count'] for w in p.windows]
        fig.add_trace(go.Bar(
            name=f'≥{thr}%',
            x=windows_labels, y=counts,
            marker_color=color,
            text=counts, textposition='outside'
        ))

    fig.update_layout(
        title='Windows where Cumulative Return Exceeded Threshold',
        xaxis_title='Rolling Window', yaxis_title='Count',
        barmode='group', height=420,
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(title='Threshold', orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=50, r=20, t=80, b=40)
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Interactive rare-event table
# ─────────────────────────────────────────────────────────────────────────────
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


def format_probability_table(table):
    """Display copy with `threshold` and `prob` rendered as percentages.

    Formatting is applied last, after any sorting: these columns become strings,
    and sorting strings would order '9.60%' above '10.00%'.
    """
    if table.empty:
        return table
    return table.assign(threshold=lambda d: d['threshold'].map('{:.2%}'.format),
                        prob=lambda d: d['prob'].map('{:.2%}'.format))


def interactive_low_probability(df_his, p=None, n_days=None, prob_max=None,
                                prob_min=None, change_type=None):
    """Live-filtered rare-event table: pick a holding period, event type and bounds.

    Renders ipywidgets controls that refilter `df_his` in place, so the table
    updates without re-running the cell. Requires ipywidgets (preinstalled on
    Colab; `pip install ipywidgets` locally) - without it, this falls back to a
    static table built from the current arguments and says so.

    Returns the widget container, or the static DataFrame when falling back.
    """
    p = _params(p)
    n_days = p.streak_days[0] if n_days is None else n_days
    if not isinstance(n_days, (int, np.integer)):
        n_days = list(n_days)[0]        # a dropdown shows one period at a time
    prob_max = p.prob_max if prob_max is None else prob_max
    prob_min = p.prob_min if prob_min is None else prob_min

    ALL = 'All'
    types = sorted(df_his['change_type'].dropna().unique().tolist())

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print('ipywidgets is not installed - showing a static table instead.')
        print('Install it for live controls:  pip install ipywidgets')
        table = low_probability_view(df_his, n_days, p, prob_max, prob_min, change_type)
        print(f'{len(table)} event(s) | holding period {n_days}d | '
              f'type {change_type or ALL} | {prob_min:.2%} < prob < {prob_max:.2%}')
        return format_probability_table(table)

    label_style = {'description_width': 'initial'}
    days_w = widgets.Dropdown(
        options=list(p.streak_days), value=n_days,
        description='Streak days:', style=label_style)
    type_w = widgets.Dropdown(
        options=[ALL] + types, value=change_type or ALL,
        description='Change type:', style=label_style)
    max_w = widgets.BoundedFloatText(
        value=prob_max, min=0.0, max=1.0, step=0.01,
        description='prob_max:', style=label_style)
    min_w = widgets.BoundedFloatText(
        value=prob_min, min=0.0, max=1.0, step=0.0001,
        description='prob_min:', style=label_style)
    out = widgets.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            if min_w.value >= max_w.value:
                print(f'prob_min ({min_w.value:.2%}) must be below '
                      f'prob_max ({max_w.value:.2%}).')
                return
            selected_type = None if type_w.value == ALL else type_w.value
            table = low_probability_view(df_his, days_w.value, p,
                                         max_w.value, min_w.value, selected_type)
            print(f'{len(table)} event(s) of {len(df_his)} | '
                  f'holding period {days_w.value}d | type {type_w.value} | '
                  f'{min_w.value:.2%} < prob < {max_w.value:.2%}')
            if table.empty:
                print('Nothing in range - widen prob_max, lower prob_min, '
                      'or switch change type.')
            else:
                display(format_probability_table(table))

    for w in (days_w, type_w, max_w, min_w):
        w.observe(render, names='value')
    render()

    controls = widgets.HBox([widgets.VBox([days_w, type_w]),
                             widgets.VBox([max_w, min_w])])
    box = widgets.VBox([controls, out])
    display(box)
    return box


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────
def export_tables(tables, out_dir='.', verbose=True):
    """Write a {filename: DataFrame} mapping to CSV; returns the paths written."""
    paths = []
    for name, table in tables.items():
        path = os.path.join(out_dir, name)
        table.to_csv(path, index=False)
        paths.append(path)
    if verbose:
        print(f'Saved {len(paths)} file(s): ' + ', '.join(tables))
    return paths
