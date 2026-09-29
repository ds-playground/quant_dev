"""Plotly figures. Each builds and returns a figure; call .show() on it."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .params import _params


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
    """The same streaks as a share of the complete windows of each length."""
    p = _params(p)
    windows_labels = [f'{w}d' for w in p.windows]
    n_windows = {w: max(len(df) - w + 1, 1) for w in p.windows}
    win_freq  = [round(len(streaks[w]['wins'])   / n_windows[w] * 100, 2) for w in p.windows]
    loss_freq = [round(len(streaks[w]['losses']) / n_windows[w] * 100, 2) for w in p.windows]

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
        title='Streak Frequency as % of Windows',
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
