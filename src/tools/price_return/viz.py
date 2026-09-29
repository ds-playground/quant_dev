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


# ── Statistics charts (price_return.stats) ──────────────────────────────────
# Colours from a colour-blind-validated categorical palette, in fixed order: blue, orange, aqua.
# Text stays in ink colours; gridlines are solid hairlines.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, BASELINE, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'


def _style(fig, title, height, y_title=None, x_title=None):
    fig.update_layout(
        title=dict(text=title, font=dict(color=INK, size=16)), height=height,
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK_2, size=12),
        hoverlabel=dict(bgcolor='white', font=dict(color=INK)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=60, r=30, t=80, b=50))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BASELINE,
                     tickfont=dict(color=MUTED), title_text=x_title)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor=BASELINE, tickfont=dict(color=MUTED), title_text=y_title)
    return fig


def plot_qq(returns):
    """Q-Q plot of daily returns against a fitted normal and a fitted Student-t.

    Points on the grey 45-degree line are where the fit describes the data. Where the normal's
    points bend away at the ends, the tails are fatter than a normal allows.
    """
    from .stats import qq_points
    fig = go.Figure()
    lo = hi = 0.0
    for colour, (label, dist) in zip(SERIES, (('Normal', 'normal'), ('Student-t', 't'))):
        q = qq_points(returns, dist) * [1, 100, 100]
        lo, hi = min(lo, q['theoretical'].min(), q['sample'].min()), max(hi, q['theoretical'].max(),
                                                                         q['sample'].max())
        fig.add_trace(go.Scatter(
            x=q['theoretical'], y=q['sample'], mode='markers', name=label,
            marker=dict(color=colour, size=8, opacity=0.75),
            hovertemplate=f'{label}<br>fitted %{{x:.2f}}%<br>observed %{{y:.2f}}%<extra></extra>'))
        top = q.iloc[-1]                                    # direct label at the largest gain
        fig.add_annotation(x=top['theoretical'], y=top['sample'], text=label, showarrow=False,
                           xanchor='left', xshift=8, font=dict(color=INK_2, size=11))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='Perfect fit',
                             line=dict(color=MUTED, width=1), hoverinfo='skip'))
    return _style(fig, 'Daily returns against fitted distributions (Q-Q)', 480,
                  y_title='Observed return (%)', x_title='Fitted distribution quantile (%)')


def plot_autocorrelation(acf):
    """Autocorrelation by lag, for returns and for squared returns, as two panels on one scale.

    `acf` comes from `stats.autocorrelation`. The shaded band is the 95% range for no
    autocorrelation; bars far outside it for squared returns mean volatility clusters.
    """
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=('Returns', 'Squared returns'))
    band = acf['band'].iloc[0]
    for col, series in ((1, 'returns'), (2, 'squared')):
        fig.add_trace(go.Bar(x=acf['lag'], y=acf[series], marker=dict(color=SERIES[0], cornerradius=4),
                             name=series, showlegend=False,
                             hovertemplate='lag %{x}<br>autocorrelation %{y:.3f}<extra></extra>'),
                      row=1, col=col)
        fig.add_hrect(y0=-band, y1=band, fillcolor=GRID, opacity=1, line_width=0, layer='below',
                      row=1, col=col)
    fig.update_layout(bargap=0.3)
    _style(fig, f'Autocorrelation by lag (shaded: 95% band for none, ±{band:.3f})', 380)
    fig.update_yaxes(title_text='Autocorrelation', row=1, col=1)
    fig.update_xaxes(title_text='Lag (days)')
    for annotation in fig.layout.annotations:
        annotation.font = dict(color=INK_2, size=13)
    return fig


def plot_drawdown(returns, top=3):
    """Underwater chart: the drawdown from the running peak, with the deepest `top` troughs labelled."""
    from .stats import drawdown_series, drawdown_table
    dd = drawdown_series(returns)['drawdown'] * 100
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd, mode='lines', name='Drawdown', fill='tozeroy',
        line=dict(color=SERIES[0], width=2), fillcolor='rgba(42,120,214,0.15)',
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:.1f}% below the peak<extra></extra>'))
    for _, row in drawdown_table(returns, top=top).iterrows():
        fig.add_annotation(x=row['trough'], y=row['depth'] * 100, text=f"{row['depth']:.1%}",
                           showarrow=False, yshift=-12, font=dict(color=INK_2, size=11))
    return _style(fig, 'Drawdown from the running peak', 380, y_title='Drawdown (%)')


def plot_rolling_risk(rolling, window=None):
    """Rolling annualized volatility and Sharpe ratio, as two stacked panels sharing the date axis.

    `rolling` comes from `stats.rolling_risk`. Different units, so separate panels rather than
    two y-axes on one plot.
    """
    label = f' ({window}-day window)' if window else ''
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Annualized volatility (%)', 'Sharpe ratio'))
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling['volatility'] * 100, mode='lines',
                             name='Volatility', line=dict(color=SERIES[0], width=2),
                             hovertemplate='%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling['sharpe'], mode='lines', name='Sharpe',
                             line=dict(color=SERIES[0], width=2),
                             hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>'), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=BASELINE, width=1), row=2, col=1)
    fig.update_layout(showlegend=False)
    _style(fig, 'Rolling risk' + label, 520)
    for annotation in fig.layout.annotations:
        annotation.font = dict(color=INK_2, size=13)
    return fig


def plot_event_probabilities(table, change_type='cumulative', change='above', n_years=None):
    """Observed probability of a move, with its bootstrap interval, against the two i.i.d. models.

    `table` is `stats.probability_intervals` merged with `stats.model_probabilities` (columns
    prob, lower, upper, prob_normal, prob_t). One event type and lookback per chart. The y axis is
    logarithmic, so rare events stay visible; a probability of zero cannot be drawn and is left out.
    """
    rows = table[(table['change_type'] == change_type) & (table['change'] == change)]
    if n_years is not None:
        rows = rows[rows['n_years'] == n_years]
    rows = rows.sort_values('threshold')
    x = rows['threshold'] * 100
    fig = go.Figure()
    observed = rows['prob'] > 0
    fig.add_trace(go.Scatter(
        x=x[observed], y=rows['prob'][observed] * 100, mode='lines+markers', name='Observed',
        line=dict(color=SERIES[0], width=2), marker=dict(size=8, color=SERIES[0]),
        error_y=dict(type='data', symmetric=False, color=SERIES[0], thickness=1.5, width=4,
                     array=(rows['upper'] - rows['prob'])[observed] * 100,
                     arrayminus=(rows['prob'] - rows['lower'])[observed] * 100),
        hovertemplate='move %{x:.2f}%<br>observed %{y:.3f}%<extra></extra>'))
    for colour, column, label in ((SERIES[1], 'prob_normal', 'Normal model'),
                                  (SERIES[2], 'prob_t', 'Student-t model')):
        keep = rows[column] > 0
        fig.add_trace(go.Scatter(
            x=x[keep], y=rows[column][keep] * 100, mode='lines', name=label,
            line=dict(color=colour, width=2),
            hovertemplate=f'move %{{x:.2f}}%<br>{label.lower()} %{{y:.3f}}%<extra></extra>'))
        if keep.any():                                      # direct label at the line's end
            last = rows[keep].iloc[-1]
            fig.add_annotation(x=last['threshold'] * 100, y=np.log10(last[column] * 100),
                               text=label, showarrow=False, xanchor='left', xshift=6,
                               font=dict(color=INK_2, size=11))
    n_days = rows['n_days'].iloc[0] if len(rows) else ''
    lookback = f', {n_years}y lookback' if n_years is not None else ''
    fig = _style(fig, f'{change_type.capitalize()} move {change} the threshold over {n_days} days'
                       f'{lookback}: observed (95% interval) vs models', 460,
                  y_title='Probability (%, log scale)', x_title='Move size (%)')
    fig.update_yaxes(type='log', dtick=1, ticksuffix='%', minor=dict(showgrid=False))
    return fig
