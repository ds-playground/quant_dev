"""Plotly charts of the out-of-sample forecasts. Each builds and returns a figure; call .show().

They draw `evaluate.Forecasts` and compute nothing beyond what the charts show (rolling
coverage, reliability bins), so the dashboard and a notebook draw the same numbers.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.tools.price_return.viz import _style, INK_2, MUTED

from .baselines import BANDS
from .metrics import interval_misses, range_scores, reliability_table

# One hue per forecast, fixed across charts: the model, then the baselines.
COLORS = {'model': '#eb6834', 'base': '#52514e', 'constant': '#8a8984', 'price_range': '#eda100',
          'vol_20': '#1baf7a', 'ewma_std_q': '#2a78d6'}
LABELS = {'model': 'model', 'base': 'base rate', 'constant': 'constant width',
          'price_range': 'price_range (250-day σ)', 'vol_20': '20-day σ',
          'ewma_std_q': 'EWMA σ × standardized quantiles'}
UP, DOWN = '#2a78d6', '#e34948'


def _interval(taus, pct):
    lo = min(taus, key=lambda t: abs(t - (1 - pct / 100) / 2))
    hi = min(taus, key=lambda t: abs(t - (1 + pct / 100) / 2))
    return lo, hi


def plot_reliability(fc, bins=20):
    """Observed up-frequency against the forecast, per bin, for the model and the base rate;
    marker area grows with the days in the bin. On the diagonal is calibrated."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='perfect calibration',
                             line=dict(color=MUTED, dash='dot', width=1), hoverinfo='skip'))
    for col, key in (('p_base', 'base'), ('p_model', 'model')):
        t = reliability_table(fc.direction[col], fc.direction['y_up'], bins)
        fig.add_trace(go.Scatter(
            x=t['forecast'], y=t['observed'], mode='markers', name=f'{fc.model if key == "model" else LABELS[key]}',
            marker=dict(size=np.clip(np.sqrt(t['days']), 6, 40), color=COLORS[key],
                        line=dict(color='white', width=2)),
            customdata=t['days'], hovertemplate='forecast %{x:.1%}<br>observed %{y:.1%}<br>%{customdata} days'))
    lo = min(fc.direction[['p_base', 'p_model']].min().min(), fc.direction['y_up'].mean()) - 0.1
    hi = max(fc.direction[['p_base', 'p_model']].max().max(), fc.direction['y_up'].mean()) + 0.1
    span = [max(0, lo), min(1, hi)]
    fig = _style(fig, f'{fc.ticker}: reliability of the direction forecasts', 520,
                 y_title='observed up frequency', x_title='forecast probability of an up day')
    fig.update_xaxes(range=span, tickformat='.0%')
    fig.update_yaxes(range=span, tickformat='.0%')
    return fig


def plot_coverage(fc, bands=('constant', 'ewma_std_q', 'model'), interval=68, window=250):
    """Rolling `window`-day coverage of the `interval`% band for each of `bands`, against its
    nominal level. A band that tracks volatility stays near the line; a constant one swings."""
    lo, hi = _interval(fc.taus, interval)
    fig = go.Figure()
    for band in bands:
        miss = pd.Series(interval_misses(fc.bands[(band, lo)], fc.bands[(band, hi)], fc.y_ret),
                         index=fc.y_ret.index)
        cover = 1 - miss.rolling(window).mean()
        fig.add_trace(go.Scatter(x=cover.index, y=cover, mode='lines', line=dict(color=COLORS[band], width=2),
                                 name=fc.model if band == 'model' else LABELS[band]))
    fig.add_hline(y=hi - lo, line=dict(color=INK_2, width=1, dash='dot'),
                  annotation_text=f'nominal {hi - lo:.1%}', annotation_position='top left')
    fig = _style(fig, f'{fc.ticker}: coverage of the {interval}% interval for the next close, '
                      f'rolling {window} days', 440, y_title='coverage')
    fig.update_yaxes(tickformat='.0%')
    return fig


def plot_forecast_bands(fc, bars, start=None, end=None, band='model'):
    """Daily candles with `band`'s 68% and 90% intervals for each close, as forecast at the close
    before. The forecast made at t is drawn at t + 1, the day it is about."""
    lo68, hi68 = _interval(fc.taus, 68)
    lo90, hi90 = _interval(fc.taus, 90)
    nxt = bars.index.to_series().shift(-1)
    prev_close = bars['Close'].reindex(fc.bands.index)
    q = fc.bands.xs(band, axis=1, level='band')
    frame = pd.DataFrame({tau: prev_close * (1 + q[tau]) for tau in (lo90, lo68, hi68, hi90)})
    frame.index = nxt.reindex(fc.bands.index).to_numpy()
    frame = frame[frame.index.notna()]
    window = bars.loc[start:end]
    frame = frame.loc[window.index.min():window.index.max()]
    fig = go.Figure()
    for (a, b), opacity, name in (((lo90, hi90), 0.12, '90% interval'), ((lo68, hi68), 0.24, '68% interval')):
        fig.add_trace(go.Scatter(x=frame.index, y=frame[b], mode='lines', line=dict(width=0, shape='hvh'),
                                 showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=frame.index, y=frame[a], mode='lines', line=dict(width=0, shape='hvh'),
                                 fill='tonexty', fillcolor=f'rgba(235,104,52,{opacity})', name=name))
    fig.add_trace(go.Candlestick(x=window.index, open=window['Open'], high=window['High'], low=window['Low'],
                                 close=window['Close'], name=fc.ticker,
                                 increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN)))
    name = fc.model if band == 'model' else LABELS[band]
    fig = _style(fig, f'{fc.ticker}: {name} intervals for each close, forecast the day before', 480,
                 y_title='price')
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def plot_folds(fc):
    """Each fold's training and test windows over time."""
    fig = go.Figure()
    for k, row in fc.folds.iterrows():
        for x0, x1, color, name in ((row['train_from'], row['train_to'], '#9ec5f4', 'train'),
                                    (row['test_from'], row['test_to'], COLORS['model'], 'test')):
            fig.add_trace(go.Scatter(x=[x0, x1], y=[k + 1, k + 1], mode='lines', line=dict(color=color, width=6),
                                     name=name, legendgroup=name, showlegend=bool(k == 0)))
    fig = _style(fig, f'{fc.ticker}: walk-forward folds', 120 + 14 * len(fc.folds), y_title='fold')
    fig.update_yaxes(autorange='reversed', showgrid=False)
    return fig


def plot_feature_importance(importance, title='Direction model: share of gain per feature'):
    """A horizontal bar per feature, largest on top (`evaluate.feature_importance`)."""
    s = importance.sort_values()
    fig = go.Figure(go.Bar(x=s, y=s.index, orientation='h', marker_color=COLORS['model'],
                           hovertemplate='%{y}: %{x:.1%}<extra></extra>'))
    fig = _style(fig, title, 120 + 22 * len(s), x_title='share of gain, mean over folds')
    fig.update_xaxes(tickformat='.0%')
    return fig


def plot_range_comparison(forecasts, bands=(*BANDS[1:], 'model')):
    """Mean pinball loss of each band relative to the constant band, per ticker (below zero is
    better than constant)."""
    fig = go.Figure()
    tickers = [fc.ticker for fc in forecasts]
    for band in bands:
        rel = []
        for fc in forecasts:
            const = range_scores(fc.quantiles('constant'), fc.y_ret, fc.taus)[0]['pinball']
            rel.append(range_scores(fc.quantiles(band), fc.y_ret, fc.taus)[0]['pinball'] / const - 1)
        fig.add_bar(x=tickers, y=rel, name=forecasts[0].model if band == 'model' else LABELS[band],
                    marker_color=COLORS[band], hovertemplate='%{x}: %{y:+.1%}<extra></extra>')
    fig = _style(fig, 'Mean pinball loss relative to the constant-width band (lower is better)', 440,
                 y_title='against constant width')
    fig.update_layout(barmode='group', bargap=0.25)
    fig.update_yaxes(tickformat='+.0%')
    return fig
