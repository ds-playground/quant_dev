"""Plotly charts of the out-of-sample forecasts. Each builds and returns a figure; call .show().

They draw `evaluate.Forecasts` and compute nothing beyond what the charts show (rolling
coverage, reliability bins), so the dashboard and a notebook draw the same numbers.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.tools.price_return.viz import _style, INK_2, MUTED

from .baselines import BANDS, WDL_CLASSES
from .metrics import interval_misses, range_scores, reliability_table

# One hue per forecast, fixed across charts: the model, then the baselines.
COLORS = {'model': '#eb6834', 'base': '#52514e', 'constant': '#8a8984', 'price_range': '#eda100',
          'vol_20': '#1baf7a', 'ewma_std_q': '#2a78d6'}
LABELS = {'model': 'model', 'base': 'base rate', 'constant': 'constant width',
          'price_range': 'price_range (250-day σ)', 'vol_20': '20-day σ',
          'ewma_std_q': 'EWMA σ × standardized quantiles'}
UP, DOWN = '#2a78d6', '#e34948'
WDL_COLORS = {-1: DOWN, 0: '#8a8984', 1: UP}


def _interval(taus, pct):
    lo = min(taus, key=lambda t: abs(t - (1 - pct / 100) / 2))
    hi = min(taus, key=lambda t: abs(t - (1 + pct / 100) / 2))
    return lo, hi


def plot_wdl_reliability(fc, bins=10, source='model'):
    """For each outcome (loss, draw, win), how often it happened against the probability
    `source` gave it, per bin; marker area grows with the days in the bin. On the diagonal is
    calibrated."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='perfect calibration',
                             line=dict(color=MUTED, dash='dot', width=1), hoverinfo='skip'))
    y = fc.y_wdl.to_numpy()
    for c, name in zip(WDL_CLASSES, ('loss', 'draw', 'win')):
        t = reliability_table(fc.wdl[(source, c)], (y == c).astype(float), bins)
        fig.add_trace(go.Scatter(
            x=t['forecast'], y=t['observed'], mode='markers+lines', name=name,
            line=dict(color=WDL_COLORS[c], width=1),
            marker=dict(size=np.clip(np.sqrt(t['days']), 6, 36), color=WDL_COLORS[c],
                        line=dict(color='white', width=1.5)),
            customdata=t['days'],
            hovertemplate=name + ': forecast %{x:.1%}<br>happened %{y:.1%}<br>%{customdata} days'))
    who = fc.model if source == 'model' else f'the {source} baseline'
    fig = _style(fig, f'{fc.ticker}: reliability of {who}\'s win/draw/loss forecasts', 520,
                 y_title='how often it happened', x_title='forecast probability')
    fig.update_xaxes(range=[0, 0.8], tickformat='.0%')
    fig.update_yaxes(range=[0, 0.8], tickformat='.0%')
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
    """Daily candles with `band`'s point forecast (its median) and 68% and 90% intervals for each
    close, as forecast at the close before. The forecast made at t is drawn at t + 1, the day it
    is about."""
    lo68, hi68 = _interval(fc.taus, 68)
    lo90, hi90 = _interval(fc.taus, 90)
    nxt = bars.index.to_series().shift(-1)
    prev_close = bars['Close'].reindex(fc.bands.index)
    q = fc.bands.xs(band, axis=1, level='band')
    frame = pd.DataFrame({tau: prev_close * (1 + q[tau]) for tau in (lo90, lo68, 0.5, hi68, hi90)})
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
    fig.add_trace(go.Scatter(x=frame.index, y=frame[0.5], mode='lines', name='point forecast (median)',
                             line=dict(color=COLORS['model'], width=1.5, shape='hvh')))
    fig.add_trace(go.Candlestick(x=window.index, open=window['Open'], high=window['High'], low=window['Low'],
                                 close=window['Close'], name=fc.ticker,
                                 increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN)))
    name = fc.model if band == 'model' else LABELS[band]
    fig = _style(fig, f'{fc.ticker}: {name} forecast of each close, made the day before', 480,
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
    fig.update_yaxes(autorange='reversed', showgrid=False,
                     tickvals=[1, *range(5, len(fc.folds) + 1, 5)])
    return fig


def plot_feature_importance(importance, title='Share of gain per feature'):
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


# One hue per model compared against, in a fixed order.
COMPARE_COLORS = ['#2a78d6', '#eda100', '#1baf7a', '#eb6834']


def plot_model_comparison(table, title=None):
    """One model against others, per ticker: the gain of `model_b` over each `model_a` as a share
    of `model_a`'s loss, with its interval (`evaluate.compare_forecasts` rows, one per ticker and
    target; the ticker is the row's `ticker` column, or its index). Right of zero, `model_b` is
    better. Two panels on their own scales: win/draw/loss (log loss) and the return (mean
    pinball)."""
    t = table.copy()
    if 'ticker' not in t.columns:
        t['ticker'] = [i[-1] if isinstance(i, tuple) else i for i in t.index]
    for c in ('gain_b_over_a', 'lower', 'upper'):
        t[c] = t[c].astype(float) / t['loss_a'].astype(float)
    targets = [x for x in ('wdl', 'return') if x in set(t['target'])]
    fig = make_subplots(rows=1, cols=len(targets), shared_yaxes=True, horizontal_spacing=0.08,
                        subplot_titles=[{'wdl': 'Win/draw/loss (log loss)',
                                         'return': 'Return (mean pinball)'}[x] for x in targets])
    tickers = list(dict.fromkeys(t['ticker']))
    others = list(dict.fromkeys(t['model_a']))
    offsets = np.linspace(-0.27, 0.27, len(others)) if len(others) > 1 else [0.0]
    for col, target in enumerate(targets, start=1):
        for k, (other, offset) in enumerate(zip(others, offsets)):
            rows = t[(t['target'] == target) & (t['model_a'] == other)]
            y = [tickers.index(x) + offset for x in rows['ticker']]
            fig.add_trace(go.Scatter(
                x=rows['gain_b_over_a'], y=y, mode='markers', name=f'vs {other}',
                legendgroup=other, showlegend=col == 1,
                marker=dict(size=9, color=COMPARE_COLORS[k % len(COMPARE_COLORS)],
                            line=dict(color='white', width=2)),
                error_x=dict(type='data', symmetric=False,
                             array=rows['upper'] - rows['gain_b_over_a'],
                             arrayminus=rows['gain_b_over_a'] - rows['lower'],
                             color=COMPARE_COLORS[k % len(COMPARE_COLORS)], thickness=2, width=0),
                customdata=np.column_stack([rows['ticker'], rows['lower'], rows['upper']]),
                hovertemplate=(f'{rows["model_b"].iloc[0] if len(rows) else ""} vs {other}, '
                               '%{customdata[0]}: %{x:+.2%} (95%: %{customdata[1]:+.2%} to '
                               '%{customdata[2]:+.2%})<extra></extra>')), row=1, col=col)
        fig.add_vline(x=0, line=dict(color=MUTED, dash='dot', width=1), row=1, col=col)
    b = t['model_b'].iloc[0]
    fig = _style(fig, title or f'{b} against the other models: loss saved, as a share of theirs '
                               '(right of zero, the ' + b + ' is better)', 160 + 70 * len(tickers))
    fig.update_xaxes(tickformat='+.0%', zeroline=False, title_text="gain, share of the other's loss")
    fig.update_yaxes(tickvals=list(range(len(tickers))), ticktext=tickers, autorange='reversed',
                     showgrid=False)
    # below the panels: above them, the legend would run into the panel titles
    fig.update_layout(legend=dict(orientation='h', yanchor='top', y=-0.16, xanchor='left', x=0),
                      margin=dict(b=110), height=fig.layout.height + 40)
    return fig


# One hue, light to dark, for shares from 0 to 1.
SEQUENTIAL = [[0.0, '#f3f7fd'], [0.5, '#8fb5e8'], [1.0, '#1d5fb0']]


def plot_wdl_confusion(forecasts, sources=('volatility', 'model'), soft=False, title=None):
    """Win/draw/loss confusion matrices (`evaluate.wdl_confusion`), one panel per source, on one
    colour scale. Rows are what happened, columns the forecast; each cell is its share of the
    row: the recall on the diagonal (hard), or the average probability given (soft). Hard cells
    also show their count of days."""
    from .evaluate import WDL_NAMES, wdl_confusion
    single = not isinstance(forecasts, (list, tuple))
    first = forecasts if single else forecasts[0]
    names = {'model': first.model, 'volatility': 'volatility baseline',
             'frequencies': 'class frequencies'}
    fig = make_subplots(rows=1, cols=len(sources), horizontal_spacing=0.12,
                        subplot_titles=[names[s] for s in sources])
    for col, source in enumerate(sources, start=1):
        m = wdl_confusion(forecasts, source, soft)
        share = m.div(m.sum(axis=1), axis=0)
        text = [[f'{share.iat[i, j]:.0%}' + ('' if soft else f'<br>{int(m.iat[i, j]):,}')
                 for j in range(3)] for i in range(3)]
        fig.add_trace(go.Heatmap(
            z=share.to_numpy(), x=list(WDL_NAMES), y=list(WDL_NAMES), zmin=0, zmax=1,
            colorscale=SEQUENTIAL, showscale=col == len(sources), xgap=2, ygap=2,
            colorbar=dict(tickformat='.0%', title=dict(text='share of row', side='right'), len=0.9),
            customdata=m.to_numpy(),
            hovertemplate=('happened %{y}, forecast %{x}: %{z:.1%} of the row'
                           + ('' if soft else ' (%{customdata:,} days)') + '<extra></extra>')),
            row=1, col=col)
        for i, actual in enumerate(WDL_NAMES):
            for j, called in enumerate(WDL_NAMES):
                fig.add_annotation(x=called, y=actual, text=text[i][j], showarrow=False,
                                   font=dict(color='white' if share.iat[i, j] > 0.55 else INK_2),
                                   row=1, col=col)
    days = int(wdl_confusion(forecasts, sources[0], soft).to_numpy().sum())
    kind = ('average probability given to each outcome' if soft
            else 'days by their most probable outcome')
    fig = _style(fig, title or f'{first.model}: win/draw/loss confusion, {kind} ({days:,} days)', 400)
    fig.update_xaxes(title_text='forecast', showgrid=False)
    fig.update_yaxes(title_text='what happened', autorange='reversed', showgrid=False)
    return fig


def _fold_labels(scores):
    """'1: Jan–Jun 2023' for each fold of a fold-score table (its test window)."""
    labels = {}
    for k, r in scores.groupby(level='fold').first().iterrows():
        a, b = pd.Timestamp(r['test_from']), pd.Timestamp(r['test_to'])
        labels[k] = f"{k}: {a:%b}–{b:%b %Y}" if a.year == b.year else f"{k}: {a:%b %Y}–{b:%b %Y}"
    return labels


def plot_wdl_confusion_folds(fc, source='model', cols=4, title=None):
    """`source`'s win/draw/loss confusion matrix for each fold of one ticker's `Forecasts`
    (`evaluate.wdl_fold_confusion`), one panel per fold on one colour scale, titled with the
    fold's test window, accuracy and macro F1. Rows are what happened, columns the forecast; each
    cell is its share of the row (the recall on the diagonal) and its count of days."""
    from .evaluate import WDL_NAMES, wdl_fold_confusion, wdl_fold_scores
    matrices = wdl_fold_confusion(fc, source)
    scores = wdl_fold_scores(fc).xs(source, level='source')
    folds = list(scores.index)
    labels = _fold_labels(wdl_fold_scores(fc))
    rows = -(-len(folds) // cols)
    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.05, vertical_spacing=0.16 / rows * 2,
                        subplot_titles=[f"{labels[k]}<br>accuracy {scores.at[k, 'accuracy']:.0%}, "
                                        f"macro F1 {scores.at[k, 'macro_f1']:.2f}" for k in folds])
    for i, k in enumerate(folds):
        r, c = i // cols + 1, i % cols + 1
        m = matrices.loc[k]
        share = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        fig.add_trace(go.Heatmap(
            z=share.to_numpy(), x=list(WDL_NAMES), y=list(WDL_NAMES), zmin=0, zmax=1,
            colorscale=SEQUENTIAL, showscale=i == 0, xgap=2, ygap=2, customdata=m.to_numpy(),
            colorbar=dict(tickformat='.0%', title=dict(text='share of row', side='right'), len=0.6),
            hovertemplate=(f'fold {k}: happened %{{y}}, forecast %{{x}}: %{{z:.1%}} of the row '
                           '(%{customdata:,} days)<extra></extra>')), row=r, col=c)
        for a, actual in enumerate(WDL_NAMES):
            for b, called in enumerate(WDL_NAMES):
                fig.add_annotation(x=called, y=actual, showarrow=False, row=r, col=c,
                                   text=f'{share.iat[a, b]:.0%}<br>{int(m.iat[a, b])}',
                                   font=dict(size=10, color='white' if share.iat[a, b] > 0.55 else INK_2))
    name = {'model': fc.model, 'volatility': 'volatility baseline',
            'frequencies': 'class frequencies'}[source]
    fig = _style(fig, title or f'{fc.ticker}: {name}, win/draw/loss confusion per fold '
                               '(rows: what happened; columns: the most probable outcome)',
                 120 + 260 * rows)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
    fig.update_yaxes(autorange='reversed', showgrid=False, tickfont=dict(size=10))
    fig.update_annotations(font_size=10)
    return fig


def plot_fold_coverage(fc, bands=None, title=None):
    """For each fold of one ticker's return `Forecasts` (`evaluate.range_fold_scores`), the share
    of its test days inside each band's 68% range (left) and 90% range (right), against the
    nominal level. `bands` defaults to the constant band, the EWMA band and the model (when
    there is one)."""
    from .evaluate import range_fold_scores
    scores = range_fold_scores(fc)
    present = scores.index.get_level_values('band').unique()
    bands = [b for b in (bands or ('constant', 'ewma_std_q', 'model')) if b in present]
    labels = _fold_labels(scores)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=['inside the 68% range', 'inside the 90% range'])
    for col, pct in enumerate((68, 90), start=1):
        lo, hi = _interval(fc.taus, pct)
        nominal = hi - lo
        for band in bands:
            s = scores.xs(band, level='band')[f'cover_{pct}']
            fig.add_trace(go.Scatter(
                x=[labels[k] for k in s.index], y=s, mode='lines+markers', legendgroup=band,
                showlegend=col == 1, name=fc.model if band == 'model' else LABELS[band],
                line=dict(color=COLORS[band], width=2), marker=dict(size=7),
                hovertemplate=f'%{{x}}: %{{y:.1%}}<extra>{LABELS.get(band, band)}</extra>'),
                row=1, col=col)
        fig.add_hline(y=nominal, line=dict(color=INK_2, width=1, dash='dot'), row=1, col=col,
                      annotation_text=f'nominal {nominal:.1%}', annotation_position='top left')
    fig = _style(fig, title or f'{fc.ticker}: share of each fold\'s test days inside the range', 480)
    fig.update_layout(margin=dict(t=140), legend=dict(y=1.16), title=dict(y=0.97, yanchor='top'))
    fig.update_yaxes(tickformat='.0%')
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))
    return fig
