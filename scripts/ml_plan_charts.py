"""The example charts in docs/ml_plan.md, made from the synthetic tickers only (no market data).

    python scripts/ml_plan_charts.py

Writes the six PNGs to docs/images/ml/ and prints the numbers the plan quotes beside them. Needs
the `stats` extra (scipy) and kaleido for the PNG export (`pip install kaleido`), which drives a
Chrome or Chromium; if kaleido cannot find one, point BROWSER_PATH at its executable.
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import plotly.graph_objects as go                                   # noqa: E402
from plotly.subplots import make_subplots                           # noqa: E402
from scipy import stats                                             # noqa: E402

from src.tools.price_return.synthetic import PROFILES, synthetic_bars   # noqa: E402

OUT = ROOT / 'docs' / 'images' / 'ml'
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7', '#e34948'
GRAY = '#b9b8b2'
UP, FLAT, DOWN = BLUE, GRAY, RED
FONT = dict(family='Inter, Helvetica, Arial, sans-serif', size=13, color=INK)


def style(fig, title, height, width=1280, top=125):
    fig.update_layout(title=dict(text=title, x=0.01, xanchor='left', y=0.985, yanchor='top', font=dict(size=17)),
                      font=FONT, paper_bgcolor='white', plot_bgcolor='white', width=width,
                      height=height, margin=dict(l=70, r=30, t=top, b=50),
                      legend=dict(orientation='h', yanchor='top', y=0.93, yref='container', x=0.01,
                                  font=dict(color=INK2)))
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=INK2), ticks='')
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK2))
    for a in fig.layout.annotations:
        if a.font.size is None:
            a.font = dict(size=14, color=INK)
    return fig


def save(fig, name):
    path = OUT / f'{name}.png'
    fig.write_image(path, scale=1)
    print(f'{path.relative_to(ROOT)}: {os.path.getsize(path) / 1024:.0f} KB')


# ── Data ─────────────────────────────────────────────────────────────────────
TICKER = 'SYN-INDEX'
bars = synthetic_bars(TICKER)
close = bars['Close']
r = close.pct_change()                                  # return of day t (known at close t)
r_next = r.shift(-1)                                    # the label: tomorrow's return
ewma_var = (r ** 2).ewm(alpha=0.06, adjust=False).mean()    # RiskMetrics, lambda = 0.94
sig_ewma = np.sqrt(ewma_var)
sig20 = r.rolling(20).std()
sig250 = r.rolling(250).std()                           # what price_range is fed (trade_days)
mu_exp = r.expanding(250).mean()


def oracle_sigma(ticker):
    """The generator's own one-step-ahead volatility for day t+1, known at close t."""
    pr = PROFILES[ticker]
    b = synthetic_bars(ticker)
    dates = b.index
    ret = b['Close'].pct_change().fillna(0).to_numpy()
    n = len(ret)
    long_run = (pr.vol / 100) ** 2
    omega = long_run * (1 - pr.alpha - pr.beta)
    multiple, mean = np.ones(n), np.full(n, pr.drift / 100)
    for first, last, m, d in pr.stress:
        span = (dates >= first) & (dates <= last)
        multiple[span], mean[span] = m, d / 100
    var = np.empty(n)
    var[0] = long_run
    for t in range(n - 1):
        shock = (ret[t] - mean[t]) / multiple[t] if t > 0 else 0.0
        var[t + 1] = omega + pr.alpha * shock ** 2 + pr.beta * var[t]
    # sigma of day t+1's return, indexed at t; the mean of day t+1 likewise
    sig_next = pd.Series(np.sqrt(np.r_[var[1:], np.nan]) * np.r_[multiple[1:], np.nan], dates)
    mean_next = pd.Series(np.r_[mean[1:], np.nan], dates)
    return sig_next, mean_next


oracle_sig, oracle_mean = oracle_sigma(TICKER)

# ── 1. Direction targets ─────────────────────────────────────────────────────
K = 0.25
lab = pd.DataFrame({'r': r_next, 'sig': sig_ewma}).dropna()
lab = lab[lab.index >= '2016-03-01']
q = lab.index.to_period('Q')
binary = np.where(lab['r'] > 0, 'up', 'down')
fixed = np.select([lab['r'] > 0.005, lab['r'] < -0.005], ['up', 'down'], 'flat')
scaled = np.select([lab['r'] > K * lab['sig'], lab['r'] < -K * lab['sig']], ['up', 'down'], 'flat')


def shares(labels):
    s = pd.crosstab(q, labels, normalize='index')
    return s.reindex(columns=['down', 'flat', 'up']).fillna(0)


panels = [('A. Binary: up if the next close is higher', shares(binary)),
          ('B. Fixed flat band: |next return| < 0.5%', shares(fixed)),
          (f'C. Volatility-scaled flat band: |next return| < {K} × the EWMA volatility known at t', shares(scaled))]
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                    subplot_titles=[p[0] for p in panels])
for i, (_, s) in enumerate(panels, start=1):
    x = s.index.to_timestamp()
    for cls, color in (('down', DOWN), ('flat', FLAT), ('up', UP)):
        fig.add_bar(x=x, y=s[cls] * 100, name=cls, marker=dict(color=color, line=dict(width=1, color='white')),
                    showlegend=(i == 3), legendgroup=cls, row=i, col=1)
    fig.update_yaxes(range=[0, 100], ticksuffix='%', dtick=50, row=i, col=1)
fig.update_layout(barmode='stack', bargap=0.08)
style(fig, f'{TICKER}: share of next-day labels per quarter under three direction targets', 720)
fig.update_layout(legend=dict(traceorder='normal'))
save(fig, 'direction_targets')
fixed_flat = shares(fixed)['flat']
scaled_flat = shares(scaled)['flat']
print('fixed flat share per quarter: min %.0f%% max %.0f%%' % (fixed_flat.min() * 100, fixed_flat.max() * 100))
print('scaled flat share per quarter: min %.0f%% max %.0f%%' % (scaled_flat.min() * 100, scaled_flat.max() * 100))
fl = pd.Series(fixed == 'flat', lab.index)
print('fixed flat vs sigma corr (daily): %.2f' % np.corrcoef(fl, lab['sig'])[0, 1])
sl = pd.Series(scaled == 'flat', lab.index)
print('scaled flat vs sigma corr (daily): %.2f' % np.corrcoef(sl, lab['sig'])[0, 1])
print('up frequency (binary): %.3f' % (lab['r'] > 0).mean())

# ── 2. Range targets ─────────────────────────────────────────────────────────
win = bars.loc['2020-01-27':'2020-04-03']
prev_close = close.shift(1).loc[win.index]
sig_prev = sig_ewma.shift(1).loc[win.index]           # forecast made at the previous close
z68, z90 = stats.norm.ppf(0.8413), stats.norm.ppf(0.95)
# high/low as multiples of sigma, from the history before the window (median, then 90% level)
hist = bars.loc[:'2020-01-24']
up_mult = ((hist['High'] / hist['Close'].shift(1) - 1) / sig_ewma.shift(1).loc[hist.index]).dropna()
dn_mult = ((1 - hist['Low'] / hist['Close'].shift(1)) / sig_ewma.shift(1).loc[hist.index]).dropna()

fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                    subplot_titles=['A. Quantiles of the next close (68% and 90% bands)',
                                    'B. The next day\'s high and low (median and 90% levels)'])
for col in (1, 2):
    fig.add_trace(go.Candlestick(x=win.index, open=win['Open'], high=win['High'], low=win['Low'],
                                 close=win['Close'], increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                 decreasing=dict(line=dict(color=RED, width=1), fillcolor=RED),
                                 showlegend=False, name=TICKER), row=1, col=col)
for z, op, name in ((z90, 0.13, '90% band'), (z68, 0.22, '68% band')):
    fig.add_trace(go.Scatter(x=win.index, y=prev_close * np.exp(z * sig_prev), mode='lines',
                             line=dict(width=0, shape='hvh'), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=win.index, y=prev_close * np.exp(-z * sig_prev), mode='lines',
                             line=dict(width=0, shape='hvh'), fill='tonexty',
                             fillcolor=f'rgba(42,120,214,{op})', name=f'close: {name}'), row=1, col=1)
fig.add_trace(go.Scatter(x=win.index, y=prev_close, mode='lines', line=dict(color=INK2, width=1, dash='dot', shape='hvh'),
                         name='previous close'), row=1, col=1)
for mult, dash, nm in ((0.5, 'solid', 'median'), (0.9, 'dash', '90% level')):
    fig.add_trace(go.Scatter(x=win.index, y=prev_close * (1 + up_mult.quantile(mult) * sig_prev), mode='lines',
                             line=dict(color=ORANGE, width=2, dash=dash, shape='hvh'), name=f'high: {nm}'), row=1, col=2)
    fig.add_trace(go.Scatter(x=win.index, y=prev_close * (1 - dn_mult.quantile(mult) * sig_prev), mode='lines',
                             line=dict(color=VIOLET, width=2, dash=dash, shape='hvh'), name=f'low: {nm}'), row=1, col=2)
fig.update_xaxes(rangeslider_visible=False)
style(fig, f'{TICKER}, Feb–Mar 2020 (its scheduled sell-off): what each range target asks for. Forecasts from an EWMA volatility at the previous close', 560)
save(fig, 'range_targets')

# ── 3. Walk-forward folds ────────────────────────────────────────────────────
dates = lab.index
first_train, step, rolling_len = 756, 63, 756
folds = []
start_test = first_train
while start_test < len(dates):
    end_test = min(start_test + step, len(dates))
    folds.append((start_test, end_test))
    start_test = end_test
print('folds:', len(folds), 'first test from', dates[folds[0][0]].date())

fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                    subplot_titles=['A. Expanding window: train on everything before the fold',
                                    'B. Rolling window: train on the last 3 years only'])
for col, kind in ((1, 'expanding'), (2, 'rolling')):
    for k, (a, b) in enumerate(folds):
        tr0 = 0 if kind == 'expanding' else max(0, a - rolling_len)
        for x0, x1, color, name in ((tr0, a, '#9ec5f4', 'train'), (a, b - 1, ORANGE, 'test (63 days)')):
            fig.add_trace(go.Scatter(x=[dates[x0], dates[x1]], y=[k + 1, k + 1], mode='lines',
                                     line=dict(color=color, width=7), name=name,
                                     showlegend=(col == 1 and k == 0), legendgroup=name), row=1, col=col)
fig.update_yaxes(autorange='reversed', title_text='fold', tickvals=[1, 5, 10, 15, 20, 25, 30], showgrid=False)
style(fig, 'Walk-forward on SYN-INDEX: 3 years before the first fold, then retrain every 63 trading days', 620)
save(fig, 'walk_forward')

# ── 3b. The purge at a fold boundary ─────────────────────────────────────────
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.06,
                    subplot_titles=['A. Next-day label (h = 1): nothing to purge',
                                    'B. 5-day label (h = 5): 4 training rows purged'])
t0, rows = 10, list(range(4, 14))
for col, h in ((1, 1), (2, 5)):
    for t in rows:
        if t >= t0:
            color, name = ORANGE, 'test row'
        elif t > t0 - h:
            color, name = MUTED, 'purged training row'
        else:
            color, name = '#86b6ef', 'training row'
        fig.add_trace(go.Scatter(x=[t, t + h], y=[t, t], mode='lines', line=dict(color=color, width=3),
                                 showlegend=False, hoverinfo='skip'), row=1, col=col)
        fig.add_trace(go.Scatter(x=[t], y=[t], mode='markers', marker=dict(size=12, color=color, symbol='square'),
                                 name=name, legendgroup=name,
                                 showlegend=(col == 2 and t in (rows[0], t0, t0 - 1))), row=1, col=col)
        fig.add_trace(go.Scatter(x=[t + h], y=[t], mode='markers', marker=dict(size=9, color='white', symbol='circle',
                                 line=dict(color=color, width=2)), showlegend=False, hoverinfo='skip'), row=1, col=col)
    fig.add_vline(x=t0 - 0.5, line=dict(color=INK, width=1, dash='dot'), row=1, col=col)
    fig.add_annotation(x=t0 - 0.5, y=rows[0] - 0.9, text='first test bar →', showarrow=False, xanchor='left',
                       font=dict(color=INK2, size=12), row=1, col=col)
    fig.update_yaxes(autorange='reversed', title_text='row (feature date t)' if col == 1 else None, dtick=1,
                     showgrid=False, row=1, col=col)
    fig.update_xaxes(title_text='bar: ■ feature date t, ○ label date t + h', range=[3.5, 19], dtick=1,
                     row=1, col=col)
style(fig, 'At each fold boundary, training rows whose label (t, t+h] reaches into the test window are purged', 560)
save(fig, 'purge')

# ── 4. Direction evaluation: reliability ─────────────────────────────────────
ev = pd.DataFrame({'y': (r_next > 0).astype(float), 'r': r}).dropna()
ev['base'] = ev['y'].shift(1).expanding(250).mean()            # up-frequency of every label known at t
ev['momentum'] = np.where(ev['r'] > 0, 0.62, 0.40)             # overconfident: "yesterday continues"
ev = ev.dropna()
ev = ev[ev.index >= dates[folds[0][0]]]


def logloss(p, y):
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def reliability(p, y, bins=np.linspace(0, 1, 21)):
    b = pd.cut(p, bins, include_lowest=True)
    g = pd.DataFrame({'p': p, 'y': y}).groupby(b, observed=True)
    return g.agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))


fig = go.Figure()
fig.add_trace(go.Scatter(x=[0.3, 0.7], y=[0.3, 0.7], mode='lines', line=dict(color=MUTED, dash='dot', width=1.5),
                         name='perfect calibration'))
summary = {}
for col, color, nm in (('base', BLUE, 'Base rate (expanding up-frequency)'),
                       ('momentum', ORANGE, 'Overconfident "momentum" (62% / 40%)')):
    rel = reliability(ev[col], ev['y'])
    ll, br = logloss(ev[col], ev['y']), brier(ev[col], ev['y'])
    summary[col] = (ll, br)
    fig.add_trace(go.Scatter(x=rel['p'], y=rel['y'], mode='markers',
                             marker=dict(size=np.clip(np.sqrt(rel['n']) * 0.9, 8, 40), color=color,
                                         line=dict(color='white', width=2)),
                             name=f'{nm}: log loss {ll:.4f}, Brier {br:.4f}'))
    for _, rw in rel.iterrows():
        fig.add_annotation(x=rw['p'], y=rw['y'], text=f"n = {int(rw['n'])}", showarrow=False, xshift=34,
                           yshift=-14, font=dict(size=12, color=INK2))
fig.update_xaxes(title_text='forecast probability of an up day', range=[0.3, 0.7], tickformat='.0%')
fig.update_yaxes(title_text='observed up frequency', range=[0.3, 0.7], tickformat='.0%')
style(fig, f'{TICKER}, out of sample from {dates[folds[0][0]].year}: reliability of two direction forecasts (marker area ∝ days)', 720, width=900, top=70)
fig.update_layout(legend=dict(orientation='v', y=0.98, yref='paper', x=0.02, yanchor='top', bgcolor='rgba(255,255,255,0.85)'))
save(fig, 'direction_eval')
print('direction:', {k: (round(a, 4), round(b, 4)) for k, (a, b) in summary.items()},
      'coin flip LL %.4f' % math.log(2), 'n', len(ev))

# ── 5. Range evaluation: coverage of the 68% band ────────────────────────────
TAUS = (0.05, 0.1587, 0.5, 0.8413, 0.95)
rv = pd.DataFrame({'y': r_next, 'mu': mu_exp, 's250': sig250, 's20': sig20, 'sew': sig_ewma,
                   'sor': oracle_sig, 'mor': oracle_mean}).dropna()
const_q = {tau: r.expanding(250).quantile(tau) for tau in TAUS}   # constant width: all past returns
zres = r / sig_ewma.shift(1)            # standardized returns, each known at its own close
for tau in TAUS:
    rv[f'const_{tau}'] = const_q[tau]
    rv[f'zq_{tau}'] = zres.expanding(250).quantile(tau)
rv = rv.dropna()
rv = rv[rv.index >= dates[folds[0][0]]]


def quantiles(name, tau):
    z = stats.norm.ppf(tau)
    if name == 'Constant width':
        return rv[f'const_{tau}']
    if name == 'price_range (250-day σ, normal)':
        return rv['mu'] + z * rv['s250']
    if name == 'Rolling 20-day σ (normal)':
        return rv['mu'] + z * rv['s20']
    if name == 'EWMA σ × past standardized quantiles':
        return rv['sew'] * rv[f'zq_{tau}']
    # the oracle knows the shocks are Student-t
    df = PROFILES[TICKER].tail_df
    tq = stats.t.ppf(tau, df) / math.sqrt(df / (df - 2))
    return rv['mor'] + tq * rv['sor']


def pinball(qhat, y, tau):
    u = y - qhat
    return float(np.mean(np.maximum(tau * u, (tau - 1) * u)))


def kupiec(miss, p):
    n, x = len(miss), int(miss.sum())
    ph = x / n
    lr = -2 * ((n - x) * math.log(1 - p) + x * math.log(p) - (n - x) * math.log(1 - ph) - x * math.log(ph))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen(miss):
    m = miss.astype(int).to_numpy()
    a, b = m[:-1], m[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    p01, p11 = n01 / (n00 + n01), n11 / (n10 + n11)
    p = (n01 + n11) / len(a)
    l0 = (n00 + n10) * math.log(1 - p) + (n01 + n11) * math.log(p)
    l1 = n00 * math.log(1 - p01) + n01 * math.log(p01) + n10 * math.log(1 - p11) + n11 * math.log(p11)
    lr = -2 * (l0 - l1)
    return lr, 1 - stats.chi2.cdf(lr, 1), p11 / p01


names = [('Constant width', ORANGE), ('price_range (250-day σ, normal)', YELLOW), ('Rolling 20-day σ (normal)', AQUA),
         ('EWMA σ × past standardized quantiles', BLUE), ('GARCH oracle (synthetic only)', VIOLET)]
fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.08, row_heights=[0.55, 0.45],
                    specs=[[{'colspan': 2}, None], [{}, {}]],
                    subplot_titles=['A. Coverage of the 68% interval for the next close, rolling 250 days',
                                    'B. Miss rate of the 68% interval, the day after a hit and after a miss',
                                    'C. Mean pinball loss over τ = 5%, 15.9%, 50%, 84.1%, 95% (basis points)'])
short = {'Constant width': 'Constant', 'price_range (250-day σ, normal)': 'price_range',
         'Rolling 20-day σ (normal)': '20-day σ', 'EWMA σ × past standardized quantiles': 'EWMA σ × std. q.',
         'GARCH oracle (synthetic only)': 'GARCH oracle'}
after_hit, after_miss, pinballs = [], [], []
table = []
for k, (nm, color) in enumerate(names):
    lo, hi = quantiles(nm, 0.1587), quantiles(nm, 0.8413)
    miss = (rv['y'] < lo) | (rv['y'] > hi)
    cov = 1 - miss.rolling(250).mean()
    fig.add_trace(go.Scatter(x=rv.index, y=cov * 100, mode='lines', line=dict(color=color, width=2), name=nm),
                  row=1, col=1)
    prev = miss.shift(1).fillna(False).astype(bool)
    after_hit.append(miss[~prev].mean() * 100)
    after_miss.append(miss[prev].mean() * 100)
    pb = np.mean([pinball(quantiles(nm, t), rv['y'], t) for t in TAUS])
    lo90, hi90 = quantiles(nm, 0.05), quantiles(nm, 0.95)
    miss90 = (rv['y'] < lo90) | (rv['y'] > hi90)
    lr_k, p_k = kupiec(miss, 1 - 0.6826)
    lr_c, p_c, ratio = christoffersen(miss)
    pinballs.append(pb * 1e4)
    table.append((nm, 1 - miss.mean(), 1 - miss90.mean(), pb * 1e4, p_k, p_c, ratio,
                  float((hi - lo).mean() * 100)))
fig.add_hline(y=68.26, line=dict(color=INK, width=1, dash='dot'), row=1, col=1)
fig.add_annotation(x=rv.index[5], y=68.26, text='nominal 68.3%', showarrow=False, yshift=10, xanchor='left',
                   font=dict(size=12, color=INK2), row=1, col=1)
fig.update_yaxes(ticksuffix='%', row=1, col=1)
labels = [short[n] for n, _ in names]
fig.add_bar(x=labels, y=after_hit, name='after a hit', marker=dict(color='#86b6ef'), row=2, col=1)
fig.add_bar(x=labels, y=after_miss, name='after a miss', marker=dict(color=INK2), row=2, col=1)
fig.add_bar(x=labels, y=pinballs, marker=dict(color=[c for _, c in names]), showlegend=False,
            text=[f'{v:.2f}' for v in pinballs], textposition='outside', row=2, col=2)
fig.update_yaxes(ticksuffix='%', row=2, col=1)
fig.update_yaxes(range=[15, 17.3], row=2, col=2)
fig.update_layout(bargap=0.3, bargroupgap=0.08)
style(fig, f'{TICKER}, out of sample 2019–2026: right coverage on average is not enough; misses must not cluster', 820)
save(fig, 'range_eval')
print(f"{'band':32s} cov68 cov90 pinball(bp) kupiec_p christ_p P(miss|miss)/P(miss|hit) width68(%)")
for row in table:
    print(f'{row[0]:32s} {row[1]:.3f} {row[2]:.3f} {row[3]:.2f} {row[4]:.3f} {row[5]:.2g} {row[6]:.2f} {row[7]:.2f}')
print('n days', len(rv), rv.index[0].date(), rv.index[-1].date())
