"""The evidence for the GRU recalibration proposal in docs/ml_plan.md (Phase 4), from the synthetic
tickers only.

    python scripts/ml_gru_recalibration.py [--jobs 4]

Runs the GRU's range model as it is and recalibrated on the five long synthetic tickers,
retraining every 63 rows (the plan's setting) and every 252, then writes
docs/images/ml/gru_recalibration.png and prints the numbers the plan quotes. Recalibrated
(`GRUReturnModel(recalibrate=True)`, the default since the owner chose it) means each tau's forecast is
shifted by the tau-quantile of the network's own errors on the validation tail of its training
window, the rows early stopping already holds out, so nothing outside the training window is
used. About 10 minutes on four cores. Needs the `ml` and `stats` extras, and
kaleido for the PNG (see scripts/ml_plan_charts.py).
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go                                   # noqa: E402
from plotly.subplots import make_subplots                           # noqa: E402

from src.tools import ml_models as ml                               # noqa: E402

LONG = ['SYN-INDEX', 'SYN-TECH', 'SYN-GOLD', 'SYN-FX', 'SYN-OIL']
OUT = ROOT / 'docs' / 'images' / 'ml' / 'gru_recalibration.png'


def run(ticker, variant, step):
    # The evidence was gathered on the 16 features, the plan's inputs at the time.
    model = ml.GRUReturnModel(features='features', threads=1, recalibrate=(variant == 'recalibrated'))
    d = ml.dataset(ml.ticker_bars(ticker))
    fc = ml.walk_forward_forecasts(d, model, ticker=ticker, step=step)
    table = ml.range_table(fc)
    return ticker, variant, step, table.loc['model'], table.loc['constant'], table.loc['ewma_std_q']


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    jobs = [(t, v, s) for s in (63, 252) for v in ('as is', 'recalibrated') for t in LONG]
    start = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run, *zip(*jobs)))
    print(f'{len(jobs)} runs in {time.time() - start:.0f} s\n')

    rows = {(t, v, s): (model, const, ewma) for t, v, s, model, const, ewma in results}
    print(f"{'ticker':10s} {'step':>4s} {'variant':13s} {'cover68':>8s} {'cover90':>8s} {'kupiec68':>8s} "
          f"{'pinball':>8s} {'vs const':>9s} {'vs EWMA':>8s}")
    for s in (63, 252):
        for t in LONG:
            for v in ('as is', 'recalibrated'):
                m, c, e = rows[(t, v, s)]
                print(f"{t:10s} {s:4d} {v:13s} {m.cover_68:8.1%} {m.cover_90:8.1%} {m.kupiec_p_68:8.3f} "
                      f"{m.pinball * 1e4:8.2f} {m.pinball / c.pinball - 1:+9.1%} {m.pinball / e.pinball - 1:+8.1%}")

    colors = {'as is': '#8a8984', 'recalibrated': '#eb6834'}
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.09, vertical_spacing=0.2,
                        subplot_titles=['A. 68% coverage, retrained every 63 rows',
                                        'B. Pinball loss against the constant band, every 63 rows',
                                        'C. 68% coverage, retrained every 252 rows',
                                        'D. Pinball loss against the constant band, every 252 rows'])
    for r, s in ((1, 63), (2, 252)):
        for v in ('as is', 'recalibrated'):
            cover = [rows[(t, v, s)][0].cover_68 for t in LONG]
            rel = [rows[(t, v, s)][0].pinball / rows[(t, v, s)][1].pinball - 1 for t in LONG]
            fig.add_bar(x=LONG, y=cover, name=f'GRU {v}', marker_color=colors[v], showlegend=(r == 1),
                        legendgroup=v, row=r, col=1)
            fig.add_bar(x=LONG, y=rel, name=f'GRU {v}', marker_color=colors[v], showlegend=False,
                        legendgroup=v, row=r, col=2)
        ewma = [rows[(t, 'as is', s)][2].pinball / rows[(t, 'as is', s)][1].pinball - 1 for t in LONG]
        fig.add_scatter(x=LONG, y=ewma, mode='markers', name='EWMA band (baseline)', showlegend=(r == 1),
                        marker=dict(color='#2a78d6', symbol='line-ew', size=26, line=dict(width=3, color='#2a78d6')),
                        row=r, col=2)
        fig.add_hline(y=0.6826, line=dict(color='#0b0b0b', width=1, dash='dot'), row=r, col=1)
        fig.update_yaxes(tickformat='.0%', range=[0.5, 0.75], row=r, col=1)
        fig.update_yaxes(tickformat='+.0%', row=r, col=2)
    fig.update_layout(title=dict(text='The GRU range model on the five long synthetic tickers: as it is, and recalibrated '
                                      'on its validation tail', x=0.01, font=dict(size=16)),
                      barmode='group', bargap=0.3, width=1280, height=760, paper_bgcolor='white',
                      plot_bgcolor='white', font=dict(family='Inter, Helvetica, Arial, sans-serif', size=13),
                      legend=dict(orientation='h', yanchor='bottom', y=1.07, x=0.01), margin=dict(t=130, l=70, r=30))
    fig.update_yaxes(gridcolor='#e6e5e0')
    fig.write_image(OUT, scale=1)
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
