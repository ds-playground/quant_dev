"""Per-ticker pipeline and cross-ticker comparison."""

import numpy as np
import pandas as pd

from .analysis import (build_historical_analysis, detect_streaks,
                       distribution_summary, low_probability_view,
                       summarize_streaks)
from .data import add_rolling_stats, daily_returns_series, load_price_data


def analyze_ticker(p, drill_n_days=3, verbose=False):
    """Run the whole pipeline for one ticker; raises so the caller can skip it."""
    df = add_rolling_stats(load_price_data(p, verbose=verbose), p)
    df_his = build_historical_analysis(daily_returns_series(df), p)
    streaks = detect_streaks(df, p)
    return {
        'params':         p,
        'df':             df,
        'df_his':         df_his,
        'drill':          low_probability_view(df_his, drill_n_days, p),
        'streaks':        streaks,
        'streak_summary': summarize_streaks(df, streaks, p),
        'dist':           distribution_summary(df, p),
    }

def compare_tickers(results, ratio_window=3):
    """Cross-ticker streak and distribution tables from `analyze_ticker` outputs.

    `results` maps ticker -> the dict `analyze_ticker` returns. Reuses each
    ticker's `streak_summary` rather than recounting streaks.
    """
    streak_rows, dist_rows = [], []
    for symbol, r in results.items():
        p = r['params']
        name = getattr(p, 'label', None) or symbol
        row = {'ticker': name}
        by_window = r['streak_summary'].set_index('Window')
        ratio = np.nan

        for w in p.windows:
            win, loss = by_window.loc[f'{w}d', ['Win Freq %', 'Loss Freq %']]
            row[f'{w}d win/loss'] = f'{win:.2f} / {loss:.2f}'
            if w == ratio_window:
                ratio = round(win / loss, 2) if loss else np.nan

        row[f'{ratio_window}d ratio'] = ratio       # keep the ratio last
        streak_rows.append(row)
        dist_rows.append(r['dist'])

    streak_table = pd.DataFrame(streak_rows)
    dist_table = pd.concat(dist_rows, ignore_index=True)
    return streak_table, dist_table
