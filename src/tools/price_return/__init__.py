"""Historical price & return analysis.

Helpers for the `price_return_analysis` notebooks: loading a price series
(simulated or Yahoo Finance), deriving rolling statistics, detecting win/loss
streaks, measuring cumulative-threshold and rare-event probabilities, and
drawing the Plotly charts for each.

Every tunable value lives on `Params`, so a notebook configures once and passes
that object around::

    from src.tools.price_return import Params, load_price_data, add_rolling_stats

    P  = Params(ticker='ES=F', start_date='2016-01-01')
    df = add_rolling_stats(load_price_data(P), P)

The implementation is split across `params`, `data`, `analysis`, `viz`, `report`,
`pipeline` and `options`; everything public is re-exported here, so importing from
`src.tools.price_return` works exactly as it did when this was one module.
"""

from .params import Params, load_ticker_config
from .data import (load_price_data, add_rolling_stats, latest_snapshot,
                   show_latest_snapshot, daily_returns_series, compound_returns)
from .analysis import (detect_streaks, summarize_streaks, analyze_cumulative,
                       summarize_cumulative, consecutive_analysis,
                       build_historical_analysis, filter_low_probability,
                       low_probability_view, distribution_summary)
from .viz import (plot_rolling_average, plot_rolling_volatility,
                  plot_price_and_returns, plot_return_distribution,
                  plot_streak_counts, plot_streak_frequency, plot_streak_timeline,
                  plot_cumulative_heatmap, plot_cumulative_counts)
from .report import (format_probability_table, interactive_low_probability,
                     export_tables)
from .pipeline import analyze_ticker, compare_tickers
from .options import price_range, move_probabilities, expected_pnl

__all__ = [
    'Params',
    # data
    'load_price_data', 'add_rolling_stats', 'latest_snapshot', 'show_latest_snapshot',
    'daily_returns_series', 'compound_returns',
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
    # multi-ticker
    'load_ticker_config', 'analyze_ticker', 'distribution_summary', 'compare_tickers',
    # option sizing
    'price_range', 'move_probabilities', 'expected_pnl',
]
