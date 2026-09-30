"""The chart registry: each name maps to one `viz.plot_*` call on the same inputs the notebooks use.

A chart builder takes the prepared parameters and the request's options and returns a Plotly
figure. Options a chart does not use are ignored; those it uses default as in the notebooks.
"""
from dataclasses import dataclass

from src.tools.price_return import (analyze_cumulative, autocorrelation, detect_streaks,
                                    plot_autocorrelation, plot_cumulative_counts,
                                    plot_cumulative_heatmap, plot_drawdown,
                                    plot_event_probabilities, plot_price_and_returns, plot_qq,
                                    plot_return_distribution, plot_rolling_average,
                                    plot_rolling_risk, plot_rolling_volatility,
                                    plot_streak_counts, plot_streak_frequency,
                                    plot_streak_timeline, rolling_risk)

from . import cache


@dataclass(frozen=True)
class ChartOptions:
    window: int | None = None        # streak-timeline: streak window; rolling-risk: rolling days
    top: int = 3                     # drawdown: troughs labelled
    n_days: int = 3                  # event-probabilities: holding period
    n_boot: int = 1000               # event-probabilities: bootstrap resamples
    change_type: str = 'cumulative'  # event-probabilities
    change: str = 'above'            # event-probabilities
    n_years: int | None = None       # event-probabilities: lookback (default: the longest)


def _rolling_risk(p, o):
    window = o.window or p.trade_days
    return plot_rolling_risk(rolling_risk(cache.returns(p), window, p.trade_days), window=window)


def _event_probabilities(p, o):
    n_years = o.n_years or max(p.lookback_years)
    return plot_event_probabilities(cache.events(p, o.n_days, o.n_boot), o.change_type, o.change,
                                    n_years=n_years)


def _streaks(p):
    df = cache.prepared_data(p)
    return df, detect_streaks(df, p)


# name -> (group, options it reads, builder(params, options) -> figure)
CHARTS = {
    'rolling-average':     ('analysis', (), lambda p, o: plot_rolling_average(cache.prepared_data(p), p)),
    'rolling-volatility':  ('analysis', (), lambda p, o: plot_rolling_volatility(cache.prepared_data(p), p)),
    'price-and-returns':   ('analysis', (), lambda p, o: plot_price_and_returns(cache.prepared_data(p), p)),
    'return-distribution': ('analysis', (), lambda p, o: plot_return_distribution(cache.prepared_data(p), p)),
    'streak-counts':       ('analysis', (), lambda p, o: plot_streak_counts(_streaks(p)[1], p)),
    'streak-frequency':    ('analysis', (), lambda p, o: plot_streak_frequency(*_streaks(p), p)),
    'streak-timeline':     ('analysis', ('window',),
                            lambda p, o: plot_streak_timeline(*_streaks(p), p, window=o.window)),
    'cumulative-heatmap':  ('analysis', (), lambda p, o: plot_cumulative_heatmap(
                                analyze_cumulative(cache.prepared_data(p), p), p)),
    'cumulative-counts':   ('analysis', (), lambda p, o: plot_cumulative_counts(
                                analyze_cumulative(cache.prepared_data(p), p), p)),
    'qq':                  ('statistics', (), lambda p, o: plot_qq(cache.returns(p))),
    'autocorrelation':     ('statistics', (), lambda p, o: plot_autocorrelation(
                                autocorrelation(cache.returns(p)))),
    'drawdown':            ('statistics', ('top',), lambda p, o: plot_drawdown(cache.returns(p), top=o.top)),
    'rolling-risk':        ('statistics', ('window',), _rolling_risk),
    'event-probabilities': ('statistics', ('n_days', 'n_boot', 'change_type', 'change', 'n_years'),
                            _event_probabilities),
}
