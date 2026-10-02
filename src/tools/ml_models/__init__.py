"""Next-day forecasts: direction and price range, judged out of sample against baselines.

The plan, with its design choices and phases, is `docs/ml_plan.md`. This is the part that needs
no machine-learning library: bars, past-only features, the labels, walk-forward folds with a
purge, the baselines and the metrics. The LightGBM and PyTorch models (the `ml` extra) build on
it in later phases.

    from src.tools.ml_models import ticker_bars, dataset, walk_forward, range_baselines

    d = dataset(ticker_bars('SYN-INDEX'))
    folds = walk_forward(len(d))
    bands = range_baselines(d, folds)
"""

from .data import bar_returns, ticker_bars
from .features import FEATURES, ewma_vol, features, wilder_rsi
from .targets import TAUS, dataset, next_direction, next_return
from .split import purged_tail, walk_forward
from .baselines import BANDS, direction_baseline, range_baselines
from .metrics import (bootstrap_means, brier, brier_decomposition, christoffersen,
                      interval_misses, kupiec, log_loss, mean_interval, pinball, range_scores,
                      reliability_table)

__all__ = ['ticker_bars', 'bar_returns',
           'FEATURES', 'features', 'ewma_vol', 'wilder_rsi',
           'TAUS', 'next_return', 'next_direction', 'dataset',
           'walk_forward', 'purged_tail',
           'BANDS', 'direction_baseline', 'range_baselines',
           'log_loss', 'brier', 'brier_decomposition', 'reliability_table',
           'pinball', 'interval_misses', 'kupiec', 'christoffersen', 'range_scores',
           'bootstrap_means', 'mean_interval']
