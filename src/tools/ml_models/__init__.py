"""Next-day forecasts: direction and price range, judged out of sample against baselines.

The plan, with its design choices and phases, is `docs/ml_plan.md`. Bars, past-only features,
the labels, walk-forward folds with a purge, the baselines and the metrics need no
machine-learning library; the LightGBM model (`gbm`) and the GRU (`sequence`) need the `ml`
extra, imported only when a model is fitted. `evaluate` runs a model through the folds beside the
baselines and holds the plan's sanity checks; `viz` draws the results.

    from src.tools.ml_models import LightGBMModel, evaluate, direction_scores, range_table

    fc = evaluate('SYN-INDEX', LightGBMModel())
    direction_scores(fc)            # log loss against the base rate, with a bootstrap interval
    range_table(fc)                 # coverage, Kupiec, Christoffersen, pinball, per band
"""

from .data import bar_returns, ticker_bars
from .features import FEATURES, ewma_vol, features, wilder_rsi
from .targets import TAUS, dataset, next_direction, next_return
from .split import purged_tail, walk_forward
from .baselines import BANDS, direction_baseline, range_baselines
from .metrics import (bootstrap_means, brier, brier_decomposition, christoffersen,
                      interval_misses, kupiec, log_loss, mean_interval, pinball, range_scores,
                      reliability_table)
from .gbm import GBM_PARAMS, LightGBMModel
from .sequence import GRU_INPUTS, GRUModel, quantile_shift, windows
from .evaluate import (Forecasts, direction_scores, evaluate, evaluate_many, feature_importance,
                       planted_signal_bars, range_check, range_table, walk_forward_forecasts)
from .viz import (plot_coverage, plot_feature_importance, plot_folds, plot_forecast_bands,
                  plot_range_comparison, plot_reliability)

__all__ = ['ticker_bars', 'bar_returns',
           'FEATURES', 'features', 'ewma_vol', 'wilder_rsi',
           'TAUS', 'next_return', 'next_direction', 'dataset',
           'walk_forward', 'purged_tail',
           'BANDS', 'direction_baseline', 'range_baselines',
           'log_loss', 'brier', 'brier_decomposition', 'reliability_table',
           'pinball', 'interval_misses', 'kupiec', 'christoffersen', 'range_scores',
           'bootstrap_means', 'mean_interval',
           'GBM_PARAMS', 'LightGBMModel', 'GRU_INPUTS', 'GRUModel', 'windows', 'quantile_shift',
           'Forecasts', 'walk_forward_forecasts', 'evaluate', 'evaluate_many',
           'direction_scores', 'range_table', 'range_check', 'feature_importance',
           'planted_signal_bars',
           'plot_reliability', 'plot_coverage', 'plot_forecast_bands', 'plot_folds',
           'plot_feature_importance', 'plot_range_comparison']
