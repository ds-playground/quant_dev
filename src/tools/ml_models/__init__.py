"""Next-day forecasts, judged out of sample against baselines: tomorrow's return (a point
forecast and its range) and tomorrow's win, draw or loss, each by its own model.

The plan, with its design choices and phases, is `docs/ml_plan.md`. Bars, past-only features,
the labels, walk-forward folds with a purge, the baselines and the metrics need no
machine-learning library; the LightGBM models (`gbm`) and the GRUs (`sequence`) need the `ml`
extra, imported only when a model is fitted. `evaluate` runs a model through the folds beside the
baselines and holds the plan's sanity checks; `registry` saves final models with their forecasts
under a unique id, to forecast with and compare against later; `viz` draws the results.

    from src.tools import ml_models as ml

    fc = ml.evaluate('SYN-INDEX', ml.LightGBMReturnModel())
    ml.point_scores(fc)             # the median's absolute error against the baselines'
    ml.range_table(fc)              # coverage, Kupiec, Christoffersen, pinball, per band
    fw = ml.evaluate('SYN-INDEX', ml.LightGBMWDLModel())
    ml.wdl_scores(fw)               # log loss against both baselines, with bootstrap intervals
    saved = ml.finalize('SYN-INDEX', ml.LightGBMWDLModel(), forecasts=fw)
    ml.save_model(saved)            # models/models/<id>.pkl and models/model_parameters/<id>.json
    ml.load_model(saved.model_id).forecast_next()   # tomorrow, without retraining
"""

from .data import bar_returns, ticker_bars
from .features import (FEATURES, STREAK_CAP, WDL_THRESHOLD, ewma_vol, features, streak,
                       wilder_rsi, win_draw_loss)
from .targets import TAUS, dataset, next_return, next_wdl
from .split import purged_tail, walk_forward
from .baselines import BANDS, WDL_BASELINES, WDL_CLASSES, range_baselines, wdl_baselines
from .metrics import (bootstrap_means, brier, brier_decomposition, christoffersen,
                      interval_misses, kupiec, log_loss, mean_interval, multiclass_brier,
                      multiclass_log_loss, pinball, range_scores, reliability_table)
from .gbm import GBM_PARAMS, LightGBMReturnModel, LightGBMWDLModel
from .sequence import GRU_INPUTS, GRUReturnModel, GRUWDLModel, quantile_shift, windows
from .evaluate import (Forecasts, compare_forecasts, daily_losses, evaluate, evaluate_many,
                       feature_importance, planted_signal_bars, point_scores, range_check,
                       range_table, walk_forward_forecasts, wdl_check, wdl_scores)
from .registry import (MODELS_DIR, PARAMETERS_DIR, SavedModel, code_hash, data_fingerprint,
                       finalize, finalize_many, forecast_next, list_models, load_model, model_id,
                       model_spec, save_model)
from .viz import (plot_coverage, plot_feature_importance, plot_folds, plot_forecast_bands,
                  plot_range_comparison, plot_wdl_reliability)

__all__ = ['ticker_bars', 'bar_returns',
           'FEATURES', 'features', 'ewma_vol', 'wilder_rsi',
           'WDL_THRESHOLD', 'STREAK_CAP', 'win_draw_loss', 'streak',
           'TAUS', 'next_return', 'next_wdl', 'dataset',
           'walk_forward', 'purged_tail',
           'BANDS', 'range_baselines',
           'WDL_CLASSES', 'WDL_BASELINES', 'wdl_baselines',
           'log_loss', 'brier', 'brier_decomposition', 'reliability_table',
           'multiclass_log_loss', 'multiclass_brier',
           'pinball', 'interval_misses', 'kupiec', 'christoffersen', 'range_scores',
           'bootstrap_means', 'mean_interval',
           'GBM_PARAMS', 'LightGBMReturnModel', 'LightGBMWDLModel',
           'GRU_INPUTS', 'GRUReturnModel', 'GRUWDLModel', 'windows', 'quantile_shift',
           'Forecasts', 'walk_forward_forecasts', 'evaluate', 'evaluate_many',
           'wdl_scores', 'wdl_check', 'point_scores', 'range_table', 'range_check',
           'feature_importance', 'daily_losses', 'compare_forecasts',
           'planted_signal_bars',
           'MODELS_DIR', 'PARAMETERS_DIR', 'SavedModel', 'model_spec', 'data_fingerprint',
           'model_id', 'code_hash', 'finalize', 'finalize_many', 'save_model', 'load_model',
           'list_models', 'forecast_next',
           'plot_wdl_reliability', 'plot_coverage', 'plot_forecast_bands', 'plot_folds',
           'plot_feature_importance', 'plot_range_comparison']
