"""The LightGBM models against the plan's sanity checks, offline, on the synthetic tickers.

The synthetic tickers draw i.i.d. Student-t shocks through GARCH(1,1): tomorrow's sign cannot be
forecast, its volatility can. So a direction model better than the base rate is a bug, and a
range model no better than a constant-width band is one too (`docs/ml_plan.md`). Controls show
the checks can fail: a planted signal must be found, and two leaks must be caught. Skipped
without the `ml` extra; marked slow (about 35 s), so `pytest -m "not slow"` leaves it out.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('lightgbm')
pytestmark = pytest.mark.slow

from src.tools import ml_models as ml                                   # noqa: E402
from src.tools.price_return import synthetic_bars, synthetic_tickers    # noqa: E402

LONG = [t for t in synthetic_tickers() if t != 'SYN-LEV']   # SYN-LEV is too short to judge
MODEL = ml.LightGBMModel()
LEAKY = dict(features=ml.FEATURES + ['leak'], max_trees=60)  # a leak needs few trees to show


@pytest.fixture(scope='module')
def long_forecasts():
    """Both models on the five long tickers, fold by fold, one process per ticker."""
    return {fc.ticker: fc for fc in ml.evaluate_many(LONG, MODEL)}


@pytest.fixture(scope='module')
def index_data():
    return ml.dataset(synthetic_bars('SYN-INDEX'))


@pytest.mark.parametrize('ticker', LONG)
def test_direction_is_no_better_than_chance(long_forecasts, ticker):
    s = ml.direction_scores(long_forecasts[ticker], level=0.99, margin=0.002)
    assert s['no_better_than_chance'], (f"{ticker}: log-loss gain {s['gain']:+.4f}, 99% interval "
                                        f"{s['lower']:+.4f} to {s['upper']:+.4f}")
    assert 0 < s['log_loss_model'] < 0.75                       # and not absurd either


def test_range_beats_the_constant_band(long_forecasts):
    table, pooled, passes = ml.range_check(list(long_forecasts.values()), level=0.95)
    assert passes, f'{table}\npooled {pooled.to_dict()}'
    assert table['lower_pinball'].all() and table['less_clustering'].all()


def test_range_forecasts_are_ordered_and_scaled(long_forecasts):
    fc = long_forecasts['SYN-INDEX']
    q = fc.bands.xs('model', axis=1, level='band').to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()                      # no crossed quantiles
    scores = ml.range_table(fc).loc['model']
    assert 0.6 < scores['cover_68'] < 0.76 and 0.84 < scores['cover_90'] < 0.95


def test_planted_signal_is_found():
    d = ml.dataset(ml.planted_signal_bars(repeat=0.6))
    s = ml.direction_scores(ml.walk_forward_forecasts(d, MODEL, ranges=False), level=0.99)
    assert s['beats_base_rate'], f"gain {s['gain']:+.4f}, 99% interval from {s['lower']:+.4f}"


def test_leaked_next_return_is_caught(index_data):
    d = index_data.assign(leak=index_data['y_ret'])
    model = ml.LightGBMModel(**LEAKY)
    s = ml.direction_scores(ml.walk_forward_forecasts(d, model, ranges=False), level=0.99)
    assert not s['no_better_than_chance'] and s['beats_base_rate']


def test_centred_window_feature_is_caught(index_data):
    # A realistic look-ahead bug: a centred 5-day average carries r(t+1) and r(t+2).
    r = ml.bar_returns(synthetic_bars('SYN-INDEX'))
    d = index_data.assign(leak=r.rolling(5, center=True).mean().reindex(index_data.index)).dropna()
    model = ml.LightGBMModel(**LEAKY)
    s = ml.direction_scores(ml.walk_forward_forecasts(d, model, ranges=False), level=0.99)
    assert not s['no_better_than_chance']


@pytest.mark.parametrize('position', [900, 1700, 2400])
def test_forecast_is_unchanged_when_later_bars_are_removed(long_forecasts, position):
    """End to end: bars, features, folds, fit. Keep the bars up to the forecast date and one more
    (whose close is only that date's label), and the forecast must be identical to the bit."""
    bars = synthetic_bars('SYN-INDEX')
    full = long_forecasts['SYN-INDEX']
    date = ml.dataset(bars).index[position]
    d = ml.dataset(bars.iloc[:bars.index.get_loc(date) + 2])
    assert d.index[-1] == date
    fold = next(f for f in ml.walk_forward(len(d)) if date in d.index[f[1]])
    cut = ml.walk_forward_forecasts(d, MODEL, folds=[fold])
    assert cut.direction.loc[date, 'p_model'] == full.direction.loc[date, 'p_model']
    pd.testing.assert_series_equal(cut.bands.loc[date, 'model'], full.bands.loc[date, 'model'],
                                   check_exact=True)


def test_direction_model_learns_almost_nothing(long_forecasts):
    # On unpredictable data early stopping keeps very few trees: the forecasts hug the base rate.
    trees = pd.concat([fc.folds['direction_trees'] for fc in long_forecasts.values()])
    assert trees.median() <= 20
    spread = [fc.direction['p_model'].std() for fc in long_forecasts.values()]
    assert max(spread) < 0.05
