"""The two LightGBM models against the plan's sanity checks, offline, on the synthetic tickers.

The synthetic tickers draw i.i.d. Student-t shocks through GARCH(1,1): tomorrow's sign cannot be
forecast, its volatility can. So:

- the win/draw/loss model may use volatility (draws are likelier in calm spells) and must beat
  the plain class frequencies, but must be no better than the volatility baseline;
- the return model's point forecast must be no better than the EWMA band's median, and its
  range must beat a constant-width band (`docs/ml_plan.md`).

Controls show the checks can fail: a planted signal must be found, and two leaks must be caught.
Skipped without the `ml` extra; marked slow, so `pytest -m "not slow"` leaves it out.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('lightgbm')
pytestmark = pytest.mark.slow

from src.tools import ml_models as ml                                   # noqa: E402
from src.tools.price_return import synthetic_bars, synthetic_tickers    # noqa: E402

LONG = [t for t in synthetic_tickers() if t != 'SYN-LEV']   # SYN-LEV is too short to judge
LEAKY = dict(features=ml.FEATURES + ['leak'], max_trees=60)  # a leak needs few trees to show


@pytest.fixture(scope='module')
def wdl_forecasts():
    """The win/draw/loss model on the five long tickers, one process per ticker."""
    return {fc.ticker: fc for fc in ml.evaluate_many(LONG, ml.LightGBMWDLModel())}


@pytest.fixture(scope='module')
def return_forecasts():
    """The return model on the five long tickers, one process per ticker."""
    return {fc.ticker: fc for fc in ml.evaluate_many(LONG, ml.LightGBMReturnModel())}


@pytest.fixture(scope='module')
def index_data():
    return ml.dataset(synthetic_bars('SYN-INDEX'))


# ── Win/draw/loss ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('ticker', LONG)
def test_wdl_is_no_better_than_the_volatility_baseline(wdl_forecasts, ticker):
    s = ml.wdl_scores(wdl_forecasts[ticker], level=0.99)
    assert s['no_better_than_volatility'], (
        f"{ticker}: log-loss gain {s['gain_vs_volatility']:+.4f}, 99% interval "
        f"{s['lower_vs_volatility']:+.4f} to {s['upper_vs_volatility']:+.4f}")
    assert 0.9 < s['log_loss_model'] < 1.15                     # and not absurd either (log 3 = 1.099)


def test_wdl_beats_the_class_frequencies(wdl_forecasts):
    # Volatility is forecastable, and with a fixed threshold it sets how likely a draw is.
    gains, pooled, above = ml.wdl_check(list(wdl_forecasts.values()), 'frequencies', 'model')
    assert above, f'{gains.to_dict()}\npooled {pooled.to_dict()}'


def test_wdl_volatility_baseline_beats_the_frequencies(wdl_forecasts):
    gains, pooled, above = ml.wdl_check(list(wdl_forecasts.values()), 'frequencies', 'volatility')
    assert above and (gains > 0).all(), f'{gains.to_dict()}\npooled {pooled.to_dict()}'


def test_wdl_leans_on_volatility(wdl_forecasts):
    shares = pd.concat([ml.feature_importance(fc) for fc in wdl_forecasts.values()], axis=1).mean(axis=1)
    volatility = shares[['vol_20', 'vol_60', 'vol_ewma', 'vol_parkinson', 'vol_gk', 'tr_close', 'abs_ret']]
    assert volatility.sum() > 0.5, shares.sort_values(ascending=False).head(8).to_dict()


def test_wdl_forecasts_are_probabilities(wdl_forecasts):
    p = wdl_forecasts['SYN-INDEX'].wdl['model'].to_numpy()
    assert ((p > 0) & (p < 1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_planted_signal_is_found():
    d = ml.dataset(ml.planted_signal_bars())                    # each sign repeats 65% of the time
    s = ml.wdl_scores(ml.walk_forward_forecasts(d, ml.LightGBMWDLModel()), level=0.99)
    assert s['beats_volatility'], (f"gain {s['gain_vs_volatility']:+.4f}, 99% interval from "
                                   f"{s['lower_vs_volatility']:+.4f}")


def test_leaked_next_return_is_caught(index_data):
    d = index_data.assign(leak=index_data['y_ret'])
    s = ml.wdl_scores(ml.walk_forward_forecasts(d, ml.LightGBMWDLModel(**LEAKY)), level=0.99)
    assert not s['no_better_than_volatility'] and s['beats_volatility']


def test_centred_window_feature_is_caught(index_data):
    # A realistic look-ahead bug: a centred 5-day average carries r(t+1) and r(t+2).
    r = ml.bar_returns(synthetic_bars('SYN-INDEX'))
    d = index_data.assign(leak=r.rolling(5, center=True).mean().reindex(index_data.index)).dropna()
    d.attrs = index_data.attrs
    s = ml.wdl_scores(ml.walk_forward_forecasts(d, ml.LightGBMWDLModel(**LEAKY)), level=0.99)
    assert not s['no_better_than_volatility']


# ── Return: the point forecast and its range ─────────────────────────────────

@pytest.mark.parametrize('ticker', LONG)
def test_point_forecast_is_no_better_than_the_ewma_median(return_forecasts, ticker):
    s = ml.point_scores(return_forecasts[ticker], level=0.99)
    assert s['no_better_than_ewma'], (f"{ticker}: gain {s['gain_vs_ewma']:+.6f}, 99% interval "
                                      f"{s['lower_vs_ewma']:+.6f} to {s['upper_vs_ewma']:+.6f}")


def test_range_beats_the_constant_band(return_forecasts):
    table, pooled, passes = ml.range_check(list(return_forecasts.values()), level=0.95)
    assert passes, f'{table}\npooled {pooled.to_dict()}'
    assert table['lower_pinball'].all() and table['less_clustering'].all()


def test_range_forecasts_are_ordered_and_scaled(return_forecasts):
    fc = return_forecasts['SYN-INDEX']
    q = fc.bands.xs('model', axis=1, level='band').to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()                      # no crossed quantiles
    scores = ml.range_table(fc).loc['model']
    assert 0.6 < scores['cover_68'] < 0.76 and 0.84 < scores['cover_90'] < 0.95


# ── No look-ahead, end to end, for both models ───────────────────────────────

@pytest.mark.parametrize('position', [900, 1700, 2400])
def test_forecast_is_unchanged_when_later_bars_are_removed(wdl_forecasts, return_forecasts, position):
    """Bars, features, folds, fit. Keep the bars up to the forecast date and one more (whose close
    is only that date's label), and both forecasts must be identical to the bit."""
    bars = synthetic_bars('SYN-INDEX')
    date = ml.dataset(bars).index[position]
    d = ml.dataset(bars.iloc[:bars.index.get_loc(date) + 2])
    assert d.index[-1] == date
    fold = next(f for f in ml.walk_forward(len(d)) if date in d.index[f[1]])
    wdl = ml.walk_forward_forecasts(d, ml.LightGBMWDLModel(), folds=[fold])
    pd.testing.assert_series_equal(wdl.wdl.loc[date, 'model'], wdl_forecasts['SYN-INDEX'].wdl.loc[date, 'model'],
                                   check_exact=True)
    ret = ml.walk_forward_forecasts(d, ml.LightGBMReturnModel(), folds=[fold])
    pd.testing.assert_series_equal(ret.bands.loc[date, 'model'], return_forecasts['SYN-INDEX'].bands.loc[date, 'model'],
                                   check_exact=True)
