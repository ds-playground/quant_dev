"""The two scikit-learn MLPs (return, win/draw/loss), offline, on the synthetic tickers. Skipped
without scikit-learn; marked slow, as the other model tests.

At the owner's starting configuration (two hidden layers as wide as the feature list, ReLU, Adam,
scikit-learn's defaults otherwise) the networks overfit the noise: their W/D/L log loss and point
error are well above the baselines'. They still pass the checks that guard against look-ahead
("no better than" the baselines), but not the range check or the planted signal. Both are kept as
strict expected failures, so a configuration that fixes them shows up; more regularization moves
every score towards the baselines (docs/ml_plan.md, *The MLP*).
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn')
pytestmark = pytest.mark.slow

from sklearn.neural_network import MLPClassifier, MLPRegressor          # noqa: E402

from src.tools import ml_models as ml                                   # noqa: E402
from src.tools.price_return import synthetic_bars                       # noqa: E402

LONG = ['SYN-INDEX', 'SYN-TECH', 'SYN-GOLD', 'SYN-FX', 'SYN-OIL']
QUICK = dict(params={'max_iter': 20})          # enough to exercise training, not to judge it


@pytest.fixture(scope='module')
def index_data():
    return ml.dataset(synthetic_bars('SYN-INDEX'))


# ── The model ────────────────────────────────────────────────────────────────

def test_the_starting_configuration_is_the_owners():
    for cls, sk in ((ml.MLPReturnModel, MLPRegressor), (ml.MLPWDLModel, MLPClassifier)):
        model = cls()
        assert model.hidden_layer_sizes == (len(ml.FEATURES),) * 2 == (18, 18)
        params = model._estimator(sk, 1000).get_params()
        defaults = sk().get_params()
        changed = {k for k in defaults if params[k] != defaults[k]}
        # only the layers (the owner's) and the seed (determinism) differ from scikit-learn's
        assert changed == {'hidden_layer_sizes', 'random_state'}, changed
        assert params['activation'] == 'relu' and params['solver'] == 'adam'
        assert params['random_state'] == 1000                  # seed 0 + the fold's first test row
    assert ml.MLPReturnModel(features=['ret_0', 'vol_20']).hidden_layer_sizes == (2, 2)
    assert ml.MLPWDLModel(params={'alpha': 1.0})._estimator(MLPClassifier, 0).alpha == 1.0


@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize('cls', [ml.MLPWDLModel, ml.MLPReturnModel])
def test_forecasts_are_scikit_learns(index_data, cls):
    """The forecasts are computed from the trained weights; they must be scikit-learn's own."""
    train, test = ml.walk_forward(len(index_data))[0]
    model = cls(**QUICK)
    state = model.fit(index_data, train, fold_start=test[0])
    X = index_data[model.features].to_numpy(dtype=float)
    Xs = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-12)
    if cls is ml.MLPWDLModel:
        net = model._estimator(MLPClassifier, test[0]).fit(Xs[train], index_data['y_wdl'].to_numpy()[train].astype(int))
        ref = net.predict_proba(Xs[test])
    else:
        fit, _ = ml.purged_tail(train)
        y = (index_data['y_ret'] / index_data['vol_ewma']).to_numpy()
        net = model._estimator(MLPRegressor, test[0]).fit(Xs[fit], y[fit])
        ref = net.predict(Xs[test])[:, None]
    np.testing.assert_allclose(model._forward(state, index_data, test), ref, rtol=0, atol=1e-12)
    if cls is ml.MLPWDLModel:              # columns in scikit-learn's class order: loss, draw, win
        assert list(net.classes_) == list(ml.WDL_CLASSES)
        np.testing.assert_allclose(model.predict(state, index_data, test), ref, rtol=0, atol=1e-12)
    assert state['n_iter'] == 20 and not state['converged']    # reported, as the fold info


def test_forecasts_are_probabilities_and_ordered_quantiles(index_data):
    train, test = ml.walk_forward(len(index_data))[0]
    p, info = ml.MLPWDLModel(**QUICK).wdl(index_data, train, test)
    assert p.shape == (len(test), 3) and ((p > 0) & (p < 1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert set(info) == {'n_iter', 'converged'}
    q, info = ml.MLPReturnModel(**QUICK).quantiles(index_data, train, test, ml.TAUS)
    assert q.shape == (len(test), len(ml.TAUS)) and (np.diff(q, axis=1) >= 0).all()
    assert len(info['shift']) == len(ml.TAUS)


def test_the_range_is_the_centre_plus_its_errors_on_the_held_out_tail(index_data):
    train, test = ml.walk_forward(len(index_data))[1]
    model = ml.MLPReturnModel(**QUICK)
    state = model.fit(index_data, train, ml.TAUS, fold_start=test[0])
    fit, val = ml.purged_tail(train)
    y = (index_data['y_ret'] / index_data['vol_ewma']).to_numpy()
    centre = model._forward(state, index_data, val)[:, 0]
    np.testing.assert_allclose(state['shift'], [np.quantile(y[val] - centre, t) for t in ml.TAUS])
    q = model.predict(state, index_data, test)
    sigma = index_data['vol_ewma'].to_numpy()[test]
    np.testing.assert_allclose(q[:, 2], (model._forward(state, index_data, test)[:, 0] + state['shift'][2]) * sigma)


def test_mlp_is_deterministic_and_seeded_per_fold(index_data):
    train, test = ml.walk_forward(len(index_data))[0]
    a = ml.MLPWDLModel(**QUICK).fit(index_data, train, fold_start=test[0])
    b = ml.MLPWDLModel(**QUICK).fit(index_data, train, fold_start=test[0])
    c = ml.MLPWDLModel(**QUICK).fit(index_data, train, fold_start=test[0] + 1)
    for wa, wb, wc in zip(a['coefs'], b['coefs'], c['coefs']):
        np.testing.assert_array_equal(wa, wb)
        assert not np.array_equal(wa, wc)
    assert ml.MLPWDLModel(**QUICK).fit(index_data, train)['coefs'][0].tolist() == a['coefs'][0].tolist()


@pytest.mark.parametrize('cls', [ml.MLPWDLModel, ml.MLPReturnModel])
def test_a_days_forecast_does_not_depend_on_the_days_forecast_with_it(index_data, cls):
    train, test = ml.walk_forward(len(index_data))[0]
    model = cls(**QUICK)
    state = model.fit(index_data, train)
    every = model.predict(state, index_data, test)
    for some in (test[:5], test[20:37], test[-1:]):
        np.testing.assert_array_equal(model.predict(state, index_data, some),
                                      every[np.searchsorted(test, some)])


def test_standardized_with_the_training_rows_only(index_data):
    train, test = ml.walk_forward(len(index_data))[0]
    state = ml.MLPWDLModel(**QUICK).fit(index_data, train)
    X = index_data[ml.FEATURES].to_numpy(dtype=float)
    np.testing.assert_allclose(state['mu'], X[train].mean(axis=0))
    np.testing.assert_allclose(state['sd'], X[train].std(axis=0) + 1e-12)


@pytest.mark.parametrize('position', [900, 2000])
def test_forecast_is_unchanged_when_later_bars_are_removed(position):
    """End to end, one fold: keep the bars up to the forecast date and one more (whose close is
    only that date's label); both models' forecasts must be identical to the bit."""
    bars = synthetic_bars('SYN-INDEX')
    full_data = ml.dataset(bars)
    date = full_data.index[position]
    cut_data = ml.dataset(bars.iloc[:bars.index.get_loc(date) + 2])
    fold = next(f for f in ml.walk_forward(len(cut_data)) if date in cut_data.index[f[1]])
    full_fold = next(f for f in ml.walk_forward(len(full_data)) if date in full_data.index[f[1]])
    for model in (ml.MLPWDLModel(**QUICK), ml.MLPReturnModel(**QUICK)):
        full = ml.walk_forward_forecasts(full_data, model, folds=[full_fold])
        cut = ml.walk_forward_forecasts(cut_data, model, folds=[fold])
        frame = 'wdl' if model.target == 'wdl' else 'bands'
        pd.testing.assert_series_equal(getattr(cut, frame).loc[date, 'model'],
                                       getattr(full, frame).loc[date, 'model'], check_exact=True)


# ── The sanity checks, on the five long tickers ──────────────────────────────

def _job(job):
    import warnings
    warnings.filterwarnings('ignore')
    kind, arg = job
    if kind == 'wdl':
        return ml.evaluate(arg, ml.MLPWDLModel())
    if kind == 'return':
        return ml.evaluate(arg, ml.MLPReturnModel())
    if kind == 'planted signal':
        return ml.walk_forward_forecasts(ml.dataset(ml.planted_signal_bars()), ml.MLPWDLModel())
    d = ml.dataset(synthetic_bars('SYN-INDEX'))
    if kind == 'leaked r(t+1)':
        d = d.assign(leak=d['y_ret'])
    else:                                   # a centred 5-day average: carries r(t+1) and r(t+2)
        r = ml.bar_returns(synthetic_bars('SYN-INDEX'))
        leak = r.rolling(5, center=True).mean().reindex(d.index)
        d, attrs = d.assign(leak=leak).dropna(), d.attrs
        d.attrs = attrs
    return ml.walk_forward_forecasts(d, ml.MLPWDLModel(features=ml.FEATURES + ['leak']))


@pytest.fixture(scope='module')
def runs():
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor
    jobs = ([('wdl', t) for t in LONG] + [('return', t) for t in LONG]
            + [('planted signal', None), ('leaked r(t+1)', None), ('centred window', None)])
    with ProcessPoolExecutor(min(len(jobs), os.cpu_count() or 1),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        return dict(zip(jobs, pool.map(_job, jobs)))


@pytest.mark.parametrize('ticker', LONG)
def test_wdl_is_no_better_than_the_volatility_baseline(runs, ticker):
    s = ml.wdl_scores(runs[('wdl', ticker)], level=0.99)
    assert s['no_better_than_volatility'], (f"{ticker}: gain {s['gain_vs_volatility']:+.4f}, 99% "
                                            f"interval from {s['lower_vs_volatility']:+.4f}")


@pytest.mark.parametrize('ticker', LONG)
def test_point_forecast_is_no_better_than_the_ewma_median(runs, ticker):
    s = ml.point_scores(runs[('return', ticker)], level=0.99)
    assert s['no_better_than_ewma'], f"{ticker}: gain {s['gain_vs_ewma']:+.6f}"


def test_range_is_calibrated_on_the_held_out_tail(runs):
    for t in LONG:
        scores = ml.range_table(runs[('return', t)]).loc['model']
        assert 0.63 < scores['cover_68'] < 0.73 and 0.86 < scores['cover_90'] < 0.94, t


@pytest.mark.xfail(strict=True, reason='at scikit-learn defaults the MLP overfits: its centre jitters, '
                                       'and pinball loss is 7-15% above the constant band on every '
                                       'long ticker (alpha=10 gives +2% on SYN-INDEX)')
def test_range_beats_the_constant_band(runs):
    table, pooled, passes = ml.range_check([runs[('return', t)] for t in LONG], level=0.95)
    assert passes, f'{table}\npooled {pooled.to_dict()}'


@pytest.mark.xfail(strict=True, reason='at scikit-learn defaults the MLP overfits: log loss 0.09 above '
                                       'the volatility baseline even with a planted signal')
def test_planted_signal_is_found(runs):
    assert ml.wdl_scores(runs[('planted signal', None)], level=0.99)['beats_volatility']


@pytest.mark.parametrize('leak', ['leaked r(t+1)', 'centred window'])
def test_leaks_are_caught(runs, leak):
    assert not ml.wdl_scores(runs[(leak, None)], level=0.99)['no_better_than_volatility']
