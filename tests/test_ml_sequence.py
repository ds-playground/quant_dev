"""The GRU sequence model, offline, on the synthetic tickers. Skipped without PyTorch; marked
slow, so `pytest -m "not slow"` leaves it out."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.slow

from src.tools import ml_models as ml                                   # noqa: E402
from src.tools.price_return import synthetic_bars                       # noqa: E402

QUICK = dict(max_epochs=3, threads=1)          # enough to exercise training, not to judge it


@pytest.fixture(scope='module')
def index_data():
    return ml.dataset(synthetic_bars('SYN-INDEX'))


def test_windows_end_at_their_row():
    X = np.arange(40, dtype=float).reshape(20, 2)
    w = ml.windows(X, np.array([4, 19]), 5)
    assert w.shape == (2, 5, 2)
    np.testing.assert_array_equal(w[0], X[0:5])                    # rows 0..4 for row 4
    np.testing.assert_array_equal(w[1], X[15:20])                  # never a row after 19
    with pytest.raises(ValueError, match='fewer than 5'):
        ml.windows(X, np.array([3]), 5)


def test_gru_forecasts_are_probabilities_and_ordered_quantiles(index_data):
    train, test = ml.walk_forward(len(index_data))[0]
    model = ml.GRUModel(**QUICK)
    p, info = model.direction(index_data, train, test)
    assert p.shape == (len(test),) and ((p > 0) & (p < 1)).all() and 1 <= info['epochs'] <= 3
    q, _ = model.quantiles(index_data, train, test, ml.TAUS)
    assert q.shape == (len(test), len(ml.TAUS)) and (np.diff(q, axis=1) >= 0).all()


def test_gru_is_deterministic_and_leaves_torch_as_it_found_it(index_data):
    train, test = ml.walk_forward(len(index_data))[2]
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    a, _ = ml.GRUModel(**QUICK).direction(index_data, train, test)
    b, _ = ml.GRUModel(**QUICK).direction(index_data, train, test)
    np.testing.assert_array_equal(a, b)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic


def test_gru_forecast_is_unchanged_when_later_bars_are_removed():
    """End to end, one fold: keep the bars up to the forecast date and one more (whose close is
    only that date's label); the forecasts must be identical to the bit."""
    bars = synthetic_bars('SYN-INDEX')
    full_data = ml.dataset(bars)
    model = ml.GRUModel(**QUICK)
    for position in (900, 2000):
        date = full_data.index[position]
        cut_data = ml.dataset(bars.iloc[:bars.index.get_loc(date) + 2])
        fold = next(f for f in ml.walk_forward(len(cut_data)) if date in cut_data.index[f[1]])
        full_fold = next(f for f in ml.walk_forward(len(full_data)) if date in full_data.index[f[1]])
        np.testing.assert_array_equal(fold[0], full_fold[0])          # the same training rows
        full = ml.walk_forward_forecasts(full_data, model, folds=[full_fold])
        cut = ml.walk_forward_forecasts(cut_data, model, folds=[fold])
        assert cut.direction.loc[date, 'p_model'] == full.direction.loc[date, 'p_model']
        pd.testing.assert_series_equal(cut.bands.loc[date, 'model'], full.bands.loc[date, 'model'],
                                       check_exact=True)


def test_quantile_shift_makes_each_forecast_cover_its_share():
    rng = np.random.default_rng(4)
    y = rng.standard_t(4, 2000)
    q = np.column_stack([np.full(2000, v) for v in (-0.5, -0.2, 0.0, 0.2, 0.5)])   # far too narrow
    shift = ml.quantile_shift(q, y, ml.TAUS)
    for i, tau in enumerate(ml.TAUS):
        assert np.mean(y <= q[:, i] + shift[i]) == pytest.approx(tau, abs=1 / 2000 + 1e-9)


def test_recalibration_is_reported_and_can_be_turned_off(index_data):
    train, test = ml.walk_forward(len(index_data))[1]
    q, info = ml.GRUModel(**QUICK).quantiles(index_data, train, test, ml.TAUS)
    assert info['shift'].shape == (len(ml.TAUS),) and (np.diff(q, axis=1) >= 0).all()
    raw, info_raw = ml.GRUModel(**QUICK, recalibrate=False).quantiles(index_data, train, test, ml.TAUS)
    assert 'shift' not in info_raw and not np.allclose(q, raw)


# ── The sanity checks, on the five long tickers ──────────────────────────────
# The plan retrains every 63 rows; the tests retrain every 252 over the same test days, about four
# times faster (the owner's choice, docs/ml_plan.md, Phase 4). The GRU has three input sets, kept
# until a model is trained on real data: the default one gets every check; all three get the
# planted signal, which the 16 features cannot find; the other two get the five-ticker checks
# under the opt-in `exhaustive` marker (`pytest -m exhaustive`). Jobs run in spawned processes,
# one thread each.

LONG = ['SYN-INDEX', 'SYN-TECH', 'SYN-GOLD', 'SYN-FX', 'SYN-OIL']
STEP = 252
DEFAULT = 'returns_range'


def _job(job):
    kind, inputs = job[0], job[1]
    if kind == 'ticker':
        return ml.evaluate(job[2], ml.GRUModel(features=inputs, threads=1), step=STEP)
    if kind == 'planted signal':
        d = ml.dataset(ml.planted_signal_bars(repeat=0.6))
        return ml.walk_forward_forecasts(d, ml.GRUModel(features=inputs, threads=1), ranges=False, step=STEP)
    d = ml.dataset(synthetic_bars('SYN-INDEX'))
    if kind == 'leaked r(t+1)':
        d = d.assign(leak=d['y_ret'])
    else:                                   # a centred 5-day average: carries r(t+1) and r(t+2)
        r = ml.bar_returns(synthetic_bars('SYN-INDEX'))
        d = d.assign(leak=r.rolling(5, center=True).mean().reindex(d.index)).dropna()
    model = ml.GRUModel(features=ml.GRU_INPUTS[inputs] + ['leak'], threads=1)
    return ml.walk_forward_forecasts(d, model, ranges=False, step=STEP)


def _run(jobs):
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor
    workers = min(len(jobs), os.cpu_count() or 1)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        return dict(zip(jobs, pool.map(_job, jobs)))


@pytest.fixture(scope='module')
def gru_runs():
    return _run([('ticker', DEFAULT, t) for t in LONG]
                + [('planted signal', inputs) for inputs in ml.GRU_INPUTS]
                + [('leaked r(t+1)', DEFAULT), ('centred window', DEFAULT)])


@pytest.fixture(scope='module')
def other_inputs_runs():
    return _run([('ticker', inputs, t) for inputs in ml.GRU_INPUTS if inputs != DEFAULT for t in LONG])


def _check_direction(forecasts, ticker):
    s = ml.direction_scores(forecasts, level=0.99, margin=0.002)
    assert s['no_better_than_chance'], (f"{ticker}: log-loss gain {s['gain']:+.4f}, 99% interval "
                                        f"{s['lower']:+.4f} to {s['upper']:+.4f}")


def _check_range(forecasts):
    table, pooled, passes = ml.range_check(forecasts, level=0.95)
    assert passes, f'{table}\npooled {pooled.to_dict()}'
    for fc in forecasts:
        # Recalibrated, the intervals are no longer too narrow (uncorrected, SYN-INDEX's 68%
        # interval covered 56% when retrained every 252 rows).
        scores = ml.range_table(fc).loc['model']
        assert 0.63 < scores['cover_68'] < 0.73 and 0.86 < scores['cover_90'] < 0.94, fc.ticker


@pytest.mark.parametrize('ticker', LONG)
def test_gru_direction_is_no_better_than_chance(gru_runs, ticker):
    _check_direction(gru_runs[('ticker', DEFAULT, ticker)], ticker)


def test_gru_range_beats_the_constant_band(gru_runs):
    _check_range([gru_runs[('ticker', DEFAULT, t)] for t in LONG])


@pytest.mark.parametrize('inputs', [
    'returns_range', 'returns',
    pytest.param('features', marks=pytest.mark.xfail(
        strict=True, reason='16 features over 60 days is too many inputs for ~1,600 training rows: '
                            'the GRU finds no planted signal (docs/ml_plan.md, Phase 4 notes)')),
])
def test_gru_finds_the_planted_signal(gru_runs, inputs):
    s = ml.direction_scores(gru_runs[('planted signal', inputs)], level=0.99)
    assert s['beats_base_rate'], f"gain {s['gain']:+.4f}, 99% interval from {s['lower']:+.4f}"


@pytest.mark.parametrize('leak', ['leaked r(t+1)', 'centred window'])
def test_gru_leaks_are_caught(gru_runs, leak):
    assert not ml.direction_scores(gru_runs[(leak, DEFAULT)], level=0.99)['no_better_than_chance']


@pytest.mark.exhaustive
@pytest.mark.parametrize('inputs', [i for i in ml.GRU_INPUTS if i != DEFAULT])
def test_gru_other_inputs_pass_the_checks(other_inputs_runs, inputs):
    for t in LONG:
        _check_direction(other_inputs_runs[('ticker', inputs, t)], t)
    _check_range([other_inputs_runs[('ticker', inputs, t)] for t in LONG])
