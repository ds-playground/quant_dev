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
