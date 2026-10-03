"""Saved final models (`ml_models.registry`): the id, the round trip through the two folders, the
final fit, and forecasting the next day. Offline, on the synthetic tickers, into temporary folders.
Skipped without LightGBM (the GRU cases also need PyTorch); marked slow, as the other model tests.
"""
import dataclasses
import json
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('lightgbm')
pytestmark = pytest.mark.slow

from src.tools import ml_models as ml                                   # noqa: E402
from src.tools.ml_models.registry import _jsonable                      # noqa: E402
from src.tools.price_return import synthetic_bars                       # noqa: E402

STEP = 500                                      # four folds: enough to save, quick to make
LGB = dict(max_trees=40)
GRU = dict(max_epochs=3, threads=1)


def _models():
    out = [ml.LightGBMWDLModel(**LGB), ml.LightGBMReturnModel(**LGB)]
    try:
        import torch  # noqa: F401
        out += [ml.GRUWDLModel(**GRU), ml.GRUReturnModel(**GRU)]
    except ImportError:
        pass
    return out


MODELS = _models()
IDS = [f'{type(m).__name__}' for m in MODELS]


@pytest.fixture(scope='module')
def bars():
    return synthetic_bars('SYN-INDEX')


@pytest.fixture(scope='module')
def saved():
    """Each model finalized on SYN-INDEX: the LightGBM W/D/L one reusing forecasts from
    `evaluate`, the others making theirs inside `finalize`."""
    fc = ml.evaluate('SYN-INDEX', ml.LightGBMWDLModel(**LGB), step=STEP)
    out = {'LightGBMWDLModel': ml.finalize('SYN-INDEX', ml.LightGBMWDLModel(**LGB), forecasts=fc,
                                           step=STEP)}
    for model in MODELS[1:]:
        out[type(model).__name__] = ml.finalize('SYN-INDEX', model, step=STEP)
    return out


# ── The id ───────────────────────────────────────────────────────────────────

def test_id_reads_at_a_glance(saved):
    names = {'LightGBMWDLModel': 'lightgbm_wdl', 'LightGBMReturnModel': 'lightgbm_return',
             'GRUWDLModel': 'gru-returns-range_wdl', 'GRUReturnModel': 'gru-returns-range_return'}
    for cls, s in saved.items():
        assert re.fullmatch(rf'{names[cls]}_SYN-INDEX_20260930_[0-9a-f]{{8}}', s.model_id), s.model_id
        assert s.parameters['data']['last'] == '2026-09-30'
        assert s.parameters['data']['last_labelled'] == '2026-09-29'    # its label is the last bar


def test_same_methodology_and_bars_give_the_same_id(saved):
    again = ml.finalize('SYN-INDEX', ml.LightGBMWDLModel(**LGB), step=STEP)
    assert again.model_id == saved['LightGBMWDLModel'].model_id
    assert again.spec_id == saved['LightGBMWDLModel'].spec_id


def test_id_changes_with_the_methodology_or_the_bars(saved, bars):
    p = saved['LightGBMWDLModel'].parameters
    spec, data = p['model'], p['data']
    assert ml.model_id(spec, data) == p['model_id']
    revised = bars.copy()
    revised.iloc[-5, revised.columns.get_loc('Close')] *= 1.0001
    variants = {
        'a setting': (ml.model_spec(ml.LightGBMWDLModel(max_trees=41), step=STEP), data),
        'the threshold': (ml.model_spec(ml.LightGBMWDLModel(**LGB), wdl_threshold=0.003, step=STEP), data),
        'the walk-forward': (ml.model_spec(ml.LightGBMWDLModel(**LGB), step=STEP + 1), data),
        'the code': ({**spec, 'code': 'changed'}, data),
        'a revised close': (spec, {**data, **ml.data_fingerprint('SYN-INDEX', 'synthetic', revised)}),
        'another ticker': (spec, {**data, **ml.data_fingerprint(
            'SYN-TECH', 'synthetic', synthetic_bars('SYN-TECH'))}),
    }
    ids = {name: ml.model_id(*v) for name, v in variants.items()}
    assert len(set(ids.values()) | {p['model_id']}) == len(ids) + 1, ids
    # spec_id is the methodology alone: shared across tickers, changed by a setting
    assert ml.registry._hash(ml.model_spec(ml.LightGBMWDLModel(**LGB), step=STEP)) == p['spec_id']
    assert ml.registry._hash(variants['a setting'][0]) != p['spec_id']


def test_code_hash_covers_the_modules_that_make_a_model():
    import inspect
    from pathlib import Path
    made_by = {Path(inspect.getfile(cls)).name for cls in ml.registry.MODEL_CLASSES.values()}
    made_by |= {Path(inspect.getfile(f)).name for f in (ml.features, ml.dataset, ml.walk_forward,
                                                         ml.ticker_bars)}
    assert made_by <= set(ml.registry.CODE_MODULES), made_by - set(ml.registry.CODE_MODULES)
    assert re.fullmatch(r'[0-9a-f]{12}', ml.code_hash())


# ── Saving and loading ───────────────────────────────────────────────────────

@pytest.mark.parametrize('cls', IDS)
def test_save_and_load_round_trip(saved, cls, tmp_path):
    s = saved[cls]
    pkl, js = ml.save_model(s, tmp_path / 'models', tmp_path / 'model_parameters')
    assert pkl == tmp_path / 'models' / f'{s.model_id}.pkl'
    assert js == tmp_path / 'model_parameters' / f'{s.model_id}.json'
    loaded = ml.load_model(s.model_id, tmp_path / 'models')
    assert js.read_text() == json.dumps(loaded.parameters, indent=2, default=_jsonable) + '\n'
    assert type(loaded.model) is type(s.model) and loaded.model.settings() == s.model.settings()
    frame = 'wdl' if s.target == 'wdl' else 'bands'
    pd.testing.assert_frame_equal(getattr(loaded.forecasts, frame), getattr(s.forecasts, frame))
    pd.testing.assert_frame_equal(loaded.forecasts.folds, s.forecasts.folds)
    assert loaded.forecasts.data is None
    # the reloaded trees or weights forecast exactly as the ones trained
    pd.testing.assert_series_equal(loaded.forecast_next(), s.forecast_next(), check_exact=True)
    assert ml.load_model(pkl).model_id == s.model_id                    # a path works too
    ordinary = tmp_path / 'ordinary.txt'
    ordinary.write_text('')
    assert pkl.stat().st_mode == js.stat().st_mode == ordinary.stat().st_mode


def test_parameters_hold_no_daily_values(saved, tmp_path):
    allowed_dates = {'created', 'first', 'last', 'first_labelled', 'last_labelled', 'test_from',
                     'test_to'}

    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from walk(v, k)
        elif isinstance(x, list):
            assert len(x) <= len(ml.FEATURES), f'{key}: a list of {len(x)}'
            for v in x:
                yield from walk(v, key)
        else:
            yield key, x

    for s in saved.values():
        _, js = ml.save_model(s, tmp_path / 'models', tmp_path / 'model_parameters')
        assert js.stat().st_size < 8000
        for key, value in walk(json.loads(js.read_text())):
            if isinstance(value, str) and re.match(r'\d{4}-\d{2}-\d{2}', value):
                assert key in allowed_dates, (key, value)


def test_save_leaves_files_already_there(saved, tmp_path):
    s = saved['LightGBMWDLModel']
    pkl, js = ml.save_model(s, tmp_path / 'models', tmp_path / 'model_parameters')
    js.write_text('kept')
    ml.save_model(s, tmp_path / 'models', tmp_path / 'model_parameters')
    assert js.read_text() == 'kept'                    # the same id is the same model
    ml.save_model(s, tmp_path / 'models', tmp_path / 'model_parameters', overwrite=True)
    assert json.loads(js.read_text())['model_id'] == s.model_id


def test_list_models_reads_the_parameters(saved, tmp_path):
    m, p = tmp_path / 'models', tmp_path / 'model_parameters'
    for s in saved.values():
        ml.save_model(s, m, p)
    gone = saved['LightGBMReturnModel'].model_id
    (m / f'{gone}.pkl').unlink()                      # parameters tracked, model not on this computer
    table = ml.list_models(p, m)
    assert set(table.index) == {s.model_id for s in saved.values()}
    assert not table.loc[gone, 'saved_here'] and table.drop(gone)['saved_here'].all()
    for s in saved.values():
        assert table.loc[s.model_id, 'loss'] == pytest.approx(ml.daily_losses(s.forecasts).mean())
    with pytest.raises(FileNotFoundError, match='git-ignored'):
        ml.load_model(gone, m)


# ── The final fit and the next day ───────────────────────────────────────────

@pytest.mark.parametrize('model', MODELS, ids=IDS)
def test_final_fit_is_the_walk_forward_fit_on_every_row(model, bars):
    """Saved on the bars up to day k, the model's forecast for the day after k must equal a
    walk-forward fold trained on the same labelled rows that tests day k, to the bit: the same
    fit, the same seed, the right row, and nothing from after day k. A threshold other than the
    default, so that one dropped along the way shows."""
    k = 1500
    s = ml.finalize('SYN-INDEX', model, bars=bars.iloc[:k + 1], step=STEP, wdl_threshold=0.003)
    nxt = s.forecast_next(bars.iloc[:k + 1])
    d = ml.dataset(bars.iloc[:k + 2], 0.003)
    n = len(d)
    assert d.index[-1] == bars.index[k] == nxt.name
    fc = ml.walk_forward_forecasts(d, model, folds=[(np.arange(n - 1), np.array([n - 1]))])
    expected = (fc.wdl if model.target == 'wdl' else fc.bands).loc[bars.index[k], 'model']
    np.testing.assert_array_equal(nxt.to_numpy(), expected.to_numpy())
    # Later bars, the same saved model: it forecasts the newer day without retraining.
    assert s.forecast_next(bars).name == bars.index[-1]


class _Capturing(ml.LightGBMWDLModel):
    """Records the frame it is asked to forecast from."""

    def predict(self, state, d, rows):
        self.frame, self.rows = d, rows
        return super().predict(state, d, rows)


def test_forecast_next_feeds_the_saved_threshold_and_the_last_bar(saved, bars):
    # A day whose return lies between 0.2% and 0.3%: a win at the default threshold, a draw at
    # the 0.3% the model is said to be saved with, so the threshold used shows in its features.
    r = ml.bar_returns(bars)
    k = next(i for i in range(1500, len(r)) if 0.002 < r.iloc[i] < 0.003)
    s = saved['LightGBMWDLModel']
    model = _Capturing(**s.model.settings())
    p = {**s.parameters, 'model': {**s.parameters['model'], 'wdl_threshold': 0.003}}
    dataclasses.replace(s, parameters=p, model=model).forecast_next(bars.iloc[:k + 1])
    frame, row = model.frame, model.rows[0]
    assert frame.index[row] == bars.index[k] and row == len(frame) - 1
    expected = ml.features(bars.iloc[:k + 1], 0.003)
    pd.testing.assert_frame_equal(frame, expected.loc[frame.index, ml.FEATURES])
    assert frame['wdl'].iloc[-1] == 0                       # a draw at 0.3%
    d = ml.dataset(bars.iloc[:k + 1], 0.003)
    assert frame.index[:-1].equals(d.index)                 # the rows the model learned from, then day k


def test_forecast_next_is_a_forecast(saved):
    p = saved['LightGBMWDLModel'].forecast_next()
    assert list(p.index) == ['loss', 'draw', 'win'] and p.sum() == pytest.approx(1)
    q = saved['LightGBMReturnModel'].forecast_next()
    assert tuple(q.index) == ml.TAUS and (np.diff(q.to_numpy()) >= 0).all()
    assert p.name == q.name == pd.Timestamp('2026-09-30')


def test_forecast_next_warns_or_refuses_on_changed_code(saved):
    s = saved['LightGBMWDLModel']
    changed = dataclasses.replace(s, parameters={**s.parameters,
                                                 'model': {**s.parameters['model'], 'code': 'old'}})
    with pytest.warns(UserWarning, match='has changed'):
        changed.forecast_next()
    gone = dataclasses.replace(s, model=ml.LightGBMWDLModel(features=['ret_0', 'retired']))
    with pytest.raises(ValueError, match='retired'):
        gone.forecast_next()


def test_finalize_refuses_forecasts_from_something_else(saved):
    fc = saved['LightGBMWDLModel'].forecasts
    cases = [(dict(model=ml.LightGBMWDLModel(max_trees=41)), 'other model settings'),
             (dict(step=STEP + 1), 'walk-forward'),
             (dict(ticker='SYN-TECH'), "not 'SYN-TECH'"),
             (dict(model=ml.LightGBMReturnModel(**LGB)), 'not LightGBM \\(return\\)'),
             (dict(wdl_threshold=0.003), 'threshold'),
             (dict(bars=synthetic_bars('SYN-INDEX').iloc[:-30]), 'other bars')]
    for change, message in cases:
        kwargs = dict(ticker='SYN-INDEX', model=ml.LightGBMWDLModel(**LGB), step=STEP, forecasts=fc)
        kwargs.update(change)
        with pytest.raises(ValueError, match=message):
            ml.finalize(**kwargs)


def test_finalize_many_matches_one_at_a_time(saved):
    jobs = [('SYN-INDEX', ml.LightGBMWDLModel(**LGB)), ('SYN-FX', ml.LightGBMWDLModel(**LGB))]
    together = ml.finalize_many(jobs, n_jobs=2, step=STEP)
    assert together[0].model_id == saved['LightGBMWDLModel'].model_id
    alone = ml.finalize('SYN-FX', ml.LightGBMWDLModel(**LGB), step=STEP)
    assert together[1].model_id == alone.model_id
    pd.testing.assert_series_equal(together[1].forecast_next(), alone.forecast_next(), check_exact=True)
