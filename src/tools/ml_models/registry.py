"""Final models, saved for later use: each trained once on every labelled row, kept with its
walk-forward forecasts, and found again by a unique id.

Two folders (the owner's layout, 2026-10-03):

- `models/models/<id>.pkl`, git-ignored: one file per model with everything in it: the
  parameters below, the trained state (LightGBM's boosters as LightGBM's own text, the GRU's
  weights and standardization) and the walk-forward out-of-sample forecasts. A new methodology
  is compared with a saved one through those forecasts (`compare_forecasts`), without retraining
  it. A final model, trained on every row, cannot be scored on the days it learned from.
- `models/model_parameters/<id>.json`, tracked: the methodology (the model's class and settings,
  features, threshold, taus, walk-forward), the data it was trained on (ticker, source, range,
  fingerprint), the final fit, summary scores, and the code and library versions. No per-day
  values and no trained weights, so no market data reaches git.

The id is `{model}_{target}_{ticker}_{last bar}_{hash}`, e.g.
`lightgbm_wdl_SYN-INDEX_20260929_3f9a1c2e`. The first four parts read at a glance. The hash, of
the methodology together with the bars' fingerprint, tells apart everything else. The same
methodology on the same bars gives the same id (training is deterministic), so saving again is a
no-op. Any change gives a new id: a setting, the code that builds the model, or a revised bar.
`spec_id`, the hash of the methodology alone, is shared by every ticker trained the same way.

    saved = ml.finalize('SYN-INDEX', ml.LightGBMWDLModel(), forecasts=fc)   # fc: its walk-forward
    ml.save_model(saved)
    saved = ml.load_model(saved.model_id)          # later, without retraining
    saved.forecast_next()                           # tomorrow's loss, draw and win probabilities
    ml.compare_forecasts(saved.forecasts, new_fc)   # a new model against the saved one

The `.pkl` files are pickles: load only ones you saved yourself.
"""
import dataclasses
import datetime as dt
import hashlib
import json
import multiprocessing
import os
import pickle
import platform
import re
import subprocess
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.price_return.params import _repo_root
from src.tools.price_return.store import symbol_to_filename
from src.tools.price_return.synthetic import synthetic_tickers

from .data import PRICE_COLUMNS, ticker_bars
from .evaluate import (Forecasts, daily_losses, point_scores, range_table, walk_forward_forecasts,
                       wdl_scores)
from .features import FEATURES, WDL_THRESHOLD, features as feature_frame
from .gbm import LightGBMReturnModel, LightGBMWDLModel
from .mlp import MLPReturnModel, MLPWDLModel
from .sequence import GRUReturnModel, GRUWDLModel
from .split import walk_forward
from .targets import TAUS, dataset

MODELS_DIR = _repo_root() / 'models' / 'models'
PARAMETERS_DIR = _repo_root() / 'models' / 'model_parameters'
FORMAT = 1
MODEL_CLASSES = {c.__name__: c for c in (LightGBMReturnModel, LightGBMWDLModel, GRUReturnModel,
                                         GRUWDLModel, MLPReturnModel, MLPWDLModel)}
# The modules that decide what a model is fed and how it is trained: a change to any of them is a
# new methodology, and a new id. Each model is hashed with the shared modules and its own, so
# adding or editing one family leaves the others' ids alone.
SHARED_MODULES = ('data.py', 'features.py', 'targets.py', 'split.py')
MODEL_MODULES = {'LightGBMReturnModel': ('gbm.py',), 'LightGBMWDLModel': ('gbm.py',),
                 'GRUReturnModel': ('sequence.py',), 'GRUWDLModel': ('sequence.py',),
                 'MLPReturnModel': ('mlp.py', 'sequence.py'), 'MLPWDLModel': ('mlp.py', 'sequence.py')}


@dataclasses.dataclass
class SavedModel:
    """A final model: `model` (rebuilt from its settings), its trained `state`, its walk-forward
    `forecasts`, and `parameters`, what `models/model_parameters/<model_id>.json` holds."""
    model_id: str
    parameters: dict
    model: object
    state: dict = dataclasses.field(repr=False)
    forecasts: Forecasts = dataclasses.field(default=None, repr=False)

    @property
    def spec_id(self):
        return self.parameters['spec_id']

    @property
    def ticker(self):
        return self.parameters['data']['ticker']

    @property
    def target(self):
        return self.parameters['model']['target']

    def forecast_next(self, bars=None, directory=None):
        """Tomorrow's forecast from this model: see `forecast_next`."""
        return forecast_next(self, bars, directory)


# ── Ids ──────────────────────────────────────────────────────────────────────

def code_modules(model_class):
    """The modules hashed for `model_class` (a class or its name): the shared ones, then its own."""
    name = model_class if isinstance(model_class, str) else model_class.__name__
    return SHARED_MODULES + MODEL_MODULES[name]


def code_hash(model_class):
    """A hash of `code_modules(model_class)`, with line endings normalized so a Windows checkout
    hashes the same."""
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in code_modules(model_class):
        h.update(name.encode())
        h.update((here / name).read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def _jsonable(x):
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (pd.Timestamp, dt.date)):
        return x.strftime('%Y-%m-%d')
    raise TypeError(f'not JSON-serializable: {type(x).__name__}')


def _hash(obj, n=8):
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'), default=_jsonable)
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def model_spec(model, wdl_threshold=WDL_THRESHOLD, taus=TAUS, first_train=756, step=63):
    """The methodology: everything that decides how `model` is built and judged, without the
    data."""
    return {'class': type(model).__name__, 'name': model.name, 'target': model.target,
            'settings': model.settings(), 'features': list(model.features),
            'wdl_threshold': wdl_threshold,
            'taus': list(taus) if model.target == 'return' else None,
            'walk_forward': {'first_train': first_train, 'step': step},
            'code': code_hash(type(model))}


def data_fingerprint(ticker, source, bars):
    """Which bars a model was trained on: their range, and a hash of every date and price."""
    dates = pd.DatetimeIndex(bars.index).asi8
    prices = np.ascontiguousarray(bars[PRICE_COLUMNS].to_numpy(dtype=float))
    sha = hashlib.sha256(dates.tobytes() + prices.tobytes()).hexdigest()[:12]
    return {'ticker': ticker, 'source': source, 'bars': len(bars),
            'first': bars.index[0].strftime('%Y-%m-%d'), 'last': bars.index[-1].strftime('%Y-%m-%d'),
            'fingerprint': sha}


def model_id(spec, data):
    """`{model}_{target}_{ticker}_{last bar}_{hash}`, the hash of the methodology and the data."""
    name = re.sub(r'[^a-z0-9]+', '-', spec['name'].lower()).strip('-')
    ticker = symbol_to_filename(data['ticker'])[:-len('.csv')]
    last = data['last'].replace('-', '')
    return f"{name}_{spec['target']}_{ticker}_{last}_{_hash({'spec': spec, 'data': data})}"


# ── Finalize ─────────────────────────────────────────────────────────────────

def _check_forecasts(fc, d, model, ticker, wdl_threshold, taus, first_train, step):
    """Raise unless `fc` are `model`'s walk-forward forecasts on `d` with these settings."""
    problems = []
    if fc.ticker != ticker:
        problems.append(f'they are for {fc.ticker!r}, not {ticker!r}')
    if (fc.model, fc.target) != (model.name, model.target):
        problems.append(f'they come from {fc.model} ({fc.target}), not {model.name} ({model.target})')
    if fc.settings is None or _hash(fc.settings) != _hash(model.settings()):
        problems.append('they were made with other model settings')
    if fc.walk_forward != {'first_train': first_train, 'step': step}:
        problems.append(f'they come from walk-forward {fc.walk_forward}, not first_train='
                        f'{first_train}, step={step}')
    if fc.target == 'wdl' and fc.wdl_threshold != wdl_threshold:
        problems.append(f'their threshold is {fc.wdl_threshold}, not {wdl_threshold}')
    if fc.target == 'return' and tuple(fc.taus) != tuple(taus):
        problems.append(f'their taus are {fc.taus}, not {tuple(taus)}')
    if not problems:
        test = np.concatenate([t for _, t in walk_forward(len(d), first_train, step)])
        y = fc.y_wdl if fc.target == 'wdl' else fc.y_ret
        label = 'y_wdl' if fc.target == 'wdl' else 'y_ret'
        if not (y.index.equals(d.index[test])
                and np.array_equal(y.to_numpy(), d[label].to_numpy()[test])):
            problems.append('they were made on other bars')
    if problems:
        raise ValueError(f'These forecasts cannot be saved with this model: {"; ".join(problems)}.')


def _versions():
    out = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}
    for name in ('lightgbm', 'torch', 'sklearn'):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            pass
    return out


def _git_commit():
    try:
        r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=_repo_root(),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _scores(fc):
    """Summary scores of the walk-forward forecasts: aggregates only, no per-day values."""
    out = {'loss': float(daily_losses(fc).mean())}
    if fc.target == 'wdl':
        out.update(wdl_scores(fc).to_dict())
    else:
        out.update(point_scores(fc).to_dict())
        out.update({f'range_{k}': v for k, v in range_table(fc).loc['model'].to_dict().items()})
    return {k: _jsonable(v) if not isinstance(v, (str, bool, int, float, type(None))) else v
            for k, v in out.items()}


def finalize(ticker, model, source=None, directory=None, wdl_threshold=WDL_THRESHOLD, taus=TAUS,
             first_train=756, step=63, forecasts=None, bars=None):
    """`model`'s final version for `ticker`, ready for `save_model`: trained on every labelled row
    of the ticker's bars (synthetic, or saved in `data/local`; or `bars` given), with its
    walk-forward `forecasts`. Pass the forecasts already made, e.g. by `evaluate`, and they are
    checked and reused; left out, they are made here.

    The final fit is the walk-forward fit with every row in the training window, seeded the same
    way, so it is the methodology the forecasts scored.
    """
    if bars is None:
        source = source or ('synthetic' if ticker in synthetic_tickers() else 'local')
        bars = ticker_bars(ticker, source, directory)
    else:
        source = source or 'given'
    d = dataset(bars, wdl_threshold)
    if forecasts is None:
        forecasts = walk_forward_forecasts(d, model, taus=taus, ticker=ticker,
                                           first_train=first_train, step=step)
    else:
        _check_forecasts(forecasts, d, model, ticker, wdl_threshold, taus, first_train, step)
    rows = np.arange(len(d))
    state = model.fit(d, rows, taus) if model.target == 'return' else model.fit(d, rows)
    spec = model_spec(model, wdl_threshold, taus, first_train, step)
    data = {**data_fingerprint(ticker, source, bars), 'labelled_rows': len(d),
            'first_labelled': d.index[0].strftime('%Y-%m-%d'),
            'last_labelled': d.index[-1].strftime('%Y-%m-%d')}
    fc = dataclasses.replace(forecasts, data=None)
    test_days = (fc.y_wdl if fc.target == 'wdl' else fc.y_ret).index
    parameters = {
        'format': FORMAT, 'model_id': model_id(spec, data), 'spec_id': _hash(spec),
        'created': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'model': spec, 'data': data,
        'final_fit': {k: _jsonable(np.asarray(v)) if isinstance(v, (list, np.ndarray)) else v
                      for k, v in state.items() if k in ('trees', 'epochs', 'best_epoch', 'shift')},
        'walk_forward': {'folds': len(fc.folds), 'test_days': len(test_days),
                         'test_from': test_days[0].strftime('%Y-%m-%d'),
                         'test_to': test_days[-1].strftime('%Y-%m-%d')},
        'scores': _scores(fc), 'versions': _versions(), 'git_commit': _git_commit()}
    return SavedModel(parameters['model_id'], parameters, model, state, fc)


def _payload(saved):
    fc = saved.forecasts
    return {'format': FORMAT, 'model_id': saved.model_id, 'parameters': saved.parameters,
            'state': type(saved.model).pack(saved.state),
            'forecasts': None if fc is None else {f.name: getattr(fc, f.name)
                                                  for f in dataclasses.fields(fc) if f.name != 'data'}}


def _from_payload(payload):
    if payload.get('format', 0) > FORMAT:
        raise ValueError(f"{payload.get('model_id')} was saved in format {payload['format']}; "
                         f'this code reads up to {FORMAT}')
    spec = payload['parameters']['model']
    cls = MODEL_CLASSES[spec['class']]
    known = {f.name for f in dataclasses.fields(Forecasts)}
    fc = payload['forecasts']
    fc = None if fc is None else Forecasts(**{k: v for k, v in fc.items() if k in known})
    return SavedModel(payload['model_id'], payload['parameters'], cls(**spec['settings']),
                      cls.unpack(payload['state']), fc)


def _finalize_job(job):
    ticker, model, kwargs = job
    return _payload(finalize(ticker, model, **kwargs))


def finalize_many(jobs, n_jobs=None, **kwargs):
    """`finalize` for each `(ticker, model)` or `(ticker, model, forecasts)` in `jobs`, in up to
    `n_jobs` spawned processes at once (default: one per CPU). Same results as one at a time."""
    jobs = [(j[0], j[1], {**kwargs, 'forecasts': j[2] if len(j) > 2 else None}) for j in jobs]
    n_jobs = min(n_jobs or os.cpu_count() or 1, len(jobs))
    if n_jobs <= 1:
        return [finalize(t, m, **kw) for t, m, kw in jobs]
    with ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        return [_from_payload(p) for p in pool.map(_finalize_job, jobs)]


# ── Save, load, list ─────────────────────────────────────────────────────────

def _write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.part')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)          # as an ordinary file, not mkstemp's owner-only
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_model(saved, models_dir=None, parameters_dir=None, overwrite=False):
    """Write `models/models/<id>.pkl` (the model, its state and forecasts) and
    `models/model_parameters/<id>.json` (its parameters). A file already there is left as it is,
    since the same id is the same model, unless `overwrite`. Returns both paths."""
    pkl = Path(models_dir or MODELS_DIR) / f'{saved.model_id}.pkl'
    js = Path(parameters_dir or PARAMETERS_DIR) / f'{saved.model_id}.json'
    if overwrite or not pkl.exists():
        _write_atomic(pkl, pickle.dumps(_payload(saved), protocol=pickle.HIGHEST_PROTOCOL))
    if overwrite or not js.exists():
        text = json.dumps(saved.parameters, indent=2, default=_jsonable) + '\n'
        _write_atomic(js, text.encode())
    return pkl, js


def load_model(model_id, models_dir=None):
    """The `SavedModel` saved as `model_id` (or at the path given): ready to forecast, with its
    walk-forward forecasts. Only load pickles you saved yourself."""
    path = Path(model_id)
    if path.suffix != '.pkl':
        path = Path(models_dir or MODELS_DIR) / f'{model_id}.pkl'
    if not path.is_file():
        raise FileNotFoundError(
            f'{path.stem} is not in {path.parent}. Its parameters may be in model_parameters, but '
            f'the model itself is only on the computer that saved it (models/models is git-ignored).')
    with open(path, 'rb') as f:
        return _from_payload(pickle.load(f))


def list_models(parameters_dir=None, models_dir=None):
    """Every model with parameters in `models/model_parameters`, one row each: what it is, what it
    was trained on, its walk-forward loss, and whether the model itself is here to load."""
    rows = []
    for path in sorted(Path(parameters_dir or PARAMETERS_DIR).glob('*.json')):
        p = json.loads(path.read_text())
        rows.append({'model_id': p['model_id'], 'model': p['model']['name'],
                     'target': p['model']['target'], 'ticker': p['data']['ticker'],
                     'source': p['data']['source'], 'wdl_threshold': p['model']['wdl_threshold'],
                     'last_bar': p['data']['last'], 'spec_id': p['spec_id'],
                     'test_days': p['walk_forward']['test_days'], 'loss': p['scores']['loss'],
                     'created': p['created'],
                     'saved_here': (Path(models_dir or MODELS_DIR) / f"{p['model_id']}.pkl").is_file()})
    columns = ['model', 'target', 'ticker', 'source', 'wdl_threshold', 'last_bar', 'spec_id',
               'test_days', 'loss', 'created', 'saved_here']
    return pd.DataFrame(rows, columns=['model_id', *columns]).set_index('model_id')


def load_latest(model, target, ticker, source=None, wdl_threshold=WDL_THRESHOLD, taus=TAUS,
                parameters_dir=None, models_dir=None):
    """The most recently saved version of `model` (its name: 'LightGBM', 'GRU (returns_range)',
    'MLP', ...) for `ticker`'s `target`, among those on this computer, or None. It must answer
    the same question on the same kind of data: the same `source` (when given), win/draw/loss
    threshold and, for the return, taus. Its settings and code may be older: that is what makes
    it a version."""
    table = list_models(parameters_dir, models_dir)
    match = ((table['model'] == model) & (table['target'] == target) & (table['ticker'] == ticker)
             & (table['wdl_threshold'] == wdl_threshold) & table['saved_here'])
    if source is not None:
        match &= table['source'] == source
    for model_id in table[match].sort_values('created', ascending=False).index:
        saved = load_model(model_id, models_dir)
        if target != 'return' or tuple(saved.forecasts.taus) == tuple(taus):
            return saved
    return None


# ── Forecast ─────────────────────────────────────────────────────────────────

def forecast_next(saved, bars=None, directory=None):
    """The saved model's forecast for the day after the last bar: the loss, draw and win
    probabilities, or the return quantiles (index: tau; the 0.5 one is the point forecast).
    Named by the last bar's date.

    `bars` default to the ticker's bars where the model was trained from (synthetic, or
    `data/local`), so after saving newer bars the same model forecasts the newer day without
    retraining. Warns if the model-building code has changed since the model was saved.
    """
    p = saved.parameters
    if p['model']['code'] != code_hash(p['model']['class']):
        warnings.warn(f'{saved.model_id}: the code that builds features and models has changed '
                      'since it was saved, so its inputs may differ from what it learned on',
                      stacklevel=2)
    missing = [f for f in saved.model.features if f not in FEATURES]
    if missing:
        raise ValueError(f'{saved.model_id} reads features this code no longer makes: {missing}')
    if bars is None:
        if p['data']['source'] not in ('synthetic', 'local'):
            raise ValueError(f'{saved.model_id} was trained on bars given by hand: pass the bars')
        bars = ticker_bars(saved.ticker, p['data']['source'], directory)
    threshold = p['model']['wdl_threshold']
    d = dataset(bars, threshold)
    last = feature_frame(bars, threshold).loc[[bars.index[-1]], FEATURES]
    if last.isna().any(axis=None):
        raise ValueError(f'the last bar, {bars.index[-1]:%Y-%m-%d}, has incomplete features')
    frame = pd.concat([d[FEATURES], last])
    out = saved.model.predict(saved.state, frame, np.array([len(frame) - 1]))[0]
    if saved.target == 'wdl':
        index = pd.Index(['loss', 'draw', 'win'], name='outcome')
    else:
        index = pd.Index(saved.state['taus'], name='tau')
    return pd.Series(out, index=index, name=bars.index[-1])
