"""Live data saved as plain CSV: download from Yahoo Finance once, then update on request.

Each symbol is one file in `data/local/` (git-ignored: it is real market data), in the same
layout as `synthetic_bars`: Date, Open, High, Low, Close, Volume, as traded (`auto_adjust=False`).
`manifest.csv` beside them records each file's range and when it was last updated.

An update downloads only from a few days before the last saved bar and merges. The overlap
catches Yahoo's revisions of recent bars: they replace the saved values and are reported, never
applied silently. A file is replaced in one step (a temporary file renamed over it), so a failed
download or an interrupted write never leaves it half-written.
"""
import datetime as dt
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd

from .params import _repo_root

LOCAL_DIR = _repo_root() / 'data' / 'local'
COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
MANIFEST = 'manifest.csv'
# Calendar days re-downloaded before the last saved bar on an update: about a trading week.
OVERLAP_DAYS = 10
# Prices are saved to 6 decimals; revisions are compared at that precision.
DECIMALS = 6
_SAFE = re.compile(r'[A-Za-z0-9=^._-]')


def _directory(directory):
    return Path(LOCAL_DIR if directory is None else directory)


def symbol_to_filename(symbol):
    """`ES=F` -> `ES=F.csv`. Characters outside letters, digits and `=^._-` are %-escaped, so
    no symbol can name a path outside the folder."""
    if not symbol or symbol.strip('.') == '':
        raise ValueError(f'Not a symbol: {symbol!r}')
    return ''.join(c if _SAFE.fullmatch(c) else quote(c, safe='') for c in symbol) + '.csv'


def filename_to_symbol(name):
    return unquote(name[:-len('.csv')])


def local_path(symbol, directory=None):
    return _directory(directory) / symbol_to_filename(symbol)


def read_local(symbol, directory=None):
    """The saved bars for `symbol`: a Date-indexed OHLCV frame."""
    path = local_path(symbol, directory)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=['Date'], index_col='Date', float_precision='round_trip')


def _download(symbol, start, end):
    """Daily bars from Yahoo, as traded, in the saved layout. Empty -> ValueError."""
    import yfinance as yf
    raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f'Yahoo Finance returned no data for {symbol!r} from {start} to {end}; '
                         f'check the symbol.')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = raw.reindex(columns=COLUMNS).dropna(subset=['Open', 'High', 'Low', 'Close'])
    bars['Volume'] = bars['Volume'].fillna(0)
    bars.index = pd.DatetimeIndex(bars.index).tz_localize(None).normalize()
    bars.index.name = 'Date'
    return _rounded(bars[~bars.index.duplicated(keep='last')].sort_index())


def _rounded(bars):
    out = bars[COLUMNS].astype(float).round(DECIMALS)
    out['Volume'] = out['Volume'].round().astype('int64')
    return out


def _write_atomic(frame, path, index=True):
    """Write a CSV next to `path` and rename it into place, so `path` is never partly written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='') as fh:
            frame.to_csv(fh, index=index)
        # mkstemp makes the file owner-only; give it the permissions any new file would get.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _revisions(saved, new):
    """Bars present in both whose values differ: one entry per changed value."""
    common = saved.index.intersection(new.index)
    changes = []
    for date in common:
        for column in COLUMNS:
            old, fresh = saved.at[date, column], new.at[date, column]
            if old != fresh:
                changes.append({'date': date.date().isoformat(), 'column': column,
                                'saved': old.item() if hasattr(old, 'item') else old,
                                'new': fresh.item() if hasattr(fresh, 'item') else fresh})
    return changes


def save_local(symbol, start_date='2016-01-01', directory=None, now=None):
    """Save `symbol`'s daily bars to data/local, or bring the saved file up to date.

    Bars up to yesterday are saved (today's is not final). The first call downloads from
    `start_date`. Later calls download from OVERLAP_DAYS before the last saved bar: new bars are added, and saved bars Yahoo has since revised are replaced and
    listed under `revised`. Bars in the saved file that the new download lacks are kept.
    Returns what changed. Raises (leaving any saved file as it was) if the download fails.
    """
    path = local_path(symbol, directory)
    now = now or dt.datetime.now(dt.timezone.utc)
    # Up to yesterday: today's bar is still changing until the close, and saving it would store
    # a partial day that the next update reports as a revision. yfinance's end is exclusive, and
    # today is the local date, as for load_price_data's default end_date.
    end = now.astimezone().date().isoformat()
    saved = read_local(symbol, directory) if path.is_file() else None

    if saved is None or saved.empty:
        new = _download(symbol, start_date, end)
        merged, added, revised = new, len(new), []
    else:
        since = (saved.index[-1] - pd.Timedelta(days=OVERLAP_DAYS)).date().isoformat()
        new = _download(symbol, since, end)
        saved = _rounded(saved)
        revised = _revisions(saved, new)
        added = int((~new.index.isin(saved.index)).sum())
        merged = pd.concat([saved[~saved.index.isin(new.index)], new]).sort_index()

    _write_atomic(merged, path)
    _record(symbol, path.name, merged, now, directory)
    return {'symbol': symbol, 'file': str(path), 'created': saved is None, 'rows': len(merged),
            'added': added, 'first': merged.index[0].date().isoformat(),
            'last': merged.index[-1].date().isoformat(), 'revised': revised}


def save_all(symbols=None, start_date='2016-01-01', directory=None, now=None, config_path=None,
             on_result=None):
    """`save_local` for each symbol: by default every ticker in the ticker config
    (configs/tickers.yaml), in its order. Duplicates are saved once.

    A symbol that fails (no data, Yahoo unreachable) is listed under `failed` with its error and
    the rest carry on; its saved file, if any, is left as it was. `on_result(entry)` is called as
    each symbol finishes, with its `save_local` result or its failure, for progress reports.
    Returns {'saved': [save_local results], 'failed': [{'symbol', 'error'}]}.
    """
    if symbols is None:
        from .params import load_ticker_config
        symbols = list(load_ticker_config(config_path, verbose=False))
    saved, failed = [], []
    for symbol in dict.fromkeys(symbols):
        try:
            entry = save_local(symbol, start_date=start_date, directory=directory, now=now)
            saved.append(entry)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the others
            entry = {'symbol': symbol, 'error': f'{type(exc).__name__}: {exc}'}
            failed.append(entry)
        if on_result:
            on_result(entry)
    return {'saved': saved, 'failed': failed}


def _record(symbol, filename, bars, now, directory):
    """Update this symbol's manifest row."""
    manifest_path = _directory(directory) / MANIFEST
    rows = pd.read_csv(manifest_path, dtype=str) if manifest_path.is_file() else pd.DataFrame(
        columns=['symbol', 'file', 'rows', 'first', 'last', 'updated'])
    rows = rows[rows['symbol'] != symbol]
    entry = pd.DataFrame([{'symbol': symbol, 'file': filename, 'rows': str(len(bars)),
                           'first': bars.index[0].date().isoformat(),
                           'last': bars.index[-1].date().isoformat(),
                           'updated': now.isoformat(timespec='seconds')}])
    rows = pd.concat([rows, entry], ignore_index=True).sort_values('symbol')
    _write_atomic(rows, manifest_path, index=False)


def local_tickers(directory=None):
    """Every saved symbol, sorted: rows, first and last date, and when it was last updated.

    Read from the files themselves, so a file added or edited by hand is listed correctly; the
    manifest only supplies `updated` (None for a file it does not know).
    """
    folder = _directory(directory)
    manifest_path = folder / MANIFEST
    updated = {}
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path, dtype=str)
        updated = dict(zip(manifest['symbol'], manifest['updated']))
    out = []
    for path in sorted(folder.glob('*.csv')) if folder.is_dir() else []:
        if path.name == MANIFEST:
            continue
        symbol = filename_to_symbol(path.name)
        dates = pd.read_csv(path, usecols=['Date'], parse_dates=['Date'])['Date']
        if dates.empty:
            continue
        out.append({'symbol': symbol, 'rows': len(dates), 'first': dates.iloc[0].date().isoformat(),
                    'last': dates.iloc[-1].date().isoformat(), 'updated': updated.get(symbol)})
    return sorted(out, key=lambda t: t['symbol'])


def local_ticker_config(config_path=None, directory=None):
    """{symbol: Params} for every saved symbol, reading its saved file (`data_source='local'`).

    Thresholds and label come from the ticker config (configs/tickers.yaml): the symbol's own
    entry when it has one, otherwise the config's defaults.
    """
    from dataclasses import replace

    from .params import config_params
    config = {}
    for entry in local_tickers(directory):
        p = config_params(entry['symbol'], config_path)
        label = p.label
        p = replace(p, data_source='local')
        p.label = label
        config[entry['symbol']] = p
    return config
