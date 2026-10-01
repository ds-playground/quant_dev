"""Run parameters and the ticker config that builds them."""

import datetime as dt
from dataclasses import dataclass, field, fields
from pathlib import Path


def _today():
    return dt.date.today().strftime('%Y-%m-%d')

@dataclass
class Params:
    """Every tunable value for a price/return analysis run.

    Defaults reproduce the notebook's out-of-the-box configuration, so
    ``Params()`` is a valid starting point and individual fields can be
    overridden by keyword.

    Units differ between groups, for historical reasons (renaming would break
    configs/tickers.yaml):

      percent  - win_threshold, loss_threshold, cum_thresholds, sim_drift, sim_vol
                 (0.5 means 0.5%, matching the `return_pct` column)
      decimal  - return_thresholds, prob_max, prob_min (0.01 means 1%, matching the
                 `PCT Change` columns and `daily_returns_series`)
      days     - trade_days, windows, roll_windows, chart_windows, timeline_window,
                 streak_days; lookback_years is in years
    """

    # ── Price data ───────────────────────────────────────────────────────
    data_source: str = 'simulated'      # 'simulated', 'synthetic', 'yahoo' or 'local' (data/local)
    ticker: str = 'AAPL'
    start_date: str = '2020-01-01'
    end_date: str = field(default_factory=_today)

    # Simulation-only knobs (ignored unless data_source == 'simulated')
    random_seed: int = 42
    sim_drift: float = 0.04             # mean daily return (%)
    sim_vol: float = 1.2                # daily return std dev (%)
    sim_start_price: float = 150.0

    # ── Rolling / streak analysis ────────────────────────────────────────
    trade_days: int = 250               # trading days per year
    win_threshold: float = 0.5          # min daily return (%) counted as a 'win'
    loss_threshold: float = -0.5        # max daily return (%) counted as a 'loss'
    windows: list = field(default_factory=lambda: [2, 3, 5])
    cum_thresholds: list = field(default_factory=lambda: [0.5, 1.0, 2.0])
    roll_windows: list = None           # defaults to [1, 2, 3, 4, 5, 10, trade_days]

    # ── Charts ───────────────────────────────────────────────────────────
    chart_windows: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 10])
    timeline_window: int = 2

    # ── Historical (rare-event) analysis ─────────────────────────────────
    # size of move tested, decimal (0.01 = a 1% return)
    return_thresholds: list = field(default_factory=lambda: [
        0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02,
        0.015, 0.01, 0.005, 0.002, 0.001, 0.0001])
    lookback_years: list = field(default_factory=lambda: [2, 5])
    streak_days: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 10, 20, 30])
    prob_max: float = 0.10              # "low probability" cutoff
    prob_min: float = 1e-4              # drop events never seen in the lookback window

    def __post_init__(self):
        if self.roll_windows is None:
            self.roll_windows = [1, 2, 3, 4, 5, 10, self.trade_days]

def _params(p):
    """Fall back to stock defaults when a caller passes nothing."""
    return Params() if p is None else p

def _repo_root():
    """Directory holding pyproject.toml, searching upward from this file.

    Located by marker rather than by counting parent directories: the count was
    silently wrong for one commit when this code moved into a package, and the
    only symptom was the ticker config quietly not being found.
    """
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / 'pyproject.toml').is_file():
            return d
    return here.parents[3]


DEFAULT_CONFIG_PATH = _repo_root() / 'configs' / 'tickers.yaml'
# The synthetic tickers (synthetic.py), generated from code: the offline list.
SYNTHETIC_CONFIG_PATH = _repo_root() / 'configs' / 'synthetic_tickers.yaml'

_DEFAULT_TICKER_CONFIG = {
    'defaults': {
        'data_source': 'yahoo',
        'start_date': '2016-01-01',
        'win_threshold': 0.2,
        'loss_threshold': -0.2,
    },
    'tickers': {
        'ES=F': {'label': 'S&P 500 futures'},
        'NQ=F': {'label': 'Nasdaq 100 futures'},
        'YM=F': {'label': 'Dow futures'},
        'RTY=F': {'label': 'Russell 2000 futures'},
    },
}

_META_KEYS = {'label'}

def _build_params(defaults, override, symbol):
    """Merge a defaults block with one ticker's overrides into a Params."""
    merged = {**(defaults or {}), **(override or {})}
    labels = {k: merged.pop(k) for k in list(merged) if k in _META_KEYS}

    valid = {f.name for f in fields(Params)}
    unknown = set(merged) - valid
    if unknown:
        raise ValueError(
            f"{symbol}: unknown config key(s) {sorted(unknown)}. "
            f"Keys must be Params fields or one of {sorted(_META_KEYS)}.")

    merged['ticker'] = symbol
    return Params(**merged), labels.get('label')

def _read_config(path):
    """The raw YAML mapping at `path` and a description of its source (built-in if absent)."""
    if not path.exists():
        return _DEFAULT_TICKER_CONFIG, 'built-in defaults'
    try:
        import yaml
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            f'Reading {path} needs PyYAML - run `pip install pyyaml`.') from exc
    with open(path, encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}, str(path)


def config_params(symbol, path=None):
    """`symbol`'s Params from a ticker config: its own entry if listed, else the `defaults` block.

    For symbols outside the configured set, such as one saved to data/local by hand, so that they
    get the same thresholds a configured ticker would. The label is the configured one, or the
    symbol.
    """
    raw, _ = _read_config(Path(DEFAULT_CONFIG_PATH if path is None else path))
    p, label = _build_params(raw.get('defaults'), (raw.get('tickers') or {}).get(symbol), symbol)
    p.label = label or symbol
    return p


def load_ticker_config(path=None, verbose=True):
    """Return an ordered {ticker: Params} mapping from a YAML config.

    Falls back to a built-in set (ES=F, NQ=F, YM=F, RTY=F) when the file is
    absent, so the notebook runs out of the box. `label` is the only key that
    may appear alongside `Params` fields; anything else raises.
    """
    path = Path(DEFAULT_CONFIG_PATH if path is None else path)
    raw, source = _read_config(path)
    if not path.exists() and verbose:
        print(f'No config at {path}; using built-in defaults: '
              + ', '.join(_DEFAULT_TICKER_CONFIG['tickers']))

    tickers = raw.get('tickers')
    if not tickers:
        raise ValueError(f'No `tickers` block in {source}.')

    config = {}
    for symbol, override in tickers.items():
        p, label = _build_params(raw.get('defaults'), override, symbol)
        p.label = label or symbol       # display name, not a Params field
        config[symbol] = p

    if verbose and path.exists():
        print(f'Loaded {len(config)} ticker(s) from {source}: ' + ', '.join(config))
    return config
