"""Offline tests for src.tools.ml_models (the part without machine-learning libraries).

References are independent of the code under test: hand-built series and hand-worked numbers,
formulas written out here from their papers, scipy's own tests (the G-test is the same likelihood
ratio as Kupiec's and Christoffersen's), `price_return`'s loader and `options.price_range`, and
`ta_tools.rsi` (TA-Lib). The no-look-ahead test is the one `ta_tools` uses: cut the bars after t,
recompute, and nothing at or before t may change.
"""
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools import ml_models as ml
from src.tools.price_return import Params, load_price_data, synthetic_bars, synthetic_tickers
from src.tools.price_return.options import price_range
from src.tools.price_return.store import local_path

REPO = Path(__file__).resolve().parent.parent
LONG = [t for t in synthetic_tickers() if t != 'SYN-LEV']   # SYN-LEV is too short to judge


@pytest.fixture(scope='module')
def index_bars():
    return synthetic_bars('SYN-INDEX')


@pytest.fixture(scope='module')
def oil_bars():
    return synthetic_bars('SYN-OIL')        # one negative settle, on 2020-04-20


@pytest.fixture(scope='module')
def index_data(index_bars):
    return ml.dataset(index_bars)


def hand_bars(closes, start='2024-01-01'):
    closes = np.asarray(closes, dtype=float)
    index = pd.bdate_range(start, periods=len(closes), name='Date')
    return pd.DataFrame({'Open': closes, 'High': closes * 1.01, 'Low': closes * 0.99,
                         'Close': closes}, index=index)


# ── The package ──────────────────────────────────────────────────────────────

def test_public_api_resolves():
    missing = [name for name in ml.__all__ if not hasattr(ml, name)]
    assert not missing, f'names in __all__ with no attribute: {missing}'


def test_importing_ml_models_loads_no_ml_library():
    # A fresh interpreter, since this test session may already have imported them.
    loaded = subprocess.run(
        [sys.executable, '-c',
         'import sys, src.tools.ml_models; print([m for m in ("lightgbm", "torch") if m in sys.modules])'],
        capture_output=True, text=True, check=True, cwd=REPO,
    ).stdout.strip()
    assert loaded == '[]'


def test_ml_notebooks_are_committed_without_outputs():
    # Run on saved data, an output would be market data, which never goes into the repo.
    notebooks = sorted((REPO / 'notebooks').glob('ml_*.ipynb'))
    assert notebooks, 'no ML notebook found'
    for path in notebooks:
        cells = [c for c in json.loads(path.read_text())['cells'] if c['cell_type'] == 'code']
        with_output = [i for i, c in enumerate(cells) if c.get('outputs') or c.get('execution_count')]
        assert not with_output, f'{path.name}: code cells {with_output} have outputs; clear them'


# ── Bars and returns ─────────────────────────────────────────────────────────

def test_ticker_bars_synthetic(index_bars):
    pd.testing.assert_frame_equal(ml.ticker_bars('SYN-INDEX'), index_bars)
    pd.testing.assert_frame_equal(ml.ticker_bars('SYN-INDEX', source='synthetic'), index_bars)


def test_ticker_bars_reads_saved_data(tmp_path, index_bars):
    saved = index_bars.iloc[:300]
    saved.to_csv(local_path('ES=F', tmp_path))
    bars = ml.ticker_bars('ES=F', directory=tmp_path)      # not synthetic, so 'local'
    np.testing.assert_allclose(bars[['Open', 'High', 'Low', 'Close']], saved[['Open', 'High', 'Low', 'Close']])
    assert list(bars.index) == list(saved.index)


def test_ticker_bars_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match='save_local'):
        ml.ticker_bars('ES=F', directory=tmp_path)
    with pytest.raises(ValueError, match='source'):
        ml.ticker_bars('SYN-INDEX', source='yahoo')


def test_bar_returns_match_the_price_return_loader(oil_bars):
    loaded = load_price_data(Params(data_source='synthetic', ticker='SYN-OIL', start_date='2016-01-01'),
                             verbose=False)
    expected = loaded.set_index('date')['return_pct'] / 100
    got = ml.bar_returns(oil_bars).reindex(expected.index)
    np.testing.assert_allclose(got, expected, rtol=1e-12, equal_nan=True)
    # The two returns spanning the negative settle are dropped, and nothing else.
    assert ml.bar_returns(oil_bars).iloc[1:].isna().sum() == 2


# ── Targets and the modelling frame ──────────────────────────────────────────

def test_next_return_and_direction_by_hand():
    bars = hand_bars([100, 101, 101, 99, 100])
    np.testing.assert_allclose(ml.next_return(bars), [0.01, 0.0, 99 / 101 - 1, 100 / 99 - 1, np.nan])
    np.testing.assert_array_equal(ml.next_direction(bars), [1.0, 0.0, 0.0, 1.0, np.nan])


def test_flat_band_scales_with_volatility():
    bars = hand_bars([100, 101, 101.2, 99, 100])
    sigma = pd.Series([0.01, 0.01, 0.005, 0.03, 0.01], index=bars.index)
    # next returns: +1%, +0.198%, -2.17%, +1.01%; bands at 0.5 sigma: 0.5%, 0.5%, 0.25%, 1.5%
    np.testing.assert_array_equal(ml.next_direction(bars, flat=0.5, sigma=sigma), [1.0, 0.0, -1.0, 0.0, np.nan])
    with pytest.raises(ValueError, match='sigma'):
        ml.next_direction(bars, flat=0.5)


def test_dataset(index_bars, index_data):
    d = index_data
    assert list(d.columns) == ml.FEATURES + ['y_ret', 'y_up', 'vol_250']
    assert not d[ml.FEATURES + ['y_ret', 'y_up']].isna().any().any()
    assert ((d['y_ret'] > 0) == (d['y_up'] == 1)).all()
    assert d.index[-1] == index_bars.index[-2]              # the last bar has no tomorrow
    # vol_250 is NaN only in its own warm-up (240 returns, from bar 240) and never removes a row:
    # the frame starts at bar 60, when vol_60 has its 60 returns.
    assert d.index[0] == index_bars.index[60]
    assert d['vol_250'].isna().sum() == 240 - 60
    np.testing.assert_allclose(d['y_ret'], ml.bar_returns(index_bars).shift(-1).loc[d.index])


def test_dataset_keeps_vol_250_through_the_negative_settle(oil_bars):
    d = ml.dataset(oil_bars)
    assert d.loc['2020-04-01':, 'vol_250'].notna().all()


# ── Features ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('column', ml.FEATURES)
def test_no_lookahead(oil_bars, column):
    """Values up to bar t must not change when the bars after t are removed."""
    full = ml.features(oil_bars)[column]
    for cut in (300, 1100, 1180, 2000):          # 1180 sits just after the negative settle
        truncated = ml.features(oil_bars.iloc[:cut + 1])[column]
        np.testing.assert_allclose(truncated, full.iloc[:cut + 1], rtol=1e-12, equal_nan=True,
                                   err_msg=f'{column} changed at or before bar {cut}')


def test_features_are_free_of_the_price_scale(index_bars):
    scaled = index_bars.copy()
    scaled[['Open', 'High', 'Low', 'Close']] *= 3.7
    pd.testing.assert_frame_equal(ml.features(scaled), ml.features(index_bars), rtol=1e-9)


def test_features_at_one_bar_by_hand(index_bars):
    t = 1500
    f = ml.features(index_bars).iloc[t]
    o, h, l, c = (index_bars[k].to_numpy() for k in ('Open', 'High', 'Low', 'Close'))
    r = [c[i] / c[i - 1] - 1 for i in range(len(c))]
    assert f['ret_0'] == pytest.approx(r[t]) and f['ret_4'] == pytest.approx(r[t - 4])
    assert f['abs_ret'] == pytest.approx(abs(r[t]))
    assert f['vol_20'] == pytest.approx(statistics.stdev(r[t - 19:t + 1]))
    assert f['vol_60'] == pytest.approx(statistics.stdev(r[t - 59:t + 1]))
    # Parkinson (1980): sigma^2 = mean(ln(H/L)^2) / (4 ln 2)
    hl = [math.log(h[i] / l[i]) ** 2 for i in range(t - 19, t + 1)]
    assert f['vol_parkinson'] == pytest.approx(math.sqrt(sum(hl) / 20 / (4 * math.log(2))))
    # Garman-Klass (1980): sigma^2 = mean(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2)
    gk = [0.5 * math.log(h[i] / l[i]) ** 2 - (2 * math.log(2) - 1) * math.log(c[i] / o[i]) ** 2
          for i in range(t - 19, t + 1)]
    assert f['vol_gk'] == pytest.approx(math.sqrt(sum(gk) / 20))
    true_range = max(h[t] - l[t], abs(h[t] - c[t - 1]), abs(l[t] - c[t - 1]))
    assert f['tr_close'] == pytest.approx(true_range / c[t])
    assert f['dist_sma20'] == pytest.approx(c[t] / (sum(c[t - 19:t + 1]) / 20) - 1)
    assert f['dist_sma50'] == pytest.approx(c[t] / (sum(c[t - 49:t + 1]) / 50) - 1)
    assert f['dow'] == index_bars.index[t].dayofweek


def test_ewma_vol_follows_its_recursion():
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 400))
    var, expected = r[0] ** 2, [abs(r[0])]
    for x in r[1:]:
        var = 0.94 * var + 0.06 * x ** 2
        expected.append(math.sqrt(var))
    got = ml.ewma_vol(r)
    assert got.iloc[:19].isna().all()
    np.testing.assert_allclose(got.iloc[19:], expected[19:], rtol=1e-12)


def test_wilder_rsi_matches_ta_tools(index_bars, oil_bars):
    pytest.importorskip('talib')
    from src.tools import ta_tools
    for bars in (index_bars, oil_bars):
        np.testing.assert_allclose(ml.wilder_rsi(bars['Close'], 14), ta_tools.rsi(bars['Close'], 14),
                                   rtol=1e-10, atol=1e-10, equal_nan=True)


def test_wilder_rsi_by_hand():
    # Changes +1, +1, -1: average gain 2/3, average loss 1/3 over 3, so RSI = 100 - 100 / 3.
    rsi = ml.wilder_rsi(pd.Series([10.0, 11, 12, 11]), 3)
    assert rsi.iloc[:3].isna().all() and rsi.iloc[3] == pytest.approx(100 - 100 / 3)
    assert ml.wilder_rsi(pd.Series([1.0, 2, 3, 4, 5]), 3).iloc[-1] == 100      # no losses


# ── Walk-forward ─────────────────────────────────────────────────────────────

def test_walk_forward_expanding():
    folds = ml.walk_forward(1000, first_train=756, step=63)
    assert len(folds) == 4
    tests = np.concatenate([test for _, test in folds])
    np.testing.assert_array_equal(tests, np.arange(756, 1000))           # every row once, in order
    assert [len(test) for _, test in folds] == [63, 63, 63, 55]
    for train, test in folds:
        np.testing.assert_array_equal(train, np.arange(0, test[0]))


@pytest.mark.parametrize('horizon', [1, 2, 5, 10])
def test_purge_leaves_no_label_overlap(horizon):
    for train, test in ml.walk_forward(2000, horizon=horizon):
        # a training label covers (t, t + h]; it must end by the first test row, and only just
        assert train[-1] + horizon == test[0]
        assert test[0] - train[-1] - 1 == horizon - 1                    # rows purged


def test_rolling_window():
    for train, test in ml.walk_forward(2000, first_train=756, step=63, horizon=5, window=500):
        assert len(train) == 500 and train[-1] + 5 == test[0]


def test_walk_forward_rejects_bad_arguments():
    for kwargs in ({'first_train': 0}, {'step': 0}, {'horizon': 0}, {'window': 0},
                   {'first_train': 3, 'horizon': 5}):
        with pytest.raises(ValueError):
            ml.walk_forward(100, **kwargs)


def test_purged_tail():
    train = np.arange(1000)
    fit, val = ml.purged_tail(train, horizon=5, tail=0.2)
    np.testing.assert_array_equal(val, np.arange(800, 1000))
    assert fit[0] == 0 and fit[-1] + 5 == val[0]
    fit1, _ = ml.purged_tail(train)
    assert fit1[-1] + 1 == 800
    with pytest.raises(ValueError):
        ml.purged_tail(train, tail=1.5)


# ── Baselines ────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def index_folds(index_data):
    return ml.walk_forward(len(index_data))


@pytest.fixture(scope='module')
def index_bands(index_data, index_folds):
    return ml.range_baselines(index_data, index_folds)


def test_direction_baseline_is_the_training_up_share(index_data, index_folds):
    p = ml.direction_baseline(index_data, index_folds)
    for train, test in index_folds[::7]:
        expected = index_data['y_up'].to_numpy()[train].mean()
        np.testing.assert_allclose(p.loc[index_data.index[test]], expected)


def test_constant_band_is_the_training_quantiles(index_data, index_folds, index_bands):
    for train, test in index_folds[::7]:
        for tau in ml.TAUS:
            expected = np.quantile(index_data['y_ret'].to_numpy()[train], tau)
            np.testing.assert_allclose(index_bands.loc[index_data.index[test], ('constant', tau)], expected)


def test_price_range_band_matches_options_price_range(index_data, index_bands):
    for date in index_bands.index[::400]:
        sigma = index_data.loc[date, 'vol_250']
        for lo, hi in ((0.1587, 0.8413), (0.05, 0.95)):
            z = statistics.NormalDist().inv_cdf(hi)
            band = price_range(1.0, sigma * z, days=1)
            assert index_bands.loc[date, ('price_range', lo)] == pytest.approx(band['low'] - 1)
            assert index_bands.loc[date, ('price_range', hi)] == pytest.approx(band['high'] - 1)


def test_baselines_use_only_training_rows(index_data, index_folds, index_bands):
    k = 10
    train, test = index_folds[k]
    changed = index_data.copy()
    changed.iloc[test[0]:, changed.columns.get_loc('y_ret')] *= 5     # every outcome from fold k on
    changed.iloc[test[0]:, changed.columns.get_loc('y_up')] = 1.0
    bands = ml.range_baselines(changed, index_folds[:k + 1])
    pd.testing.assert_frame_equal(bands, index_bands.loc[bands.index])
    p = ml.direction_baseline(changed, index_folds[:k + 1])
    pd.testing.assert_series_equal(p, ml.direction_baseline(index_data, index_folds[:k + 1]))


def _quantiles(bands, band):
    return {tau: bands[(band, tau)] for tau in ml.TAUS}


def test_ewma_band_beats_the_constant_band():
    """The baselines' sanity check: on GARCH data a band that tracks volatility must beat a
    constant one, with misses that cluster less, on every long ticker and significantly pooled."""
    pooled = []
    for ticker in LONG:
        d = ml.dataset(synthetic_bars(ticker))
        bands = ml.range_baselines(d, ml.walk_forward(len(d)))
        y = d.loc[bands.index, 'y_ret']
        const, const_daily = ml.range_scores(_quantiles(bands, 'constant'), y)
        ewma, ewma_daily = ml.range_scores(_quantiles(bands, 'ewma_std_q'), y)
        assert ewma['pinball'] < const['pinball'], ticker
        assert ewma['cluster_ratio_68'] < const['cluster_ratio_68'], ticker
        assert abs(ewma['cover_68'] - 0.6826) < 0.02, ticker            # and it covers as it claims
        pooled.append(ml.bootstrap_means((const_daily - ewma_daily) / const_daily.mean()))
    draws = np.mean(pooled, axis=0)
    assert np.quantile(draws, 0.025) > 0


# ── Metrics ──────────────────────────────────────────────────────────────────

def test_log_loss_and_brier_by_hand():
    np.testing.assert_allclose(ml.log_loss([0.8, 0.8, 0.5], [1, 0, 1]),
                               [-math.log(0.8), -math.log(0.2), math.log(2)])
    np.testing.assert_allclose(ml.brier([0.8, 0.8, 0.5], [1, 0, 1]), [0.04, 0.64, 0.25])
    assert np.isfinite(ml.log_loss([0.0, 1.0], [1, 0])).all()            # clipped, not infinite
    with pytest.raises(ValueError):
        ml.log_loss([0.5, 0.5], [1])


def test_brier_decomposition_by_hand():
    p = np.array([0.3] * 4 + [0.6] * 6)
    y = np.array([1, 0, 0, 0] + [1, 1, 0, 1, 0, 1])
    parts = ml.brier_decomposition(p, y)
    base = 0.5
    assert parts['reliability'] == pytest.approx((4 * (0.3 - 0.25) ** 2 + 6 * (0.6 - 4 / 6) ** 2) / 10)
    assert parts['resolution'] == pytest.approx((4 * (0.25 - base) ** 2 + 6 * (4 / 6 - base) ** 2) / 10)
    assert parts['uncertainty'] == pytest.approx(0.25)
    assert parts['brier'] == pytest.approx(parts['reliability'] - parts['resolution'] + parts['uncertainty'])
    table = ml.reliability_table(p, y)
    assert list(table['days']) == [4, 6]
    np.testing.assert_allclose(table['observed'], [0.25, 4 / 6])


def test_pinball_by_hand_and_minimized_by_the_quantile():
    np.testing.assert_allclose(ml.pinball([0, 0], [1, -1], 0.9), [0.9, 0.1])
    x = np.random.default_rng(5).standard_t(4, 5001)
    for tau in ml.TAUS:
        q = np.quantile(x, tau)
        best = ml.pinball(np.full_like(x, q), x, tau).mean()
        for shift in (-0.05, 0.05):
            assert best <= ml.pinball(np.full_like(x, q + shift), x, tau).mean()


def test_interval_misses_include_the_bounds():
    np.testing.assert_array_equal(ml.interval_misses([-1, -1, -1], [1, 1, 1], [-1, 1, 1.5]),
                                  [False, False, True])


def test_kupiec_is_scipys_g_test():
    stats = pytest.importorskip('scipy.stats')
    misses = np.zeros(500, dtype=bool)
    misses[::4] = True                                  # 125 misses where 0.3174 x 500 = 158.7 expected
    lr, p = ml.kupiec(misses, 0.3174)
    g, p_ref = stats.power_divergence([125, 375], [0.3174 * 500, 0.6826 * 500], lambda_='log-likelihood')
    assert lr == pytest.approx(g) and p == pytest.approx(p_ref)
    lr0, p0 = ml.kupiec(np.zeros(100, dtype=bool), 0.05)  # no misses at all
    assert lr0 == pytest.approx(-2 * 100 * math.log(0.95)) and 0 < p0 < 1


def test_christoffersen_is_scipys_g_test_of_independence():
    stats = pytest.importorskip('scipy.stats')
    rng = np.random.default_rng(8)
    misses = np.zeros(600, dtype=int)
    for t in range(1, 600):                             # a miss makes the next one likelier
        misses[t] = rng.random() < (0.6 if misses[t - 1] else 0.2)
    lr, p, ratio = ml.christoffersen(misses)
    a, b = misses[:-1], misses[1:]
    table = np.array([[np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))],
                      [np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))]])
    g, p_ref, _, _ = stats.chi2_contingency(table, correction=False, lambda_='log-likelihood')
    assert lr == pytest.approx(g) and p == pytest.approx(p_ref)
    assert ratio == pytest.approx((table[1, 1] / table[1].sum()) / (table[0, 1] / table[0].sum()))
    assert ratio > 2 and p < 1e-6


def test_christoffersen_by_hand():
    # transitions of 0,0,1,1,0,1,0,0: 00 x2, 01 x2, 10 x2, 11 x1
    _, _, ratio = ml.christoffersen([0, 0, 1, 1, 0, 1, 0, 0])
    assert ratio == pytest.approx((1 / 3) / (2 / 4))


def test_range_scores():
    y = np.array([-0.02, -0.005, 0.0, 0.004, 0.03, 0.001])
    q = {0.05: np.full(6, -0.015), 0.1587: np.full(6, -0.006), 0.5: np.zeros(6),
         0.8413: np.full(6, 0.006), 0.95: np.full(6, 0.015)}
    scores, daily = ml.range_scores(q, y)
    assert scores['cover_68'] == pytest.approx(4 / 6) and scores['cover_90'] == pytest.approx(4 / 6)
    assert scores['width_68'] == pytest.approx(0.012)
    expected = np.mean([ml.pinball(q[tau], y, tau) for tau in ml.TAUS], axis=0)
    np.testing.assert_allclose(daily, expected)
    assert scores['pinball'] == pytest.approx(expected.mean())


def test_mean_interval():
    x = np.random.default_rng(2).normal(0.3, 1, 3000)
    out = ml.mean_interval(x, level=0.95)
    assert out['estimate'] == pytest.approx(x.mean())
    assert out['lower'] < x.mean() < out['upper']
    assert out['upper'] - out['lower'] == pytest.approx(2 * 1.96 / math.sqrt(3000), rel=0.25)
    pd.testing.assert_series_equal(out, ml.mean_interval(x, level=0.95))       # seeded
