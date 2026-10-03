"""Offline tests for src.tools.ml_models (the part without machine-learning libraries).

References are independent of the code under test: hand-built series and hand-worked numbers,
formulas written out here from their papers, scipy's own tests (the G-test is the same likelihood
ratio as Kupiec's and Christoffersen's), `price_return`'s loader and `options.price_range`, and
`ta_tools.rsi` (TA-Lib). The no-look-ahead test is the one `ta_tools` uses: cut the bars after t,
recompute, and nothing at or before t may change.
"""
import dataclasses
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

def test_next_return_and_win_draw_loss_by_hand():
    bars = hand_bars([100, 101, 101, 99, 100, 100.1])
    # returns: +1%, 0%, -1.98%, +1.01%, +0.1%
    np.testing.assert_allclose(ml.next_return(bars), [0.01, 0.0, 99 / 101 - 1, 100 / 99 - 1, 0.001, np.nan])
    np.testing.assert_array_equal(ml.next_wdl(bars), [1.0, 0.0, -1.0, 1.0, 0.0, np.nan])     # at 0.2%
    np.testing.assert_array_equal(ml.next_wdl(bars, threshold=0.015), [0.0, 0.0, -1.0, 0.0, 0.0, np.nan])
    assert ml.WDL_THRESHOLD == 0.002                                    # the owner's default
    # exactly at the threshold is a draw, either way
    r = pd.Series([0.002, -0.002, 0.0021, -0.0021, np.nan])
    np.testing.assert_array_equal(ml.win_draw_loss(r), [0.0, 0.0, 1.0, -1.0, np.nan])


def test_streak_by_hand():
    outcomes = pd.Series([1, 1, 1, 0, -1, -1, 1, np.nan, -1, 0, 0, 1])
    np.testing.assert_array_equal(ml.streak(outcomes), [1, 2, 3, 0, -1, -2, 1, np.nan, -1, 0, 0, 1])
    long_run = pd.Series([1.0] * 14 + [-1.0] * 12)
    got = ml.streak(long_run)
    assert got.iloc[13] == 10 and got.iloc[9] == 10 and got.iloc[8] == 9    # capped at 10
    assert got.iloc[14] == -1 and got.iloc[-1] == -10
    assert ml.STREAK_CAP == 10


def test_dataset(index_bars, index_data):
    d = index_data
    assert list(d.columns) == ml.FEATURES + ['y_ret', 'y_wdl', 'vol_250']
    assert not d[ml.FEATURES + ['y_ret', 'y_wdl']].isna().any().any()
    np.testing.assert_array_equal(d['y_wdl'], np.select([d['y_ret'] > 0.002, d['y_ret'] < -0.002], [1, -1], 0))
    assert d.attrs['wdl_threshold'] == 0.002
    # today's outcome and streak are tomorrow's label of the day before
    np.testing.assert_array_equal(d['wdl'].iloc[1:], d['y_wdl'].iloc[:-1])
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
    assert f['wdl'] == (1 if r[t] > 0.002 else -1 if r[t] < -0.002 else 0)
    run = 0                                       # the streak, counted back from t
    while run < 10 and np.sign(r[t - run]) == f['wdl'] != 0 and abs(r[t - run]) > 0.002:
        run += 1
    assert f['streak'] == f['wdl'] * run


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


@pytest.fixture(scope='module')
def index_wdl(index_data, index_folds):
    return ml.wdl_baselines(index_data, index_folds)


def test_wdl_frequencies_are_the_training_shares(index_data, index_folds, index_wdl):
    assert list(index_wdl.columns) == [(b, c) for b in ml.WDL_BASELINES for c in ml.WDL_CLASSES]
    np.testing.assert_allclose(index_wdl.T.groupby(level='baseline').sum().T, 1.0)
    for train, test in index_folds[::7]:
        y = index_data['y_wdl'].to_numpy()[train]
        for c in ml.WDL_CLASSES:
            expected = (np.sum(y == c) + 0.5) / (len(y) + 1.5)        # half a count each: never 0
            np.testing.assert_allclose(index_wdl.loc[index_data.index[test], ('frequencies', c)], expected)


def test_wdl_volatility_baseline_by_hand(index_data, index_folds, index_wdl):
    # The share of the training window's standardized returns that, scaled by today's volatility,
    # would clear the threshold either way.
    train, test = index_folds[5]
    z = (index_data['y_ret'] / index_data['vol_ewma']).to_numpy()[train]
    for row in test[::20]:
        cut = 0.002 / index_data['vol_ewma'].iloc[row]
        counts = {1: np.sum(z > cut), -1: np.sum(z < -cut)}
        counts[0] = len(z) - counts[1] - counts[-1]
        for c in ml.WDL_CLASSES:
            assert index_wdl.loc[index_data.index[row], ('volatility', c)] == pytest.approx(
                (counts[c] + 0.5) / (len(z) + 1.5))
    # Calm days make draws likelier than volatile ones do.
    calm = index_data.loc[index_wdl.index, 'vol_ewma'] < index_data['vol_ewma'].quantile(0.2)
    wild = index_data.loc[index_wdl.index, 'vol_ewma'] > index_data['vol_ewma'].quantile(0.8)
    assert index_wdl.loc[calm, ('volatility', 0)].mean() > 1.5 * index_wdl.loc[wild, ('volatility', 0)].mean()


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


def test_baselines_use_only_training_rows(index_data, index_folds, index_bands, index_wdl):
    k = 10
    train, test = index_folds[k]
    changed = index_data.copy()
    changed.iloc[test[0]:, changed.columns.get_loc('y_ret')] *= 5     # every outcome from fold k on
    changed.iloc[test[0]:, changed.columns.get_loc('y_wdl')] = 1.0
    bands = ml.range_baselines(changed, index_folds[:k + 1])
    pd.testing.assert_frame_equal(bands, index_bands.loc[bands.index])
    wdl = ml.wdl_baselines(changed, index_folds[:k + 1])
    pd.testing.assert_frame_equal(wdl, index_wdl.loc[wdl.index])


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

def test_multiclass_log_loss_and_brier_by_hand():
    p = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.0, 1.0, 0.0]])
    y = np.array([1, -1, 0])
    np.testing.assert_allclose(ml.multiclass_log_loss(p, y, ml.WDL_CLASSES),
                               [-math.log(0.5), -math.log(0.6), 0.0], atol=1e-12)
    np.testing.assert_allclose(ml.multiclass_brier(p, y, ml.WDL_CLASSES),
                               [0.04 + 0.09 + 0.25, 0.16 + 0.09 + 0.01, 0.0])
    with pytest.raises(ValueError, match='outside'):
        ml.multiclass_log_loss(p, np.array([1, 2, 0]), ml.WDL_CLASSES)
    # uniform probabilities cost log(3), whatever happens
    np.testing.assert_allclose(ml.multiclass_log_loss(np.full((3, 3), 1 / 3), y, ml.WDL_CLASSES), math.log(3))


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


# ── The walk-forward runner, with stub models (no LightGBM needed) ──────────

class StubReturnModel:
    """Quantiles that encode their own row, so the runner's alignment can be checked."""
    name = 'stub'
    target = 'return'

    def quantiles(self, d, train, test, taus):
        sigma = d['vol_ewma'].to_numpy()[test][:, None]
        z = np.array([statistics.NormalDist().inv_cdf(t) for t in taus])[None, :]
        return sigma * z, {'trees': [1] * len(taus), 'importance': {'ret_0': 3.0, 'vol_20': 1.0}}


class StubWDLModel:
    """Probabilities that encode their own row: odd rows lean to a win, even rows to a loss."""
    name = 'stub'
    target = 'wdl'

    def wdl(self, d, train, test):
        odd = (np.asarray(test) % 2)[:, None]
        p = np.where(odd, [[0.2, 0.3, 0.5]], [[0.5, 0.3, 0.2]])
        return p, {'trees': 1, 'importance': {'streak': 1.0, 'wdl': 1.0}}


@pytest.fixture(scope='module')
def stub_forecasts(index_data):
    return ml.walk_forward_forecasts(index_data, StubReturnModel(), ticker='SYN-INDEX')


@pytest.fixture(scope='module')
def stub_wdl(index_data):
    return ml.walk_forward_forecasts(index_data, StubWDLModel(), ticker='SYN-INDEX')


def test_runner_aligns_return_forecasts_with_their_rows(index_data, index_folds, stub_forecasts):
    fc = stub_forecasts
    test_rows = np.concatenate([test for _, test in index_folds])
    assert fc.target == 'return' and fc.wdl is None
    assert list(fc.y_ret.index) == list(index_data.index[test_rows])
    np.testing.assert_allclose(fc.y_ret, index_data['y_ret'].to_numpy()[test_rows])
    for tau in ml.TAUS:
        expected = index_data['vol_ewma'].to_numpy()[test_rows] * statistics.NormalDist().inv_cdf(tau)
        np.testing.assert_allclose(fc.bands[('model', tau)], expected)
    pd.testing.assert_series_equal(fc.point(), fc.bands[('model', 0.5)])
    assert len(fc.folds) == len(index_folds)
    assert (fc.folds['test_from'] > fc.folds['train_to']).all()


def test_runner_aligns_wdl_forecasts_with_their_rows(index_data, index_folds, stub_wdl):
    fc = stub_wdl
    test_rows = np.concatenate([test for _, test in index_folds])
    assert fc.target == 'wdl' and fc.bands is None and fc.wdl_threshold == 0.002
    np.testing.assert_allclose(fc.y_wdl, index_data['y_wdl'].to_numpy()[test_rows])
    np.testing.assert_allclose(fc.wdl[('model', 1)], np.where(test_rows % 2, 0.5, 0.2))
    pd.testing.assert_frame_equal(fc.wdl[list(ml.WDL_BASELINES)],
                                  ml.wdl_baselines(index_data, index_folds))


def test_runner_refuses_a_model_without_a_target(index_data):
    with pytest.raises(ValueError, match='target'):
        ml.walk_forward_forecasts(index_data, object())


def test_wdl_scores_by_hand(stub_wdl):
    s = ml.wdl_scores(stub_wdl)
    y = stub_wdl.y_wdl.to_numpy()
    losses = {src: ml.multiclass_log_loss(stub_wdl.wdl[src].to_numpy(), y, ml.WDL_CLASSES)
              for src in (*ml.WDL_BASELINES, 'model')}
    assert s['log_loss_model'] == pytest.approx(losses['model'].mean())
    assert s['gain_vs_volatility'] == pytest.approx((losses['volatility'] - losses['model']).mean())
    assert s['share_draw'] == pytest.approx(np.mean(y == 0))
    assert s['lower_vs_volatility'] < s['gain_vs_volatility'] < s['upper_vs_volatility']
    # leaning at random to a win or a loss is worse than the baselines: "no better", not "beats"
    assert s['no_better_than_volatility'] and not s['beats_volatility']
    gains, pooled, above = ml.wdl_check([stub_wdl])
    assert gains['SYN-INDEX'] == pytest.approx((losses['frequencies'] - losses['model']).mean())
    assert pooled['lower'] < pooled['estimate'] < pooled['upper'] and not above
    with pytest.raises(ValueError, match='not return'):
        ml.point_scores(stub_wdl)


def test_wdl_scores_between_no_better_and_beats(stub_wdl):
    # The volatility baseline, nudged toward the outcome on half the days and away from it on the
    # other half: a gain near zero whose interval straddles it, so "no better" and not "beats".
    vol = stub_wdl.wdl['volatility'].to_numpy()
    onehot = stub_wdl.y_wdl.to_numpy()[:, None] == np.asarray(ml.WDL_CLASSES)[None, :]
    toward = np.random.default_rng(0).random(len(vol)) < 0.5
    nudge = np.where(toward[:, None], onehot - vol, -(onehot - vol)) * 0.05
    p = np.clip(vol + nudge, 1e-3, None)
    wdl = stub_wdl.wdl.copy()
    for i, c in enumerate(ml.WDL_CLASSES):
        wdl[('model', c)] = p[:, i] / p.sum(axis=1)
    s = ml.wdl_scores(dataclasses.replace(stub_wdl, wdl=wdl))
    assert s['lower_vs_volatility'] < 0 < s['upper_vs_volatility']
    assert s['no_better_than_volatility'] and not s['beats_volatility']


def _as_model(fc, column):
    """`fc` with one baseline's columns as its model's, to compare two known forecasts."""
    if fc.target == 'wdl':
        wdl = fc.wdl.copy()
        for c in ml.WDL_CLASSES:
            wdl[('model', c)] = wdl[(column, c)]
        return dataclasses.replace(fc, wdl=wdl, model=column)
    bands = fc.bands.copy()
    for tau in fc.taus:
        bands[('model', tau)] = bands[(column, tau)]
    return dataclasses.replace(fc, bands=bands, model=column)


def test_daily_losses_by_hand(stub_wdl, stub_forecasts):
    y = stub_wdl.y_wdl.to_numpy()
    np.testing.assert_allclose(ml.daily_losses(stub_wdl),
                               ml.multiclass_log_loss(stub_wdl.wdl['model'].to_numpy(), y, ml.WDL_CLASSES))
    y = stub_forecasts.y_ret.to_numpy()
    by_hand = np.mean([ml.pinball(stub_forecasts.bands[('model', t)], y, t) for t in ml.TAUS], axis=0)
    np.testing.assert_allclose(ml.daily_losses(stub_forecasts), by_hand)


def test_compare_forecasts_by_hand(stub_wdl, stub_forecasts):
    a, b = _as_model(stub_wdl, 'frequencies'), _as_model(stub_wdl, 'volatility')
    c = ml.compare_forecasts(a, b)
    gain = ml.daily_losses(a) - ml.daily_losses(b)
    assert c['days'] == len(gain) and c['gain_b_over_a'] == pytest.approx(gain.mean())
    assert c['lower'] < c['gain_b_over_a'] < c['upper']
    assert c['b_better'] == (c['lower'] > 0) and c['a_better'] == (c['upper'] < 0)
    same = ml.compare_forecasts(a, a)
    assert same['gain_b_over_a'] == 0 and not same['b_better'] and not same['a_better']
    # only the shared days count
    late = dataclasses.replace(b, y_wdl=b.y_wdl.iloc[500:], wdl=b.wdl.iloc[500:])
    part = ml.compare_forecasts(a, late)
    assert part['days'] == len(gain) - 500 and part['first'] == gain.index[500]
    assert part['gain_b_over_a'] == pytest.approx(gain.iloc[500:].mean())
    r = ml.compare_forecasts(_as_model(stub_forecasts, 'constant'), _as_model(stub_forecasts, 'ewma_std_q'))
    y = stub_forecasts.y_ret
    assert r['mae_b'] == pytest.approx((y - stub_forecasts.point('ewma_std_q')).abs().mean())
    with pytest.raises(ValueError, match='cannot compare'):
        ml.compare_forecasts(stub_wdl, stub_forecasts)
    with pytest.raises(ValueError, match='not the same data'):
        ml.compare_forecasts(a, dataclasses.replace(b, y_wdl=-b.y_wdl))
    with pytest.raises(ValueError, match='threshold'):
        ml.compare_forecasts(a, dataclasses.replace(b, wdl_threshold=0.003))


def test_runner_records_the_settings_and_the_walk_forward(index_data, stub_wdl):
    assert stub_wdl.settings is None                       # the stub has no settings()
    assert stub_wdl.walk_forward == {'first_train': 756, 'step': 63}
    by_hand = ml.walk_forward_forecasts(index_data, StubWDLModel(),
                                        folds=ml.walk_forward(len(index_data))[:2])
    assert by_hand.walk_forward is None


def test_model_comparison_chart(stub_wdl, stub_forecasts):
    rows = []
    for fc, cols in ((stub_wdl, ('frequencies', 'volatility')), (stub_forecasts, ('constant', 'ewma_std_q'))):
        a = dataclasses.replace(_as_model(fc, cols[0]), model='A')
        b = dataclasses.replace(_as_model(fc, cols[1]), model='new')
        for other in (a, dataclasses.replace(a, model='B')):
            rows.append({**ml.compare_forecasts(other, b).to_dict(), 'ticker': 'SYN-INDEX'})
    table = pd.DataFrame(rows)
    fig = ml.plot_model_comparison(table)
    assert len(fig.data) == 4                       # two models compared against, two panels
    first = fig.data[0]
    assert first.name == 'vs A' and fig.data[1].name == 'vs B'
    expected = table.loc[0, 'gain_b_over_a'] / table.loc[0, 'loss_a']
    assert first.x[0] == pytest.approx(expected)
    assert first.error_x.array[0] == pytest.approx((table.loc[0, 'upper'] - table.loc[0, 'gain_b_over_a'])
                                                   / table.loc[0, 'loss_a'])
    assert [a.text for a in fig.layout.annotations][:2] == ['Win/draw/loss (log loss)', 'Return (mean pinball)']


def test_wdl_confusion_chart(stub_wdl):
    fig = ml.plot_wdl_confusion(stub_wdl)
    assert [t.type for t in fig.data] == ['heatmap', 'heatmap']        # volatility, then the model
    hard = ml.wdl_confusion(stub_wdl)
    np.testing.assert_allclose(fig.data[1].z, hard.div(hard.sum(axis=1), axis=0).to_numpy())
    np.testing.assert_allclose(np.asarray(fig.data[1].z).sum(axis=1), 1)
    assert [a.text for a in fig.layout.annotations][:2] == ['volatility baseline', 'stub']
    cells = [a.text for a in fig.layout.annotations][2:]
    assert len(cells) == 18 and '<br>' in cells[0]                  # share and count, per cell
    soft = ml.plot_wdl_confusion(stub_wdl, sources=('model',), soft=True)
    assert len(soft.data) == 1 and '<br>' not in soft.layout.annotations[1].text


def test_saved_models_stay_local_and_their_parameters_are_tracked():
    root = Path(__file__).resolve().parents[1]

    def ignored(path):
        return subprocess.run(['git', 'check-ignore', '-q', path], cwd=root).returncode == 0

    assert ignored('models/models/lightgbm_wdl_SPY_20260930_0123abcd.pkl')
    for name in ('lightgbm_wdl_SPY_20260930_0123abcd', 'gru-returns-range_return_TMP_dev_copy_1'):
        assert not ignored(f'models/model_parameters/{name}.json'), name


def test_confusion_matrix_by_hand():
    m = ml.confusion_matrix([-1, -1, 0, 1, 1, 1], [1, -1, 0, 1, 1, 0], (-1, 0, 1))
    np.testing.assert_array_equal(m, [[1, 0, 1],      # two losses: one called a loss, one a win
                                      [0, 1, 0],
                                      [0, 1, 2]])
    with pytest.raises(ValueError, match='forecasts outside'):
        ml.confusion_matrix([1], [2], (-1, 0, 1))
    with pytest.raises(ValueError, match='outcomes outside'):
        ml.confusion_matrix([5], [1], (-1, 0, 1))


def test_wdl_confusion_by_hand(stub_wdl):
    # The stub leans to a win on odd rows and to a loss on even rows.
    y = stub_wdl.y_wdl.to_numpy()
    p = stub_wdl.wdl['model'].to_numpy()
    call = np.where(p[:, 2] > p[:, 0], 1, -1)
    hard = ml.wdl_confusion(stub_wdl)
    expected = pd.crosstab(y, call).reindex(index=ml.WDL_CLASSES, columns=ml.WDL_CLASSES, fill_value=0)
    np.testing.assert_array_equal(hard.to_numpy(), expected.to_numpy())
    assert list(hard.index) == list(hard.columns) == ['loss', 'draw', 'win']
    assert (hard['draw'] == 0).all() and hard.to_numpy().sum() == len(y)
    soft = ml.wdl_confusion(stub_wdl, soft=True)
    for i, c in enumerate(ml.WDL_CLASSES):
        np.testing.assert_allclose(soft.iloc[i], p[y == c].sum(axis=0))
    np.testing.assert_allclose(soft.sum(axis=1), [np.sum(y == c) for c in ml.WDL_CLASSES])
    vol_call = np.asarray(ml.WDL_CLASSES)[stub_wdl.wdl['volatility'].to_numpy().argmax(axis=1)]
    vol = ml.wdl_confusion(stub_wdl, source='volatility')
    np.testing.assert_array_equal(vol.to_numpy(), pd.crosstab(y, vol_call).reindex(
        index=ml.WDL_CLASSES, columns=ml.WDL_CLASSES, fill_value=0).to_numpy())
    assert not vol.equals(hard)
    # several tickers' days are pooled
    np.testing.assert_array_equal(ml.wdl_confusion([stub_wdl, stub_wdl]), 2 * hard.to_numpy())
    with pytest.raises(ValueError, match='not wdl'):
        ml.wdl_confusion(dataclasses.replace(stub_wdl, target='return'))


def test_regression_scores_match_scikit_learn():
    metrics = pytest.importorskip('sklearn.metrics')
    rng = np.random.default_rng(3)
    y = rng.standard_t(4, 500) * 0.01
    f = 0.3 * y + rng.normal(0, 0.008, 500)
    s = ml.regression_scores(y, f)
    assert s['r2'] == pytest.approx(metrics.r2_score(y, f))
    assert s['mae'] == pytest.approx(metrics.mean_absolute_error(y, f))
    assert s['rmse'] == pytest.approx(np.sqrt(metrics.mean_squared_error(y, f)))
    assert s['corr'] == pytest.approx(np.corrcoef(f, y)[0, 1])
    assert s['ic'] == pytest.approx(pd.Series(f).corr(pd.Series(y), method='spearman'))


def test_regression_scores_by_hand():
    y = np.array([0.02, -0.01, 0.0, 0.03])
    f = np.array([0.01, 0.01, 0.02, -0.01])
    s = ml.regression_scores(y, f)
    assert s['r2_vs_zero'] == pytest.approx(1 - np.sum((f - y) ** 2) / np.sum(y ** 2))
    assert s['bias'] == pytest.approx(np.mean(f - y))          # above zero: forecasts too high
    assert s['hit_rate'] == pytest.approx(1 / 3)              # the day y = 0 is left out
    assert s['forecast_up'] == pytest.approx(0.75)
    flat = ml.regression_scores(y, np.zeros(4))
    assert flat['r2_vs_zero'] == 0 and np.isnan(flat['corr']) and np.isnan(flat['hit_rate'])
    assert flat['r2'] < 0                                      # zero is not the test days' mean


def test_classification_scores_match_scikit_learn():
    metrics = pytest.importorskip('sklearn.metrics')
    rng = np.random.default_rng(5)
    classes = (-1, 0, 1)
    actual = rng.choice(classes, 600, p=[0.3, 0.2, 0.5])
    predicted = np.where(rng.random(600) < 0.4, actual, rng.choice([-1, 1], 600))
    predicted[predicted == 0] = 1                                                    # never a draw
    summary, per_class = ml.classification_scores(actual, predicted, classes)
    assert summary['accuracy'] == pytest.approx(metrics.accuracy_score(actual, predicted))
    assert summary['balanced_accuracy'] == pytest.approx(metrics.balanced_accuracy_score(actual, predicted))
    for average in ('macro', 'weighted'):
        assert summary[f'{average}_f1'] == pytest.approx(metrics.f1_score(
            actual, predicted, labels=list(classes), average=average, zero_division=0))
    assert summary['mcc'] == pytest.approx(metrics.matthews_corrcoef(actual, predicted))
    p, r, f, n = metrics.precision_recall_fscore_support(actual, predicted, labels=list(classes),
                                                          zero_division=0)
    np.testing.assert_allclose(per_class['precision'], p)
    np.testing.assert_allclose(per_class['recall'], r)
    np.testing.assert_allclose(per_class['f1'], f)
    np.testing.assert_array_equal(per_class['support'], n)
    assert per_class.loc[0, 'called'] == 0 and per_class.loc[0, 'precision'] == 0


def test_a_constant_call_scores_as_chance():
    actual = np.array([-1, -1, 0, 1, 1, 1, 1, 0])
    summary, _ = ml.classification_scores(actual, np.ones(8, dtype=int), (-1, 0, 1))
    assert summary['accuracy'] == pytest.approx(0.5)            # the share of wins
    assert summary['balanced_accuracy'] == pytest.approx(1 / 3)
    assert summary['mcc'] == 0


def test_wdl_metrics_by_hand(stub_wdl):
    table, gains = ml.wdl_metrics(stub_wdl)
    y = stub_wdl.y_wdl.to_numpy()
    classes = np.asarray(ml.WDL_CLASSES)
    recall, losses = {}, {}
    for source in (*ml.WDL_BASELINES, 'model'):
        probs = stub_wdl.wdl[source].to_numpy()
        call = classes[probs.argmax(axis=1)]
        summary, per_class = ml.classification_scores(y, call, ml.WDL_CLASSES)
        row = table.loc[source]
        assert row['days'] == len(y)
        for k in ('accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'mcc'):
            assert row[k] == pytest.approx(summary[k]), (source, k)
        assert row['called_win'] == pytest.approx(np.mean(call == 1))
        losses[source] = ml.multiclass_log_loss(probs, y, ml.WDL_CLASSES)
        assert row['log_loss'] == pytest.approx(losses[source].mean())
        assert row['brier'] == pytest.approx(ml.multiclass_brier(probs, y, ml.WDL_CLASSES).mean())
        recall[source] = summary['balanced_accuracy']
    assert table.loc['model', 'called_draw'] == 0               # the stub never favours a draw
    for b in ml.WDL_BASELINES:
        g = gains.loc[('balanced_accuracy', b)]
        assert g['estimate'] == pytest.approx(recall['model'] - recall[b])
        assert g['lower'] <= g['estimate'] <= g['upper']
        assert gains.loc[('log_loss', b), 'estimate'] == pytest.approx((losses[b] - losses['model']).mean())
    pooled, _ = ml.wdl_metrics([stub_wdl, stub_wdl])
    assert pooled.loc['model', 'days'] == 2 * len(y)
    assert pooled.loc['model', 'macro_f1'] == pytest.approx(table.loc['model', 'macro_f1'])


def test_wdl_class_report(stub_wdl):
    report = ml.wdl_class_report(stub_wdl)
    assert list(report.index) == ['loss', 'draw', 'win']
    assert report['support'].sum() == len(stub_wdl.y_wdl)
    assert report.loc['draw', 'recall'] == 0 and report.loc['draw', 'called'] == 0
    assert report['called'].sum() == pytest.approx(1)


def test_return_metrics_by_hand(stub_forecasts):
    table = ml.return_metrics(stub_forecasts)
    y = stub_forecasts.y_ret.to_numpy()
    assert list(table.index) == ['model', 'no change', *ml.BANDS]
    s = ml.regression_scores(y, stub_forecasts.point())
    row = table.loc['model']
    assert row['r2'] == pytest.approx(s['r2'])
    assert np.isnan(row['ic']) and np.isnan(s['ic'])          # the stub's median is zero every day
    assert row['mae_bp'] == pytest.approx(s['mae'] * 1e4)
    assert row['rmse_bp'] == pytest.approx(s['rmse'] * 1e4)
    ranges = ml.range_table(stub_forecasts)
    assert row['cover_68'] == pytest.approx(ranges.loc['model', 'cover_68'])
    assert row['pinball_bp'] == pytest.approx(ranges.loc['model', 'pinball'] * 1e4)
    flat = table.loc['no change']
    assert flat['r2_vs_no_change'] == 0 and flat['mae_bp'] == pytest.approx(np.mean(np.abs(y)) * 1e4)
    assert np.isnan(flat['cover_68'])
    ewma = ml.regression_scores(y, stub_forecasts.point('ewma_std_q'))
    assert table.loc['ewma_std_q', 'r2_vs_no_change'] == pytest.approx(ewma['r2_vs_zero'])
    pooled = ml.return_metrics([stub_forecasts, stub_forecasts])
    assert pooled.loc['model', 'days'] == 2 * len(y)
    assert pooled.loc['model', 'r2'] == pytest.approx(row['r2'])
    with pytest.raises(ValueError, match='not return'):
        ml.return_metrics(dataclasses.replace(stub_forecasts, target='wdl'))


def test_point_scores_by_hand(stub_forecasts):
    s = ml.point_scores(stub_forecasts)
    y = stub_forecasts.y_ret.to_numpy()
    for band in (*ml.BANDS, 'model'):
        assert s[f'mae_{band}'] == pytest.approx(np.mean(np.abs(y - stub_forecasts.point(band))))
    gain = np.abs(y - stub_forecasts.point('ewma_std_q')) - np.abs(y - stub_forecasts.point('model'))
    assert s['gain_vs_ewma'] == pytest.approx(gain.mean())
    with pytest.raises(ValueError, match='not wdl'):
        ml.wdl_scores(stub_forecasts)


def test_range_table_and_check(stub_forecasts):
    table = ml.range_table(stub_forecasts)
    assert list(table.index) == [*ml.BANDS, 'model']
    scores, _ = ml.range_scores(stub_forecasts.quantiles('ewma_std_q'), stub_forecasts.y_ret)
    assert table.loc['ewma_std_q', 'pinball'] == pytest.approx(scores['pinball'])
    per_ticker, pooled, passes = ml.range_check([stub_forecasts])
    assert per_ticker.loc['SYN-INDEX', 'gain_vs_constant'] == pytest.approx(
        1 - table.loc['model', 'pinball'] / table.loc['constant', 'pinball'])
    assert pooled['lower'] < pooled['estimate'] < pooled['upper']
    assert passes == bool(per_ticker.iloc[0]['lower_pinball'] and per_ticker.iloc[0]['less_clustering']
                          and pooled['lower'] > 0)


def test_feature_importance_shares(stub_forecasts, stub_wdl):
    assert ml.feature_importance(stub_forecasts).to_dict() == pytest.approx({'ret_0': 0.75, 'vol_20': 0.25})
    assert ml.feature_importance(stub_wdl).to_dict() == pytest.approx({'streak': 0.5, 'wdl': 0.5})


def test_evaluate_many_matches_one_at_a_time():
    for model in (StubReturnModel(), StubWDLModel()):
        one = [ml.evaluate(t, model) for t in ('SYN-GOLD', 'SYN-FX')]
        many = ml.evaluate_many(['SYN-GOLD', 'SYN-FX'], model, n_jobs=2)
        for a, b in zip(one, many):
            assert a.ticker == b.ticker and a.target == b.target
            for part in ('bands', 'wdl'):
                if getattr(a, part) is not None:
                    pd.testing.assert_frame_equal(getattr(a, part), getattr(b, part))


def test_evaluate_passes_the_threshold_on():
    fc = ml.evaluate('SYN-FX', StubWDLModel(), wdl_threshold=0.005)
    assert fc.wdl_threshold == 0.005
    d = ml.dataset(synthetic_bars('SYN-FX'), wdl_threshold=0.005)
    np.testing.assert_array_equal(fc.y_wdl, d.loc[fc.y_wdl.index, 'y_wdl'])
    assert (fc.y_wdl == 0).mean() > 0.6                          # a wide band: mostly draws


def test_planted_signal_bars():
    bars = ml.planted_signal_bars()
    sign = np.sign(bars['Close'].pct_change().dropna())
    repeats = (sign.to_numpy()[1:] == sign.to_numpy()[:-1]).mean()
    assert repeats == pytest.approx(0.65, abs=0.02)                    # the signal is really there
    original = synthetic_bars('SYN-INDEX')
    np.testing.assert_allclose(bars['Close'].pct_change().abs().iloc[1:],
                               original['Close'].pct_change().abs().iloc[1:], rtol=1e-6)  # sizes kept
    assert (bars['High'] >= bars[['Open', 'Close']].max(axis=1) - 1e-9).all()


# ── Charts ───────────────────────────────────────────────────────────────────

def test_charts_draw_the_forecasts(index_bars, stub_forecasts, stub_wdl):
    fc = stub_forecasts
    fig = ml.plot_coverage(fc, bands=('constant', 'model'), window=250)
    miss = ml.interval_misses(fc.bands[('model', 0.1587)], fc.bands[('model', 0.8413)], fc.y_ret)
    expected = 1 - pd.Series(miss, index=fc.y_ret.index).rolling(250).mean()
    np.testing.assert_allclose(np.asarray(fig.data[1].y, dtype=float), expected, equal_nan=True)

    fig = ml.plot_wdl_reliability(stub_wdl)
    win = ml.reliability_table(stub_wdl.wdl[('model', 1)], (stub_wdl.y_wdl == 1).astype(float), 10)
    np.testing.assert_allclose(fig.data[3].y, win['observed'])          # traces: diagonal, l, d, w

    fig = ml.plot_forecast_bands(fc, index_bars, '2024-01-01', '2024-03-31')
    # On a day, each line is the close before it times (1 + that day's forecast): the 68% upper
    # edge, and the point forecast, the median.
    day = pd.Timestamp('2024-02-15')
    before = index_bars.index[index_bars.index.get_loc(day) - 1]
    at_day = lambda trace: dict(zip(pd.to_datetime(trace.x), trace.y))[day]
    close = index_bars.loc[before, 'Close']
    assert at_day(fig.data[2]) == pytest.approx(close * (1 + fc.bands.loc[before, ('model', 0.8413)]))
    assert fig.data[4].name.startswith('point forecast')
    assert at_day(fig.data[4]) == pytest.approx(close * (1 + fc.bands.loc[before, ('model', 0.5)]))

    assert len(ml.plot_folds(fc).data) == 2 * len(fc.folds)
    assert list(ml.plot_feature_importance(ml.feature_importance(fc)).data[0].y) == ['vol_20', 'ret_0']
    assert len(ml.plot_range_comparison([fc]).data) == len(ml.BANDS)
