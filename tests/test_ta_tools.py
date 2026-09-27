"""Offline tests for src.tools.ta_tools."""
import subprocess
import sys

import pandas as pd
import pytest

from src.tools import ta_tools
from src.tools.ta_tools import backend


def test_public_api_resolves():
    missing = [name for name in ta_tools.__all__ if not hasattr(ta_tools, name)]
    assert not missing, f"names in __all__ with no attribute: {missing}"


def test_importing_ta_tools_does_not_load_pandas_ta():
    # A fresh interpreter, since this test session may already have imported it.
    loaded = subprocess.run(
        [sys.executable, "-c",
         "import sys, src.tools.ta_tools; print('pandas_ta' in sys.modules)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert loaded == "False"


def test_pandas_ta_loads_on_demand():
    pytest.importorskip("pandas_ta")
    assert backend.pandas_ta().__name__ == "pandas_ta"


def test_provides_records_the_source(monkeypatch):
    monkeypatch.setattr(backend, "CAPABILITIES", {})

    @backend.provides("custom")
    def some_primitive():
        return 1

    assert backend.CAPABILITIES == {"some_primitive": "custom"}
    assert some_primitive() == 1


def test_provides_rejects_unknown_sources():
    with pytest.raises(ValueError, match="source must be one of"):
        backend.provides("ta")


def test_make_bars_shape_and_reproducibility():
    bars = ta_tools.make_bars(n=250, seed=7)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) == 250
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert bars.index.is_monotonic_increasing
    assert not bars.isna().any().any()

    pd.testing.assert_frame_equal(bars, ta_tools.make_bars(n=250, seed=7))
    assert not bars.equals(ta_tools.make_bars(n=250, seed=8))


def test_make_bars_are_valid_ohlc():
    bars = ta_tools.make_bars(n=2000, seed=3)
    body_top = bars[["open", "close"]].max(axis=1)
    body_bottom = bars[["open", "close"]].min(axis=1)
    assert (bars["high"] >= body_top).all()
    assert (bars["low"] <= body_bottom).all()
    assert (bars["low"] > 0).all()


# ── Phase 3: wrapped primitives ──────────────────────────────────────────────
import numpy as np

BARS = ta_tools.make_bars(n=400, seed=11)
HIGH, LOW, CLOSE = BARS["high"], BARS["low"], BARS["close"]


def leading_nans(series):
    return int(np.argmax(series.notna().to_numpy()))


def pine_seeded(values, length, alpha):
    """Pine's ta.ema/ta.rma: SMA of the first `length` values, then the alpha recursion."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    start = int(np.flatnonzero(~np.isnan(values))[0])
    first = start + length - 1
    out[first] = values[start:first + 1].mean()
    for i in range(first + 1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def wma(values, length):
    weights = np.arange(1, length + 1, dtype=float)
    return pd.Series(values).rolling(length).apply(
        lambda w: np.dot(w, weights) / weights.sum(), raw=True).to_numpy()


def pine_alma(values, length, offset, sigma, floor):
    """TradingView's reference ta.alma, written out independently of pandas_ta."""
    m = np.floor(offset * (length - 1)) if floor else offset * (length - 1)
    s = length / sigma
    weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * s ** 2))
    weights /= weights.sum()
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(length - 1, len(values)):
        out[i] = np.dot(values[i - length + 1:i + 1], weights)
    return out


CONTRACT = [
    ("sma", lambda: ta_tools.sma(CLOSE, 20), "sma_20", 19),
    ("ema", lambda: ta_tools.ema(CLOSE, 20), "ema_20", 19),
    ("wma", lambda: ta_tools.wma(CLOSE, 20), "wma_20", 19),
    ("rsi", lambda: ta_tools.rsi(CLOSE, 14), "rsi_14", 14),
    ("stdev", lambda: ta_tools.stdev(CLOSE, 20), "stdev_20", 19),
    ("atr", lambda: ta_tools.atr(HIGH, LOW, CLOSE, 14), "atr_14", 13),
    ("hma", lambda: ta_tools.hma(CLOSE, 16), "hma_16", 18),
    ("alma", lambda: ta_tools.alma(CLOSE, 9), "alma_9", 8),
]


@pytest.mark.parametrize("label,call,name,warmup", CONTRACT, ids=[c[0] for c in CONTRACT])
def test_primitive_contract(label, call, name, warmup):
    if ta_tools.CAPABILITIES[label] == "pandas_ta":
        pytest.importorskip("pandas_ta")
    out = call()
    assert isinstance(out, pd.Series)
    assert out.index.equals(CLOSE.index)
    assert out.dtype == "float64"
    assert out.name == name
    assert leading_nans(out) == warmup
    assert out.iloc[warmup:].notna().all(), "no gaps after the warm-up"


def test_capabilities_record_each_primitive_source():
    expected = {"sma": "talib", "ema": "talib", "wma": "talib", "stdev": "talib",
                "atr": "custom", "true_range": "custom", "bb": "talib", "rsi": "talib",
                "hma": "pandas_ta", "alma": "pandas_ta"}
    assert {k: ta_tools.CAPABILITIES[k] for k in expected} == expected


def test_non_series_input_is_rejected():
    with pytest.raises(TypeError, match="expected a pandas Series"):
        ta_tools.sma(CLOSE.to_numpy(), 5)


def test_integer_input_is_accepted():
    ints = pd.Series(np.arange(1, 31), index=CLOSE.index[:30])
    assert ta_tools.sma(ints, 3).iloc[-1] == pytest.approx(29.0)


def test_sma_of_a_constant_is_the_constant():
    flat = pd.Series(7.5, index=CLOSE.index)
    assert (ta_tools.sma(flat, 10).dropna() == 7.5).all()


def test_length_one_follows_pine_where_talib_refuses():
    # TA-Lib raises TA_BAD_PARAM for length 1 on SMA, EMA, WMA, STDDEV, BBANDS and RSI. Pine
    # accepts it, and the Pine scripts' inputs allow it (e.g. LinReg Candles' signal_length min is 1).
    pd.testing.assert_series_equal(ta_tools.sma(CLOSE, 1), CLOSE.rename("sma_1"))
    pd.testing.assert_series_equal(ta_tools.ema(CLOSE, 1), CLOSE.rename("ema_1"))
    assert (ta_tools.stdev(CLOSE, 1) == 0).all()
    pd.testing.assert_series_equal(ta_tools.wma(CLOSE, 1), CLOSE.rename("wma_1"))
    bands = ta_tools.bb(CLOSE, 1)
    for column in bands:
        np.testing.assert_array_equal(bands[column], CLOSE)
    change = CLOSE.diff()
    rsi1 = ta_tools.rsi(CLOSE, 1)
    assert np.isnan(rsi1.iloc[0])
    assert ((rsi1[change > 0] == 100).all() and (rsi1[change <= 0] == 0).all())


def test_ema_matches_pine_definition():
    expected = pine_seeded(CLOSE, 20, 2 / 21)
    np.testing.assert_allclose(ta_tools.ema(CLOSE, 20), expected, rtol=1e-12, equal_nan=True)


def test_stdev_is_population_as_in_pine():
    population = CLOSE.rolling(20).std(ddof=0)
    np.testing.assert_allclose(ta_tools.stdev(CLOSE, 20), population, rtol=1e-9, equal_nan=True)
    # pandas' default (ddof=1) is the easy mistake; make sure it really differs.
    assert not np.allclose(ta_tools.stdev(CLOSE, 20).dropna(), CLOSE.rolling(20).std().dropna())


def pine_atr(high, low, close, length):
    """Pine's ta.atr written out independently: Wilder-smoothed true range, bar 0's being high - low."""
    prev = close.shift(1).to_numpy()
    h, l = high.to_numpy(), low.to_numpy()
    tr = np.where(np.isnan(prev), h - l,
                  np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)]))
    return pine_seeded(tr, length, 1 / length)


def test_atr_matches_pine_exactly():
    np.testing.assert_allclose(ta_tools.atr(HIGH, LOW, CLOSE, 14), pine_atr(HIGH, LOW, CLOSE, 14),
                               rtol=1e-12, equal_nan=True)


def test_talib_atr_is_why_atr_is_built_here():
    # TA-Lib has no bar-0 true range, so it starts a bar later and disagrees early on.
    talib_atr = backend.talib.ATR(HIGH, LOW, CLOSE, timeperiod=14).to_numpy()
    pine = pine_atr(HIGH, LOW, CLOSE, 14)
    assert leading_nans(pd.Series(talib_atr)) == leading_nans(pd.Series(pine)) + 1
    relative = np.abs(talib_atr - pine) / pine
    assert relative[14] > 1e-3 and relative[-1] < 1e-9


def test_true_range():
    tr = ta_tools.true_range(HIGH, LOW, CLOSE)
    assert tr.iloc[0] == HIGH.iloc[0] - LOW.iloc[0]
    assert (tr >= HIGH - LOW).all()
    gaps = pd.concat([(HIGH - CLOSE.shift(1)).abs(), (LOW - CLOSE.shift(1)).abs()], axis=1).max(axis=1)
    assert (tr.iloc[1:] >= gaps.iloc[1:]).all()


def test_hma_matches_wma_definition():
    pytest.importorskip("pandas_ta")
    for length in (9, 15, 16):
        half, root = int(length / 2), int(np.sqrt(length))
        inner = 2 * wma(CLOSE, half) - wma(CLOSE, length)
        expected = wma(np.nan_to_num(inner, nan=np.nan), root)
        np.testing.assert_allclose(ta_tools.hma(CLOSE, length), expected,
                                   rtol=1e-9, equal_nan=True, err_msg=f"length={length}")


def test_alma_equals_pine_alma_with_floor():
    pytest.importorskip("pandas_ta")
    ours = ta_tools.alma(CLOSE, 9, offset=0.85, sigma=6.0)
    floored = pine_alma(CLOSE, 9, 0.85, 6.0, floor=True)
    unfloored = pine_alma(CLOSE, 9, 0.85, 6.0, floor=False)
    np.testing.assert_allclose(ours, floored, rtol=1e-9, equal_nan=True)
    assert not np.allclose(ours.dropna(), unfloored[~np.isnan(unfloored)]), \
        "differs from Pine's default floor=false, as documented"


def test_talib_primitives_agree_with_pandas_ta_own_implementation():
    pta = pytest.importorskip("pandas_ta")
    # talib=False forces pandas_ta's own code; otherwise it would silently call TA-Lib.
    np.testing.assert_allclose(ta_tools.sma(CLOSE, 20), pta.sma(CLOSE, 20, talib=False),
                               rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(ta_tools.ema(CLOSE, 20), pta.ema(CLOSE, 20, talib=False),
                               rtol=1e-9, equal_nan=True)


def test_wma_matches_pine_definition():
    weights = np.arange(1, 21, dtype=float)          # newest bar gets the largest weight
    expected = CLOSE.rolling(20).apply(lambda w: w @ weights / weights.sum(), raw=True)
    np.testing.assert_allclose(ta_tools.wma(CLOSE, 20), expected, rtol=1e-10, equal_nan=True)


def pine_rsi(close, length):
    """Pine's RSI: rma of up and down moves; defined independently of TA-Lib."""
    change = close.diff().to_numpy()
    up = np.where(np.isnan(change), np.nan, np.maximum(change, 0))
    down = np.where(np.isnan(change), np.nan, np.maximum(-change, 0))
    up, down = pine_seeded(up, length, 1 / length), pine_seeded(down, length, 1 / length)
    return 100 - 100 / (1 + up / down)


def test_rsi_matches_pine_definition():
    np.testing.assert_allclose(ta_tools.rsi(CLOSE, 14), pine_rsi(CLOSE, 14),
                               rtol=1e-10, equal_nan=True)


def test_rsi_extremes_follow_talib_convention():
    index = CLOSE.index[:40]
    rising = pd.Series(np.arange(40.0), index=index)
    assert ta_tools.rsi(rising, 14).iloc[-1] == 100
    assert ta_tools.rsi(rising[::-1].set_axis(index), 14).iloc[-1] == 0
    # Flat window: TA-Lib returns 0, TradingView's built-in RSI script 100. Documented, not "fixed".
    assert ta_tools.rsi(pd.Series(50.0, index=index), 14).iloc[-1] == 0


def test_bb_frame_and_values():
    bands = ta_tools.bb(CLOSE, 20, 2.0)
    assert isinstance(bands, pd.DataFrame)
    assert list(bands.columns) == ["bb_mid_20", "bb_upper_20", "bb_lower_20"]
    assert bands.index.equals(CLOSE.index)
    assert (bands.dtypes == "float64").all()
    assert all(leading_nans(bands[c]) == 19 for c in bands)

    basis = CLOSE.rolling(20).mean()
    spread = 2.0 * CLOSE.rolling(20).std(ddof=0)    # population, as Pine's ta.bb
    np.testing.assert_allclose(bands["bb_mid_20"], basis, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(bands["bb_upper_20"], basis + spread, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(bands["bb_lower_20"], basis - spread, rtol=1e-10, equal_nan=True)

    valid = bands.dropna()
    assert (valid["bb_upper_20"] >= valid["bb_mid_20"]).all()
    assert (valid["bb_mid_20"] >= valid["bb_lower_20"]).all()


def test_bb_joins_onto_the_bars():
    joined = BARS.join(ta_tools.bb(CLOSE, 20))
    assert list(joined.columns[-3:]) == ["bb_mid_20", "bb_upper_20", "bb_lower_20"]
    assert len(joined) == len(BARS)


# ── Phase 4: Pine primitives no library provides ─────────────────────────────
INDEX = pd.bdate_range("2024-01-01", periods=12)


def test_linreg_on_a_straight_line_reproduces_the_line():
    line = pd.Series(3.0 + 0.5 * np.arange(60), index=CLOSE.index[:60])
    for offset in (0, 1, 4):
        fitted = ta_tools.linreg(line, 11, offset)
        # A line fitted to a line is the line; evaluated `offset` bars back it gives that bar's value.
        np.testing.assert_allclose(fitted.iloc[10:], line.shift(offset).iloc[10:], rtol=1e-10,
                                   equal_nan=True, err_msg=f"offset={offset}")


def test_linreg_matches_numpy_polyfit():
    values = CLOSE.to_numpy()
    for offset in (0, 2, 5):
        fitted = ta_tools.linreg(CLOSE, 14, offset).to_numpy()
        for bar in (13, 150, 399):
            slope, intercept = np.polyfit(np.arange(14), values[bar - 13:bar + 1], 1)
            assert fitted[bar] == pytest.approx(intercept + slope * (14 - 1 - offset), rel=1e-10)


def test_linreg_is_equivariant():
    base = ta_tools.linreg(CLOSE, 11, 2)
    np.testing.assert_allclose(ta_tools.linreg(3 * CLOSE + 7, 11, 2), 3 * base + 7,
                               rtol=1e-10, equal_nan=True)


def test_linreg_accepts_its_own_output():
    once = ta_tools.linreg(CLOSE, 11)
    twice = ta_tools.linreg(once, 11)
    assert leading_nans(once) == 10 and leading_nans(twice) == 20
    assert twice.iloc[20:].notna().all()


def test_rma_matches_pine_definition():
    np.testing.assert_allclose(ta_tools.rma(CLOSE, 14), pine_seeded(CLOSE, 14, 1 / 14),
                               rtol=1e-12, equal_nan=True)


def test_rma_seeds_from_the_first_valid_value():
    gappy = CLOSE.copy()
    gappy.iloc[:5] = np.nan
    out = ta_tools.rma(gappy, 14)
    assert leading_nans(out) == 5 + 13
    assert out.iloc[18] == pytest.approx(CLOSE.iloc[5:19].mean())


def test_length_one_linreg_and_rma_return_the_input():
    np.testing.assert_array_equal(ta_tools.linreg(CLOSE, 1), CLOSE)
    np.testing.assert_allclose(ta_tools.rma(CLOSE, 1), CLOSE)


def pivot_oracle(values, left, right, high=True):
    """Brute force: a pivot is >= every bar on its left and > every bar on its right."""
    values = np.asarray(values, dtype=float) * (1 if high else -1)
    out = np.full(len(values), np.nan)
    for i in range(left, len(values) - right):
        centre = values[i]
        if (all(centre >= v for v in values[i - left:i])
                and all(centre > v for v in values[i + 1:i + right + 1])):
            out[i + right] = centre
    return out * (1 if high else -1)


@pytest.mark.parametrize("left,right", [(1, 1), (3, 2), (5, 5), (14, 14)])
def test_pivots_match_brute_force_including_ties(left, right):
    tied = (CLOSE * 2).round() / 2        # half-point steps force plenty of equal values
    np.testing.assert_array_equal(ta_tools.pivot_high(tied, left, right),
                                  pivot_oracle(tied, left, right, high=True))
    np.testing.assert_array_equal(ta_tools.pivot_low(tied, left, right),
                                  pivot_oracle(tied, left, right, high=False))


def test_pivots_are_published_right_bars_after_the_pivot():
    left, right = 4, 3
    published = ta_tools.pivot_high(HIGH, left, right)
    for position in np.flatnonzero(published.notna().to_numpy()):
        pivot_bar = position - right
        assert published.iloc[position] == HIGH.iloc[pivot_bar]
        assert HIGH.iloc[pivot_bar] == HIGH.iloc[pivot_bar - left:pivot_bar + right + 1].max()


def test_a_flat_top_yields_one_pivot_at_its_last_bar():
    top = pd.Series([1, 2, 5, 5, 5, 2, 1, 0, 0, 0, 0, 0], index=INDEX, dtype=float)
    published = ta_tools.pivot_high(top, 2, 2)
    assert published.notna().sum() == 1
    assert published.iloc[4 + 2] == 5


NO_LOOKAHEAD = {
    "sma": lambda b: ta_tools.sma(b["close"], 10),
    "ema": lambda b: ta_tools.ema(b["close"], 10),
    "wma": lambda b: ta_tools.wma(b["close"], 10),
    "rsi": lambda b: ta_tools.rsi(b["close"], 14),
    "stdev": lambda b: ta_tools.stdev(b["close"], 10),
    "atr": lambda b: ta_tools.atr(b["high"], b["low"], b["close"], 14),
    "true_range": lambda b: ta_tools.true_range(b["high"], b["low"], b["close"]),
    "bb_upper": lambda b: ta_tools.bb(b["close"], 20)["bb_upper_20"],
    "hma": lambda b: ta_tools.hma(b["close"], 16),
    "alma": lambda b: ta_tools.alma(b["close"], 9),
    "linreg": lambda b: ta_tools.linreg(b["close"], 11, 3),
    "rma": lambda b: ta_tools.rma(b["close"], 14),
    "pivot_high": lambda b: ta_tools.pivot_high(b["high"], 5, 5),
    "pivot_low": lambda b: ta_tools.pivot_low(b["low"], 5, 5),
    # Whole frames, as float so lrc_bull compares alongside the price columns.
    "linreg_candles_sma": lambda b: ta_tools.linreg_candles(
        b["open"], b["high"], b["low"], b["close"]).astype(float),
    "linreg_candles_ema": lambda b: ta_tools.linreg_candles(
        b["open"], b["high"], b["low"], b["close"], 14, 9, sma_signal=False).astype(float),
}


@pytest.mark.parametrize("label", list(NO_LOOKAHEAD))
def test_no_lookahead(label):
    """Values up to bar t must not change when the bars after t are removed."""
    if label in ("hma", "alma"):
        pytest.importorskip("pandas_ta")
    full = NO_LOOKAHEAD[label](BARS)
    for cut in (60, 150, 299):
        truncated = NO_LOOKAHEAD[label](BARS.iloc[:cut + 1])
        np.testing.assert_allclose(truncated, full.iloc[:cut + 1], rtol=1e-10, equal_nan=True,
                                   err_msg=f"{label} changed at or before bar {cut}")


def test_crossover_and_crossunder():
    a = pd.Series([1, 2, 3, 2, 1, 2, 5, 1, 1, 3, 3, 0], index=INDEX, dtype=float)
    assert ta_tools.crossover(a, 1.5).astype(int).tolist() == [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    assert ta_tools.crossunder(a, 1.5).astype(int).tolist() == [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
    # Against a series: 2 -> 3 is a cross over 2, but rising from 1 to exactly 2 is not.
    b = pd.Series(2.0, index=INDEX)
    assert ta_tools.crossover(a, b).astype(int).tolist() == [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]


def test_barssince():
    flags = pd.Series([False, True, False, False, True, False], index=INDEX[:6])
    assert ta_tools.barssince(flags).tolist()[1:] == [0, 1, 2, 0, 1]
    assert np.isnan(ta_tools.barssince(flags).iloc[0])
    assert ta_tools.barssince(pd.Series(False, index=INDEX)).isna().all()


def test_change_and_nz():
    x = pd.Series(np.arange(12, dtype=float) ** 2, index=INDEX)
    pd.testing.assert_series_equal(ta_tools.change(x, 2), (x - x.shift(2)).rename("change_2"))
    assert ta_tools.nz(ta_tools.change(x, 2)).iloc[:2].tolist() == [0.0, 0.0]


def test_recurse_runs_bar_by_bar_state():
    frame = pd.DataFrame({"v": [1, 2, 3, 2, 1, 2, 5, 1]}, index=INDEX[:8], dtype=float)
    run = ta_tools.recurse(frame, lambda s, bar: {"run": s["run"] + 1 if bar.v > 1 else 0},
                           {"run": 0})
    assert run["run"].tolist() == [0, 1, 2, 3, 0, 1, 2, 0]


def test_recurse_is_safe_against_in_place_mutation():
    frame = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=INDEX[:3])

    def mutating(state, bar):
        state["total"] += bar.v          # mutates the dict it was given
        return state

    assert ta_tools.recurse(frame, mutating, {"total": 0.0})["total"].tolist() == [1.0, 3.0, 6.0]


def test_recurse_reproduces_rma():
    alpha = 1 / 14
    seed = CLOSE.iloc[:14].mean()
    frame = pd.DataFrame({"x": CLOSE, "i": np.arange(len(CLOSE))})

    def step(state, bar):
        if bar.i < 13:
            return {"rma": np.nan}
        if bar.i == 13:
            return {"rma": seed}
        return {"rma": alpha * bar.x + (1 - alpha) * state["rma"]}

    np.testing.assert_allclose(ta_tools.recurse(frame, step, {"rma": np.nan})["rma"],
                               ta_tools.rma(CLOSE, 14), rtol=1e-12, equal_nan=True)


def test_capabilities_record_phase_four_sources():
    assert ta_tools.CAPABILITIES["linreg"] == "derived"
    for name in ("rma", "pivot_high", "pivot_low", "change", "crossover", "crossunder",
                 "barssince", "nz", "recurse"):
        assert ta_tools.CAPABILITIES[name] == "custom", name


# ── Phase 5: LinReg Candles and Slope ────────────────────────────────────────
OHLC = [BARS["open"], HIGH, LOW, CLOSE]
LRC_COLUMNS = ["lrc_open", "lrc_high", "lrc_low", "lrc_close", "lrc_signal", "lrc_slope",
               "lrc_bull"]


def pine_linreg_candles(bars, linreg_length, signal_length, sma_signal):
    """The Pine script written out with numpy.polyfit and a hand-rolled SMA / seeded EMA."""
    out = {}
    for column in ("open", "high", "low", "close"):
        values = bars[column].to_numpy()
        fitted = np.full(len(values), np.nan)
        for i in range(linreg_length - 1, len(values)):
            slope, intercept = np.polyfit(np.arange(linreg_length),
                                          values[i - linreg_length + 1:i + 1], 1)
            fitted[i] = intercept + slope * (linreg_length - 1)
        out[column] = fitted
    lclose = out["close"]
    if sma_signal:
        signal = pd.Series(lclose).rolling(signal_length).mean().to_numpy()
    else:
        signal = pine_seeded(lclose, signal_length, 2 / (signal_length + 1))
    return out, signal


def test_linreg_candles_frame_and_warmup():
    out = ta_tools.linreg_candles(*OHLC, linreg_length=11, signal_length=7)
    assert list(out.columns) == LRC_COLUMNS
    assert out.index.equals(BARS.index)
    assert out["lrc_bull"].dtype == bool
    for column in LRC_COLUMNS[:6]:
        assert out[column].dtype == "float64", column
    for column, warmup in [("lrc_open", 10), ("lrc_close", 10), ("lrc_signal", 16),
                           ("lrc_slope", 17)]:
        assert leading_nans(out[column]) == warmup, column
        assert out[column].iloc[warmup:].notna().all(), column
    assert not out["lrc_bull"].iloc[:10].any()


@pytest.mark.parametrize("sma_signal", [True, False], ids=["sma", "ema"])
def test_linreg_candles_match_the_pine_script(sma_signal):
    out = ta_tools.linreg_candles(*OHLC, linreg_length=11, signal_length=9, sma_signal=sma_signal)
    candles, signal = pine_linreg_candles(BARS, 11, 9, sma_signal)
    for column, expected in candles.items():
        np.testing.assert_allclose(out[f"lrc_{column}"], expected, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(out["lrc_signal"], signal, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(out["lrc_slope"], np.diff(signal, prepend=np.nan), rtol=1e-8,
                               atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(out["lrc_bull"], candles["open"] < candles["close"])


def test_linreg_candles_of_straight_lines():
    t = np.arange(80, dtype=float)
    line = {c: pd.Series(base + 0.25 * t, index=BARS.index[:80])
            for c, base in zip(("open", "high", "low", "close"), (10.0, 11.0, 9.0, 10.5))}
    out = ta_tools.linreg_candles(line["open"], line["high"], line["low"], line["close"], 11, 5)
    # LinReg of a line is the line; its SMA lags by (5-1)/2 bars, so the slope is the line's.
    np.testing.assert_allclose(out["lrc_close"].iloc[10:], line["close"].iloc[10:], rtol=1e-10)
    np.testing.assert_allclose(out["lrc_signal"].iloc[14:], line["close"].iloc[12:-2], rtol=1e-10)
    np.testing.assert_allclose(out["lrc_slope"].iloc[15:], 0.25, rtol=1e-8)
    assert out["lrc_bull"].iloc[10:].all()


def test_linreg_candles_without_linear_regression_pass_the_raw_candles():
    out = ta_tools.linreg_candles(*OHLC, signal_length=5, lin_reg=False)
    np.testing.assert_array_equal(out[["lrc_open", "lrc_high", "lrc_low", "lrc_close"]],
                                  BARS[["open", "high", "low", "close"]])
    np.testing.assert_allclose(out["lrc_signal"], ta_tools.sma(CLOSE, 5), equal_nan=True)
    np.testing.assert_array_equal(out["lrc_bull"], BARS["open"] < BARS["close"])


def test_linreg_candles_at_length_one():
    # Pine accepts 1 for both lengths (minval = 1): raw candles, and a signal equal to the close.
    for sma_signal in (True, False):
        out = ta_tools.linreg_candles(*OHLC, linreg_length=1, signal_length=1,
                                      sma_signal=sma_signal)
        np.testing.assert_array_equal(out["lrc_close"], CLOSE)
        np.testing.assert_array_equal(out["lrc_signal"], CLOSE)


def test_linreg_candles_join_onto_the_bars():
    joined = BARS.join(ta_tools.linreg_candles(*OHLC))
    assert len(joined.columns) == len(BARS.columns) + len(LRC_COLUMNS)


def test_capabilities_record_phase_five_sources():
    assert ta_tools.CAPABILITIES["linreg_candles"] == "derived"
