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


def fake_yfinance(monkeypatch, frame):
    """Stand in for yfinance so load_bars is tested offline."""
    calls = []

    def download(ticker, **kwargs):
        calls.append((ticker, kwargs))
        return frame

    monkeypatch.setitem(sys.modules, "yfinance", type(sys)("yfinance"))
    monkeypatch.setattr(sys.modules["yfinance"], "download", download, raising=False)
    return calls


def test_load_bars_matches_the_make_bars_shape(monkeypatch):
    dates = pd.to_datetime(["2023-01-03", "2023-01-04"])
    columns = pd.MultiIndex.from_product([["Adj Close", "Close", "High", "Low", "Open", "Volume"],
                                          ["AAPL"]], names=["Price", "Ticker"])
    raw = pd.DataFrame([[1.0, 2.0, 3.0, 1.5, 2.5, 100], [1.1, 2.1, 3.1, 1.6, 2.6, 200]],
                       index=pd.DatetimeIndex(dates, name="Date"), columns=columns)
    calls = fake_yfinance(monkeypatch, raw)
    bars = ta_tools.load_bars("AAPL", "2023-01-01")
    assert list(bars.columns) == list(ta_tools.make_bars(n=5).columns)
    assert bars.index.name == "date" and (bars.dtypes == "float64").all()
    assert bars["close"].tolist() == [2.0, 2.1]            # the traded close, not Adj Close
    assert calls[0][1]["auto_adjust"] is False


def test_load_bars_rejects_an_unknown_ticker(monkeypatch):
    fake_yfinance(monkeypatch, pd.DataFrame())
    with pytest.raises(ValueError, match="No price data"):
        ta_tools.load_bars("NOPE", "2023-01-01")


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
    # Realtime trendlines only: backpaint looks ahead by design (tested separately).
    "trendlines_atr": lambda b: ta_tools.trendlines(b["high"], b["low"], b["close"],
                                                    10).astype(float),
    "trendlines_stdev": lambda b: ta_tools.trendlines(b["high"], b["low"], b["close"], 12,
                                                      calc_method="stdev").astype(float),
    "trendlines_linreg": lambda b: ta_tools.trendlines(b["high"], b["low"], b["close"], 7,
                                                       calc_method="linreg").astype(float),
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


# ── Phase 6: Trendlines with Breaks ──────────────────────────────────────────
TL_COLUMNS = ["tl_upper", "tl_lower", "tl_upper_slope", "tl_lower_slope", "tl_pivot_high",
              "tl_pivot_low", "tl_upos", "tl_dnos", "tl_upper_break", "tl_lower_break"]
TL_LINES = TL_COLUMNS[:6]


def pine_trendline_slope(bars, length, mult, method):
    """Pine's three slope formulas, written out literally (n is bar_index)."""
    close = bars["close"]
    if method == "atr":
        return pine_atr(bars["high"], bars["low"], close, length) / length * mult
    if method == "stdev":
        return close.rolling(length).std(ddof=0).to_numpy() / length * mult
    n = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    mean = lambda x: x.rolling(length).mean()
    variance = n.rolling(length).var(ddof=0)
    return (abs(mean(close * n) - mean(close) * mean(n)) / variance / 2 * mult).to_numpy()


def pine_trendlines(bars, length, mult, method):
    """Realtime Trendlines with Breaks in closed form, with no bar-by-bar recursion.

    Between pivots each line is anchored at its latest pivot k, so Pine's `upper` is
    ph[k] - slope[k] * (t - k); a latch is set by the first break after its pivot.
    """
    close = bars["close"].to_numpy()
    slope = np.asarray(pine_trendline_slope(bars, length, mult, method))
    t = np.arange(len(close))
    out = {}
    for side, pivots, sign, broke in (
            ("upper", pivot_oracle(bars["high"], length, length, high=True), -1, np.greater),
            ("lower", pivot_oracle(bars["low"], length, length, high=False), 1, np.less)):
        is_pivot = ~np.isnan(pivots)
        k = pd.Series(np.where(is_pivot, t, np.nan)).ffill().to_numpy()
        seen = ~np.isnan(k)
        anchor, held = np.full(len(t), np.nan), np.full(len(t), np.nan)
        anchor[seen] = pivots[k[seen].astype(int)]
        held[seen] = slope[k[seen].astype(int)]
        realtime = anchor + sign * held * (t - k) + sign * held * length
        with np.errstate(invalid="ignore"):
            # Before the first pivot Pine's line is its var initial value, 0.
            crossed = broke(close, np.where(seen, realtime, 0.0)) & ~is_pivot
        latch = pd.Series(crossed).groupby(np.cumsum(is_pivot)).cummax().astype("int64").to_numpy()
        out[side] = realtime
        out[f"{side}_slope"] = held
        out[f"{side}_latch"] = latch
        out[f"{side}_break"] = np.diff(latch, prepend=latch[0]) > 0
    return out


def trendlines(bars, **kwargs):
    return ta_tools.trendlines(bars["high"], bars["low"], bars["close"], **kwargs)


def test_trendlines_frame():
    out = trendlines(BARS)
    assert list(out.columns) == TL_COLUMNS
    assert out.index.equals(BARS.index)
    assert (out[TL_LINES].dtypes == "float64").all()
    assert (out[["tl_upos", "tl_dnos"]].dtypes == "int64").all()
    assert (out[["tl_upper_break", "tl_lower_break"]].dtypes == bool).all()


@pytest.mark.parametrize("method,mult", [("atr", 1.0), ("stdev", 1.0), ("linreg", 1.0),
                                         ("atr", 2.5)])
def test_trendlines_match_the_pine_script(method, mult):
    out = trendlines(BARS, length=10, mult=mult, calc_method=method)
    ref = pine_trendlines(BARS, 10, mult, method)
    for side in ("upper", "lower"):
        np.testing.assert_allclose(out[f"tl_{side}"], ref[side], rtol=1e-9, equal_nan=True)
        np.testing.assert_allclose(out[f"tl_{side}_slope"], ref[f"{side}_slope"], rtol=1e-7,
                                   equal_nan=True)
        np.testing.assert_array_equal(out[f"tl_{side}_break"], ref[f"{side}_break"])
    np.testing.assert_array_equal(out["tl_upos"], ref["upper_latch"])
    np.testing.assert_array_equal(out["tl_dnos"], ref["lower_latch"])
    assert out["tl_upper_break"].sum() > 0 and out["tl_lower_break"].sum() > 0


def test_trendlines_rails_move_by_the_slope_between_pivots():
    out = trendlines(BARS, length=10)
    between = out["tl_upper"].notna() & out["tl_pivot_high"].isna()
    np.testing.assert_allclose(out["tl_upper"].diff()[between],
                               -out["tl_upper_slope"][between], rtol=1e-9)
    between = out["tl_lower"].notna() & out["tl_pivot_low"].isna()
    np.testing.assert_allclose(out["tl_lower"].diff()[between],
                               out["tl_lower_slope"][between], rtol=1e-9)
    assert (out["tl_upper_slope"].dropna() >= 0).all()


def test_a_break_fires_at_most_once_per_pivot():
    out = trendlines(BARS, length=5)
    for pivot, event, latch in (("tl_pivot_high", "tl_upper_break", "tl_upos"),
                                ("tl_pivot_low", "tl_lower_break", "tl_dnos")):
        segment = out[pivot].notna().cumsum()
        assert out[event].groupby(segment).sum().max() == 1
        assert set(out[latch].unique()) <= {0, 1}
    breaks = out["tl_upper_break"]
    assert (BARS["close"][breaks] > out["tl_upper"][breaks]).all()
    breaks = out["tl_lower_break"]
    assert (BARS["close"][breaks] < out["tl_lower"][breaks]).all()


def test_trendlines_before_the_first_pivot():
    out = trendlines(BARS, length=10)
    first = out["tl_pivot_high"].first_valid_index()
    before = out.loc[:first].iloc[:-1]
    assert before["tl_upper"].isna().all() and before["tl_upper_slope"].isna().all()
    # Pine's `var upper = 0.` puts the line at zero, so the latch sets on bar 0 without a break.
    assert (before["tl_upos"] == 1).all() and not before["tl_upper_break"].any()


def test_backpaint_draws_the_same_lines_from_the_pivot_bar():
    length = 10
    realtime = trendlines(BARS, length=length)
    backpaint = trendlines(BARS, length=length, backpaint=True)
    # Backpaint plots Pine's `upper` itself, `length` bars earlier; realtime plots it projected
    # `length` bars on, so undoing the projection and the shift must recover the backpaint line.
    upper = realtime["tl_upper"] + realtime["tl_upper_slope"] * length
    lower = realtime["tl_lower"] - realtime["tl_lower_slope"] * length
    np.testing.assert_allclose(backpaint["tl_upper"], upper.shift(-length), rtol=1e-12,
                               equal_nan=True)
    np.testing.assert_allclose(backpaint["tl_lower"], lower.shift(-length), rtol=1e-12,
                               equal_nan=True)
    for column in ("tl_upper_slope", "tl_lower_slope", "tl_pivot_high", "tl_pivot_low"):
        np.testing.assert_array_equal(backpaint[column], realtime[column].shift(-length))
    # Each backpainted line starts on the pivot bar, at the pivot's own high or low.
    at = backpaint["tl_pivot_high"].notna()
    np.testing.assert_array_equal(backpaint["tl_upper"][at], HIGH[at])
    at = backpaint["tl_pivot_low"].notna()
    np.testing.assert_array_equal(backpaint["tl_lower"][at], LOW[at])
    assert backpaint[TL_LINES].iloc[-length:].isna().all().all()
    # Breaks and latches are plotted without an offset in Pine, so they never move.
    pd.testing.assert_frame_equal(backpaint[TL_COLUMNS[6:]], realtime[TL_COLUMNS[6:]])


def test_backpaint_looks_ahead_by_design():
    full = trendlines(BARS, length=10, backpaint=True)["tl_upper"]
    changed = []
    for cut in (60, 150, 299):
        truncated = trendlines(BARS.iloc[:cut + 1], length=10, backpaint=True)["tl_upper"]
        changed.append(not np.allclose(truncated, full.iloc[:cut + 1], equal_nan=True))
    assert any(changed)


def test_zero_mult_gives_flat_lines_at_the_pivot_price():
    out = trendlines(BARS, length=10, mult=0.0)
    np.testing.assert_array_equal(out["tl_upper"], out["tl_pivot_high"].ffill())
    np.testing.assert_array_equal(out["tl_lower"], out["tl_pivot_low"].ffill())


def test_trendlines_reject_unknown_methods():
    with pytest.raises(ValueError, match="calc_method must be one of"):
        trendlines(BARS, calc_method="Atr")


def test_capabilities_record_phase_six_sources():
    assert ta_tools.CAPABILITIES["trendlines"] == "derived"


# ── Phase 7: equivariance ────────────────────────────────────────────────────
# Prices mapped by x -> K*x + C (K > 0, so order and validity are kept). A price-level output
# maps the same way; a spread (a distance between prices) only scales by K; an oscillator or a
# signal does not change at all.
K, C = 3.0, 7.0
MOVED = BARS.copy()
MOVED[["open", "high", "low", "close"]] = BARS[["open", "high", "low", "close"]] * K + C


def expect(base, kind):
    return {"level": base * K + C, "spread": base * K, "invariant": base}[kind]


def assert_equivariant(moved, base, kind, label):
    # atol covers outputs that pass through zero (slopes); rtol the rest.
    np.testing.assert_allclose(moved, expect(base, kind), rtol=1e-9, atol=1e-9,
                               equal_nan=True, err_msg=label)


EQUIVARIANCE = [
    ("sma", lambda b: ta_tools.sma(b["close"], 20), "level"),
    ("ema", lambda b: ta_tools.ema(b["close"], 20), "level"),
    ("wma", lambda b: ta_tools.wma(b["close"], 20), "level"),
    ("hma", lambda b: ta_tools.hma(b["close"], 16), "level"),
    ("alma", lambda b: ta_tools.alma(b["close"], 9), "level"),
    ("rma", lambda b: ta_tools.rma(b["close"], 14), "level"),
    ("bb_mid", lambda b: ta_tools.bb(b["close"], 20)["bb_mid_20"], "level"),
    ("bb_upper", lambda b: ta_tools.bb(b["close"], 20)["bb_upper_20"], "level"),
    ("bb_lower", lambda b: ta_tools.bb(b["close"], 20)["bb_lower_20"], "level"),
    ("stdev", lambda b: ta_tools.stdev(b["close"], 20), "spread"),
    ("true_range", lambda b: ta_tools.true_range(b["high"], b["low"], b["close"]), "spread"),
    ("atr", lambda b: ta_tools.atr(b["high"], b["low"], b["close"], 14), "spread"),
    ("rsi", lambda b: ta_tools.rsi(b["close"], 14), "invariant"),
]


@pytest.mark.parametrize("label,call,kind", EQUIVARIANCE, ids=[e[0] for e in EQUIVARIANCE])
def test_primitives_are_equivariant(label, call, kind):
    if label in ("hma", "alma"):
        pytest.importorskip("pandas_ta")
    assert_equivariant(call(MOVED), call(BARS), kind, label)


LRC_KINDS = {"lrc_open": "level", "lrc_high": "level", "lrc_low": "level", "lrc_close": "level",
             "lrc_signal": "level", "lrc_slope": "spread"}


@pytest.mark.parametrize("sma_signal", [True, False])
def test_linreg_candles_are_equivariant(sma_signal):
    base = ta_tools.linreg_candles(*OHLC, 11, 7, sma_signal=sma_signal)
    moved = ta_tools.linreg_candles(MOVED["open"], MOVED["high"], MOVED["low"], MOVED["close"],
                                    11, 7, sma_signal=sma_signal)
    for column, kind in LRC_KINDS.items():
        assert_equivariant(moved[column], base[column], kind, column)
    pd.testing.assert_series_equal(moved["lrc_bull"], base["lrc_bull"])


TL_KINDS = {"tl_upper": "level", "tl_lower": "level", "tl_pivot_high": "level",
            "tl_pivot_low": "level", "tl_upper_slope": "spread", "tl_lower_slope": "spread"}


@pytest.mark.parametrize("method", ["atr", "stdev", "linreg"])
@pytest.mark.parametrize("backpaint", [False, True])
def test_trendlines_are_equivariant(method, backpaint):
    base = trendlines(BARS, length=10, calc_method=method, backpaint=backpaint)
    moved = trendlines(MOVED, length=10, calc_method=method, backpaint=backpaint)
    for column, kind in TL_KINDS.items():
        assert_equivariant(moved[column], base[column], kind, f"{method}: {column}")
    # Pivots, breaks and latches are decided by comparing prices, which the map preserves.
    pd.testing.assert_frame_equal(moved[TL_COLUMNS[6:]], base[TL_COLUMNS[6:]])


# ── Phase 8: data sources ────────────────────────────────────────────────────
from src.tools.ta_tools.data import YAHOO_LOOKBACK_DAYS, _normalise


def raw_bars(n=4, **extra):
    index = pd.bdate_range("2024-01-02", periods=n)
    columns = {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, **extra}
    return pd.DataFrame({k: np.full(n, v) if np.isscalar(v) else v for k, v in columns.items()},
                        index=index)


def test_normalise_orders_columns_and_casts():
    frame = raw_bars(plot=[1, 2, 3, 4], volume=[5, 6, 7, 8])
    frame["open"] = frame["open"].astype(int)
    bars = _normalise(frame)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume", "plot"]
    assert (bars[["open", "high", "low", "close", "volume"]].dtypes == "float64").all()
    assert bars.index.name == "date"
    assert bars.index.dtype == ta_tools.make_bars(n=2).index.dtype


def test_normalise_keeps_volume_optional():
    assert list(_normalise(raw_bars()).columns) == ["open", "high", "low", "close"]


def test_normalise_accepts_leading_nan_and_nan_outside_prices():
    frame = raw_bars(close=[np.nan, 10.0, 10.5, 11.0], plot=[1.0, np.nan, 2.0, np.nan])
    bars = _normalise(frame)
    assert bars["close"].isna().sum() == 1 and bars["plot"].isna().sum() == 2


@pytest.mark.parametrize("close,match", [
    ([10.0, np.nan, 10.5, 11.0], "NaN at 1 bars after its first value"),
    ([10.0, 10.5, 11.0, np.nan], "NaN at 1 bars after its first value"),
    ([np.nan] * 4, "no values"),
])
def test_normalise_rejects_price_gaps(close, match):
    with pytest.raises(ValueError, match=match):
        _normalise(raw_bars(close=close))


def test_normalise_rejects_unsorted_and_repeated_timestamps():
    frame = raw_bars()
    with pytest.raises(ValueError, match="oldest first"):
        _normalise(frame.iloc[::-1])
    with pytest.raises(ValueError, match="1 repeated timestamps"):
        _normalise(frame.iloc[[0, 1, 1, 2]])


def test_normalise_rejects_missing_prices_and_non_time_index():
    with pytest.raises(ValueError, match=r"missing \['low'\]"):
        _normalise(raw_bars().drop(columns="low"))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        _normalise(raw_bars().reset_index(drop=True))


def yahoo_frame(index, rows):
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close", "Volume"])


def test_load_bars_still_drops_an_incomplete_yahoo_row(monkeypatch):
    index = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
    fake_yfinance(monkeypatch, yahoo_frame(index, [[1, 2, 0.5, 1.5, 10],
                                                   [np.nan, np.nan, np.nan, 1.6, np.nan],
                                                   [1.6, 2.1, 1.1, 1.7, 30]]))
    bars = ta_tools.load_bars("AAPL", "2023-01-01")
    assert bars.index.tolist() == [index[0], index[2]]


def test_load_bars_passes_the_interval_and_keeps_intraday_timezone(monkeypatch):
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
    index = pd.date_range(start + pd.Timedelta(hours=9.5), periods=3, freq="5min",
                          tz="America/New_York")
    calls = fake_yfinance(monkeypatch, yahoo_frame(index, [[1, 2, 0.5, 1.5, 10]] * 3))
    bars = ta_tools.load_bars("AAPL", start, interval="5m")
    assert calls[0][1]["interval"] == "5m"
    assert str(bars.index.tz) == "America/New_York" and bars.index.name == "date"
    assert ta_tools.load_bars("AAPL", "2023-01-01") is not None     # daily: no limit
    assert calls[1][1]["interval"] == "1d"


def days_ago(days):
    return (pd.Timestamp.today().normalize() - pd.Timedelta(days=days)).date().isoformat()


@pytest.mark.parametrize("interval,start,end,match", [
    ("5m", days_ago(90), None, "last 60 days only"),
    ("1h", days_ago(800), None, "last 730 days only"),
    ("1m", days_ago(45), None, "last 30 days only"),
    ("1m", days_ago(20), days_ago(5), "7 days per request"),
    ("3m", days_ago(2), None, "interval must be one of"),
])
def test_load_bars_refuses_what_yahoo_does_not_serve(monkeypatch, interval, start, end, match):
    calls = fake_yfinance(monkeypatch, pd.DataFrame())
    with pytest.raises(ValueError, match=match):
        ta_tools.load_bars("AAPL", start, end, interval=interval)
    assert calls == [], "refused before calling Yahoo"


def test_load_bars_accepts_requests_inside_yahoo_limits(monkeypatch):
    index = pd.date_range("2024-01-02 09:30", periods=2, freq="1min", tz="America/New_York")
    fake_yfinance(monkeypatch, yahoo_frame(index, [[1, 2, 0.5, 1.5, 10]] * 2))
    for interval, days in YAHOO_LOOKBACK_DAYS.items():
        ta_tools.load_bars("AAPL", days_ago(min(days - 1, 5)), interval=interval)


# A TradingView "Export chart data" file: daily NASDAQ bars stamped at the 09:30 New York open,
# one plotted series with gaps (as a backpainted line has), then Volume. Values are made up.
TV_EXPORT = """time,open,high,low,close,Upper,Volume
1704205800,187.15,188.44,183.89,185.64,,82488700
1704292200,184.22,185.88,183.43,184.25,190.5,58414500
1704378600,182.15,183.09,180.88,181.91,,71983600
1704465000,181.99,182.76,180.17,181.18,188.1,62303300
"""
TV_EXPORT_ISO = TV_EXPORT.replace("1704205800", "2024-01-02T09:30:00-05:00") \
    .replace("1704292200", "2024-01-03T09:30:00-05:00") \
    .replace("1704378600", "2024-01-04T09:30:00-05:00") \
    .replace("1704465000", "2024-01-05T09:30:00-05:00")


def write(tmp_path, text, name="bars.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_read_bars_reads_a_tradingview_export(tmp_path):
    bars = ta_tools.read_bars(write(tmp_path, TV_EXPORT))
    assert list(bars.columns) == ["open", "high", "low", "close", "volume", "Upper"]
    assert (bars.dtypes == "float64").all()
    assert bars.index.name == "date" and str(bars.index.tz) == "UTC"
    assert bars.index[0] == pd.Timestamp("2024-01-02 14:30", tz="UTC")
    assert bars["close"].tolist() == [185.64, 184.25, 181.91, 181.18]
    assert bars["Upper"].isna().tolist() == [True, False, True, False]


def test_read_bars_reads_unix_and_iso_times_alike(tmp_path):
    unix = ta_tools.read_bars(write(tmp_path, TV_EXPORT, "unix.csv"))
    iso = ta_tools.read_bars(write(tmp_path, TV_EXPORT_ISO, "iso.csv"))
    pd.testing.assert_frame_equal(unix, iso)


def test_read_bars_daily_gives_load_bars_dates(tmp_path):
    bars = ta_tools.read_bars(write(tmp_path, TV_EXPORT), daily=True, tz="America/New_York")
    assert bars.index.tz is None
    assert bars.index.equals(pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=4),
                                              name="date"))
    local = ta_tools.read_bars(write(tmp_path, TV_EXPORT), tz="America/New_York")
    assert local.index[0] == pd.Timestamp("2024-01-02 09:30", tz="America/New_York")


def test_read_bars_gives_back_exactly_the_floats_written(tmp_path):
    bars = ta_tools.make_bars(n=300, seed=1)
    path = tmp_path / "bars.csv"
    bars.to_csv(path, index_label="time")
    pd.testing.assert_frame_equal(ta_tools.read_bars(path, daily=True), bars, check_exact=True,
                                  check_freq=False)


def test_read_bars_daily_takes_the_date_in_the_exchange_timezone(tmp_path):
    # A Tokyo daily bar stamped at midnight local time is 15:00 UTC the day before.
    text = "time,open,high,low,close\n2024-01-03T15:00:00Z,1,2,0.5,1.5\n"
    utc = ta_tools.read_bars(write(tmp_path, text), daily=True)
    tokyo = ta_tools.read_bars(write(tmp_path, text), daily=True, tz="Asia/Tokyo")
    assert utc.index[0] == pd.Timestamp("2024-01-03")
    assert tokyo.index[0] == pd.Timestamp("2024-01-04")


def test_read_bars_maps_another_layout(tmp_path):
    text = ("Date,Open,High,Low,Close,Adj Close\n"
            "2024-01-02,1,2,0.5,1.5,1.4\n"
            "2024-01-03,1.5,2,1,1.8,1.7\n")
    bars = ta_tools.read_bars(write(tmp_path, text), columns={"Date": "time"}, daily=True)
    assert list(bars.columns) == ["open", "high", "low", "close", "Adj Close"]
    assert bars.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_read_bars_needs_a_time_column(tmp_path):
    with pytest.raises(ValueError, match="no time column"):
        ta_tools.read_bars(write(tmp_path, "Date,open,high,low,close\n2024-01-02,1,2,0.5,1.5\n"))


def test_read_bars_rejects_intraday_bars_squeezed_to_dates(tmp_path):
    text = "time,open,high,low,close\n1704205800,1,2,0.5,1.5\n1704206100,1,2,0.5,1.5\n"
    with pytest.raises(ValueError, match="repeated timestamps"):
        ta_tools.read_bars(write(tmp_path, text), daily=True)
