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
    ("atr", lambda: ta_tools.atr(HIGH, LOW, CLOSE, 14), "atr_14", 14),
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
                "atr": "talib", "bb": "talib", "rsi": "talib",
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


def test_atr_starts_one_bar_after_pine_and_converges():
    prev = CLOSE.shift(1).to_numpy()
    h, l = HIGH.to_numpy(), LOW.to_numpy()
    tr = np.where(np.isnan(prev), h - l,
                  np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)]))
    pine = pine_seeded(tr, 14, 1 / 14)
    ours = ta_tools.atr(HIGH, LOW, CLOSE, 14).to_numpy()

    assert leading_nans(pd.Series(ours)) == leading_nans(pd.Series(pine)) + 1
    relative = np.abs(ours - pine) / pine
    assert relative[14] > 1e-3, "documented early divergence from Pine"
    assert relative[-1] < 1e-9, "converges to Pine"


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
