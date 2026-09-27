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
