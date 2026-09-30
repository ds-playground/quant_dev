"""Shared fixtures: one yfinance stub for every Yahoo-path test, the API client on it, and a
clean API cache around every test."""
import sys
import types

import pandas as pd
import pytest

from src.api import cache
from src.tools.price_return import store
from src.tools.price_return.data import DEMO_DIR


@pytest.fixture(autouse=True)
def fresh_api_cache():
    """Every test starts and ends with empty API caches, even when it fails part-way."""
    cache.cache_clear()
    yield
    cache.cache_clear()


class YahooStub:
    """Stands in for yfinance, offline: serves data/demo/CL_demo.csv as CL=F, in yfinance's own
    shape ((Price, Ticker) columns, and an Adj Close that differs from Close, so reading the
    wrong column shows), and only up to `until`, a movable "today". Other symbols get an empty
    frame, as Yahoo returns for an unknown one. More symbols can be added to `bars`."""

    def __init__(self):
        bars = pd.read_csv(DEMO_DIR / "CL_demo.csv", parse_dates=["Date"], index_col="Date")
        self.bars = {"CL=F": bars}
        self.until = pd.Timestamp("2026-06-30")
        self.fail = None                     # an exception to raise instead of answering
        self.calls = []

    def download(self, symbol, start=None, end=None, auto_adjust=True, **kwargs):
        assert auto_adjust is False          # prices as traded, never adjusted
        self.calls.append((symbol, start, end))
        if self.fail:
            raise self.fail
        if symbol not in self.bars:
            return pd.DataFrame()
        bars = self.bars[symbol]
        bars = bars[(bars.index >= pd.Timestamp(start)) & (bars.index < pd.Timestamp(end))
                    & (bars.index <= self.until)].copy()
        bars.insert(0, "Adj Close", bars["Close"] * 0.98)
        bars.columns = pd.MultiIndex.from_product([bars.columns, [symbol]], names=["Price", "Ticker"])
        return bars


@pytest.fixture
def yahoo(monkeypatch, tmp_path):
    """The yfinance stub in place of yfinance, and data/local redirected to a temporary folder."""
    stub = YahooStub()
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=stub.download))
    monkeypatch.setattr(store, "LOCAL_DIR", tmp_path / "local")
    return stub


@pytest.fixture
def api(yahoo):
    """A test client for the API, with the yfinance stub and a temporary data/local."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api.app import app
    return TestClient(app), yahoo
