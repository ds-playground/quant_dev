"""Offline tests for data/local, the store of saved live data (src/tools/price_return/store.py).

A yfinance stub serves data/demo/CL_demo.csv as CL=F, in yfinance's own shape, up to a movable
"today", so saving and updating can be replayed without a network.
"""
import datetime as dt
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.tools import price_return as pr
from src.tools.price_return import store
from src.tools.price_return.data import DEMO_DIR


class YahooStub:
    """Serves demo bars under a Yahoo symbol, only up to `until`, as yf.download would."""

    def __init__(self):
        bars = pd.read_csv(DEMO_DIR / "CL_demo.csv", parse_dates=["Date"], index_col="Date")
        self.bars = {"CL=F": bars}
        self.until = pd.Timestamp("2026-06-30")
        self.fail = None                     # an exception to raise instead of answering
        self.calls = []

    def download(self, symbol, start=None, end=None, auto_adjust=True, **kwargs):
        assert auto_adjust is False
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
    stub = YahooStub()
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=stub.download))
    monkeypatch.setattr(store, "LOCAL_DIR", tmp_path / "local")
    return stub


def expected_bars(stub, symbol="CL=F", start="2016-01-01", until=None):
    """What the saved file should hold: the stub's bars in the saved layout, rounded as saved."""
    bars = stub.bars[symbol]
    bars = bars[(bars.index >= pd.Timestamp(start)) & (bars.index <= (until or stub.until))]
    out = bars[["Open", "High", "Low", "Close", "Volume"]].round(6)
    out["Volume"] = out["Volume"].round().astype("int64")
    return out


NOW = dt.datetime(2026, 6, 30, 22, 0, tzinfo=dt.timezone.utc)


def test_first_save_writes_exactly_the_downloaded_bars(yahoo):
    result = pr.save_local("CL=F", now=NOW)
    saved = pr.read_local("CL=F")
    pd.testing.assert_frame_equal(saved, expected_bars(yahoo), check_freq=False)
    assert result["created"] and result["added"] == result["rows"] == len(saved)
    assert (result["first"], result["last"]) == ("2016-01-04", "2026-06-30")
    assert result["revised"] == []
    assert yahoo.calls == [("CL=F", "2016-01-01", "2026-07-01")]      # end exclusive: tomorrow
    assert (store.LOCAL_DIR / "CL=F.csv").read_text().startswith("Date,Open,High,Low,Close,Volume\n")
    plain = store.LOCAL_DIR / "plain.txt"
    plain.write_text("")                              # the mode any new file gets here
    for name in ("CL=F.csv", "manifest.csv"):
        assert (store.LOCAL_DIR / name).stat().st_mode == plain.stat().st_mode


def test_update_downloads_only_the_recent_days_and_adds_the_new_bars(yahoo):
    pr.save_local("CL=F", now=NOW)
    yahoo.until = pd.Timestamp("2026-09-29")
    later = dt.datetime(2026, 9, 30, 8, 0, tzinfo=dt.timezone.utc)
    result = pr.save_local("CL=F", now=later)
    assert yahoo.calls[-1] == ("CL=F", "2026-06-20", "2026-10-01")     # 10 days before the last bar
    new_days = expected_bars(yahoo).loc["2026-07-01":]
    assert result["added"] == len(new_days) > 50
    assert not result["created"] and result["revised"] == []
    pd.testing.assert_frame_equal(pr.read_local("CL=F"), expected_bars(yahoo), check_freq=False)


def test_a_revised_bar_is_replaced_and_reported(yahoo):
    pr.save_local("CL=F", now=NOW)
    old_close = yahoo.bars["CL=F"].at[pd.Timestamp("2026-06-26"), "Close"]
    yahoo.bars["CL=F"] = yahoo.bars["CL=F"].copy()
    yahoo.bars["CL=F"].at[pd.Timestamp("2026-06-26"), "Close"] = old_close + 0.25
    result = pr.save_local("CL=F", now=NOW)
    assert result["added"] == 0
    assert result["revised"] == [{"date": "2026-06-26", "column": "Close",
                                  "saved": round(old_close, 6), "new": round(old_close + 0.25, 6)}]
    assert pr.read_local("CL=F").at[pd.Timestamp("2026-06-26"), "Close"] == round(old_close + 0.25, 6)


def test_bars_missing_from_a_new_download_are_kept(yahoo):
    pr.save_local("CL=F", now=NOW)
    yahoo.bars["CL=F"] = yahoo.bars["CL=F"].drop(pd.Timestamp("2026-06-25"))
    pr.save_local("CL=F", now=NOW)
    assert pd.Timestamp("2026-06-25") in pr.read_local("CL=F").index


@pytest.mark.parametrize("failure", [ConnectionError("Yahoo unreachable"), "empty"])
def test_a_failed_update_leaves_the_files_byte_for_byte(yahoo, failure):
    pr.save_local("CL=F", now=NOW)
    files = {p.name: p.read_bytes() for p in store.LOCAL_DIR.iterdir()}
    if failure == "empty":
        yahoo.bars["CL=F"] = yahoo.bars["CL=F"].iloc[:0]
        error = ValueError
    else:
        yahoo.fail, error = failure, ConnectionError
    with pytest.raises(error):
        pr.save_local("CL=F", now=NOW)
    assert {p.name: p.read_bytes() for p in store.LOCAL_DIR.iterdir()} == files   # no temp files left


def test_an_unknown_symbol_saves_nothing(yahoo):
    with pytest.raises(ValueError, match="no data for 'NOPE'"):
        pr.save_local("NOPE", now=NOW)
    assert not store.LOCAL_DIR.exists() or not any(store.LOCAL_DIR.iterdir())


def test_the_local_source_equals_the_yahoo_path_on_the_same_bars(yahoo):
    pr.save_local("CL=F", now=NOW)
    dates = {"start_date": "2019-01-01", "end_date": "2021-01-01"}
    local = pr.load_price_data(pr.Params(data_source="local", ticker="CL=F", **dates), verbose=False)
    live = pr.load_price_data(pr.Params(data_source="yahoo", ticker="CL=F", **dates), verbose=False)
    pd.testing.assert_frame_equal(local, live)
    assert "2020-04-20" not in set(local["date"].dt.strftime("%Y-%m-%d"))   # the negative close


def test_an_unsaved_symbol_says_what_is_saved(yahoo):
    pr.save_local("CL=F", now=NOW)
    with pytest.raises(ValueError, match=r"No saved data for 'ES=F'.*Saved: CL=F"):
        pr.load_price_data(pr.Params(data_source="local", ticker="ES=F"), verbose=False)


def test_local_tickers_and_manifest_describe_the_files(yahoo):
    pr.save_local("CL=F", now=NOW)
    [entry] = pr.local_tickers()
    assert entry == {"symbol": "CL=F", "rows": len(expected_bars(yahoo)), "first": "2016-01-04",
                     "last": "2026-06-30", "updated": "2026-06-30T22:00:00+00:00"}
    manifest = pd.read_csv(store.LOCAL_DIR / "manifest.csv", dtype=str)
    assert manifest.to_dict("records") == [{"symbol": "CL=F", "file": "CL=F.csv", "rows": str(entry["rows"]),
                                            "first": "2016-01-04", "last": "2026-06-30",
                                            "updated": "2026-06-30T22:00:00+00:00"}]


def test_local_config_takes_thresholds_from_the_ticker_config(yahoo):
    pr.save_local("CL=F", now=NOW)
    yahoo.bars["ZZZ"] = yahoo.bars["CL=F"]
    pr.save_local("ZZZ", now=NOW)
    config = pr.local_ticker_config()
    assert list(config) == ["CL=F", "ZZZ"]
    listed = pr.load_ticker_config(verbose=False)["CL=F"]
    assert config["CL=F"].label == listed.label == "Crude oil futures"
    assert config["CL=F"].data_source == "local"
    assert config["CL=F"].win_threshold == listed.win_threshold
    assert (config["ZZZ"].label, config["ZZZ"].win_threshold) == ("ZZZ", 0.2)   # the defaults block


@pytest.mark.parametrize("symbol, filename", [
    ("ES=F", "ES=F.csv"), ("^GSPC", "^GSPC.csv"), ("EURUSD=X", "EURUSD=X.csv"),
    ("BRK.B", "BRK.B.csv"), ("A/B", "A%2FB.csv"), ("X Y", "X%20Y.csv"), ("..\\x", "..%5Cx.csv"),
])
def test_symbols_round_trip_through_safe_file_names(symbol, filename):
    assert store.symbol_to_filename(symbol) == filename
    assert store.filename_to_symbol(filename) == symbol
    assert "/" not in filename and "\\" not in filename


@pytest.mark.parametrize("symbol", ["", ".", ".."])
def test_names_that_are_not_symbols_are_refused(symbol):
    with pytest.raises(ValueError, match="Not a symbol"):
        store.symbol_to_filename(symbol)


def test_saved_data_is_ignored_by_git():
    result = subprocess.run(["git", "check-ignore", "-q", "data/local/ES=F.csv"],
                            cwd=store._repo_root(), check=False)
    assert result.returncode == 0


# ── The API over data/local ─────────────────────────────────────────────────
@pytest.fixture
def api(yahoo):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import cache
    from src.api.app import app
    cache.cache_clear()
    yield TestClient(app), yahoo
    cache.cache_clear()


def test_update_endpoint_saves_then_updates(api):
    client, yahoo = api
    assert client.get("/api/local").json()["tickers"] == []
    first = client.post("/api/local/CL=F/update").json()
    assert first["created"] and first["rows"] == len(expected_bars(yahoo)) == first["added"]
    assert first["last"] == "2026-06-30"
    yahoo.until = pd.Timestamp("2026-07-31")
    second = client.post("/api/local/CL=F/update").json()
    assert not second["created"] and second["added"] == len(expected_bars(yahoo).loc["2026-07-01":])
    assert client.get("/api/local").json()["tickers"] == pr.local_tickers()


def test_symbols_with_url_characters_save_under_their_own_name(api):
    client, yahoo = api
    yahoo.bars["^GSPC"] = pd.read_csv(DEMO_DIR / "SPX_demo.csv", parse_dates=["Date"], index_col="Date")
    assert client.post("/api/local/%5EGSPC/update").status_code == 200
    assert (store.LOCAL_DIR / "^GSPC.csv").is_file()
    assert [t["symbol"] for t in client.get("/api/local").json()["tickers"]] == ["^GSPC"]


def test_update_failures_are_reported_and_leave_the_file(api):
    client, yahoo = api
    client.post("/api/local/CL=F/update")
    before = (store.LOCAL_DIR / "CL=F.csv").read_bytes()
    yahoo.fail = ConnectionError("Yahoo unreachable")
    response = client.post("/api/local/CL=F/update")
    assert response.status_code == 502 and "Yahoo unreachable" in response.json()["detail"]
    yahoo.fail = None
    response = client.post("/api/local/NOPE/update")
    assert response.status_code == 422 and "no data for 'NOPE'" in response.json()["detail"]
    assert (store.LOCAL_DIR / "CL=F.csv").read_bytes() == before
    assert client.post("/api/local/CL=F/update?start_date=2016-1-1").status_code == 422


def test_ticker_sets_show_what_is_saved(api):
    client, _ = api
    client.post("/api/local/CL=F/update")
    local = client.get("/api/tickers?set=local").json()
    [cl] = local["tickers"]
    assert (cl["symbol"], cl["label"], cl["params"]["data_source"]) == ("CL=F", "Crude oil futures", "local")
    assert cl["saved"]["last"] == "2026-06-30"
    yahoo_set = {t["symbol"]: t for t in client.get("/api/tickers?set=yahoo").json()["tickers"]}
    assert yahoo_set["CL=F"]["saved"] == cl["saved"] and yahoo_set["ES=F"]["saved"] is None
    assert yahoo_set["CL=F"]["params"]["data_source"] == "yahoo"


def test_local_answers_equal_the_yahoo_path_and_follow_updates(api):
    client, yahoo = api
    client.post("/api/local/CL=F/update")
    body = {"ticker": "CL=F", "start_date": "2019-01-01"}
    local = client.post("/api/overview", json={**body, "data_source": "local"}).json()
    live = client.post("/api/overview", json={**body, "data_source": "yahoo"}).json()
    assert local == {**live, "data_source": "local"}
    yahoo.until = pd.Timestamp("2026-09-29")
    client.post("/api/local/CL=F/update")
    later = client.post("/api/overview", json={**body, "data_source": "local"}).json()
    assert later["end"][:10] == "2026-09-29" and later["rows"] > local["rows"]   # not a stale cache


def test_multi_ticker_runs_over_saved_data(api):
    client, _ = api
    assert client.get("/api/multi-ticker?set=local").status_code == 422      # nothing saved yet
    client.post("/api/local/CL=F/update")
    body = client.get("/api/multi-ticker?set=local").json()
    assert [t["symbol"] for t in body["tickers"]] == ["CL=F"] and body["failed"] == []


def test_refresh_script_updates_and_reports_failures(yahoo, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "update_local_data", store._repo_root() / "scripts" / "update_local_data.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    assert script.main(["CL=F"]) == 0
    assert "CL=F: saved" in capsys.readouterr().out
    yahoo.bars["CL=F"] = yahoo.bars["CL=F"].copy()
    yahoo.bars["CL=F"].at[pd.Timestamp("2026-06-26"), "Close"] += 1          # inside the overlap
    assert script.main(["CL=F", "NOPE"]) == 1
    out = capsys.readouterr().out
    assert "CL=F: +0 bars" in out and "1 revised value(s)" in out and "Close:" in out
    assert "NOPE: FAILED" in out and "1 of 2 failed: NOPE" in out


def test_a_write_that_dies_halfway_leaves_the_saved_file(yahoo, monkeypatch):
    pr.save_local("CL=F", now=NOW)
    files = {p.name: p.read_bytes() for p in store.LOCAL_DIR.iterdir()}
    real = pd.DataFrame.to_csv

    def dies_halfway(self, fh=None, *args, **kwargs):
        fh.write(real(self, None, *args, **kwargs)[:1000])     # part of the file, then a crash
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", dies_halfway)
    yahoo.until = pd.Timestamp("2026-07-31")
    with pytest.raises(OSError, match="disk full"):
        pr.save_local("CL=F", now=NOW)
    assert {p.name: p.read_bytes() for p in store.LOCAL_DIR.iterdir()} == files
