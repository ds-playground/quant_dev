"""Offline tests for src.api: each endpoint must return exactly what the package computes."""
import dataclasses
import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api import cache
from src.api.app import app
from src.api.schemas import ParamsIn, to_params
from src.api.serialize import clean, figure_to_json, frame_to_records
from src.tools import price_return as pr

client = TestClient(app)
SIMULATED = {"data_source": "simulated", "start_date": "2020-01-01", "random_seed": 7}


def via_json(value):
    """What a client receives: the value through a real JSON encode and decode."""
    return json.loads(json.dumps(value, allow_nan=False))


# ── Parameters ──────────────────────────────────────────────────────────────
def test_params_model_mirrors_every_params_field_and_default():
    names = [f.name for f in dataclasses.fields(pr.Params)]
    assert list(ParamsIn.model_fields) == names
    assert to_params(ParamsIn()) == pr.Params()
    body = ParamsIn(**SIMULATED, windows=[2, 4], win_threshold=0.3)
    assert to_params(body) == pr.Params(**SIMULATED, windows=[2, 4], win_threshold=0.3)


@pytest.mark.parametrize("payload", [{"win_thresold": 0.2}, {"trade_days": "many"},
                                     {"windows": "2,3"}])
def test_bad_parameters_are_rejected(payload):
    response = client.post("/api/overview", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]


# ── Serialization ───────────────────────────────────────────────────────────
def test_clean_makes_every_value_json_ready():
    assert clean(np.float64(0.1)) == 0.1 and type(clean(np.float64(0.1))) is float
    assert clean(np.int64(3)) == 3 and type(clean(np.int64(3))) is int
    assert clean(np.bool_(True)) is True
    assert clean(np.nan) is None and clean(np.inf) is None and clean(pd.NaT) is None
    assert clean(pd.Timestamp("2024-01-02")) == "2024-01-02T00:00:00"
    assert clean(dt.date(2024, 1, 2)) == "2024-01-02"
    assert clean({"a": [np.float64(1.5), None]}) == {"a": [1.5, None]}


def test_floats_survive_json_exactly():
    values = np.random.default_rng(0).normal(0, 0.01, 1000)
    frame = pd.DataFrame({"x": values})
    assert [row["x"] for row in via_json(frame_to_records(frame))] == values.tolist()


def test_figure_to_json_has_data_and_layout():
    params = pr.Params(**SIMULATED)
    fig = pr.plot_price_and_returns(pr.load_price_data(params, verbose=False), params)
    as_json = via_json(figure_to_json(fig))
    assert set(as_json) >= {"data", "layout"}
    assert len(as_json["data"]) == len(fig.data)


# ── Endpoints ───────────────────────────────────────────────────────────────
def test_health():
    assert client.get("/api/health").json() == {"status": "ok", "version": app.version}


@pytest.mark.parametrize("query, path", [("", pr.params.DEMO_CONFIG_PATH),
                                         ("?set=demo", pr.params.DEMO_CONFIG_PATH),
                                         ("?set=yahoo", pr.params.DEFAULT_CONFIG_PATH)])
def test_tickers_are_the_ticker_config(query, path):
    body = client.get("/api/tickers" + query).json()
    config = pr.load_ticker_config(path, verbose=False)
    assert body["set"] == ("yahoo" if "yahoo" in query else "demo")
    assert body["source"] == str(path)
    assert [t["symbol"] for t in body["tickers"]] == list(config)
    for entry in body["tickers"]:
        p = config[entry["symbol"]]
        assert entry["label"] == p.label
        assert entry["params"] == via_json(clean(dataclasses.asdict(p)))


def test_default_tickers_are_the_demo_files_and_load_offline():
    body = client.get("/api/tickers").json()
    assert sorted(t["symbol"] for t in body["tickers"]) == pr.demo_tickers()
    assert all(t["params"]["data_source"] == "demo" for t in body["tickers"])
    first = body["tickers"][0]
    response = client.post("/api/overview", json=first["params"])
    assert response.status_code == 200
    assert response.json()["ticker"] == first["symbol"] == "SPX"


def test_unknown_ticker_set_is_rejected():
    assert client.get("/api/tickers?set=bloomberg").status_code == 422


def test_overview_matches_the_package_exactly():
    response = client.post("/api/overview", json=SIMULATED)
    assert response.status_code == 200
    body = response.json()
    p = pr.Params(**SIMULATED)
    df = pr.add_rolling_stats(pr.load_price_data(p, verbose=False), p)
    assert body["rows"] == len(df)
    assert body["start"] == df["date"].iloc[0].isoformat()
    assert body["end"] == df["date"].iloc[-1].isoformat()
    assert body["last_price"] == df["price"].iloc[-1]
    assert body["snapshot"] == pr.latest_snapshot(df, p).to_dict()
    assert body["distribution"] == pr.distribution_summary(df, p).iloc[0].to_dict()


def test_an_impossible_request_is_a_422():
    response = client.post("/api/overview", json={"data_source": "csv"})
    assert response.status_code == 422
    assert "data_source must be" in response.json()["detail"]


@pytest.mark.parametrize("error,status", [
    (ValueError("No price data for ticker 'NOPE'"), 422),
    (ConnectionError("Yahoo unreachable"), 502),
])
def test_data_source_failures_are_reported_not_crashed(monkeypatch, error, status):
    def fail(*args, **kwargs):
        raise error
    cache.cache_clear()
    monkeypatch.setattr(cache, "load_price_data", fail)
    response = client.post("/api/overview", json={"ticker": "NOPE", "data_source": "yahoo"})
    assert response.status_code == status
    assert str(error) in response.json()["detail"]
    cache.cache_clear()


def test_prices_are_cached_by_the_data_fields_only(monkeypatch):
    calls = []
    real = cache.load_price_data

    def counting(p, verbose=True):
        calls.append(p.ticker)
        return real(p, verbose=verbose)

    cache.cache_clear()
    monkeypatch.setattr(cache, "load_price_data", counting)
    client.post("/api/overview", json=SIMULATED)
    client.post("/api/overview", json={**SIMULATED, "win_threshold": 0.9, "trade_days": 252})
    assert len(calls) == 1                     # analysis settings reuse the prices
    client.post("/api/overview", json={**SIMULATED, "random_seed": 8})
    assert len(calls) == 2                     # a different series is a new download
    cache.cache_clear()


def test_overview_on_demo_data_matches_the_package_exactly():
    payload = {"data_source": "demo", "ticker": "CL", "start_date": "2019-06-01",
               "end_date": "2021-06-01"}
    body = client.post("/api/overview", json=payload).json()
    p = pr.Params(**payload)
    df = pr.add_rolling_stats(pr.load_price_data(p, verbose=False), p)
    assert body["rows"] == len(df)
    assert body["last_price"] == df["price"].iloc[-1]
    assert body["snapshot"] == via_json(clean(pr.latest_snapshot(df, p).to_dict()))
    assert body["distribution"] == via_json(frame_to_records(pr.distribution_summary(df, p))[0])


def test_unknown_demo_ticker_is_a_422_naming_the_demo_tickers():
    response = client.post("/api/overview", json={"data_source": "demo", "ticker": "ES=F"})
    assert response.status_code == 422
    assert "SPX" in response.json()["detail"]
