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
from src.api.serialize import clean, figure_to_json, frame_to_records, series_to_dict
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


# ── Phase 2: analysis, rare events, charts, statistics, multi-ticker ─────────
DEMO = {"data_source": "demo", "ticker": "SPX", "start_date": "2019-01-01"}


def direct(payload):
    """The notebook's own calls on the same parameters, bypassing the API and its caches."""
    p = pr.Params(**payload)
    df = pr.add_rolling_stats(pr.load_price_data(p, verbose=False), p)
    return p, df, pr.daily_returns_series(df)


def table(frame):
    return via_json({"columns": [str(c) for c in frame.columns],
                     "records": frame_to_records(frame)})


@pytest.mark.parametrize("payload", [DEMO, SIMULATED])
def test_streak_and_cumulative_summaries_match_the_package(payload):
    p, df, _ = direct(payload)
    body = client.post("/api/streaks", json=payload).json()
    assert body["summary"] == table(pr.summarize_streaks(df, pr.detect_streaks(df, p), p))
    body = client.post("/api/cumulative", json=payload).json()
    assert body["summary"] == table(pr.summarize_cumulative(pr.analyze_cumulative(df, p), p))
    assert body["summary"]["columns"][0] == "Window"          # order kept, not alphabetical


@pytest.mark.parametrize("query, n_days, change_type", [
    ("", 3, None),
    ("?n_days=1&n_days=10&change_type=consecutive", [1, 10], "consecutive"),
    ("?n_days=5&change_type=cumulative", [5], "cumulative"),
])
def test_rare_events_match_low_probability_view(query, n_days, change_type):
    payload = {**DEMO, "prob_max": 0.2, "prob_min": 0.001}
    p, _, r = direct(payload)
    expected = pr.low_probability_view(pr.build_historical_analysis(r, p), n_days, p,
                                       change_type=change_type)
    body = client.post("/api/rare-events" + query, json=payload).json()
    assert body["events"] == table(expected) and len(expected) > 0
    assert (body["prob_max"], body["prob_min"]) == (0.2, 0.001)


def test_moving_the_probability_bounds_refilters_the_cached_table(monkeypatch):
    calls = []
    real = cache.build_historical_analysis
    monkeypatch.setattr(cache, "build_historical_analysis",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    cache.cache_clear()
    wide = client.post("/api/rare-events", json={**DEMO, "prob_max": 0.5}).json()
    narrow = client.post("/api/rare-events", json={**DEMO, "prob_max": 0.01}).json()
    assert len(calls) == 1
    assert 0 < len(narrow["events"]["records"]) < len(wide["events"]["records"])
    client.post("/api/rare-events", json={**DEMO, "lookback_years": [1, 3]})
    assert len(calls) == 2                     # a new lookback is a new table
    cache.cache_clear()


def test_rare_events_reject_a_holding_period_not_computed():
    response = client.post("/api/rare-events?n_days=7", json=DEMO)
    assert response.status_code == 422 and "streak_days" in response.json()["detail"]


def _reference_charts(p, df, r):
    """Every chart built as the notebooks build it."""
    streaks, cum = pr.detect_streaks(df, p), pr.analyze_cumulative(df, p)
    return {
        "rolling-average": pr.plot_rolling_average(df, p),
        "rolling-volatility": pr.plot_rolling_volatility(df, p),
        "price-and-returns": pr.plot_price_and_returns(df, p),
        "return-distribution": pr.plot_return_distribution(df, p),
        "streak-counts": pr.plot_streak_counts(streaks, p),
        "streak-frequency": pr.plot_streak_frequency(df, streaks, p),
        "streak-timeline": pr.plot_streak_timeline(df, streaks, p),
        "cumulative-heatmap": pr.plot_cumulative_heatmap(cum, p),
        "cumulative-counts": pr.plot_cumulative_counts(cum, p),
        "qq": pr.plot_qq(r),
        "autocorrelation": pr.plot_autocorrelation(pr.autocorrelation(r)),
        "drawdown": pr.plot_drawdown(r),
        "rolling-risk": pr.plot_rolling_risk(pr.rolling_risk(r, 250, p.trade_days), window=250),
        "event-probabilities": pr.plot_event_probabilities(
            pr.event_probability_table(r, 3, p, n_boot=100), "cumulative", "above",
            n_years=max(p.lookback_years)),
    }


def test_every_chart_is_the_notebook_figure():
    pytest.importorskip("scipy")
    p, df, r = direct(DEMO)
    reference = _reference_charts(p, df, r)
    listed = client.get("/api/charts").json()["charts"]
    assert [c["name"] for c in listed] == list(reference)
    assert [c["group"] for c in listed].count("statistics") == 5
    for name, fig in reference.items():
        response = client.post(f"/api/charts/{name}?n_boot=100", json=DEMO)
        assert response.status_code == 200, name
        assert response.json() == figure_to_json(fig), name


def test_chart_options_reach_the_plot():
    pytest.importorskip("scipy")
    p, df, r = direct(DEMO)
    cases = {
        "streak-timeline?window=3": pr.plot_streak_timeline(df, pr.detect_streaks(df, p), p,
                                                           window=3),
        "drawdown?top=5": pr.plot_drawdown(r, top=5),
        "rolling-risk?window=60": pr.plot_rolling_risk(pr.rolling_risk(r, 60, p.trade_days),
                                                       window=60),
        "event-probabilities?n_days=5&n_boot=100&change_type=consecutive&change=below&n_years=2":
            pr.plot_event_probabilities(pr.event_probability_table(r, 5, p, n_boot=100),
                                        "consecutive", "below", n_years=2),
    }
    for path, fig in cases.items():
        assert client.post(f"/api/charts/{path}", json=DEMO).json() == figure_to_json(fig), path


def test_unknown_chart_is_a_404_listing_the_charts():
    response = client.post("/api/charts/pie", json=DEMO)
    assert response.status_code == 404 and "rolling-average" in response.json()["detail"]


def test_statistics_sections_match_the_package():
    pytest.importorskip("scipy")
    p, _, r = direct(DEMO)
    post = lambda section, q="": client.post(f"/api/statistics/{section}{q}", json=DEMO).json()
    assert post("distribution") == via_json({
        "moments": series_to_dict(pr.return_moments(r)),
        "jarque_bera": series_to_dict(pr.jarque_bera(r)),
        "student_t": series_to_dict(pr.fit_student_t(r)),
        "tail_index": series_to_dict(pr.tail_index(r)),
        "value_at_risk": table(pr.value_at_risk(r))})
    assert post("dependence") == via_json({
        "autocorrelation": table(pr.autocorrelation(r)),
        "ljung_box": table(pr.ljung_box(r)),
        "variance_ratio": table(pr.variance_ratio(r)),
        "arch_lm": series_to_dict(pr.arch_lm(r))})
    assert post("drawdowns", "?top=4") == via_json({
        "risk_ratios": series_to_dict(pr.risk_ratios(r, p.trade_days)),
        "drawdowns": table(pr.drawdown_table(r, top=4))})
    body = post("probabilities", "?n_days=2&n_boot=100")
    assert body == {"n_days": 2, "n_boot": 100,
                    "events": table(pr.event_probability_table(r, 2, p, n_boot=100))}


@pytest.mark.parametrize("path", ["/api/statistics/tails", "/api/statistics/probabilities?n_boot=5000",
                                  "/api/statistics/drawdowns?top=0"])
def test_bad_statistics_requests_are_rejected(path):
    assert client.post(path, json=DEMO).status_code == 422


def test_a_missing_stats_extra_is_a_501_with_the_install_hint(monkeypatch):
    import src.api.app as api_app

    def no_scipy(returns):
        raise ImportError('This needs scipy; from the repo root run: pip install -e ".[stats]"')
    monkeypatch.setattr(api_app, "fit_student_t", no_scipy)
    response = client.post("/api/statistics/distribution", json=DEMO)
    assert response.status_code == 501 and ".[stats]" in response.json()["detail"]


def _write_config(tmp_path, tickers):
    lines = ["defaults: {data_source: demo, start_date: '2019-01-01'}", "tickers:"]
    lines += [f"  {symbol}: {{label: {label}}}" for symbol, label in tickers]
    path = tmp_path / "tickers.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_multi_ticker_is_rare_case_run(tmp_path, monkeypatch):
    import src.api.app as api_app
    path = _write_config(tmp_path, [("SPX", "S&P demo"), ("NOPE", "Missing"), ("KO", "KO demo")])
    monkeypatch.setitem(api_app.TICKER_SETS, "demo", path)
    body = client.get("/api/multi-ticker?drill_n_days=2").json()

    config = pr.load_ticker_config(path, verbose=False)
    results = {s: pr.analyze_ticker(config[s], drill_n_days=2) for s in ("SPX", "KO")}
    streak_table, dist_table = pr.compare_tickers(results, ratio_window=2)
    assert body["streaks"] == table(streak_table)
    assert body["distribution"] == table(dist_table)
    assert [t["symbol"] for t in body["tickers"]] == ["SPX", "KO"]
    assert [t["label"] for t in body["tickers"]] == ["S&P demo", "KO demo"]
    for entry in body["tickers"]:
        r = results[entry["symbol"]]
        assert entry["drill"] == table(r["drill"]) and entry["rows"] == len(r["df"])
    assert [f["symbol"] for f in body["failed"]] == ["NOPE"]
    assert "No demo data" in body["failed"][0]["error"]


def test_multi_ticker_with_every_ticker_failing_is_a_502(tmp_path, monkeypatch):
    import src.api.app as api_app
    monkeypatch.setitem(api_app.TICKER_SETS, "demo", _write_config(tmp_path, [("NOPE", "x")]))
    response = client.get("/api/multi-ticker")
    assert response.status_code == 502
    assert response.json()["detail"]["failed"][0]["symbol"] == "NOPE"


POST_ENDPOINTS = ["/api/overview", "/api/streaks", "/api/cumulative", "/api/rare-events",
                  "/api/charts/qq", "/api/charts/event-probabilities",
                  "/api/statistics/distribution", "/api/statistics/probabilities"]


@pytest.mark.parametrize("path", POST_ENDPOINTS)
def test_every_endpoint_reports_data_failures(path, monkeypatch):
    response = client.post(path, json={"data_source": "demo", "ticker": "ES=F"})
    assert response.status_code == 422 and "Demo tickers" in response.json()["detail"]

    def offline(*args, **kwargs):
        raise ConnectionError("Yahoo unreachable")
    cache.cache_clear()
    monkeypatch.setattr(cache, "load_price_data", offline)
    response = client.post(path, json={"data_source": "yahoo", "ticker": "ES=F"})
    assert response.status_code == 502 and "Yahoo unreachable" in response.json()["detail"]
    cache.cache_clear()
