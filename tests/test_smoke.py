"""End-to-end smoke test for the price-return pipeline.

Runs against simulated data so it needs no network. This checks that the
pipeline holds together and that the public API stays importable; it does not
verify the numerical methods against independent reference values.
"""
import pandas as pd
import pytest

from src.tools import price_return as pr


@pytest.fixture(scope="module")
def params():
    return pr.Params(data_source="simulated", start_date="2020-01-01")


@pytest.fixture(scope="module")
def prices(params):
    return pr.load_price_data(params, verbose=False)


@pytest.fixture(scope="module")
def enriched(prices, params):
    return pr.add_rolling_stats(prices, params)


def test_public_api_resolves():
    missing = [name for name in pr.__all__ if not hasattr(pr, name)]
    assert not missing, f"names in __all__ with no attribute: {missing}"


def test_load_price_data(prices):
    assert not prices.empty
    assert list(prices.columns) == ["date", "price", "return_pct"]
    assert pd.api.types.is_datetime64_any_dtype(prices["date"])
    assert prices["date"].is_monotonic_increasing


def test_simulation_is_reproducible(params):
    a = pr.load_price_data(params, verbose=False)
    b = pr.load_price_data(params, verbose=False)
    pd.testing.assert_frame_equal(a, b)


def test_add_rolling_stats_columns(enriched, params):
    for d in params.roll_windows:
        for suffix in ("", " Av", " STD"):
            assert f"PCT Change {d}{suffix}" in enriched.columns
    assert "PCT Change Annualized" in enriched.columns
    assert enriched["PCT Change Annualized"].notna().any()


def test_streaks(enriched, params):
    streaks = pr.detect_streaks(enriched, params)
    assert set(streaks) == set(params.windows)

    summary = pr.summarize_streaks(enriched, streaks, params)
    assert len(summary) == len(params.windows)
    assert {"Window", "Win Streaks", "Loss Streaks"} <= set(summary.columns)

    # A window of all wins must itself hold only days clearing the win threshold.
    for window, buckets in streaks.items():
        for entry in buckets["wins"]:
            assert len(entry["returns"]) == window
            assert min(entry["returns"]) >= params.win_threshold
        for entry in buckets["losses"]:
            assert max(entry["returns"]) <= params.loss_threshold


def test_cumulative(enriched, params):
    cum = pr.analyze_cumulative(enriched, params)
    assert cum
    summary = pr.summarize_cumulative(cum, params)
    assert not summary.empty


def test_historical_probabilities(enriched, params):
    returns = pr.daily_returns_series(enriched)
    assert returns.index.name == "date"

    historical = pr.build_historical_analysis(returns, params)
    assert not historical.empty

    prob = historical["prob"] if "prob" in historical.columns else None
    if prob is not None:
        assert prob.between(0, 1).all(), "probabilities must be in [0, 1]"


def test_export_tables(enriched, params, tmp_path):
    written = pr.export_tables({"smoke.csv": enriched.head()}, out_dir=str(tmp_path),
                               verbose=False)
    assert written
    for path in written:
        assert pd.read_csv(path).shape[0] == 5
