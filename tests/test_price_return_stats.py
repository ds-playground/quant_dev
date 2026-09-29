"""Offline tests for price_return's option helpers and statistics, against independent references."""
import math

import numpy as np
import pandas as pd
import pytest

from src.tools import price_return as pr


# ── Phase 1: option helpers ─────────────────────────────────────────────────
def test_price_range_days_is_the_lognormal_band():
    band = pr.price_range(100, 0.01, days=4)
    assert band["vol"] == pytest.approx(0.02)                    # 1% * sqrt(4)
    assert band["low"] == pytest.approx(100 * math.exp(-0.02))
    assert band["high"] == pytest.approx(100 * math.exp(0.02))
    assert band["horizon"] == "4 days"


def test_price_range_hours_scales_by_the_share_of_the_session():
    assert pr.price_range(50, 0.012, hours=6.5)["vol"] == pytest.approx(0.012)     # a full day
    assert pr.price_range(50, 0.012, hours=1.625)["vol"] == pytest.approx(0.006)   # a quarter: sqrt
    assert pr.price_range(50, 0.012, hours=2, session_hours=8)["vol"] == pytest.approx(0.006)
    assert pr.price_range(50, 0.012, hours=0)["vol"] == 0


def test_price_range_rounds_outwards_to_the_tick():
    band = pr.price_range(100, 0.01, days=4, tick=0.1)            # 98.0199 .. 102.0201
    assert (band["low"], band["high"]) == (98.0, 102.1)
    band = pr.price_range(100, 0.01, days=4, tick=0.25)
    assert (band["low"], band["high"]) == (98.0, 102.25)


def test_price_range_keeps_a_price_already_on_the_tick():
    # 4.1 / 0.1 is 40.99999..., so a naive ceil/floor would move the band a whole tick.
    band = pr.price_range(4.1, 0.0, days=1, tick=0.1)
    assert (band["low"], band["high"]) == (4.1, 4.1)


@pytest.mark.parametrize("kwargs,match", [
    ({}, "exactly one"),
    ({"days": 2, "hours": 1}, "exactly one"),
    ({"hours": 7}, "between 0 and session_hours"),
    ({"hours": -1}, "between 0 and session_hours"),
    ({"days": -1}, "days must be >= 0"),
])
def test_price_range_rejects_bad_horizons(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pr.price_range(100, 0.01, **kwargs)


def hand_frame():
    """Two rows that fall outside the trailing window, then the five it covers."""
    return pd.DataFrame({
        "PCT Change 1": [0.5, -0.5, 0.02, -0.03, 0.005, 0.015, -0.012],
        "PCT Change 1 STD": [np.nan] * 6 + [0.018],
        "PCT Change 4": [0.5, -0.5, np.nan, 0.03, -0.01, 0.019, 0.021],
        "PCT Change 4 STD": [np.nan] * 6 + [0.025],
    })


def lookup(table, horizon, method, direction):
    row = table.set_index(["horizon", "method", "direction"]).loc[(horizon, method, direction)]
    return row["count"], row["n_obs"], row["prob"]


def test_move_probabilities_counts_the_trailing_windows_by_hand():
    params = pr.Params(trade_days=5)
    table = pr.move_probabilities(hand_frame(), params, horizons=(1, 4), scaled=0.01, fixed=0.025)
    # 1 day, threshold 0.01: above 0.02, 0.015; below -0.03, -0.012.
    assert lookup(table, 1, "scaled", "above") == (2, 5, 0.4)
    assert lookup(table, 1, "scaled", "below") == (2, 5, 0.4)
    assert lookup(table, 1, "scaled", "exceeding") == (4, 5, 0.8)
    # fixed 0.025: only -0.03 is beyond it.
    assert lookup(table, 1, "fixed", "above") == (0, 5, 0.0)
    assert lookup(table, 1, "fixed", "exceeding") == (1, 5, 0.2)
    # actual: the latest 1-day STD, 0.018.
    assert lookup(table, 1, "actual", "exceeding") == (2, 5, 0.4)
    # 4 days: scaled threshold 0.01 * sqrt(4) = 0.02; the NaN window is not complete.
    assert lookup(table, 4, "scaled", "above") == (2, 4, 0.5)    # 0.03, 0.021
    assert lookup(table, 4, "scaled", "below") == (0, 4, 0.0)
    assert lookup(table, 4, "fixed", "above") == (1, 4, 0.25)    # fixed does not scale
    assert lookup(table, 4, "actual", "above") == (1, 4, 0.25)   # latest 4-day STD, 0.025
    scaled_4 = table[(table["horizon"] == 4) & (table["method"] == "scaled")]
    assert scaled_4["threshold"].iloc[0] == pytest.approx(0.02)


def test_move_probabilities_only_includes_the_methods_given():
    table = pr.move_probabilities(hand_frame(), pr.Params(trade_days=5), horizons=(1,))
    assert set(table["method"]) == {"actual"}
    assert len(table) == 3


def test_move_probabilities_needs_the_horizon_columns():
    with pytest.raises(KeyError, match="horizon"):
        pr.move_probabilities(hand_frame(), pr.Params(trade_days=5), horizons=(1, 10))


def test_move_probabilities_runs_on_the_pipeline_frame():
    params = pr.Params(data_source="simulated", start_date="2020-01-01")
    df = pr.add_rolling_stats(pr.load_price_data(params, verbose=False), params)
    table = pr.move_probabilities(df, params, scaled=0.01, fixed=0.02)
    assert len(table) == 3 * 3 * 3                     # horizons x methods x directions
    assert (table["n_obs"] == params.trade_days).all()
    above, below, exceeding = (table.set_index(["horizon", "method", "direction"])["count"]
                               .unstack("direction")[["above", "below", "exceeding"]].T.to_numpy())
    np.testing.assert_array_equal(above + below, exceeding)


def test_expected_pnl_is_the_probability_weighted_outcome():
    probs = pd.DataFrame({"horizon": [1, 1, 5], "method": "fixed", "threshold": 0.02,
                          "direction": ["above", "below", "above"], "count": [1, 1, 1],
                          "n_obs": 4, "prob": [0.25, 0.5, 0.1]})
    grid = {1: {"above": [0.2, -1.0], "below": [0.15, -0.3]}}
    out = pr.expected_pnl(probs, grid, contract_size=100)
    assert len(out) == 2                                # horizon 5 is not in the grid
    by_direction = out.set_index("direction")["expected_pnl"]
    assert by_direction["above"] == pytest.approx((0.75 * 0.2 + 0.25 * -1.0) * 100)   # -10
    assert by_direction["below"] == pytest.approx((0.5 * 0.15 + 0.5 * -0.3) * 100)    # -7.5


# ── Phase 3: one n-day return, complete-window frequencies, episodes ─────────
def price_frame(return_pct):
    """A load_price_data-shaped frame from percent daily returns."""
    r = np.asarray(return_pct, dtype=float)
    return pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=len(r)),
                         "price": 100 * np.cumprod(1 + r / 100), "return_pct": r})


def dated(decimal_returns):
    r = pd.Series(decimal_returns, dtype=float)
    r.index = pd.bdate_range("2024-01-01", periods=len(r), name="date")
    return r


def test_compound_returns_multiplies_rather_than_adds():
    r = pd.Series([0.1, 0.1, -0.5, 0.2])
    np.testing.assert_allclose(pr.compound_returns(r, 2), [np.nan, 0.21, -0.45, -0.4],
                               equal_nan=True)                      # summing gives 0.2, -0.4, -0.3
    np.testing.assert_allclose(pr.compound_returns(r, 3), [np.nan, np.nan, -0.395, -0.34],
                               equal_nan=True)
    pd.testing.assert_series_equal(pr.compound_returns(r, 1), r)    # exact at one day


def test_add_rolling_stats_compounds_every_horizon():
    params = pr.Params(trade_days=3, roll_windows=[1, 2, 3])
    df = pr.add_rolling_stats(price_frame([10, 10, -50, 20]), params)
    np.testing.assert_allclose(df["PCT Change 2"], [np.nan, 0.21, -0.45, -0.4], equal_nan=True)
    np.testing.assert_allclose(df["PCT Change Annualized"], [np.nan, np.nan, -0.395, -0.34],
                               equal_nan=True)
    # ...and each horizon's return is the price change over it.
    price = df["price"].to_numpy()
    assert df["PCT Change 3"].iloc[3] == pytest.approx(price[3] / price[0] - 1)


def test_consecutive_analysis_compounds_the_cumulative_move():
    params = pr.Params()
    up = pr.consecutive_analysis(dated([0.1, 0.1]), 0.205, 2, 1, params).set_index(
        ["change_type", "change"])
    assert up.loc[("cumulative", "above"), "count"] == 1       # +21% compounded; +20% summed
    down = pr.consecutive_analysis(dated([-0.1, -0.1]), 0.195, 2, 1, params).set_index(
        ["change_type", "change"])
    assert down.loc[("cumulative", "below"), "count"] == 0     # -19% compounded; -20% summed


def test_consecutive_analysis_divides_by_complete_windows():
    table = pr.consecutive_analysis(dated([0.02] * 5), 0.01, 3, 1, pr.Params()).set_index(
        ["change_type", "change"])
    row = table.loc[("consecutive", "above")]
    assert (row["n_obs"], row["n_windows"], row["count"]) == (5, 3, 3)
    assert row["prob"] == 1.0                                   # 3 of 3 windows, not 3 of 5 days


def test_streak_frequency_divides_by_complete_windows():
    params = pr.Params(windows=[2], win_threshold=0.5, loss_threshold=-0.5)
    df = price_frame([1.0] * 5)
    streaks = pr.detect_streaks(df, params)
    summary = pr.summarize_streaks(df, streaks, params).iloc[0]
    assert summary["Win Streaks"] == 4 and summary["Win Freq %"] == 100.0   # 4 of 4, not 4 of 5
    fig = pr.plot_streak_frequency(df, streaks, params)
    assert list(fig.data[0].y) == [100.0]


def test_episodes_count_each_unbroken_run_once():
    # Win, win, win, loss, win, win: 2-day win windows end on days 2, 3 and 6 - two runs.
    params = pr.Params(windows=[2], win_threshold=0.5, loss_threshold=-0.5, cum_thresholds=[1.5])
    df = price_frame([1, 1, 1, -1, 1, 1])
    summary = pr.summarize_streaks(df, pr.detect_streaks(df, params), params).iloc[0]
    assert (summary["Win Streaks"], summary["Win Episodes"]) == (3, 2)
    assert (summary["Loss Streaks"], summary["Loss Episodes"]) == (0, 0)

    cum = pr.analyze_cumulative(df, params)[2][1.5]             # 2-day compounded >= 1.5%
    assert (cum["count"], cum["episodes"]) == (3, 2)
    assert pr.summarize_cumulative(pr.analyze_cumulative(df, params), params)[
        "≥1.5% Episodes"].iloc[0] == 2

    table = pr.consecutive_analysis(dated([0.01, 0.01, 0.01, -0.01, 0.01, 0.01]), 0.005, 2, 1,
                                    pr.Params()).set_index(["change_type", "change"])
    row = table.loc[("consecutive", "above")]
    assert (row["count"], row["episodes"]) == (3, 2)
