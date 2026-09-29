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
