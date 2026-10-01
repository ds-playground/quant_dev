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


# ── Phase 4: statistics ─────────────────────────────────────────────────────
def normal_returns(n, seed, mu=0.0005, sd=0.01):
    return dated(np.random.default_rng(seed).normal(mu, sd, n))


def garch_returns(n, seed, omega=2e-6, alpha=0.1, beta=0.85):
    """A GARCH(1,1) path: volatility clusters by construction."""
    rng = np.random.default_rng(seed)
    r, var = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = math.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    return dated(r)


def ar1_returns(n, seed, phi, sd=0.01):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + sd * rng.standard_normal()
    return dated(r)


def test_return_moments_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    x = np.random.default_rng(1).standard_t(5, 3000) * 0.01
    m = pr.return_moments(x)
    assert m["n"] == 3000 and m["std"] == pytest.approx(np.std(x, ddof=1))
    assert m["skew"] == pytest.approx(stats.skew(x, bias=False))
    assert m["excess_kurtosis"] == pytest.approx(stats.kurtosis(x, bias=False))


def test_jarque_bera_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for x in (np.random.default_rng(2).normal(size=500),
              np.random.default_rng(3).standard_t(4, 2000)):
        ours, ref = pr.jarque_bera(x), stats.jarque_bera(x)
        assert ours["statistic"] == pytest.approx(ref.statistic)
        assert ours["p_value"] == pytest.approx(ref.pvalue, abs=1e-12)


def test_fit_student_t_recovers_the_degrees_of_freedom():
    pytest.importorskip("scipy")
    fit = pr.fit_student_t(np.random.default_rng(4).standard_t(4, 20000) * 0.01)
    assert 3.3 < fit["df"] < 4.8
    assert fit["scale"] == pytest.approx(0.01, rel=0.05)


def test_qq_points_are_straight_for_the_right_distribution():
    pytest.importorskip("scipy")
    normal = pr.qq_points(np.random.default_rng(5).normal(0, 0.01, 4000), "normal")
    assert np.corrcoef(normal["theoretical"], normal["sample"])[0, 1] > 0.999
    fat = np.random.default_rng(6).standard_t(3, 4000) * 0.01
    # Compare over the central 98% of points: the single most extreme point of a fat-tailed
    # sample is too noisy to separate any two fits.
    gap = {d: (pr.qq_points(fat, d)["sample"] - pr.qq_points(fat, d)["theoretical"]).abs()
           .iloc[40:-40].max() for d in ("normal", "t")}
    assert gap["t"] < gap["normal"] / 3            # the normal misses the fat tails badly
    with pytest.raises(ValueError, match="dist"):
        pr.qq_points(fat, "cauchy")


def test_tail_index_recovers_a_pareto_alpha():
    u = np.random.default_rng(7).random(50000)
    gains = 0.01 * (1 - u) ** (-1 / 3)                    # Pareto with alpha = 3, scaled to returns
    right = pr.tail_index(gains, k=2500)
    assert right["right"] == pytest.approx(3, abs=0.2)
    assert pr.tail_index(-gains, k=2500)["left"] == pytest.approx(3, abs=0.2)


def test_value_at_risk_by_hand():
    x = dated([-0.05, -0.02, 0.01, 0.03, 0.04])
    table = pr.value_at_risk(x, horizons=(1, 2), levels=(0.8,)).set_index(["horizon", "method"])
    # 20% quantile, linear interpolation: -0.05 + 0.8 * 0.03 = -0.026; shortfall: the -5% day.
    assert table.loc[(1, "historical"), "var"] == pytest.approx(0.026)
    assert table.loc[(1, "historical"), "es"] == pytest.approx(0.05)
    # 2-day compounded returns: -0.069, -0.0102, 0.0403, 0.0712; quantile -0.069 + 0.6 * 0.0588.
    assert table.loc[(2, "historical"), "var"] == pytest.approx(0.069 - 0.6 * 0.0588)


def test_value_at_risk_on_a_normal_sample():
    stats = pytest.importorskip("scipy.stats")
    mu, sd = 0.0005, 0.01
    table = pr.value_at_risk(normal_returns(200_000, 8, mu, sd), horizons=(1,),
                             levels=(0.95, 0.99)).set_index(["level", "method"])
    for level in (0.95, 0.99):
        z = stats.norm.ppf(1 - level)
        exact_var = -(mu + sd * z)
        exact_es = -(mu - sd * stats.norm.pdf(z) / (1 - level))
        for method in ("historical", "normal", "cornish_fisher"):
            assert table.loc[(level, method), "var"] == pytest.approx(exact_var, rel=0.02), method
            assert table.loc[(level, method), "es"] == pytest.approx(exact_es, rel=0.03), method


def test_cornish_fisher_raises_var_for_fat_tails():
    table = pr.value_at_risk(dated(np.random.default_rng(9).standard_t(4, 20000) * 0.01),
                             horizons=(1,), levels=(0.99,)).set_index("method")
    assert table.loc["cornish_fisher", "var"] > table.loc["normal", "var"]


def test_cornish_fisher_moves_the_normal_quantile_towards_the_true_one():
    stats = pytest.importorskip("scipy.stats")
    df, scale = 10, 0.01                                  # excess kurtosis 6 / (df - 4) = 1
    x = stats.t.rvs(df, scale=scale, size=400_000, random_state=np.random.default_rng(22))
    table = pr.value_at_risk(dated(x), horizons=(1,), levels=(0.99,)).set_index("method")
    true_var = -stats.t.ppf(0.01, df, scale=scale)
    miss = {m: abs(table.loc[m, "var"] - true_var) for m in ("normal", "cornish_fisher")}
    assert miss["cornish_fisher"] < miss["normal"]        # closer, though it overshoots at 99%
    # The textbook expansion from the population moments: skew 0, excess kurtosis 1.
    z = stats.norm.ppf(0.01)
    textbook = -(scale * math.sqrt(df / (df - 2))) * (z + (z ** 3 - 3 * z) * 1.0 / 24)
    assert table.loc["cornish_fisher", "var"] == pytest.approx(textbook, rel=0.01)


def test_autocorrelation_matches_the_formula():
    x = np.array([0.01, -0.02, 0.015, 0.003, -0.007, 0.02, -0.01, 0.004, 0.0, -0.012])
    d = x - x.mean()
    expected = [sum(d[t] * d[t - k] for t in range(k, len(x))) / sum(d * d) for k in (1, 2, 3)]
    table = pr.autocorrelation(x, lags=3)
    np.testing.assert_allclose(table["returns"], expected)
    assert table["band"].iloc[0] == pytest.approx(1.96 / math.sqrt(10))


def test_ljung_box_matches_the_formula():
    stats = pytest.importorskip("scipy.stats")
    x = np.random.default_rng(10).normal(size=40)
    d = x - x.mean()
    rho = [sum(d[t] * d[t - k] for t in range(k, 40)) / sum(d * d) for k in range(1, 6)]
    q = 40 * 42 * sum(rho[k - 1] ** 2 / (40 - k) for k in range(1, 6))
    row = pr.ljung_box(x, lags=(5,)).set_index("series").loc["returns"]
    assert row["statistic"] == pytest.approx(q)
    assert row["p_value"] == pytest.approx(stats.chi2.sf(q, 5))


def test_ljung_box_finds_volatility_clustering():
    pytest.importorskip("scipy")
    table = pr.ljung_box(garch_returns(3000, 11), lags=(10,)).set_index("series")
    assert table.loc["squared", "p_value"] < 1e-4
    assert pr.ljung_box(normal_returns(3000, 12), lags=(10,)).set_index("series").loc[
        "squared", "p_value"] > 0.01


def test_variance_ratio_on_random_walk_trend_and_reversion():
    iid = pr.variance_ratio(normal_returns(5000, 13), periods=(2, 5)).set_index("q")
    assert (iid["variance_ratio"] - 1).abs().max() < 0.06
    assert (iid["z_robust"].abs() < 2.5).all()
    np.testing.assert_allclose(iid["z"], iid["z_robust"], atol=0.15)   # constant vol: they agree

    for phi, sign in ((0.2, 1), (-0.2, -1)):
        r = ar1_returns(5000, 14, phi)
        vr = pr.variance_ratio(r, periods=(2, 5)).set_index("q")
        y = np.log1p(r.to_numpy())
        rho = [np.corrcoef(y[k:], y[:-k])[0, 1] for k in range(1, 5)]
        # Independent of the Lo-MacKinlay algebra: VR(q) ~ 1 + 2 * sum_k (1 - k/q) rho_k.
        for q in (2, 5):
            approx = 1 + 2 * sum((1 - k / q) * rho[k - 1] for k in range(1, q))
            assert vr.loc[q, "variance_ratio"] == pytest.approx(approx, abs=0.02)
        assert sign * vr.loc[2, "z_robust"] > 5


def test_arch_lm_at_one_lag_is_n_times_the_squared_correlation():
    pytest.importorskip("scipy")
    r = garch_returns(2000, 15).to_numpy()
    e2 = (r - r.mean()) ** 2
    expected = (len(r) - 1) * np.corrcoef(e2[1:], e2[:-1])[0, 1] ** 2
    assert pr.arch_lm(r, lags=1)["statistic"] == pytest.approx(expected)


def test_arch_lm_separates_clustered_from_constant_volatility():
    pytest.importorskip("scipy")
    assert pr.arch_lm(garch_returns(3000, 16))["p_value"] < 1e-4
    assert pr.arch_lm(normal_returns(3000, 17))["p_value"] > 0.05


def test_drawdowns_on_a_hand_built_path():
    r = dated([0.1, -0.2, 0.05, 0.2, -0.1])           # wealth 1.1, .88, .924, 1.1088, .99792
    np.testing.assert_allclose(pr.drawdown_series(r)["drawdown"], [0, -0.2, -0.16, 0, -0.1],
                               atol=1e-12)
    table = pr.drawdown_table(r)
    dates = r.index
    first = table.iloc[0]
    assert first["depth"] == pytest.approx(-0.2)
    assert (first["peak"], first["trough"], first["recovery"]) == (dates[0], dates[1], dates[3])
    assert (first["days_to_trough"], first["days_to_recover"]) == (1, 2)
    second = table.iloc[1]
    assert second["depth"] == pytest.approx(-0.1) and pd.isna(second["recovery"])
    assert pr.max_drawdown(r)["depth"] == pytest.approx(-0.2)


def test_drawdown_from_the_starting_capital():
    r = dated([-0.1, 0.2])
    row = pr.max_drawdown(r)
    assert row["depth"] == pytest.approx(-0.1) and pd.isna(row["peak"])
    assert (row["trough"], row["recovery"]) == (r.index[0], r.index[1])
    assert pr.max_drawdown(dated([0.01, 0.02]))["depth"] == 0.0


def test_risk_ratios_by_hand():
    r = dated([0.01, -0.01, 0.02, 0.0])
    ratios = pr.risk_ratios(r, trade_days=250)
    std = math.sqrt(0.0005 / 3)                         # deviations .005, -.015, .015, -.005
    assert ratios["sharpe"] == pytest.approx(0.005 / std * math.sqrt(250))
    assert ratios["sortino"] == pytest.approx(math.sqrt(250))   # downside deviation is 0.005
    growth = 1.01 * 0.99 * 1.02
    assert ratios["annual_return"] == pytest.approx(growth ** (250 / 4) - 1)
    assert ratios["max_drawdown"] == pytest.approx(0.99 - 1)
    assert ratios["calmar"] == pytest.approx((growth ** 62.5 - 1) / 0.01)


def test_rolling_risk_ends_on_the_last_window():
    params = pr.Params(data_source="simulated", start_date="2020-01-01")
    r = pr.daily_returns_series(pr.add_rolling_stats(pr.load_price_data(params, verbose=False),
                                                     params))
    last = pr.rolling_risk(r, window=250).iloc[-1]
    full = pr.risk_ratios(r.iloc[-250:])
    assert last["volatility"] == pytest.approx(full["annual_volatility"])
    assert last["sharpe"] == pytest.approx(full["sharpe"])
    assert last["sortino"] == pytest.approx(full["sortino"])


def test_stationary_bootstrap_blocks_have_the_mean_length():
    idx = pr.stationary_bootstrap(1000, mean_block=10, n_boot=50, seed=18)
    assert idx.shape == (50, 1000) and idx.min() >= 0 and idx.max() < 1000
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % 1000).mean()
    assert continues == pytest.approx(0.9, abs=0.01)


def test_bootstrap_intervals_cover_the_truth_at_about_the_nominal_rate():
    true_p = 1 - 0.8413447460685429                    # P(Z > 1)
    covered = 0
    for seed in range(200):
        x = np.random.default_rng(1000 + seed).normal(0, 0.01, 500)
        band = pr.bootstrap_interval(x, lambda a: np.mean(a > 0.01), n_boot=300, mean_block=1,
                                     seed=seed)
        covered += band["lower"] <= true_p <= band["upper"]
    assert 0.88 <= covered / 200 <= 0.99


def test_probability_intervals_agree_with_consecutive_analysis():
    params = pr.Params(data_source="simulated", start_date="2016-01-01")
    r = pr.daily_returns_series(pr.add_rolling_stats(pr.load_price_data(params, verbose=False),
                                                     params))
    keys = ["change_type", "change", "threshold", "n_days", "n_years"]
    for n_days in (1, 3, 10):
        bands = pr.probability_intervals(r, n_days, params, n_boot=200)
        reference = pd.concat([pr.consecutive_analysis(r, thr, n_days, n_years, params)
                               for thr in params.return_thresholds
                               for n_years in params.lookback_years])
        merged = bands.merge(reference, on=keys, suffixes=("", "_ref"))
        assert len(merged) == len(bands) == len(reference)
        np.testing.assert_allclose(merged["prob"], merged["prob_ref"], atol=1e-12)
        inside = (merged["lower"] <= merged["prob"] + 1e-12) & (merged["prob"] - 1e-12 <= merged["upper"])
        assert inside.mean() > 0.95


def test_event_probability_table_joins_counts_intervals_and_models_row_for_row():
    pytest.importorskip("scipy")
    params = pr.Params(data_source="synthetic", ticker="SYN-INDEX", start_date="2016-01-01")
    r = pr.daily_returns_series(pr.add_rolling_stats(pr.load_price_data(params, verbose=False),
                                                     params))
    table = pr.event_probability_table(r, 3, params, n_boot=100)
    keys = ["change_type", "change", "threshold", "n_days", "n_years"]
    # One row per threshold x lookback x event type, nothing lost or duplicated in the joins.
    assert len(table) == len(params.return_thresholds) * len(params.lookback_years) * 4
    assert not table.duplicated(keys).any()
    # Each column is its source's, on the same row.
    bands = pr.probability_intervals(r, 3, params, n_boot=100)
    models = pr.model_probabilities(r, 3, params)
    for source, columns in ((bands, ["prob", "lower", "upper"]),
                            (models, ["prob_normal", "prob_t"])):
        joined = table.merge(source, on=keys, suffixes=("", "_src"))
        for c in columns:
            assert (joined[c] == joined[f"{c}_src"]).all()
    for row in table.sample(10, random_state=0).itertuples():
        ref = pr.consecutive_analysis(r, row.threshold, 3, row.n_years, params)
        ref = ref[(ref["change_type"] == row.change_type) & (ref["change"] == row.change)].iloc[0]
        assert (row.count, row.episodes) == (ref["count"], ref["episodes"])
        assert row.prob == pytest.approx(ref["count"] / ref["n_windows"], abs=1e-12)


def test_model_probabilities_normal_is_exact_and_matches_iid_data():
    pytest.importorskip("scipy")
    params = pr.Params()
    r = normal_returns(5 * params.trade_days, 19, mu=0.0003, sd=0.01)
    model = pr.model_probabilities(r, 5, params, thresholds=[0.01, 0.02], lookback_years=[5])
    model = model.set_index(["change_type", "change", "threshold"])
    y = np.log1p(r.to_numpy())
    sims = np.random.default_rng(20).normal(y.mean(), y.std(ddof=1), (400_000, 5)).sum(axis=1)
    assert model.loc[("cumulative", "above", 0.02), "prob_normal"] == pytest.approx(
        np.mean(sims >= math.log1p(0.02)), abs=0.003)
    # On i.i.d. normal data the model and the observed frequency agree.
    observed = pr.consecutive_analysis(r, 0.02, 5, 5, params).set_index(["change_type", "change"])
    assert model.loc[("cumulative", "above", 0.02), "prob_normal"] == pytest.approx(
        observed.loc[("cumulative", "above"), "prob"], abs=0.04)


def test_student_t_model_gives_fat_tails_more_weight():
    pytest.importorskip("scipy")
    r = dated(np.random.default_rng(21).standard_t(3, 1250) * 0.008)
    model = pr.model_probabilities(r, 1, pr.Params(), thresholds=[0.04], lookback_years=[5])
    row = model.set_index(["change_type", "change"]).loc[("consecutive", "above")]
    assert row["prob_t"] > 3 * row["prob_normal"]


# ── Phase 5: statistics charts ──────────────────────────────────────────────
def test_statistics_charts_build():
    pytest.importorskip("scipy")
    params = pr.Params()
    r = garch_returns(1500, 23)
    table = pr.probability_intervals(r, 3, params, n_boot=100).merge(
        pr.model_probabilities(r, 3, params, n_sims=20_000),
        on=["change_type", "change", "threshold", "n_days", "n_years"])
    figures = {
        "qq": pr.plot_qq(r),
        "acf": pr.plot_autocorrelation(pr.autocorrelation(r)),
        "drawdown": pr.plot_drawdown(r),
        "rolling": pr.plot_rolling_risk(pr.rolling_risk(r, 250), window=250),
        "events": pr.plot_event_probabilities(table, "cumulative", "above", n_years=5),
    }
    assert [t.name for t in figures["qq"].data] == ["Normal", "Student-t", "Perfect fit"]
    assert [t.name for t in figures["events"].data] == ["Observed", "Normal model",
                                                        "Student-t model"]
    assert figures["events"].layout.yaxis.type == "log"
    assert len(figures["drawdown"].layout.annotations) == 3          # deepest troughs labelled
    for name, fig in figures.items():
        # One y-scale per plot: panels may sit side by side or stacked, never overlaid.
        assert not any(getattr(fig.layout[a], "overlaying", None)
                       for a in fig.layout if a.startswith("yaxis")), name


def test_streak_timeline_shades_each_streak_once_over_the_full_height():
    params = pr.Params(data_source="synthetic", ticker="SYN-OIL", start_date="2019-01-01",
                       end_date="2021-01-01")
    df = pr.load_price_data(params, verbose=False)
    streaks = pr.detect_streaks(df, params)
    for window in params.windows:
        shapes = pr.plot_streak_timeline(df, streaks, params, window=window).layout.shapes
        rects = [(pd.Timestamp(s.x0), pd.Timestamp(s.x1), s.fillcolor) for s in shapes
                 if s.type == "rect"]
        expected = ([(w["start"], w["end"], "#1D9E75") for w in streaks[window]["wins"]]
                    + [(l["start"], l["end"], "#D85A30") for l in streaks[window]["losses"]])
        assert rects == expected and len(rects) > 0
        assert all((s.yref, s.y0, s.y1) == ("y domain", 0, 1) for s in shapes if s.type == "rect")
        lines = sorted(s.y0 for s in shapes if s.type == "line")
        assert lines == [params.loss_threshold, params.win_threshold]


def _event_table(prob, normal, student):
    """A hand-built event table: one cumulative-above row per threshold, 5-year lookback."""
    n = len(prob)
    return pd.DataFrame({
        "change_type": ["cumulative"] * n, "change": ["above"] * n,
        "threshold": np.linspace(0.005, 0.03, n), "n_days": [3] * n, "n_years": [5] * n,
        "prob": prob, "lower": np.array(prob) * 0.8, "upper": np.array(prob) * 1.2,
        "prob_normal": normal, "prob_t": student})


def test_event_chart_ticks_suit_the_range():
    wide = _event_table([0.3, 0.1, 0.01, 0.001], [0.3, 0.08, 0.005, 0.0002], [0.3, 0.1, 0.01, 0.001])
    narrow = _event_table([0.6, 0.4, 0.2, 0.1], [0.55, 0.4, 0.25, 0.12], [0.6, 0.42, 0.22, 0.11])
    assert pr.plot_event_probabilities(wide, n_years=5).layout.yaxis.dtick == 1
    assert pr.plot_event_probabilities(narrow, n_years=5).layout.yaxis.dtick == "D2"


def test_event_chart_keeps_the_model_labels_apart():
    def labels(normal_end, t_end):
        table = _event_table([0.5, 0.2, 0.05, 0.01], [0.5, 0.2, 0.05, normal_end],
                             [0.5, 0.2, 0.05, t_end])
        fig = pr.plot_event_probabilities(table, n_years=5)
        values = table[["prob", "lower", "upper", "prob_normal", "prob_t"]].to_numpy().ravel()
        decades = np.log10(values.max() / values.min())
        px_per_decade = 330 / decades                 # the plot's height over the axis span
        return {a.text: (a.y, a.yshift or 0) for a in fig.layout.annotations}, px_per_decade

    close, px = labels(0.0100, 0.0102)
    (y_n, shift_n), (y_t, shift_t) = close["Normal model"], close["Student-t model"]
    assert shift_t > 0 > shift_n                      # the higher line's label moves up
    assert abs(y_t - y_n) * px + (shift_t - shift_n) == pytest.approx(14)   # one label height apart
    far, _ = labels(0.0001, 0.01)
    assert far["Normal model"][1] == far["Student-t model"][1] == 0


def test_histogram_threshold_labels_sit_outside_their_lines():
    params = pr.Params(win_threshold=0.2, loss_threshold=-0.2)
    df = pr.load_price_data(params, verbose=False)
    labels = {a.text: a for a in pr.plot_return_distribution(df, params).layout.annotations}
    assert labels["Win thr"].x == 0.2 and labels["Win thr"].xanchor == "left"
    assert labels["Loss thr"].x == -0.2 and labels["Loss thr"].xanchor == "right"


def test_autocorrelation_chart_draws_the_band_in_both_panels():
    acf = pr.autocorrelation(normal_returns(500, 24))
    fig = pr.plot_autocorrelation(acf)
    bands = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(bands) == 2
    assert all(s.y1 == pytest.approx(acf["band"].iloc[0]) for s in bands)


def test_event_chart_leaves_out_zero_probabilities():
    table = pd.DataFrame({"change_type": "cumulative", "change": "above", "n_days": 3,
                          "n_years": 2, "threshold": [0.01, 0.02, 0.05],
                          "prob": [0.1, 0.01, 0.0], "lower": [0.08, 0.0, 0.0],
                          "upper": [0.12, 0.03, 0.0], "prob_normal": [0.09, 0.005, 1e-6],
                          "prob_t": [0.1, 0.01, 1e-4]})
    fig = pr.plot_event_probabilities(table)
    observed = fig.data[0]
    assert list(observed.x) == [1.0, 2.0] and list(observed.y) == [10.0, 1.0]
    assert len(fig.data[1].x) == 3                                  # models are never zero
