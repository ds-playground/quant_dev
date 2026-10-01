"""Offline tests for the synthetic series (src/tools/price_return/synthetic.py), the `synthetic`
data source, and the Yahoo data path checked against the same bars.

The series exist so that the analysis can be exercised offline on data shaped like a market's, so
the tests check those properties (fat tails, volatility clustering, a sell-off, one negative
settle) rather than exact numbers, which depend on NumPy's random stream.
"""
import numpy as np
import pandas as pd
import pytest

from src.tools import price_return as pr
from src.tools.price_return.params import SYNTHETIC_CONFIG_PATH
from src.tools.price_return.synthetic import END, PROFILES

ALL = {"start_date": "2000-01-01", "end_date": "2100-01-01"}


def returns(ticker):
    """Daily returns from the closes, leaving out any that span a non-positive close."""
    close = pr.synthetic_bars(ticker)["Close"]
    return close.pct_change().where((close > 0) & (close.shift() > 0)).dropna()


@pytest.mark.parametrize("ticker", pr.synthetic_tickers())
def test_bars_are_well_formed_business_days_and_the_same_every_time(ticker):
    bars = pr.synthetic_bars(ticker)
    assert list(bars.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert bars.index.name == "Date" and bars.index.is_unique and bars.index.is_monotonic_increasing
    assert bars.index[0] == pd.Timestamp(PROFILES[ticker].start)
    assert bars.index[-1] == pd.Timestamp(END) and (bars.index.dayofweek < 5).all()
    assert not bars.isna().any().any() and (bars["Volume"] > 0).all()
    assert (bars["Low"] <= bars[["Open", "Close"]].min(axis=1)).all()
    assert (bars["High"] >= bars[["Open", "Close"]].max(axis=1)).all()
    bars.iloc[0, 0] = -1                                     # a caller's edit stays its own
    pd.testing.assert_frame_equal(pr.synthetic_bars(ticker), pr.synthetic_bars(ticker))
    assert pr.synthetic_bars(ticker).iloc[0, 0] != -1


def test_each_ticker_is_its_own_series():
    """Each ticker has its own seed, so their returns are unrelated, not one path rescaled."""
    frame = pd.DataFrame({t: returns(t) for t in pr.synthetic_tickers()}).dropna()
    corr = frame.corr().to_numpy()
    assert len(frame) > 900 and np.abs(corr[np.triu_indices_from(corr, 1)]).max() < 0.2


@pytest.mark.parametrize("ticker", pr.synthetic_tickers())
def test_returns_look_like_a_market_s(ticker):
    r = returns(ticker)
    assert 0.6 < r.std() * 100 / PROFILES[ticker].vol < 1.4       # near the profile's volatility
    assert r.kurt() > 2                                           # fat tails (0 for a normal)
    size = (r - r.mean()).abs()
    assert size.autocorr(1) > 0.1                                 # big moves follow big moves
    assert abs(r.autocorr(1)) < 0.1                               # direction does not persist


def test_the_index_sells_off_in_its_stress_period():
    close = pr.synthetic_bars("SYN-INDEX")["Close"]
    first, last, *_ = PROFILES["SYN-INDEX"].stress[0]
    peak = close.loc[:first].max()
    assert close.loc[first:last].min() / peak - 1 < -0.15


@pytest.mark.parametrize("ticker", pr.synthetic_tickers())
def test_synthetic_source_gives_the_closes_and_their_returns(ticker):
    bars = pr.synthetic_bars(ticker)
    df = pr.load_price_data(pr.Params(data_source="synthetic", ticker=ticker, **ALL), verbose=False)
    close = bars["Close"].to_numpy()
    expected = (close[1:] / close[:-1] - 1) * 100
    keep = (close[1:] > 0) & (close[:-1] > 0)
    assert list(df.columns) == ["date", "price", "return_pct"]
    assert (df["date"].to_numpy() == bars.index.to_numpy()[1:][keep]).all()
    assert (df["price"].to_numpy() == close[1:][keep]).all()
    assert np.allclose(df["return_pct"], expected[keep], rtol=1e-12, atol=0)


def test_dates_are_filtered_with_start_inclusive_and_end_exclusive():
    p = pr.Params(data_source="synthetic", ticker="SYN-INDEX", start_date="2020-03-02",
                  end_date="2020-03-09")
    df = pr.load_price_data(p, verbose=False)
    # 2020-03-02 is the first close loaded, so it has no return; 2020-03-09 is excluded.
    assert [d.strftime("%Y-%m-%d") for d in df["date"]] == [
        "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06"]


def test_the_negative_settle_drops_the_two_returns_spanning_it(capsys):
    assert pr.synthetic_bars("SYN-OIL").loc["2020-04-20", "Close"] < 0
    df = pr.load_price_data(pr.Params(data_source="synthetic", ticker="SYN-OIL", **ALL))
    dates = set(df["date"].dt.strftime("%Y-%m-%d"))
    assert "2020-04-20" not in dates and "2020-04-21" not in dates and "2020-04-22" in dates
    assert (df["price"] > 0).all()
    assert "SYN-OIL has 1 non-positive price" in capsys.readouterr().out


def test_unknown_ticker_names_the_synthetic_tickers():
    with pytest.raises(ValueError, match="Synthetic tickers: SYN-INDEX, SYN-TECH"):
        pr.load_price_data(pr.Params(data_source="synthetic", ticker="ES=F"), verbose=False)


def test_a_range_outside_the_series_says_what_it_covers():
    p = pr.Params(data_source="synthetic", ticker="SYN-LEV", start_date="2016-01-01",
                  end_date="2022-01-01")
    with pytest.raises(ValueError, match="covers 2022-08-10 to 2026-09-30"):
        pr.load_price_data(p, verbose=False)


def test_synthetic_config_lists_every_synthetic_ticker_on_the_synthetic_source():
    config = pr.load_ticker_config(SYNTHETIC_CONFIG_PATH, verbose=False)
    assert list(config) == pr.synthetic_tickers()
    assert all(p.data_source == "synthetic" and p.label.startswith("Synthetic")
               for p in config.values())


def test_the_yahoo_path_reads_what_the_synthetic_path_reads(yahoo):
    """The Yahoo branch of load_price_data, offline: the `yahoo` fixture serves SYN-OIL's bars as
    CL=F in yfinance's shape, with an Adj Close that differs from Close. The Yahoo path must give
    the synthetic path's frame for the same bars (so it reads Close, flattens yfinance's columns
    and ends before end_date), and an unknown symbol must fail clearly."""
    dates = {"start_date": "2019-01-01", "end_date": "2021-01-01"}
    live = pr.load_price_data(pr.Params(data_source="yahoo", ticker="CL=F", **dates), verbose=False)
    synthetic = pr.load_price_data(pr.Params(data_source="synthetic", ticker="SYN-OIL", **dates),
                                   verbose=False)
    pd.testing.assert_frame_equal(live, synthetic)
    with pytest.raises(ValueError, match="No price data for ticker 'ES=F'"):
        pr.load_price_data(pr.Params(data_source="yahoo", ticker="ES=F", **dates), verbose=False)
