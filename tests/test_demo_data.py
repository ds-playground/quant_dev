"""Offline tests for the processed demo data in data/demo, the `demo` data source, and the Yahoo
data path checked against the same files."""
import numpy as np
import pandas as pd
import pytest

from src.tools import price_return as pr
from src.tools.price_return.data import DEMO_DIR
from src.tools.price_return.params import DEMO_CONFIG_PATH

MANIFEST = pd.read_csv(DEMO_DIR / "manifest.csv")
ALL = {"start_date": "2000-01-01", "end_date": "2100-01-01"}


def read_file(ticker):
    """The file as written, read independently of load_price_data."""
    return pd.read_csv(DEMO_DIR / f"{ticker}_demo.csv", parse_dates=["Date"],
                       float_precision="round_trip")


def test_every_file_is_in_the_manifest_and_matches_it():
    assert sorted(MANIFEST["file"].str[:-len("_demo.csv")]) == pr.demo_tickers()
    for row in MANIFEST.itertuples():
        bars = read_file(row.file[:-len("_demo.csv")])
        assert list(bars.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert len(bars) == row.rows
        assert str(bars["Date"].iloc[0].date()) == row.first
        assert str(bars["Date"].iloc[-1].date()) == row.last
        assert bars["Date"].is_monotonic_increasing and bars["Date"].is_unique
        assert not bars.isna().any().any()
        assert (bars["Low"] <= bars[["Open", "Close"]].min(axis=1)).all()
        assert (bars["High"] >= bars[["Open", "Close"]].max(axis=1)).all()


@pytest.mark.parametrize("ticker", pr.demo_tickers())
def test_demo_source_gives_the_file_closes_and_their_returns(ticker):
    bars = read_file(ticker)
    df = pr.load_price_data(pr.Params(data_source="demo", ticker=ticker, **ALL), verbose=False)
    # Written out: each close against the one before, dropping any return that spans a
    # non-positive close (CL's 2020-04-20), and the first day, which has no return.
    close = bars["Close"].to_numpy()
    returns = (close[1:] / close[:-1] - 1) * 100
    keep = (close[1:] > 0) & (close[:-1] > 0)
    assert list(df.columns) == ["date", "price", "return_pct"]
    assert (df["date"].to_numpy() == bars["Date"].to_numpy()[1:][keep]).all()
    assert (df["price"].to_numpy() == close[1:][keep]).all()        # exactly the file's closes
    assert np.allclose(df["return_pct"], returns[keep], rtol=1e-12, atol=0)
    expected_rows = MANIFEST.set_index("file").loc[f"{ticker}_demo.csv", "rows"] - 1
    assert len(df) == expected_rows - (2 if ticker == "CL" else 0)


def test_dates_are_filtered_with_start_inclusive_and_end_exclusive():
    p = pr.Params(data_source="demo", ticker="SPX", start_date="2020-03-02",
                  end_date="2020-03-09")
    df = pr.load_price_data(p, verbose=False)
    # 2020-03-02 is the first close loaded, so it has no return; 2020-03-09 is excluded.
    assert [d.strftime("%Y-%m-%d") for d in df["date"]] == [
        "2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06"]


def test_cl_negative_close_drops_the_two_returns_spanning_it(capsys):
    df = pr.load_price_data(pr.Params(data_source="demo", ticker="CL", **ALL))
    assert "2020-04-20" not in set(df["date"].dt.strftime("%Y-%m-%d"))
    assert "2020-04-21" not in set(df["date"].dt.strftime("%Y-%m-%d"))
    assert (df["price"] > 0).all()
    assert "CL" in capsys.readouterr().out


def test_unknown_ticker_names_the_demo_tickers():
    with pytest.raises(ValueError, match="Demo tickers: AAPL, CL, EURUSD"):
        pr.load_price_data(pr.Params(data_source="demo", ticker="ES=F"), verbose=False)


def test_a_range_outside_the_file_says_what_it_covers():
    p = pr.Params(data_source="demo", ticker="TSLL", start_date="2016-01-01",
                  end_date="2022-01-01")
    with pytest.raises(ValueError, match="covers 2022-08-09 to"):
        pr.load_price_data(p, verbose=False)


def test_demo_config_lists_every_demo_file_on_the_demo_source():
    config = pr.load_ticker_config(DEMO_CONFIG_PATH, verbose=False)
    assert sorted(config) == pr.demo_tickers()
    assert list(config)[0] == "SPX"
    assert all(p.data_source == "demo" and p.label.endswith(")") for p in config.values())


def test_readme_says_the_data_is_processed_and_not_for_trading():
    readme = (DEMO_DIR / "README.md").read_text()
    assert "not market data" in readme
    assert "education" in readme and "trading" in readme
    for row in MANIFEST.itertuples():
        assert row.file in readme and row.yahoo_symbol in readme


def test_the_yahoo_path_reads_what_the_demo_path_reads(yahoo):
    """The Yahoo branch of load_price_data, offline: the `yahoo` fixture serves CL_demo.csv as CL=F
    in yfinance's shape, with an Adj Close that differs from Close. The Yahoo path must give the
    demo path's frame for the same file (so it reads Close, flattens yfinance's columns and ends
    before end_date), and an unknown symbol must fail clearly."""
    dates = {"start_date": "2019-01-01", "end_date": "2021-01-01"}
    live = pr.load_price_data(pr.Params(data_source="yahoo", ticker="CL=F", **dates), verbose=False)
    demo = pr.load_price_data(pr.Params(data_source="demo", ticker="CL", **dates), verbose=False)
    pd.testing.assert_frame_equal(live, demo)
    with pytest.raises(ValueError, match="No price data for ticker 'ES=F'"):
        pr.load_price_data(pr.Params(data_source="yahoo", ticker="ES=F", **dates), verbose=False)
