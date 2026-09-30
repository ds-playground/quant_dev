"""The FastAPI app. Endpoints call `src.tools` and serialize; they compute nothing themselves."""
import dataclasses
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from src.tools.price_return import distribution_summary, latest_snapshot, load_ticker_config
from src.tools.price_return.params import DEFAULT_CONFIG_PATH, DEMO_CONFIG_PATH

from . import cache
from .schemas import ParamsIn, to_params
from .serialize import clean, frame_to_records, series_to_dict

app = FastAPI(title='quant_dev API', version='0.1.0',
              description='The price-return analysis behind JSON, for the dashboard.')


def _load(p):
    """Prices plus rolling statistics, with data-source failures turned into HTTP errors.

    `load_price_data` raises ValueError for requests that cannot succeed (an unknown ticker, an
    unknown data source): 422. Anything else is the data source failing (network, Yahoo): 502.
    """
    try:
        return cache.prepared_data(p)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - any data-source failure is reported, not a 500
        raise HTTPException(status_code=502,
                            detail=f'Data source failed: {type(exc).__name__}: {exc}') from exc


@app.get('/api/health')
def health():
    return {'status': 'ok', 'version': app.version}


# The ticker lists the dashboard offers: the processed demo files, which work offline, or the
# Yahoo Finance tickers analysed by the notebooks.
TICKER_SETS = {'demo': DEMO_CONFIG_PATH, 'yahoo': DEFAULT_CONFIG_PATH}


@app.get('/api/tickers')
def tickers(ticker_set: Literal['demo', 'yahoo'] = Query('demo', alias='set')):
    """The tickers of a config (demo_tickers.yaml or tickers.yaml), with labels and parameters."""
    path = TICKER_SETS[ticker_set]
    config = load_ticker_config(path, verbose=False)
    return {
        'set': ticker_set,
        'source': str(path) if path.exists() else 'built-in defaults',
        'tickers': [{'symbol': symbol, 'label': getattr(p, 'label', symbol),
                     'params': clean(dataclasses.asdict(p))} for symbol, p in config.items()],
    }


@app.post('/api/overview')
def overview(body: ParamsIn):
    """The series loaded, the latest rolling snapshot, and the return-distribution summary."""
    p = to_params(body)
    df = _load(p)
    return {
        'ticker': p.ticker,
        'data_source': p.data_source,
        'rows': len(df),
        'start': clean(df['date'].iloc[0]),
        'end': clean(df['date'].iloc[-1]),
        'last_price': clean(df['price'].iloc[-1]),
        'snapshot': series_to_dict(latest_snapshot(df, p)),
        'distribution': frame_to_records(distribution_summary(df, p))[0],
    }
