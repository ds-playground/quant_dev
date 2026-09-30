"""The FastAPI app. Endpoints call `src.tools` and serialize; they compute nothing themselves."""
import dataclasses
from typing import Literal

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.tools.price_return import (analyze_cumulative, arch_lm, autocorrelation,
                                    compare_tickers, detect_streaks, distribution_summary,
                                    drawdown_table, fit_student_t, jarque_bera, latest_snapshot,
                                    ljung_box, load_ticker_config, low_probability_view,
                                    return_moments, risk_ratios, summarize_cumulative,
                                    summarize_streaks, tail_index, value_at_risk, variance_ratio,
                                    local_ticker_config, local_tickers, save_local, store,
                                    config_params, demo_tickers)
from src.tools.price_return.params import DEFAULT_CONFIG_PATH, DEMO_CONFIG_PATH

from . import cache, dashboard
from .charts import CHARTS, ChartOptions
from .schemas import ParamsIn, to_params
from .serialize import clean, figure_to_json, frame_to_records, frame_to_table, series_to_dict

app = FastAPI(title='quant_dev API', version='0.3.0',
              description='The price-return analysis behind JSON, for the dashboard.')

ChangeType = Literal['consecutive', 'cumulative']
TickerSet = Literal['demo', 'yahoo', 'local']
# Bootstrap resamples: the notebook's 1,000 by default, capped so one request stays within
# seconds (about 2 s per 1,000 on ten years of daily data).
N_BOOT = Query(1000, ge=100, le=2000)


@app.exception_handler(ImportError)
def _missing_extra(request, exc):
    """The statistics functions that need scipy raise ImportError with an install hint."""
    return JSONResponse(status_code=501, content={'detail': str(exc)})


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
    return {'status': 'ok', 'version': app.version, 'dashboard_built': dashboard.is_built()}


# The ticker lists the dashboard offers: the processed demo files, which work offline; the
# Yahoo Finance tickers analysed by the notebooks; and the live data saved to data/local.
TICKER_SETS = {'demo': DEMO_CONFIG_PATH, 'yahoo': DEFAULT_CONFIG_PATH}


def _ticker_config(ticker_set):
    """{symbol: Params} for a set, and where it came from."""
    if ticker_set == 'local':
        return local_ticker_config(), str(store.LOCAL_DIR)
    path = TICKER_SETS[ticker_set]
    return load_ticker_config(path, verbose=False), str(path) if path.exists() else 'built-in defaults'


def _entry(symbol, p, saved):
    return {'symbol': symbol, 'label': getattr(p, 'label', symbol),
            'params': clean(dataclasses.asdict(p)), 'saved': saved.get(symbol)}


@app.get('/api/tickers')
def tickers(ticker_set: TickerSet = Query('demo', alias='set')):
    """A set's tickers with labels and parameters, and, for any ticker saved in data/local, what
    is saved (rows, first and last date, last update)."""
    config, source = _ticker_config(ticker_set)
    saved = {t['symbol']: t for t in local_tickers()}
    return {'set': ticker_set, 'source': source,
            'tickers': [_entry(symbol, p, saved) for symbol, p in config.items()]}


# Yahoo symbols: letters, digits and ^ = . _ - (e.g. AAPL, ^GSPC, ES=F, EURUSD=X, BRK-B).
SYMBOL = r'^[A-Za-z0-9^=._-]{1,20}$'


def _ticker_params(symbol, ticker_set):
    """Params for any symbol in a set: its configured entry, or the config's defaults. The demo
    set has only its files, and the saved set only what is saved."""
    if ticker_set == 'demo':
        if symbol not in demo_tickers():
            raise HTTPException(status_code=422, detail=f'No demo data for {symbol!r}. Demo '
                                                        f'tickers: {", ".join(demo_tickers())}.')
        return config_params(symbol, DEMO_CONFIG_PATH)
    if ticker_set == 'local':
        config = local_ticker_config()
        if symbol not in config:
            raise HTTPException(status_code=422, detail=f'{symbol!r} is not saved. Saved: '
                                                        f'{", ".join(config) or "none"}.')
        return config[symbol]
    p = config_params(symbol, TICKER_SETS['yahoo'])
    label = p.label
    p = dataclasses.replace(p, data_source='yahoo')
    p.label = label
    return p


@app.get('/api/ticker')
def ticker(symbol: str = Query(pattern=SYMBOL), ticker_set: TickerSet = Query('yahoo', alias='set')):
    """One symbol, listed or not: its label and parameters (its own config entry, or the
    config's defaults) and what is saved. For looking up a ticker outside the configured list;
    whether Yahoo has data for it shows when its data is loaded."""
    saved = {t['symbol']: t for t in local_tickers()}
    return _entry(symbol, _ticker_params(symbol, ticker_set), saved)


@app.get('/api/local')
def local():
    """The live data saved in data/local: each symbol's rows, date range and last update."""
    return {'directory': str(store.LOCAL_DIR), 'tickers': local_tickers()}


@app.post('/api/local/{symbol:path}/update')
def update_local(symbol: str, start_date: str = Query('2016-01-01', pattern=r'^\d{4}-\d{2}-\d{2}$')):
    """Save a symbol's Yahoo data to data/local, or bring the saved file up to date. Returns what
    changed, including any saved bars Yahoo has revised. A failed download is a 422 (no such
    symbol) or 502 (Yahoo failing) and leaves the file as it was."""
    try:
        return clean(save_local(symbol, start_date=start_date))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - any download failure is reported, not a 500
        raise HTTPException(status_code=502,
                            detail=f'Download failed: {type(exc).__name__}: {exc}') from exc


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


@app.post('/api/streaks')
def streaks(body: ParamsIn):
    """`summarize_streaks`: per window, win and loss streak counts, episodes and frequencies."""
    p = to_params(body)
    df = _load(p)
    return {'summary': frame_to_table(summarize_streaks(df, detect_streaks(df, p), p))}


@app.post('/api/cumulative')
def cumulative(body: ParamsIn):
    """`summarize_cumulative`: per window, how often the compounded return cleared each threshold."""
    p = to_params(body)
    df = _load(p)
    return {'summary': frame_to_table(summarize_cumulative(analyze_cumulative(df, p), p))}


@app.post('/api/rare-events')
def rare_events(body: ParamsIn,
                n_days: list[int] = Query([3], description='holding periods, from streak_days'),
                change_type: ChangeType | None = Query(None, description='default: both')):
    """`low_probability_view` over `build_historical_analysis`: the rare-but-observed events,
    with prob between the parameters' prob_min and prob_max. The full table is cached, so moving
    the bounds or the holding period only refilters it."""
    p = to_params(body)
    unknown = sorted(set(n_days) - set(p.streak_days))
    if unknown:
        raise HTTPException(status_code=422,
                            detail=f'n_days {unknown} not in streak_days {p.streak_days}')
    _load(p)
    df_his = cache.history(p)
    view = low_probability_view(df_his, n_days, p, change_type=change_type)
    return {'n_days': n_days, 'change_type': change_type, 'prob_max': p.prob_max,
            'prob_min': p.prob_min, 'total_events': len(df_his), 'events': frame_to_table(view)}


@app.get('/api/charts')
def chart_list():
    """The chart names, each with its group and the query options it reads."""
    return {'charts': [{'name': name, 'group': group, 'options': list(options)}
                       for name, (group, options, _) in CHARTS.items()]}


@app.post('/api/charts/{name}')
def chart(body: ParamsIn,
          name: str = Path(description='a name from GET /api/charts'),
          window: int | None = Query(None, ge=1),
          top: int = Query(3, ge=1, le=10),
          n_days: int = Query(3, ge=1, le=250),
          n_boot: int = N_BOOT,
          change_type: ChangeType = 'cumulative',
          change: Literal['above', 'below'] = 'above',
          n_years: int | None = Query(None, ge=1)):
    """One `viz.plot_*` figure as Plotly JSON, identical to the notebook's."""
    if name not in CHARTS:
        raise HTTPException(status_code=404,
                            detail=f'No chart {name!r}; charts: {", ".join(CHARTS)}')
    p = to_params(body)
    _load(p)
    options = ChartOptions(window, top, n_days, n_boot, change_type, change, n_years)
    return figure_to_json(CHARTS[name][2](p, options))


def _distribution(p, r, n_days, n_boot, top):
    return {'moments': series_to_dict(return_moments(r)),
            'jarque_bera': series_to_dict(jarque_bera(r)),
            'student_t': series_to_dict(fit_student_t(r)),
            'tail_index': series_to_dict(tail_index(r)),
            'value_at_risk': frame_to_table(value_at_risk(r))}


def _dependence(p, r, n_days, n_boot, top):
    return {'autocorrelation': frame_to_table(autocorrelation(r)),
            'ljung_box': frame_to_table(ljung_box(r)),
            'variance_ratio': frame_to_table(variance_ratio(r)),
            'arch_lm': series_to_dict(arch_lm(r))}


def _drawdowns(p, r, n_days, n_boot, top):
    return {'risk_ratios': series_to_dict(risk_ratios(r, p.trade_days)),
            'drawdowns': frame_to_table(drawdown_table(r, top=top))}


def _probabilities(p, r, n_days, n_boot, top):
    return {'n_days': n_days, 'n_boot': n_boot,
            'events': frame_to_table(cache.events(p, n_days, n_boot))}


STATISTICS = {'distribution': _distribution, 'dependence': _dependence,
              'drawdowns': _drawdowns, 'probabilities': _probabilities}


@app.post('/api/statistics/{section}')
def statistics(body: ParamsIn,
               section: Literal['distribution', 'dependence', 'drawdowns', 'probabilities'],
               n_days: int = Query(3, ge=1, le=250, description='probabilities: holding period'),
               n_boot: int = N_BOOT,
               top: int = Query(5, ge=1, le=20, description='drawdowns: how many')):
    """The statistics notebook's four sections, on the daily returns of the parameters' series.
    `probabilities` returns every event (`event_probability_table`); filter by prob to show the
    rare ones, as the notebook does."""
    p = to_params(body)
    _load(p)
    return STATISTICS[section](p, cache.returns(p), n_days, n_boot, top)


def _compare(entries, drill_n_days):
    """`analyze_ticker` for each (key, set, Params), then `compare_tickers`. A ticker that fails
    is listed under `failed` rather than failing the request."""
    results, tickers, failed = {}, [], []
    for key, ticker_set, p in entries:
        try:
            r = results[key] = cache.ticker_result(p, drill_n_days)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not sink the others
            failed.append({'symbol': p.ticker, 'set': ticker_set,
                           'error': f'{type(exc).__name__}: {exc}'})
            continue
        tickers.append({'symbol': p.ticker, 'set': ticker_set, 'label': p.label,
                        'rows': len(r['df']), 'start': clean(r['df']['date'].iloc[0]),
                        'end': clean(r['df']['date'].iloc[-1]), 'drill': frame_to_table(r['drill'])})
    if not results:
        raise HTTPException(status_code=502, detail={'message': 'Every ticker failed',
                                                     'failed': failed})
    streak_table, dist_table = compare_tickers(results, ratio_window=drill_n_days)
    return {'drill_n_days': drill_n_days, 'tickers': tickers,
            'streaks': frame_to_table(streak_table), 'distribution': frame_to_table(dist_table),
            'failed': failed}


@app.get('/api/multi-ticker')
def multi_ticker(ticker_set: TickerSet = Query('demo', alias='set'),
                 drill_n_days: int = Query(3, ge=1, le=250)):
    """`rare_case_run` over every ticker of a set (the streak ratio uses `drill_n_days`, as the
    notebook does)."""
    config, _ = _ticker_config(ticker_set)
    if not config:
        raise HTTPException(status_code=422, detail=f'The {ticker_set} set has no tickers yet.')
    return {'set': ticker_set,
            **_compare([(symbol, ticker_set, p) for symbol, p in config.items()], drill_n_days)}


class Pick(BaseModel):
    symbol: str = Field(pattern=SYMBOL)
    set: TickerSet


class Selection(BaseModel):
    tickers: list[Pick] = Field(min_length=1, max_length=30)
    drill_n_days: int = Field(3, ge=1, le=250)


SOURCE_NAMES = {'demo': 'demo', 'yahoo': 'live', 'local': 'saved'}


@app.post('/api/multi-ticker')
def multi_ticker_selection(body: Selection):
    """`rare_case_run` over a chosen list, which may mix sets: configured Yahoo tickers (live),
    saved CSVs, demo files. When sources are mixed, each row's name says its source, so the same
    symbol live and saved can be compared."""
    picks = list(dict.fromkeys((pick.symbol, pick.set) for pick in body.tickers))
    mixed = len({ticker_set for _, ticker_set in picks}) > 1
    entries = []
    for symbol, ticker_set in picks:
        p = _ticker_params(symbol, ticker_set)
        source = SOURCE_NAMES[ticker_set]
        if mixed and source not in p.label.lower():         # demo labels already say "(demo)"
            label = f'{p.label} ({source})'
            p = dataclasses.replace(p)
            p.label = label
        entries.append((f'{ticker_set}:{symbol}', ticker_set, p))
    return {'set': None, **_compare(entries, body.drill_n_days)}


# Last, so that every API route (and /docs) matches first.
dashboard.add_dashboard_routes(app)
