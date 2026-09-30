"""The FastAPI app. Endpoints call `src.tools` and serialize; they compute nothing themselves."""
import dataclasses
from typing import Literal

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from src.tools.price_return import (analyze_cumulative, arch_lm, autocorrelation,
                                    compare_tickers, detect_streaks, distribution_summary,
                                    drawdown_table, fit_student_t, jarque_bera, latest_snapshot,
                                    ljung_box, load_ticker_config, low_probability_view,
                                    return_moments, risk_ratios, summarize_cumulative,
                                    summarize_streaks, tail_index, value_at_risk, variance_ratio)
from src.tools.price_return.params import DEFAULT_CONFIG_PATH, DEMO_CONFIG_PATH

from . import cache
from .charts import CHARTS, ChartOptions
from .schemas import ParamsIn, to_params
from .serialize import clean, figure_to_json, frame_to_records, frame_to_table, series_to_dict

app = FastAPI(title='quant_dev API', version='0.2.0',
              description='The price-return analysis behind JSON, for the dashboard.')

ChangeType = Literal['consecutive', 'cumulative']
TickerSet = Literal['demo', 'yahoo']
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
    return {'status': 'ok', 'version': app.version}


# The ticker lists the dashboard offers: the processed demo files, which work offline, or the
# Yahoo Finance tickers analysed by the notebooks.
TICKER_SETS = {'demo': DEMO_CONFIG_PATH, 'yahoo': DEFAULT_CONFIG_PATH}


@app.get('/api/tickers')
def tickers(ticker_set: TickerSet = Query('demo', alias='set')):
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


@app.get('/api/multi-ticker')
def multi_ticker(ticker_set: TickerSet = Query('demo', alias='set'),
                 drill_n_days: int = Query(3, ge=1, le=250)):
    """`rare_case_run`: `analyze_ticker` for every ticker of a set, then `compare_tickers`
    (the streak ratio uses `drill_n_days`, as the notebook does). A ticker that fails is listed
    under `failed` rather than failing the request."""
    config = load_ticker_config(TICKER_SETS[ticker_set], verbose=False)
    results, tickers, failed = {}, [], []
    for symbol, p in config.items():
        try:
            r = results[symbol] = cache.ticker_result(p, drill_n_days)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not sink the others
            failed.append({'symbol': symbol, 'error': f'{type(exc).__name__}: {exc}'})
            continue
        tickers.append({'symbol': symbol, 'label': p.label, 'rows': len(r['df']),
                        'start': clean(r['df']['date'].iloc[0]),
                        'end': clean(r['df']['date'].iloc[-1]),
                        'drill': frame_to_table(r['drill'])})
    if not results:
        raise HTTPException(status_code=502, detail={'message': 'Every ticker failed',
                                                     'failed': failed})
    streak_table, dist_table = compare_tickers(results, ratio_window=drill_n_days)
    return {'set': ticker_set, 'drill_n_days': drill_n_days, 'tickers': tickers,
            'streaks': frame_to_table(streak_table), 'distribution': frame_to_table(dist_table),
            'failed': failed}
