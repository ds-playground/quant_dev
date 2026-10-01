// Typed client for the API in src/api. The types mirror the JSON the endpoints return; the
// API computes everything, so this file only fetches.

/** The Params dataclass (src/tools/price_return/params.py), as JSON. Every POST body is one.
 *  tests/test_api.py checks that these field names match the dataclass. */
export interface Params {
  data_source: 'demo' | 'yahoo' | 'local' | 'simulated';
  ticker: string;
  start_date: string;
  end_date: string;
  random_seed: number;
  sim_drift: number;
  sim_vol: number;
  sim_start_price: number;
  trade_days: number;
  win_threshold: number;
  loss_threshold: number;
  windows: number[];
  cum_thresholds: number[];
  roll_windows: number[] | null;
  chart_windows: number[];
  timeline_window: number;
  return_thresholds: number[];
  lookback_years: number[];
  streak_days: number[];
  prob_max: number;
  prob_min: number;
}

export type TickerSet = 'demo' | 'yahoo' | 'local';
export type Cell = string | number | boolean | null;

/** A DataFrame: column names in order, and one record per row. */
export interface Table {
  columns: string[];
  records: Record<string, Cell>[];
}

/** What data/local holds for a symbol. */
export interface Saved {
  symbol: string;
  rows: number;
  first: string;
  last: string;
  updated: string | null;
}

export interface Ticker {
  symbol: string;
  label: string;
  params: Params;
  saved: Saved | null;
}

/** The outcome of saving or updating a symbol in data/local. */
export interface SaveResult {
  symbol: string;
  file: string;
  created: boolean;
  rows: number;
  added: number;
  first: string;
  last: string;
  revised: { date: string; column: string; saved: number; new: number }[];
}

/** The outcome of saving the whole default list: each symbol's result, and each failure. */
export interface SaveAllResult {
  saved: SaveResult[];
  failed: { symbol: string; error: string }[];
}

export interface TickerList {
  set: TickerSet;
  source: string;
  tickers: Ticker[];
}

export interface Overview {
  ticker: string;
  data_source: string;
  rows: number;
  start: string;
  end: string;
  last_price: number;
  snapshot: {
    annualized_return: number | null;
    annualized_volatility: number | null;
    daily_avg_return: number | null;
    daily_std_dev: number | null;
  };
  distribution: Record<string, Cell>;
}

export interface Summary {
  summary: Table;
}

export interface RareEvents {
  n_days: number[];
  change_type: ChangeType | null;
  prob_max: number;
  prob_min: number;
  total_events: number;
  events: Table;
}

export type ChangeType = 'consecutive' | 'cumulative';

export interface ChartInfo {
  name: string;
  group: 'analysis' | 'statistics';
  options: string[];
}

/** Plotly figure JSON, as `fig.to_json()` writes it. */
export interface Figure {
  data: object[];
  layout: Record<string, unknown>;
}

export interface MultiTicker {
  set: TickerSet | null;              // null for a chosen selection
  drill_n_days: number;
  tickers: { symbol: string; set: TickerSet; label: string; rows: number; start: string; end: string;
             drill: Table }[];
  streaks: Table;
  distribution: Table;
  failed: { symbol: string; set: TickerSet; error: string }[];
}

/** One ticker of one set, as chosen for the multi-ticker comparison. */
export interface Pick {
  symbol: string;
  set: TickerSet;
}

type Query = Record<string, string | number | (string | number)[] | null | undefined>;

/** An error response from the API: its status and the server's explanation. */
export class ApiError extends Error {
  // `raw` is the server's detail as sent, for errors that carry more than a message.
  constructor(public status: number, public detail: string, public raw?: unknown) {
    super(`${status}: ${detail}`);
  }
}

function queryString(query: Query = {}): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === null || value === undefined) continue;
    for (const v of Array.isArray(value) ? value : [value]) search.append(key, String(v));
  }
  const text = search.toString();
  return text ? `?${text}` : '';
}

/** FastAPI's validation errors are a list of {loc, msg}; everything else is a string. */
function describe(detail: unknown): string {
  if (Array.isArray(detail)) {
    return detail.map((d) => `${d.msg}: ${(d.loc ?? []).slice(-1)[0] ?? ''}`).join('; ');
  }
  if (detail && typeof detail === 'object' && 'message' in detail) return String(detail.message);
  return String(detail);
}

async function request<T>(method: 'GET' | 'POST', path: string, body?: unknown,
                          query?: Query): Promise<T> {
  const response = await fetch(`/api${path}${queryString(query)}`, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let detail: unknown = response.statusText;
    try {
      detail = (await response.json()).detail;
    } catch {
      // not JSON: keep the status text
    }
    throw new ApiError(response.status, describe(detail), detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; version: string }>('GET', '/health'),
  tickers: (set: TickerSet) => request<TickerList>('GET', '/tickers', undefined, { set }),
  overview: (p: Params) => request<Overview>('POST', '/overview', p),
  streaks: (p: Params) => request<Summary>('POST', '/streaks', p),
  cumulative: (p: Params) => request<Summary>('POST', '/cumulative', p),
  rareEvents: (p: Params, nDays: number[], changeType: ChangeType | null) =>
    request<RareEvents>('POST', '/rare-events', p, { n_days: nDays, change_type: changeType }),
  charts: () => request<{ charts: ChartInfo[] }>('GET', '/charts'),
  chart: (p: Params, name: string, options: Query = {}) =>
    request<Figure>('POST', `/charts/${name}`, p, options),
  statistics: <T = Record<string, unknown>>(p: Params, section: string, options: Query = {}) =>
    request<T>('POST', `/statistics/${section}`, p, options),
  local: () => request<{ directory: string; tickers: Saved[] }>('GET', '/local'),
  saveLocal: (symbol: string, startDate = '2016-01-01') =>
    request<SaveResult>('POST', `/local/${encodeURIComponent(symbol)}/update`, undefined,
                        { start_date: startDate }),
  saveAll: (startDate = '2016-01-01') =>
    request<SaveAllResult>('POST', '/local/update-all', undefined, { start_date: startDate }),
  ticker: (symbol: string, set: TickerSet = 'yahoo') =>
    request<Ticker>('GET', '/ticker', undefined, { symbol, set }),
  compare: (tickers: Pick[], drillNDays = 3) =>
    request<MultiTicker>('POST', '/multi-ticker', { tickers, drill_n_days: drillNDays }),
  multiTicker: (set: TickerSet, drillNDays = 3) =>
    request<MultiTicker>('GET', '/multi-ticker', undefined, { set, drill_n_days: drillNDays }),
};
