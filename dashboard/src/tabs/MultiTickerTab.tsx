import { useQueries, useQuery } from '@tanstack/react-query';
import type { Dispatch, SetStateAction } from 'react';

import { api, type Pick, type TickerSet } from '../api';
import { DataTable } from '../components/DataTable';
import { QueryState } from '../components/QueryState';
import { isoDate, pctCell } from '../format';

const GROUPS: { set: TickerSet; title: string }[] = [
  { set: 'yahoo', title: 'Default list · Yahoo Finance (live)' },
  { set: 'local', title: 'Saved CSV (offline)' },
  { set: 'demo', title: 'Demo data (offline)' },
];
const MAX = 30;                               // the API's limit per comparison
const SOURCE: Record<TickerSet, string> = { yahoo: 'live', local: 'saved', demo: 'demo' };
const key = (p: Pick) => `${p.set}:${p.symbol}`;

/** rare_case_run over a chosen list: any mix of the configured Yahoo tickers (and those added in
 *  the ticker picker), saved CSVs and demo files, each with its configured thresholds. Runs on
 *  request: the first run downloads (Yahoo) and analyses every ticker; later runs are cached. */
/** What the tab keeps while other tabs are open: held by App, so it lasts until a page reload. */
export interface MultiTickerState {
  chosen: Set<string>;                       // `${set}:${symbol}` keys; empty on a fresh page
  drill: number;
  run: { picks: Pick[]; drill: number } | null;
}

export const initialMultiTicker: MultiTickerState = { chosen: new Set(), drill: 3, run: null };

export function MultiTickerTab({ state, setState, addedSymbols = [] }: {
  state: MultiTickerState;
  setState: Dispatch<SetStateAction<MultiTickerState>>;
  addedSymbols?: string[];
}) {
  const { drill, run } = state;
  const setDrill = (d: number) => setState((s) => ({ ...s, drill: d }));
  const setRun = (r: MultiTickerState['run']) => setState((s) => ({ ...s, run: r }));
  const lists = useQueries({
    queries: GROUPS.map((g) => ({ queryKey: ['tickers', g.set], queryFn: () => api.tickers(g.set) })),
  });
  const options: Record<TickerSet, { symbol: string; label: string }[]> = { yahoo: [], local: [], demo: [] };
  GROUPS.forEach((g, i) => {
    const listed = lists[i].data?.tickers.map((t) => ({ symbol: t.symbol, label: t.label })) ?? [];
    options[g.set] = g.set === 'yahoo'
      ? [...listed, ...addedSymbols.filter((a) => !listed.some((t) => t.symbol === a))
                                   .map((a) => ({ symbol: a, label: a }))]
      : listed;
  });

  // Nothing is chosen on a fresh page; the choice then lasts across tabs until a reload.
  const selection = state.chosen;
  const setChosen = (update: (c: Set<string>) => Set<string>) =>
    setState((s) => ({ ...s, chosen: update(s.chosen) }));
  const picks: Pick[] = GROUPS.flatMap((g) => options[g.set]
    .filter((t) => selection.has(key({ symbol: t.symbol, set: g.set })))
    .map((t) => ({ symbol: t.symbol, set: g.set })));

  const toggle = (k: string, on: boolean) =>
    setChosen((c) => { const next = new Set(c); if (on) next.add(k); else next.delete(k); return next; });
  const setGroup = (set: TickerSet, on: boolean) =>
    setChosen((c) => {
      const next = new Set(c);
      for (const t of options[set]) {
        const k = key({ symbol: t.symbol, set });
        if (on) next.add(k); else next.delete(k);
      }
      return next;
    });

  const query = useQuery({
    queryKey: ['compare', run],
    queryFn: () => api.compare(run!.picks, run!.drill),
    enabled: run !== null,
  });
  const current = run;
  const tooMany = picks.length > MAX;

  return (
    <div className="stack">
      <section className="card">
        <h2>Tickers to compare <span className="h2-note">({picks.length} chosen{tooMany ? `; at most ${MAX}` : ''})</span></h2>
        <div className="pick-groups">
          {GROUPS.map((g, i) => (
            <fieldset key={g.set} className="pick-group">
              <legend>{g.title}</legend>
              <div className="pick-actions">
                <button type="button" className="link" onClick={() => setGroup(g.set, true)}
                        disabled={!options[g.set].length}>All</button>
                <button type="button" className="link" onClick={() => setGroup(g.set, false)}
                        disabled={!options[g.set].length}>None</button>
              </div>
              {lists[i].isPending ? <p className="note">Loading…</p>
                : !options[g.set].length ? (
                  <p className="note">
                    {g.set === 'local' ? 'Nothing saved yet: use Save to CSV on a Yahoo ticker.' : 'None.'}
                  </p>
                ) : (
                  <ul className="pick-list">
                    {options[g.set].map((t) => {
                      const k = key({ symbol: t.symbol, set: g.set });
                      return (
                        <li key={k}>
                          <label className="check">
                            <input type="checkbox" checked={selection.has(k)}
                                   onChange={(e) => toggle(k, e.target.checked)} />
                            <span>{t.label}{t.label === t.symbol ? '' : <span className="muted"> {t.symbol}</span>}</span>
                          </label>
                        </li>
                      );
                    })}
                  </ul>
                )}
            </fieldset>
          ))}
        </div>
        <div className="toolbar inset">
          <label>
            <span>Holding period for the drill tables and ratio</span>
            <select value={drill} onChange={(e) => setDrill(Number(e.target.value))}>
              {[1, 2, 3, 5].map((d) => <option key={d} value={d}>{d} {d === 1 ? 'day' : 'days'}</option>)}
            </select>
          </label>
          <div className="actions">
            <button type="button" className="primary" disabled={query.isFetching || !picks.length || tooMany}
                    onClick={() => setRun({ picks, drill })}>
              {query.isFetching ? `Analysing ${run?.picks.length} tickers…`
                : `Compare ${picks.length} ticker${picks.length === 1 ? '' : 's'}`}
            </button>
          </div>
        </div>
      </section>

      {current === null ? (
        <section className="card">
          <p className="lede">
            Runs the whole analysis for each chosen ticker, with its own thresholds from its config,
            and compares them. Choose from any of the three sources; the same symbol live and saved
            is compared side by side. The first run takes a few seconds per ticker (Yahoo tickers
            download first); later runs are cached.
          </p>
        </section>
      ) : (
        <QueryState query={query} what="every ticker">
          {(data) => (
            <>
              {data.failed.length ? (
                <section className="card" role="alert">
                  <h2>{data.failed.length} of {data.failed.length + data.tickers.length} tickers failed</h2>
                  <ul className="failures">
                    {data.failed.map((f) => (
                      <li key={`${f.set}:${f.symbol}`}><strong>{f.symbol}</strong> ({SOURCE[f.set]}): {f.error}</li>
                    ))}
                  </ul>
                </section>
              ) : null}
              <section className="card">
                <h2>Streak frequency (% of complete windows), win / loss</h2>
                <p className="lede">
                  Each ticker uses its own thresholds, so the rows are comparable across asset classes.
                  The ratio is win over loss frequency for {data.drill_n_days}-day streaks: above 1,
                  runs up outnumber runs down.
                </p>
                <DataTable table={data.streaks} />
              </section>
              <section className="card">
                <h2>Return distribution</h2>
                <p className="lede">
                  Daily returns in %. Streaks count threshold-clearing days, so they track the median
                  rather than the mean; skew separates the two.
                </p>
                <DataTable table={data.distribution} />
              </section>
              <section className="card">
                <h2>Rare events per ticker, {data.drill_n_days}-day holding period</h2>
                {data.tickers.map((t) => (
                  <details key={`${t.set}:${t.symbol}`}>
                    <summary>
                      <strong>{t.label}</strong> ({t.symbol}): {t.drill.records.length} rare events;
                      {' '}{t.rows.toLocaleString('en-GB')} days, {isoDate(t.start)} to {isoDate(t.end)}
                    </summary>
                    <DataTable table={t.drill}
                               columns={['change_type', 'change', 'threshold', 'n_years', 'count', 'episodes', 'prob', 'last_occurred']}
                               labels={{ change_type: 'type', change: 'direction', threshold: 'move',
                                         n_years: 'lookback (y)', last_occurred: 'last seen' }}
                               formats={{ threshold: pctCell(2), prob: pctCell(3) }} />
                  </details>
                ))}
              </section>
            </>
          )}
        </QueryState>
      )}
    </div>
  );
}
