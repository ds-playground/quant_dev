import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';

import { api, type TickerSet } from '../api';
import { DataTable } from '../components/DataTable';
import { QueryState } from '../components/QueryState';
import { isoDate, pctCell } from '../format';

const SET_NAMES: Record<TickerSet, string> = { demo: 'demo', yahoo: 'Yahoo Finance', local: 'saved' };

/** rare_case_run: every ticker of the selected set, each with its configured thresholds, then
 *  the cross-ticker streak and distribution tables. Runs on request: the first run downloads
 *  (Yahoo) and analyses every ticker; later runs are cached. */
export function MultiTickerTab({ tickerSet }: { tickerSet: TickerSet }) {
  const [drill, setDrill] = useState(3);
  const [run, setRun] = useState<{ set: TickerSet; drill: number } | null>(null);
  const current = run && run.set === tickerSet ? run : null;
  const query = useQuery({
    queryKey: ['multi-ticker', current],
    queryFn: () => api.multiTicker(current!.set, current!.drill),
    enabled: current !== null,
  });

  return (
    <div className="stack">
      <div className="toolbar" role="group" aria-label="Multi-ticker run">
        <label>
          <span>Holding period for the drill tables and ratio</span>
          <select value={drill} onChange={(e) => setDrill(Number(e.target.value))}>
            {[1, 2, 3, 5].map((d) => <option key={d} value={d}>{d} {d === 1 ? 'day' : 'days'}</option>)}
          </select>
        </label>
        <div className="actions">
          <button type="button" className="primary" disabled={query.isFetching}
                  onClick={() => setRun({ set: tickerSet, drill })}>
            {query.isFetching ? `Analysing every ${SET_NAMES[tickerSet]} ticker…`
              : `Run every ${SET_NAMES[tickerSet]} ticker`}
          </button>
        </div>
      </div>

      {current === null ? (
        <section className="card">
          <p className="lede">
            Runs the whole analysis for every ticker in the {SET_NAMES[tickerSet]} set, each with its
            own thresholds from its config, and compares them. The first run takes a few seconds per
            ticker; later runs are cached. The Data control above picks the set.
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
                    {data.failed.map((f) => <li key={f.symbol}><strong>{f.symbol}</strong>: {f.error}</li>)}
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
                  <details key={t.symbol}>
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
