import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { useState } from 'react';

import { api, type ChangeType, type Params } from '../api';
import { DataTable } from '../components/DataTable';
import { QueryState } from '../components/QueryState';
import { isoDate, pctCell } from '../format';
import { useDebounced } from '../useDebounced';

const COLUMNS = ['change_type', 'change', 'threshold', 'n_years', 'n_windows', 'count', 'episodes',
                 'prob', 'last_occurred'];
const LABELS = { change_type: 'type', change: 'direction', threshold: 'move', n_years: 'lookback (y)',
                 n_windows: 'windows', last_occurred: 'last seen' };
const FORMATS = { threshold: pctCell(2), prob: pctCell(3),
                  last_occurred: (v: unknown) => isoDate(v as string | null) };

/** The rare-event table of price_return_analysis's interactive widget: for one holding period,
 *  the moves rarer than prob_max that did happen (prob above prob_min), and when last seen.
 *  The server keeps the full table, so each control change only refilters it. */
export function RareEventsTab({ params }: { params: Params }) {
  const [nDays, setNDays] = useState(params.streak_days.includes(3) ? 3 : params.streak_days[0]);
  const [changeType, setChangeType] = useState<ChangeType | ''>('');
  const [probMax, setProbMax] = useState(String(params.prob_max * 100));
  const [probMin, setProbMin] = useState(String(params.prob_min * 100));
  const bounds = useDebounced({ max: Number(probMax) / 100, min: Number(probMin) / 100 });
  const valid = bounds.max > bounds.min && bounds.min >= 0 && bounds.max <= 1;
  const days = params.streak_days.includes(nDays) ? nDays : params.streak_days[0];

  const query = useQuery({
    queryKey: ['rare-events', params, days, changeType, bounds],
    queryFn: () => api.rareEvents({ ...params, prob_max: bounds.max, prob_min: bounds.min }, [days],
                                  changeType || null),
    placeholderData: keepPreviousData,
    enabled: valid,
  });

  return (
    <div className="stack">
      <div className="toolbar" role="group" aria-label="Rare-event filters">
        <label>
          <span>Holding period</span>
          <select value={days} onChange={(e) => setNDays(Number(e.target.value))}>
            {params.streak_days.map((d) => <option key={d} value={d}>{d} {d === 1 ? 'day' : 'days'}</option>)}
          </select>
        </label>
        <label>
          <span>Event type</span>
          <select value={changeType} onChange={(e) => setChangeType(e.target.value as ChangeType | '')}>
            <option value="">Both</option>
            <option value="cumulative">Cumulative (compounded move)</option>
            <option value="consecutive">Consecutive (every day)</option>
          </select>
        </label>
        <label>
          <span>Rarer than (%)</span>
          <input type="number" className="short" min="0" max="100" step="1" value={probMax}
                 onChange={(e) => setProbMax(e.target.value)} />
        </label>
        <label>
          <span>More common than (%)</span>
          <input type="number" className="short" min="0" max="100" step="0.01" value={probMin}
                 onChange={(e) => setProbMin(e.target.value)} />
        </label>
        {!valid ? <p className="error inline" role="alert">The upper bound must be above the lower one, both between 0 and 100%.</p> : null}
      </div>

      <section className="card">
        <QueryState query={query} what="the rare events">
          {(data) => (
            <>
              <h2>
                {data.events.records.length} rare events over {days} {days === 1 ? 'day' : 'days'}
                <span className="h2-note"> ({data.total_events} measured across all holding periods)</span>
              </h2>
              <p className="lede">
                <em>prob</em> is the share of complete {days}-day windows in the lookback where the move
                happened. A probability resting on few <em>episodes</em> (unbroken runs) is fragile; the
                Statistics tab puts intervals on it.
                {params.lookback_years.length > 1
                  ? ` Compare the ${params.lookback_years.join('- and ')}-year rows: a large gap means the estimate depends on the regime.`
                  : ''}
              </p>
              <DataTable table={data.events} columns={COLUMNS} labels={LABELS} formats={FORMATS} />
              {!data.events.records.length ? (
                <p className="note">
                  None for this holding period: every move measured is either more common than{' '}
                  {probMax}% or never happened. Raise <em>Rarer than</em>, or try another holding period.
                </p>
              ) : null}
            </>
          )}
        </QueryState>
      </section>
    </div>
  );
}
