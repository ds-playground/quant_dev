import { useEffect, useState, type FormEvent } from 'react';

import type { Params, Ticker, TickerSet } from '../api';

interface Draft {
  start_date: string;
  end_date: string;
  win_threshold: string;
  loss_threshold: string;
}

const draftOf = (p: Params): Draft => ({
  start_date: p.start_date,
  end_date: p.end_date,
  win_threshold: String(p.win_threshold),
  loss_threshold: String(p.loss_threshold),
});

/** The one row of controls above the tabs. Choosing a data set or a ticker applies at once, with
 *  that ticker's configured parameters; dates and thresholds apply with the Apply button. */
export function ParamsPanel({ tickerSet, onTickerSet, tickers, params, onApply }: {
  tickerSet: TickerSet;
  onTickerSet: (set: TickerSet) => void;
  tickers: Ticker[];
  params: Params | null;
  onApply: (params: Params) => void;
}) {
  const [draft, setDraft] = useState<Draft | null>(params && draftOf(params));
  useEffect(() => setDraft(params && draftOf(params)), [params]);

  const edited = params && draft ? {
    ...params,
    start_date: draft.start_date,
    end_date: draft.end_date,
    win_threshold: Number(draft.win_threshold),
    loss_threshold: Number(draft.loss_threshold),
  } : null;
  const problem = !edited ? null
    : !edited.start_date || !edited.end_date || edited.start_date >= edited.end_date
      ? 'Start date must be before end date.'
      : !(edited.win_threshold > 0) ? 'Win threshold must be above 0.'
        : !(edited.loss_threshold < 0) ? 'Loss threshold must be below 0.' : null;
  const dirty = !!(params && draft) && JSON.stringify(draft) !== JSON.stringify(draftOf(params));

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (edited && dirty && !problem) onApply(edited);
  };
  const field = (key: keyof Draft) => ({
    value: draft?.[key] ?? '',
    disabled: !draft,
    onChange: (e: { target: { value: string } }) => setDraft((d) => d && { ...d, [key]: e.target.value }),
  });
  const selected = tickers.find((t) => t.symbol === params?.ticker);

  return (
    <form className="controls" onSubmit={submit} aria-label="Analysis parameters">
      <label>
        <span>Data</span>
        <select value={tickerSet} onChange={(e) => onTickerSet(e.target.value as TickerSet)}>
          <option value="demo">Demo data (offline)</option>
          <option value="yahoo">Yahoo Finance</option>
        </select>
      </label>
      <label>
        <span>Ticker</span>
        <select
          value={params?.ticker ?? ''}
          disabled={!tickers.length}
          onChange={(e) => {
            const ticker = tickers.find((t) => t.symbol === e.target.value);
            if (ticker) onApply(ticker.params);
          }}
        >
          {tickers.map((t) => (
            <option key={t.symbol} value={t.symbol}>{t.label} ({t.symbol})</option>
          ))}
        </select>
      </label>
      <label>
        <span>Start</span>
        <input type="date" {...field('start_date')} />
      </label>
      <label>
        <span>End (exclusive)</span>
        <input type="date" {...field('end_date')} />
      </label>
      <label>
        <span>Win above (%)</span>
        <input type="number" step="0.05" inputMode="decimal" className="short" {...field('win_threshold')} />
      </label>
      <label>
        <span>Loss below (%)</span>
        <input type="number" step="0.05" inputMode="decimal" className="short" {...field('loss_threshold')} />
      </label>
      <div className="actions">
        <button type="submit" disabled={!dirty || !!problem}>Apply</button>
        <button type="button" className="ghost" disabled={!selected || !dirty}
                onClick={() => selected && setDraft(draftOf(params!))}>
          Undo edits
        </button>
      </div>
      {dirty && problem ? <p className="error inline" role="alert">{problem}</p> : null}
    </form>
  );
}
