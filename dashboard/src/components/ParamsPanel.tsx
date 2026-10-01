import { useEffect, useState, type FormEvent } from 'react';

import { SYMBOL, tidySymbol } from '../addedTickers';
import type { Params, Ticker, TickerSet } from '../api';
import { DownloadAll, SavedData } from './SavedData';

const OTHER = '__other__';

const optionText = (t: Ticker, tickerSet: TickerSet) =>
  `${t.label}${t.label === t.symbol ? '' : ` (${t.symbol})`}`
  + (tickerSet === 'local' && t.saved ? `, to ${t.saved.last}` : '');

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
export function ParamsPanel({ tickerSet, onTickerSet, tickers, params, onApply, addedSymbols = [],
                              onAddTicker, onRemoveTicker }: {
  tickerSet: TickerSet;
  onTickerSet: (set: TickerSet) => void;
  tickers: Ticker[];
  params: Params | null;
  onApply: (params: Params) => void;
  addedSymbols?: string[];                          // tickers added beyond the configured list
  onAddTicker?: (symbol: string) => Promise<void>;  // offered when set: "Other ticker…"
  onRemoveTicker?: (symbol: string) => void;
}) {
  const [draft, setDraft] = useState<Draft | null>(params && draftOf(params));
  const [adding, setAdding] = useState(false);
  const [symbol, setSymbol] = useState('');
  const [addError, setAddError] = useState<string | null>(null);
  const [looking, setLooking] = useState(false);
  useEffect(() => setAdding(false), [tickerSet]);
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
  const configured = tickers.filter((t) => !addedSymbols.includes(t.symbol));
  const extra = tickers.filter((t) => addedSymbols.includes(t.symbol));

  const lookUp = async () => {
    const wanted = tidySymbol(symbol);
    if (!SYMBOL.test(wanted)) {
      setAddError('A Yahoo symbol: letters, digits and ^ = . _ - only (e.g. AAPL, ^GSPC, SI=F).');
      return;
    }
    setLooking(true);
    setAddError(null);
    try {
      await onAddTicker!(wanted);
      setAdding(false);
      setSymbol('');
    } catch (error) {
      setAddError(String((error as Error).message ?? error));
    } finally {
      setLooking(false);
    }
  };

  return (
    <form className="controls" onSubmit={submit} aria-label="Analysis parameters">
      <label>
        <span>Data</span>
        <select value={tickerSet} onChange={(e) => onTickerSet(e.target.value as TickerSet)}>
          <option value="synthetic">Synthetic data (offline)</option>
          <option value="yahoo">Yahoo Finance (live)</option>
          <option value="local">Saved CSV (offline)</option>
        </select>
      </label>
      <label>
        <span>Ticker</span>
        <select
          value={adding ? OTHER : params?.ticker ?? ''}
          disabled={!tickers.length}
          onChange={(e) => {
            if (e.target.value === OTHER) {
              setAdding(true);
              setAddError(null);
              return;
            }
            setAdding(false);
            const ticker = tickers.find((t) => t.symbol === e.target.value);
            if (ticker) onApply(ticker.params);
          }}
        >
          {onAddTicker ? (
            <>
              <optgroup label="Configured">
                {configured.map((t) => <option key={t.symbol} value={t.symbol}>{optionText(t, tickerSet)}</option>)}
              </optgroup>
              {extra.length ? (
                <optgroup label="Added">
                  {extra.map((t) => <option key={t.symbol} value={t.symbol}>{optionText(t, tickerSet)}</option>)}
                </optgroup>
              ) : null}
              <option value={OTHER}>Other ticker…</option>
            </>
          ) : tickers.map((t) => <option key={t.symbol} value={t.symbol}>{optionText(t, tickerSet)}</option>)}
        </select>
      </label>
      {adding ? (
        <div className="other-ticker" role="group" aria-label="Other ticker">
          <label>
            <span>Yahoo symbol</span>
            <input type="text" className="short" value={symbol} autoFocus placeholder="e.g. AAPL"
                   spellCheck={false} autoCapitalize="characters"
                   onChange={(e) => setSymbol(e.target.value)}
                   onKeyDown={(e) => {
                     if (e.key === 'Enter') { e.preventDefault(); void lookUp(); }
                     if (e.key === 'Escape') setAdding(false);
                   }} />
          </label>
          <button type="button" className="primary" disabled={!symbol.trim() || looking} onClick={() => void lookUp()}>
            {looking ? 'Loading…' : 'Load'}
          </button>
          <button type="button" className="ghost" onClick={() => setAdding(false)}>Cancel</button>
          {addError ? <p className="error inline" role="alert">{addError}</p> : null}
        </div>
      ) : selected && addedSymbols.includes(selected.symbol) && onRemoveTicker ? (
        <div className="other-ticker">
          <button type="button" className="ghost" onClick={() => onRemoveTicker(selected.symbol)}
                  aria-label={`Remove ${selected.symbol} from the list`}
                  title={`Remove ${selected.symbol} from the Added list`}>
            Remove
          </button>
        </div>
      ) : null}
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
      <SavedData tickerSet={tickerSet} ticker={selected} />
      {tickerSet !== 'synthetic' && tickers.length ? <DownloadAll /> : null}
    </form>
  );
}
