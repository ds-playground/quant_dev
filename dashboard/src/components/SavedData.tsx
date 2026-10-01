import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import type { SaveAllResult, SaveResult, Ticker, TickerSet } from '../api';
import { api, ApiError } from '../api';

// Saved data older than this (calendar days, covering a weekend and a holiday) is flagged.
const STALE_DAYS = 4;

export function ageInDays(date: string, today = new Date()): number {
  return Math.floor((today.getTime() - new Date(`${date}T00:00:00Z`).getTime()) / 86_400_000);
}

export function describeSave(r: SaveResult): string {
  const range = `${r.rows.toLocaleString('en-GB')} bars, ${r.first} to ${r.last}`;
  if (r.created) return `Saved ${r.symbol}: ${range}.`;
  const revised = r.revised.length
    ? ` ${r.revised.length} saved value${r.revised.length > 1 ? 's' : ''} revised by Yahoo: `
      + r.revised.slice(0, 3).map((c) => `${c.date} ${c.column} ${c.saved} → ${c.new}`).join('; ')
      + (r.revised.length > 3 ? '; …' : '') + '.'
    : '';
  return `Updated ${r.symbol}: +${r.added} bar${r.added === 1 ? '' : 's'}, now ${range}.${revised}`;
}

/** One line for the whole list: how many were saved for the first time, updated, and failed. */
export function describeSaveAll(r: SaveAllResult): string {
  const created = r.saved.filter((s) => s.created).length;
  const updated = r.saved.length - created;
  const total = r.saved.length + r.failed.length;
  const parts = [created ? `${created} saved` : '', updated ? `${updated} updated` : '',
                 r.failed.length ? `${r.failed.length} failed` : ''].filter(Boolean);
  return `${total} ticker${total === 1 ? '' : 's'}: ${parts.join(', ')}.`;
}

/** The failures a 502 carries when every download failed (the server's detail), if any. */
export function failuresOf(error: unknown): SaveAllResult['failed'] {
  const raw = error instanceof ApiError ? error.raw : null;
  return raw && typeof raw === 'object' && 'failed' in raw ? (raw.failed as SaveAllResult['failed']) : [];
}

/** Download or update every ticker in configs/tickers.yaml into data/local, in one request
 *  (POST /api/local/update-all). The request returns when all are done; the result lists each
 *  ticker, and every view refetches. */
export function DownloadAll() {
  const client = useQueryClient();
  const configured = useQuery({ queryKey: ['tickers', 'yahoo'], queryFn: () => api.tickers('yahoo') });
  const save = useMutation({ mutationFn: () => api.saveAll(), onSettled: () => client.invalidateQueries() });
  const count = configured.data?.tickers.length;
  const failed = save.data?.failed ?? failuresOf(save.error);

  return (
    <div className="saved download-all" aria-live="polite">
      <button type="button" className="ghost" disabled={save.isPending} onClick={() => save.mutate()}>
        {save.isPending ? `Downloading ${count ? `${count} tickers` : 'the default tickers'}…`
          : `Download all ${count ? `${count} ` : ''}default tickers`}
      </button>
      {save.isPending ? <span className="note">A few seconds a ticker the first time; updates are quicker.</span> : null}
      {save.data ? <span className="note ok">{describeSaveAll(save.data)}</span> : null}
      {save.error ? <span className="error">{save.error.message}</span> : null}
      {save.data?.saved.length || failed.length ? (
        <details className="note">
          <summary>Each ticker</summary>
          <ul className="failures">
            {save.data?.saved.map((r) => <li key={r.symbol}>{describeSave(r)}</li>)}
            {failed.map((f) => <li key={f.symbol} className="error"><strong>{f.symbol}</strong>: {f.error}</li>)}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

/** Saving live data as CSV: with the Yahoo set, save or update the chosen ticker; with the saved
 *  set, show how current it is and update it. Afterwards every view refetches. */
export function SavedData({ tickerSet, ticker }: { tickerSet: TickerSet; ticker: Ticker | undefined }) {
  const client = useQueryClient();
  const save = useMutation({
    mutationFn: (symbol: string) => api.saveLocal(symbol),
    onSuccess: () => client.invalidateQueries(),
  });
  if (tickerSet === 'synthetic' || !ticker) return null;

  const saved = ticker.saved;
  const age = saved ? ageInDays(saved.last) : null;
  const stale = age !== null && age > STALE_DAYS;
  const label = save.isPending ? 'Downloading…' : saved ? 'Update CSV' : 'Save to CSV';

  return (
    <div className="saved" aria-live="polite">
      <span className="saved-state">
        {saved ? (
          <>
            Saved to <strong>{saved.last}</strong>
            {stale ? <span className="flag" title={`The last saved bar is ${age} days old`}>⚠ {age} days old</span> : null}
          </>
        ) : tickerSet === 'yahoo' ? 'Not saved as CSV' : null}
      </span>
      <button type="button" className="ghost" disabled={save.isPending}
              onClick={() => save.mutate(ticker.symbol)}>
        {label}
      </button>
      {save.data && save.variables === ticker.symbol ? <span className="note ok">{describeSave(save.data)}</span> : null}
      {save.error ? <span className="error">{save.error.message}</span> : null}
    </div>
  );
}
