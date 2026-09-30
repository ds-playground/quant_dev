import { useMutation, useQueryClient } from '@tanstack/react-query';

import type { SaveResult, Ticker, TickerSet } from '../api';
import { api } from '../api';

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

/** Saving live data as CSV: with the Yahoo set, save or update the chosen ticker; with the saved
 *  set, show how current it is and update it. Afterwards every view refetches. */
export function SavedData({ tickerSet, ticker }: { tickerSet: TickerSet; ticker: Ticker | undefined }) {
  const client = useQueryClient();
  const save = useMutation({
    mutationFn: (symbol: string) => api.saveLocal(symbol),
    onSuccess: () => client.invalidateQueries(),
  });
  if (tickerSet === 'demo' || !ticker) return null;

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
