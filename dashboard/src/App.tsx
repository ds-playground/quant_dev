import { useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

import { useAddedTickers } from './addedTickers';
import { api, type Params, type Ticker, type TickerSet } from './api';
import { ParamsPanel } from './components/ParamsPanel';
import { MultiTickerTab } from './tabs/MultiTickerTab';
import { OverviewTab } from './tabs/OverviewTab';
import { RareEventsTab } from './tabs/RareEventsTab';
import { StatisticsTab } from './tabs/StatisticsTab';
import { StreaksTab } from './tabs/StreaksTab';
import { useTheme, type ThemeChoice } from './theme';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'streaks', label: 'Streaks & cumulative' },
  { id: 'rare', label: 'Rare events' },
  { id: 'statistics', label: 'Statistics' },
  { id: 'multi', label: 'Multi-ticker' },
] as const;
type TabId = (typeof TABS)[number]['id'];

export default function App() {
  const [themeChoice, setThemeChoice, mode] = useTheme();
  const [tab, setTab] = useState<TabId>('overview');
  const [tickerSet, setTickerSet] = useState<TickerSet>('demo');
  const [params, setParams] = useState<Params | null>(null);

  const health = useQuery({ queryKey: ['health'], queryFn: api.health, retry: false });
  const tickers = useQuery({ queryKey: ['tickers', tickerSet], queryFn: () => api.tickers(tickerSet) });

  // Yahoo symbols added beyond the configured list (kept in this browser), each looked up afresh.
  const added = useAddedTickers();
  const client = useQueryClient();
  const lookups = useQueries({
    queries: added.symbols.map((symbol) => ({
      queryKey: ['ticker', 'yahoo', symbol],
      queryFn: () => api.ticker(symbol, 'yahoo'),
      enabled: tickerSet === 'yahoo',
    })),
  });
  const configured = tickers.data?.tickers ?? [];
  const extra: Ticker[] = tickerSet !== 'yahoo' ? [] : lookups
    .map((q) => q.data)
    .filter((t): t is Ticker => !!t && !configured.some((c) => c.symbol === t.symbol));
  const list = [...configured, ...extra];
  const lookupsPending = tickerSet === 'yahoo' && lookups.some((q) => q.isPending);

  // A new ticker list starts on its first ticker (SPX for the demo set), with its parameters.
  useEffect(() => {
    if (!tickers.data || lookupsPending) return;
    if (list.length && !list.some((t) => t.symbol === params?.ticker)) setParams(list[0].params);
  }, [tickers.data, lookupsPending, list, params?.ticker]);

  const addTicker = async (symbol: string) => {
    const ticker = await client.fetchQuery({ queryKey: ['ticker', 'yahoo', symbol],
                                             queryFn: () => api.ticker(symbol, 'yahoo') });
    if (!configured.some((c) => c.symbol === symbol)) added.add(symbol);
    setParams(ticker.params);
  };

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Price-return dashboard</h1>
          <p className="subtitle">
            The <code>price_return</code> analysis, served by the API in <code>src/api</code>
            {health.data ? ` (v${health.data.version})` : ''}.
          </p>
        </div>
        <label className="theme">
          <span>Theme</span>
          <select value={themeChoice} onChange={(e) => setThemeChoice(e.target.value as ThemeChoice)}>
            <option value="auto">Auto</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </label>
      </header>

      {health.isError ? (
        <p className="error" role="alert">
          The API is not answering. Start it from the repo root with{' '}
          <code>uvicorn src.api.app:app --reload</code>, then reload this page.
        </p>
      ) : null}

      <ParamsPanel
        tickerSet={tickerSet}
        onTickerSet={(set) => { setTickerSet(set); setParams(null); }}
        tickers={list}
        addedSymbols={tickerSet === 'yahoo' ? extra.map((t) => t.symbol) : []}
        onAddTicker={tickerSet === 'yahoo' ? addTicker : undefined}
        onRemoveTicker={added.remove}
        params={params}
        onApply={setParams}
      />

      {params?.data_source === 'demo' ? (
        <p className="notice" role="note">
          <strong>Demo data:</strong> processed from Yahoo Finance with 0.01% random noise. Not
          market data; for education only (see <code>data/demo/README.md</code>).
        </p>
      ) : null}

      <nav className="tabs" role="tablist" aria-label="Views">
        {TABS.map((t) => (
          <button key={t.id} role="tab" id={`tab-${t.id}`} aria-selected={tab === t.id}
                  aria-controls="panel" onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      <main id="panel" role="tabpanel" aria-labelledby={`tab-${tab}`}>
        {!params ? (
          tickers.error ? <p className="error">{tickers.error.message}</p>
            : tickers.data && !tickers.data.tickers.length ? (
              <section className="card">
                <h2>Nothing saved yet</h2>
                <p className="lede">
                  Saved CSVs are live Yahoo data kept in <code>data/local/</code> (not committed), so
                  later runs are offline and fast. To save one, choose <strong>Yahoo Finance (live)</strong>,
                  pick a ticker and press <strong>Save to CSV</strong>. Or save every configured ticker
                  from the repo root with <code>python scripts/update_local_data.py</code>.
                </p>
              </section>
            ) : <p className="loading">Loading tickers…</p>
        ) : tab === 'overview' ? (
          <OverviewTab params={params} mode={mode} />
        ) : tab === 'streaks' ? (
          <StreaksTab params={params} mode={mode} />
        ) : tab === 'rare' ? (
          <RareEventsTab key={params.ticker} params={params} />
        ) : tab === 'statistics' ? (
          <StatisticsTab params={params} mode={mode} />
        ) : (
          <MultiTickerTab tickerSet={tickerSet} addedSymbols={added.symbols} />
        )}
      </main>
    </div>
  );
}
