import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

import { api, type Params, type TickerSet } from './api';
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

  // A new ticker list starts on its first ticker (SPX for the demo set), with its parameters.
  useEffect(() => {
    const list = tickers.data?.tickers;
    if (list?.length && !list.some((t) => t.symbol === params?.ticker)) setParams(list[0].params);
  }, [tickers.data, params?.ticker]);

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
        tickers={tickers.data?.tickers ?? []}
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
          tickers.error ? <p className="error">{tickers.error.message}</p> : <p className="loading">Loading tickers…</p>
        ) : tab === 'overview' ? (
          <OverviewTab params={params} mode={mode} />
        ) : tab === 'streaks' ? (
          <StreaksTab params={params} mode={mode} />
        ) : tab === 'rare' ? (
          <RareEventsTab key={params.ticker} params={params} />
        ) : tab === 'statistics' ? (
          <StatisticsTab params={params} mode={mode} />
        ) : (
          <MultiTickerTab tickerSet={tickerSet} />
        )}
      </main>
    </div>
  );
}
