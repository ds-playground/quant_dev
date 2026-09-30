import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { useState } from 'react';

import { api, type Params } from '../api';
import type { Mode } from '../chartTheme';
import { DataTable } from '../components/DataTable';
import { Chart } from '../components/PlotlyChart';
import { QueryState } from '../components/QueryState';

/** Win and loss streaks (every day beyond its threshold) and cumulative moves (the compounded
 *  return over a window clearing a threshold), as in price_return_analysis. */
export function StreaksTab({ params, mode }: { params: Params; mode: Mode }) {
  const [window, setWindow] = useState(params.timeline_window);
  const streaks = useQuery({ queryKey: ['streaks', params], queryFn: () => api.streaks(params),
                             placeholderData: keepPreviousData });
  const cumulative = useQuery({ queryKey: ['cumulative', params], queryFn: () => api.cumulative(params),
                                placeholderData: keepPreviousData });
  const timelineWindow = params.windows.includes(window) ? window : params.windows[0];

  return (
    <div className="stack">
      <section className="card">
        <h2>Streaks: every day beyond the threshold</h2>
        <p className="lede">
          A win streak is a window of <em>d</em> days each above +{params.win_threshold}%; a loss
          streak, each below {params.loss_threshold}%. Frequency is the share of complete windows.
          Windows overlap, so <em>Episodes</em> counts each unbroken run once.
        </p>
        <QueryState query={streaks} what="the streaks">
          {(data) => <DataTable table={data.summary} />}
        </QueryState>
      </section>
      <div className="grid-2">
        <Chart name="streak-counts" params={params} mode={mode} label="Streak counts" />
        <Chart name="streak-frequency" params={params} mode={mode} label="Streak frequency" />
      </div>
      <div className="toolbar" role="group" aria-label="Timeline">
        <label>
          <span>Timeline window</span>
          <select value={timelineWindow} onChange={(e) => setWindow(Number(e.target.value))}>
            {params.windows.map((w) => <option key={w} value={w}>{w} days</option>)}
          </select>
        </label>
      </div>
      <Chart name="streak-timeline" params={params} mode={mode} options={{ window: timelineWindow }}
             label="Streak timeline" />

      <section className="card">
        <h2>Cumulative moves: the compounded return clears the threshold</h2>
        <p className="lede">
          Unlike a streak, single days may go the other way as long as the compounded move over the
          window clears the threshold ({params.cum_thresholds.map((t) => `${t}%`).join(', ')}).
        </p>
        <QueryState query={cumulative} what="the cumulative moves">
          {(data) => <DataTable table={data.summary} />}
        </QueryState>
      </section>
      {/* Two heatmaps with their own scales: it needs the full width. */}
      <Chart name="cumulative-heatmap" params={params} mode={mode} label="Cumulative heatmap" />
      <Chart name="cumulative-counts" params={params} mode={mode} label="Cumulative counts" />
    </div>
  );
}
