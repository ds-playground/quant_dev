import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';

import { api, type Figure, type Params } from '../api';
import { themed, type Mode } from '../chartTheme';

type PlotlyModule = typeof import('plotly.js-dist-min');
type PlotArgs = Parameters<PlotlyModule['react']>;

// plotly.js is large (about 4.6 MB), so it loads once, on the first chart, not with the app.
let plotly: Promise<PlotlyModule> | null = null;
const loadPlotly = () => (plotly ??= import('plotly.js-dist-min').then((m) => (m.default ?? m) as PlotlyModule));

/** Draws a figure from the API. The figure is viz.py's; only dark-mode colours change here. */
export function Plot({ figure, mode, label }: { figure: Figure; mode: Mode; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadPlotly()
      .then((Plotly) => {
        if (cancelled || !ref.current) return;
        const { data, layout } = themed(figure, mode);
        // The figure's own width, if any, would fight the card: let the card size it.
        const { width: _width, ...rest } = layout;
        return Plotly.react(ref.current, data as PlotArgs[1], { ...rest, autosize: true } as PlotArgs[2],
                            { responsive: true, displaylogo: false });
      })
      .catch((error: unknown) => setFailed(String(error)));
    return () => {
      cancelled = true;
    };
  }, [figure, mode]);

  useEffect(() => {
    const node = ref.current;
    return () => {
      if (node) loadPlotly().then((Plotly) => Plotly.purge(node));
    };
  }, []);

  const height = typeof figure.layout.height === 'number' ? figure.layout.height : 450;
  if (failed) return <p className="error">Could not draw the chart: {failed}</p>;
  return <div ref={ref} className="plot" style={{ minHeight: height }} role="img" aria-label={label} />;
}

/** A chart card: fetches one chart for the parameters and keeps the last one on screen,
 *  dimmed, while a new one loads. */
export function Chart({ name, params, mode, options = {}, label }: {
  name: string;
  params: Params;
  mode: Mode;
  options?: Record<string, string | number>;
  label: string;
}) {
  const query = useQuery({
    queryKey: ['chart', name, params, options],
    queryFn: () => api.chart(params, name, options),
    placeholderData: keepPreviousData,
  });
  return (
    <figure className="card chart" aria-busy={query.isFetching}>
      {query.error ? <p className="error">{String(query.error.message)}</p> : null}
      {query.data ? (
        <div className={query.isPlaceholderData ? 'stale' : undefined}>
          <Plot figure={query.data} mode={mode} label={label} />
        </div>
      ) : query.isPending ? (
        <p className="loading">Loading {label.toLowerCase()}…</p>
      ) : null}
    </figure>
  );
}
