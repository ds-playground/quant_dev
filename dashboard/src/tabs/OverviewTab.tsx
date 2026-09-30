import { keepPreviousData, useQuery } from '@tanstack/react-query';

import { api, type Params } from '../api';
import type { Mode } from '../chartTheme';
import { Chart } from '../components/PlotlyChart';
import { DataTable } from '../components/DataTable';
import { StatTiles } from '../components/StatTiles';
import { isoDate, num, pct } from '../format';

// Non-breaking hyphens, so a narrow tile wraps between the two dates, not inside one.
const keepTogether = (date: string) => date.replace(/-/g, '\u2011');

/** The series at a glance: latest statistics, price and returns, the return distribution, and
 *  the rolling return and volatility by holding period. The charts are requested once the series
 *  has loaded: they all read it, so a series that fails shows one error, not five. */
export function OverviewTab({ params, mode }: { params: Params; mode: Mode }) {
  const overview = useQuery({
    queryKey: ['overview', params],
    queryFn: () => api.overview(params),
    placeholderData: keepPreviousData,
  });
  const o = overview.data;

  // Every chart loads the same series, so if the series fails there is one error, not six.
  if (overview.error && !o) {
    return (
      <section className="card" role="alert">
        <h2>Could not load {params.ticker}</h2>
        <p className="error">{overview.error.message}</p>
      </section>
    );
  }

  return (
    <div className="stack">
      {overview.error ? <p className="error" role="alert">{overview.error.message}</p> : null}
      {o ? (
        <StatTiles
          stale={overview.isPlaceholderData}
          stats={[
            { label: 'Last price', value: num(o.last_price), detail: `close on ${isoDate(o.end)}` },
            { label: 'Annualized return', value: pct(o.snapshot.annualized_return, 1),
              detail: `last ${params.trade_days} trading days` },
            { label: 'Annualized volatility', value: pct(o.snapshot.annualized_volatility, 1),
              detail: `daily std dev × √${params.trade_days}` },
            { label: 'Daily mean return', value: pct(o.snapshot.daily_avg_return, 3),
              detail: `std dev ${pct(o.snapshot.daily_std_dev, 2)}` },
            { label: 'Daily returns', value: o.rows.toLocaleString('en-GB'),
              detail: `${keepTogether(isoDate(o.start))} to ${keepTogether(isoDate(o.end))}` },
          ]}
        />
      ) : overview.isPending ? <p className="loading">Loading the series…</p> : null}

      {o ? (
        <>
          <Chart name="price-and-returns" params={params} mode={mode} label="Price and daily returns" />
          <div className="grid-2">
            <Chart name="return-distribution" params={params} mode={mode} label="Return distribution" />
            <section className={`card${overview.isPlaceholderData ? ' stale' : ''}`}>
              <h2>Return distribution</h2>
              <DataTable
                table={{
                  columns: ['statistic', 'value'],
                  records: Object.entries(o.distribution).map(([statistic, value]) => ({ statistic, value })),
                }}
              />
              <p className="note">
                Daily returns in %. <em>days &gt; +thr</em> and <em>days &lt; −thr</em> are the share
                of days beyond the win ({params.win_threshold}%) and loss ({params.loss_threshold}%)
                thresholds. Streaks count those days, so they track the median more than the mean.
              </p>
            </section>
          </div>
          <Chart name="rolling-average" params={params} mode={mode} label="Rolling average return" />
          <Chart name="rolling-volatility" params={params} mode={mode} label="Rolling volatility" />
        </>
      ) : null}
    </div>
  );
}
