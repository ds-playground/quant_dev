import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { useState } from 'react';

import { api, type Cell, type ChangeType, type Params, type Table } from '../api';
import type { Mode } from '../chartTheme';
import { DataTable, KeyValues } from '../components/DataTable';
import { Chart } from '../components/PlotlyChart';
import { QueryState } from '../components/QueryState';
import { StatTiles } from '../components/StatTiles';
import { fixedCell, num, pct, pctCell } from '../format';
import { varTable } from '../tables';

type Values = Record<string, Cell>;
interface Distribution { moments: Values; jarque_bera: Values; student_t: Values; tail_index: Values; value_at_risk: Table }
interface Dependence { autocorrelation: Table; ljung_box: Table; variance_ratio: Table; arch_lm: Values }
interface Drawdowns { risk_ratios: Values; drawdowns: Table }
interface Probabilities { n_days: number; n_boot: number; events: Table }

function useSection<T>(params: Params, section: string, options: Record<string, number> = {}, enabled = true) {
  return useQuery({
    queryKey: ['statistics', section, params, options],
    queryFn: () => api.statistics<T>(params, section, options),
    placeholderData: keepPreviousData,
    enabled,
  });
}

const pct3 = pctCell(3);

/** The statistics notebook's four sections for the same series. The rare-event uncertainty runs
 *  a bootstrap (seconds), so it waits for a request. */
export function StatisticsTab({ params, mode }: { params: Params; mode: Mode }) {
  const distribution = useSection<Distribution>(params, 'distribution');
  const dependence = useSection<Dependence>(params, 'dependence');
  const drawdowns = useSection<Drawdowns>(params, 'drawdowns', { top: 5 });

  return (
    <div className="stack">
      <section className="card">
        <h2>1. Distribution and tails</h2>
        <p className="lede">
          How far the daily returns are from normal. Excess kurtosis is 0 for a normal; a tiny
          Jarque–Bera p-value says the difference is not noise. A Student-t with few degrees of
          freedom, or a Hill tail index below 4, means fat tails.
        </p>
        <QueryState query={distribution} what="the distribution statistics">
          {(d) => (
            <>
              <div className="grid-3">
                <KeyValues caption="Moments of daily returns" values={d.moments}
                           formats={{ n: fixedCell(0), mean: pct3, std: pct3, min: pct3, max: pct3 }} />
                <KeyValues caption="Jarque–Bera test" values={{ statistic: d.jarque_bera.statistic, p_value: d.jarque_bera.p_value }} />
                <KeyValues caption="Student-t fit and Hill tail index"
                           values={{ 't df': d.student_t.df, 't loc': d.student_t.loc, 't scale': d.student_t.scale,
                                     'alpha, losses': d.tail_index.left, 'alpha, gains': d.tail_index.right,
                                     'tail size k': d.tail_index.k }}
                           formats={{ 't loc': pct3, 't scale': pct3, 'tail size k': fixedCell(0) }} />
              </div>
              <h3>Value at risk and expected shortfall (losses, compounded over the horizon)</h3>
              <DataTable table={varTable(d.value_at_risk)}
                         formats={Object.fromEntries(varTable(d.value_at_risk).columns.slice(2).map((c) => [c, pctCell(2)]))} />
            </>
          )}
        </QueryState>
      </section>
      <Chart name="qq" params={params} mode={mode} label="Q-Q plot" />

      <section className="card">
        <h2>2. Dependence and volatility clustering</h2>
        <p className="lede">
          Memory in direction (returns) or in size (squared returns). Small Ljung–Box p-values on
          squared returns and a small ARCH-LM p-value mean big moves follow big moves. A variance
          ratio below 1 (z_robust below −2) means multi-day moves partly reverse.
        </p>
        <QueryState query={dependence} what="the dependence tests">
          {(d) => (
            <div className="grid-2">
              <DataTable caption="Ljung–Box" table={d.ljung_box} />
              <DataTable caption="Variance ratio" table={d.variance_ratio} />
              <KeyValues caption={`ARCH-LM (${d.arch_lm.lags} lags)`}
                         values={{ statistic: d.arch_lm.statistic, p_value: d.arch_lm.p_value, 'R²': d.arch_lm.r_squared }} />
            </div>
          )}
        </QueryState>
      </section>
      <Chart name="autocorrelation" params={params} mode={mode} label="Autocorrelation" />

      <section className="card">
        <h2>3. Drawdowns and risk-adjusted returns</h2>
        <QueryState query={drawdowns} what="the drawdowns">
          {(d) => (
            <>
              <StatTiles stats={[
                { label: 'Annual return', value: pct(d.risk_ratios.annual_return as number, 1) },
                { label: 'Annual volatility', value: pct(d.risk_ratios.annual_volatility as number, 1) },
                { label: 'Sharpe', value: num(d.risk_ratios.sharpe as number) },
                { label: 'Sortino', value: num(d.risk_ratios.sortino as number) },
                { label: 'Calmar', value: num(d.risk_ratios.calmar as number) },
                { label: 'Max drawdown', value: pct(d.risk_ratios.max_drawdown as number, 1) },
              ]} />
              <h3>Deepest drawdowns</h3>
              <DataTable table={d.drawdowns} formats={{ depth: pctCell(1) }} />
              <p className="note">An empty recovery means the drawdown is still open.</p>
            </>
          )}
        </QueryState>
      </section>
      <div className="grid-2">
        <Chart name="drawdown" params={params} mode={mode} label="Drawdown" />
        <Chart name="rolling-risk" params={params} mode={mode} label="Rolling risk" />
      </div>

      <ProbabilitySection params={params} mode={mode} />
    </div>
  );
}

function ProbabilitySection({ params, mode }: { params: Params; mode: Mode }) {
  const [nDays, setNDays] = useState(params.streak_days.includes(3) ? 3 : params.streak_days[0]);
  const [nBoot, setNBoot] = useState(1000);
  const [changeType, setChangeType] = useState<ChangeType>('cumulative');
  const [rareOnly, setRareOnly] = useState(true);
  const [run, setRun] = useState<{ n_days: number; n_boot: number } | null>(null);
  const query = useSection<Probabilities>(params, 'probabilities', run ?? {}, run !== null);
  const longest = Math.max(...params.lookback_years);

  const events = query.data?.events;
  const shown: Table | undefined = events && {
    columns: events.columns,
    records: events.records
      .filter((r) => !rareOnly || ((r.prob as number) < params.prob_max && (r.prob as number) > params.prob_min))
      .sort((a, b) => Number(a.n_years) - Number(b.n_years) || String(a.change_type).localeCompare(String(b.change_type))
        || String(a.change).localeCompare(String(b.change)) || Number(a.threshold) - Number(b.threshold)),
  };

  return (
    <>
      <section className="card">
        <h2>4. How precise are the rare-event probabilities?</h2>
        <p className="lede">
          Each probability gets a 95% interval from a block bootstrap (resampling runs of days, so
          volatility clustering survives), and is set against an i.i.d. normal and a fitted
          Student-t. A wide interval or few episodes means the number is not precise enough to
          price on alone; observed well above both models means fatter tails or more dependence.
        </p>
        <div className="toolbar inset" role="group" aria-label="Bootstrap settings">
          <label>
            <span>Holding period</span>
            <select value={nDays} onChange={(e) => setNDays(Number(e.target.value))}>
              {params.streak_days.map((d) => <option key={d} value={d}>{d} {d === 1 ? 'day' : 'days'}</option>)}
            </select>
          </label>
          <label>
            <span>Resamples</span>
            <select value={nBoot} onChange={(e) => setNBoot(Number(e.target.value))}>
              {[500, 1000, 2000].map((n) => <option key={n} value={n}>{n.toLocaleString('en-GB')}</option>)}
            </select>
          </label>
          <div className="actions">
            <button type="button" className="primary" disabled={query.isFetching}
                    onClick={() => setRun({ n_days: nDays, n_boot: nBoot })}>
              {query.isFetching ? 'Running the bootstrap…' : run ? 'Run again' : 'Run the bootstrap'}
            </button>
          </div>
          {events ? (
            <label className="check">
              <input type="checkbox" checked={rareOnly} onChange={(e) => setRareOnly(e.target.checked)} />
              <span>Rare only ({pct(params.prob_min, 2)} to {pct(params.prob_max, 0)})</span>
            </label>
          ) : null}
        </div>
        {run === null ? (
          <p className="note">Takes a few seconds on ten years of daily data; the result is cached.</p>
        ) : (
          <QueryState query={query} what="the bootstrap">
            {() => shown ? (
              <>
                <h3>{shown.records.length} events, {run.n_days}-day holding period, {run.n_boot.toLocaleString('en-GB')} resamples</h3>
                <DataTable table={shown}
                           columns={['n_years', 'change_type', 'change', 'threshold', 'count', 'episodes',
                                     'prob', 'lower', 'upper', 'prob_normal', 'prob_t']}
                           labels={{ n_years: 'lookback (y)', change_type: 'type', change: 'direction', threshold: 'move',
                                     prob: 'observed', prob_normal: 'normal', prob_t: 'Student-t' }}
                           formats={{ threshold: pctCell(2), prob: pct3, lower: pct3, upper: pct3,
                                      prob_normal: pct3, prob_t: pct3 }} />
              </>
            ) : null}
          </QueryState>
        )}
      </section>
      {query.data && run ? (
        <>
          <div className="toolbar" role="group" aria-label="Event chart">
            <label>
              <span>Event type</span>
              <select value={changeType} onChange={(e) => setChangeType(e.target.value as ChangeType)}>
                <option value="cumulative">Cumulative</option>
                <option value="consecutive">Consecutive</option>
              </select>
            </label>
          </div>
          <div className="stack">
            {(['above', 'below'] as const).map((change) => (
              <Chart key={change} name="event-probabilities" params={params} mode={mode}
                     options={{ n_days: run.n_days, n_boot: run.n_boot, change_type: changeType, change, n_years: longest }}
                     label={`Event probabilities, ${change}`} />
            ))}
          </div>
        </>
      ) : null}
    </>
  );
}
