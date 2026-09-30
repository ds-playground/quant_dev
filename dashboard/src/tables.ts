// Reshaping API tables for display. Nothing here computes a statistic.
import type { Cell, Table } from './api';

/** value_at_risk's long table (horizon, level, method, var, es) with one column per method for
 *  VaR and for ES, one row per horizon and level: the statistics notebook's pivot. */
export function varTable(t: Table): Table {
  const methods = [...new Set(t.records.map((r) => String(r.method)))];
  const rows = new Map<string, Record<string, Cell>>();
  for (const r of t.records) {
    const key = `${r.horizon}|${r.level}`;
    const row = rows.get(key) ?? { horizon: r.horizon, level: r.level };
    row[`VaR ${r.method}`] = r.var;
    row[`ES ${r.method}`] = r.es;
    rows.set(key, row);
  }
  return {
    columns: ['horizon', 'level', ...methods.map((m) => `VaR ${m}`), ...methods.map((m) => `ES ${m}`)],
    records: [...rows.values()],
  };
}
