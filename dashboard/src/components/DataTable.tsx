import type { Cell, Table } from '../api';
import { cell } from '../format';

export type Formats = Record<string, (value: Cell) => string>;

/** A table from the API, columns in the server's order. Numbers align right. `formats` and
 *  `labels` change how a column shows, never what it holds. */
export function DataTable({ table, caption, maxRows, formats = {}, labels = {}, columns }: {
  table: Table;
  caption?: string;
  maxRows?: number;
  formats?: Formats;
  labels?: Record<string, string>;
  columns?: string[];                  // a subset or reordering of table.columns
}) {
  const shown = columns ?? table.columns;
  const rows = maxRows ? table.records.slice(0, maxRows) : table.records;
  const numeric = new Set(shown.filter((c) => table.records.some((r) => typeof r[c] === 'number')));
  return (
    <div className="table-wrap">
      <table>
        {caption ? <caption>{caption}</caption> : null}
        <thead>
          <tr>
            {shown.map((c) => (
              <th key={c} scope="col" className={numeric.has(c) ? 'num' : undefined}>{labels[c] ?? c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((record, i) => (
            <tr key={i}>
              {shown.map((c) => (
                <td key={c} className={numeric.has(c) ? 'num' : undefined}>
                  {(formats[c] ?? cell)(record[c] ?? null)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {!table.records.length ? <p className="note">No rows.</p> : null}
      {maxRows && table.records.length > maxRows ? (
        <p className="note">Showing {maxRows} of {table.records.length} rows.</p>
      ) : null}
    </div>
  );
}

/** A one-level object ({name: value}) as a two-column table. */
export function KeyValues({ values, formats = {}, labels = {}, caption }: {
  values: Record<string, Cell>;
  formats?: Formats;
  labels?: Record<string, string>;
  caption?: string;
}) {
  return (
    <div className="table-wrap">
      <table className="kv">
        {caption ? <caption>{caption}</caption> : null}
        <tbody>
          {Object.entries(values).map(([key, value]) => (
            <tr key={key}>
              <th scope="row">{labels[key] ?? key}</th>
              <td className="num">{(formats[key] ?? cell)(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
