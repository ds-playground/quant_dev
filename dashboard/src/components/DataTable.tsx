import type { Table } from '../api';
import { cell } from '../format';

/** A table from the API, columns in the server's order. Numbers align right. */
export function DataTable({ table, caption, maxRows }: { table: Table; caption?: string; maxRows?: number }) {
  const rows = maxRows ? table.records.slice(0, maxRows) : table.records;
  const numeric = new Set(
    table.columns.filter((c) => table.records.some((r) => typeof r[c] === 'number')),
  );
  return (
    <div className="table-wrap">
      <table>
        {caption ? <caption>{caption}</caption> : null}
        <thead>
          <tr>
            {table.columns.map((c) => (
              <th key={c} scope="col" className={numeric.has(c) ? 'num' : undefined}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((record, i) => (
            <tr key={i}>
              {table.columns.map((c) => (
                <td key={c} className={numeric.has(c) ? 'num' : undefined}>{cell(record[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {maxRows && table.records.length > maxRows ? (
        <p className="note">Showing {maxRows} of {table.records.length} rows.</p>
      ) : null}
    </div>
  );
}
