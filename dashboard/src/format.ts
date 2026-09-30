// Display formatting only: the API sends raw values, as format_probability_table does for the
// notebooks, and the dashboard decides how to show them.
import type { Cell } from './api';

const DASH = '–';

/** A decimal as a percentage: 0.1472 -> "14.72%". */
export function pct(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? DASH
    : `${(value * 100).toFixed(digits)}%`;
}

/** A number with thousands separators and a fixed number of decimals. */
export function num(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? DASH
    : value.toLocaleString('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** An ISO date or datetime as YYYY-MM-DD; midnight times carry no information for daily data. */
export function isoDate(value: string | null | undefined): string {
  return value ? value.slice(0, 10) : DASH;
}

const ISO_DATE = /^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?$/;

/** One table cell: integers as they are, other numbers to 4 significant digits, dates short. */
export function cell(value: Cell): string {
  if (value === null) return DASH;
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toLocaleString('en-GB');
    return Math.abs(value) >= 1000 ? num(value, 1) : String(Number(value.toPrecision(4)));
  }
  if (typeof value === 'string' && ISO_DATE.test(value)) return isoDate(value);
  return String(value);
}
