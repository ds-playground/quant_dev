import { describe, expect, it } from 'vitest';

import { cell, isoDate, num, pct } from './format';

describe('format', () => {
  it('shows decimals as percentages', () => {
    expect(pct(0.14716)).toBe('14.72%');
    expect(pct(0.000583, 3)).toBe('0.058%');
    expect(pct(-0.339, 1)).toBe('-33.9%');
  });

  it('shows missing and non-finite values as a dash', () => {
    for (const value of [null, undefined, NaN, Infinity]) {
      expect(pct(value)).toBe('–');
      expect(num(value)).toBe('–');
    }
    expect(cell(null)).toBe('–');
  });

  it('formats cells by type', () => {
    expect(cell(2699)).toBe('2,699');
    expect(cell(0.096153846)).toBe('0.09615');
    expect(cell(-37.621931)).toBe('-37.62');
    expect(cell(7671.704656)).toBe('7,671.7');
    expect(cell('2020-03-23T00:00:00')).toBe('2020-03-23');
    expect(cell('2026-09-16')).toBe('2026-09-16');
    expect(cell('16.72 / 10.86')).toBe('16.72 / 10.86');
    expect(cell(true)).toBe('true');
  });

  it('cuts ISO datetimes to the date', () => {
    expect(isoDate('2016-01-05T00:00:00')).toBe('2016-01-05');
    expect(isoDate(null)).toBe('–');
  });
});
