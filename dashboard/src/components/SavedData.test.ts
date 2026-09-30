import { describe, expect, it } from 'vitest';

import type { SaveResult } from '../api';
import { ageInDays, describeSave } from './SavedData';

const base: SaveResult = { symbol: 'CL=F', file: 'data/local/CL=F.csv', created: false, rows: 2700,
                           added: 0, first: '2016-01-04', last: '2026-09-29', revised: [] };

describe('describeSave', () => {
  it('says what a first save wrote', () => {
    expect(describeSave({ ...base, created: true, added: 2700 }))
      .toBe('Saved CL=F: 2,700 bars, 2016-01-04 to 2026-09-29.');
  });

  it('says what an update added and lists revisions', () => {
    expect(describeSave({ ...base, added: 1 })).toBe('Updated CL=F: +1 bar, now 2,700 bars, 2016-01-04 to 2026-09-29.');
    const revised = { ...base, added: 3, revised: [{ date: '2026-09-24', column: 'Close', saved: 70.1, new: 70.4 }] };
    expect(describeSave(revised)).toBe('Updated CL=F: +3 bars, now 2,700 bars, 2016-01-04 to 2026-09-29.'
      + ' 1 saved value revised by Yahoo: 2026-09-24 Close 70.1 → 70.4.');
  });
});

describe('ageInDays', () => {
  it('counts whole calendar days since the last saved bar', () => {
    expect(ageInDays('2026-09-29', new Date('2026-09-30T10:00:00Z'))).toBe(1);
    expect(ageInDays('2026-09-25', new Date('2026-09-30T10:00:00Z'))).toBe(5);
  });
});
