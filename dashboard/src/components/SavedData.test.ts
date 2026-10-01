import { describe, expect, it } from 'vitest';

import { ApiError, type SaveResult } from '../api';
import { ageInDays, describeSave, describeSaveAll, failuresOf } from './SavedData';

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

describe('describeSaveAll', () => {
  it('counts first saves, updates and failures', () => {
    const failure = { symbol: 'NG=F', error: 'ConnectionError: Yahoo unreachable' };
    expect(describeSaveAll({ saved: [{ ...base, created: true }, base, base], failed: [failure] }))
      .toBe('4 tickers: 1 saved, 2 updated, 1 failed.');
    expect(describeSaveAll({ saved: [base], failed: [] })).toBe('1 ticker: 1 updated.');
  });
});

describe('failuresOf', () => {
  it('lists the failures a 502 carries when every download failed, and nothing otherwise', () => {
    const failed = [{ symbol: 'ES=F', error: 'ConnectionError: Yahoo unreachable' }];
    expect(failuresOf(new ApiError(502, 'Every download failed', { message: 'Every download failed', failed })))
      .toEqual(failed);
    expect(failuresOf(new ApiError(422, 'bad', [{ msg: 'x' }]))).toEqual([]);
    expect(failuresOf(new Error('network'))).toEqual([]);
  });
});
