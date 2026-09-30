import { describe, expect, it } from 'vitest';

import { SYMBOL, tidySymbol } from './addedTickers';

describe('symbols', () => {
  it('tidies what is typed', () => {
    expect(tidySymbol('  aapl ')).toBe('AAPL');
    expect(tidySymbol('^gspc')).toBe('^GSPC');
  });

  it('accepts Yahoo symbols and nothing that could be a path', () => {
    for (const s of ['AAPL', '^GSPC', 'ES=F', 'EURUSD=X', 'BRK-B', 'BRK.B', 'RDS_A']) expect(SYMBOL.test(s)).toBe(true);
    for (const s of ['', 'A B', '../X', 'A/B', 'X'.repeat(21)]) expect(SYMBOL.test(s)).toBe(false);
  });
});
