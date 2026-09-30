import { describe, expect, it } from 'vitest';

import { varTable } from './tables';

describe('varTable', () => {
  it('puts each method in its own VaR and ES column, one row per horizon and level', () => {
    const long = {
      columns: ['horizon', 'level', 'method', 'var', 'es', 'n_windows'],
      records: [
        { horizon: 1, level: 0.95, method: 'historical', var: 0.02, es: 0.03, n_windows: 100 },
        { horizon: 1, level: 0.95, method: 'normal', var: 0.018, es: 0.022, n_windows: 100 },
        { horizon: 5, level: 0.99, method: 'historical', var: 0.06, es: 0.08, n_windows: 96 },
        { horizon: 5, level: 0.99, method: 'normal', var: 0.05, es: 0.055, n_windows: 96 },
      ],
    };
    expect(varTable(long)).toEqual({
      columns: ['horizon', 'level', 'VaR historical', 'VaR normal', 'ES historical', 'ES normal'],
      records: [
        { horizon: 1, level: 0.95, 'VaR historical': 0.02, 'ES historical': 0.03, 'VaR normal': 0.018, 'ES normal': 0.022 },
        { horizon: 5, level: 0.99, 'VaR historical': 0.06, 'ES historical': 0.08, 'VaR normal': 0.05, 'ES normal': 0.055 },
      ],
    });
  });
});
