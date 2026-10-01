import { afterEach, describe, expect, it, vi } from 'vitest';

import { api, ApiError, type Params } from './api';

const params = { ticker: 'SYN-INDEX', data_source: 'synthetic' } as Params;

function mockFetch(status: number, body: unknown) {
  const fetch = vi.fn(async (_url: string, _init?: RequestInit) => new Response(JSON.stringify(body), { status }));
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

afterEach(() => vi.unstubAllGlobals());

describe('api', () => {
  it('posts the parameters as JSON and repeats list query values', async () => {
    const fetch = mockFetch(200, { events: { columns: [], records: [] } });
    await api.rareEvents(params, [3, 5], 'cumulative');
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe('/api/rare-events?n_days=3&n_days=5&change_type=cumulative');
    expect(init?.method).toBe('POST');
    expect(JSON.parse(init?.body as string)).toEqual(params);
  });

  it('leaves out empty query values', async () => {
    const fetch = mockFetch(200, {});
    await api.rareEvents(params, [3], null);
    expect(fetch.mock.calls[0][0]).toBe('/api/rare-events?n_days=3');
  });

  it('turns an error response into an ApiError with the server explanation', async () => {
    mockFetch(422, { detail: "No synthetic data for ticker 'NOPE'." });
    await expect(api.overview(params)).rejects.toEqual(
      new ApiError(422, "No synthetic data for ticker 'NOPE'.", "No synthetic data for ticker 'NOPE'."));
  });

  it('names the field in a validation error', async () => {
    mockFetch(422, { detail: [{ loc: ['query', 'n_boot'], msg: 'Input should be less than or equal to 2000' }] });
    await expect(api.statistics(params, 'probabilities', { n_boot: 5000 }))
      .rejects.toMatchObject({ status: 422, detail: 'Input should be less than or equal to 2000: n_boot' });
  });
});
