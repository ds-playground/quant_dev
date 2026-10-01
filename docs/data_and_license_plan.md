# Data and license plan (`dev/data_and_license`)

## Picking this up

- **Branch:** `dev/data_and_license`, cut from `master` at `8f4747e` (the merge of the documentation
  update, PR #11).
- **Working rules:** one phase per request, then stop. Each phase is committed and pushed, and
  its commit is recorded here. Tests stay offline.
- **Owner's decisions (2026-10-01):**
  - market data is no longer committed: users download the default tickers themselves, into the
    git-ignored `data/local`;
  - the demo data goes, and tests, screenshots and examples use a synthetic series;
  - the code is MIT-licensed;
  - after this branch merges, the history is rewritten to remove `data/demo`, and the archive
    branches are deleted.

## Context

`data/demo` held ten tickers of Yahoo Finance data with 0.01% noise added. The noise does not make
it anyone's data but Yahoo's, so committing it to a public repo risks Yahoo's terms, and no license
of ours can cover it. `save_local` and `scripts/update_local_data.py` can already save live data
locally. The missing pieces are a one-click download of the default list, and offline test data
that is not derived from the market.

## Phases

| # | Phase | Deliverable |
|---|---|---|
| **0** ✅ | License | `LICENSE` (MIT, Jones Wan); README License section with what it does not cover; `license` in `pyproject.toml` and `package.json` — **done, `c304972`** |
| **1** ✅ | Save all | `save_all` in `store.py`: `save_local` over the default list, one failure not stopping the rest; `POST /api/local/update-all`; the refresh script uses it; tests — **done, `97c1fe0`** |
| **2** ✅ | Download-all button | the dashboard button, with each ticker's result — **done, `9c1e012`** |
| **3** ✅ | Synthetic data | a seeded synthetic series for the tests, end-to-end checks and screenshots; the `demo` source, `data/demo`, its script and config removed — **done, `8f9922f`** |
| **4** ✅ | Docs and notebooks | screenshots and examples retaken on the synthetic series; notebooks read saved data when there is some, with the synthetic tickers for offline runs — **done, `d84b9da`** |
| **5** | History rewrite (after merge) | `data/demo` removed from every commit; archive branches deleted. Run only with the owner's confirmation, after showing the commands |

## Phase 0: License ✅

- `LICENSE` is the standard MIT text, "Copyright (c) 2026 Jones Wan".
- The README's License section replaces "all rights reserved". It names what MIT does not cover:
  - the Pine scripts (their upstream licenses);
  - `ta_tools`' `trendlines`, a port of LuxAlgo's CC BY-NC-SA 4.0 script, which its docstring also
    notes;
  - market data.
- `pyproject.toml` declares `license = "MIT"`, so it needs setuptools 77 or newer. A built wheel
  carries `License-Expression: MIT` and the file.

## Phase 4: Docs and notebooks ✅

- **Screenshots:**
  - all eight retaken with `npm run screenshots` on SYN-INDEX (six tickers in Multi-ticker), and
    each one inspected;
  - every number quoted in `docs/dashboard.md` re-read from them. For example, the bootstrap
    example now shows the normal model under-predicting a 3-day fall of more than 5% by about
    60 times, while the Student-t matches, as it should for Student-t shocks.
- **`docs/api.md`:** the overview, rare-event, chart, comparison and Python examples re-captured
  from `python -m src.api` on SYN-INDEX, SYN-OIL and SYN-FX. The Python example now shows an open
  drawdown (no recovery by `end_date`), which the page explains. The error example, the
  defaults, the data-source table and the sequence diagram's timing are updated.
- **Notebooks (no outputs committed, so source edits only, each cell keeping its format):**
  - `api_examples`: the synthetic set by default, or `'local'` / `'yahoo'`;
  - `price_return_statistics`: a synthetic ticker runs offline, a saved one reads `data/local`,
    and anything else downloads;
  - `rare_case_run`: saved tickers come from `data/local` and the rest download; one line
    switches to the synthetic tickers;
  - `price_return_analysis`: the `data_source` comment lists the sources.

  All four were run headless (nbclient) with no errors: the first two as committed, and the
  statistics and multi-ticker notebooks on their offline settings, since Yahoo is blocked where
  this was written.
- **README:** the notebook table and a changelog line. Every "demo" left in the README is in the
  changelog, which records history.
- **Checks:** 337 pytest, the typecheck, 21 unit and 10 end-to-end tests; five Mermaid blocks
  render; every relative link and anchor resolves.

## Phase 3: Synthetic data ✅

- **Examples shown, approved as shown:**
  - six tickers' price paths: SYN-INDEX, SYN-TECH, SYN-GOLD, SYN-FX, SYN-OIL, SYN-LEV;
  - SYN-INDEX's returns, fat-tailed distribution and autocorrelation.

  The owner chose a `synthetic` data source generated on use, not committed files.
- **Built:**
  - `src/tools/price_return/synthetic.py`: GARCH(1,1) with Student-t shocks, seeded from the
    ticker name. It adds a scheduled sell-off per ticker, a cap on daily moves, and SYN-OIL's
    negative settle on 2020-04-20. Business days to 2026-09-30, no files.
  - `synthetic_tickers()` and `synthetic_bars()` are exported.
  - `data_source='synthetic'`; `configs/synthetic_tickers.yaml` (FX and the leveraged fund get
    their own thresholds).
  - The API's offline ticker set is `synthetic` (version 0.4.0).
  - The dashboard: "Synthetic data (offline)", with a notice that it is not market data.
- **Removed:** `data/demo/`, `scripts/make_demo_data.py`, `configs/demo_tickers.yaml`, the
  `'demo'` source, `demo_tickers`, `DEMO_CONFIG_PATH` and `tests/test_demo_data.py`.
- **Tests:**
  - `tests/test_synthetic_data.py` checks properties, not exact numbers, since NumPy may change
    its random stream:
    - well-formed business-day bars, the same on every call;
    - a caller's edit does not reach the cache;
    - independent tickers (pairwise correlation under 0.2);
    - volatility near the profile's;
    - excess kurtosis above 2;
    - lag-1 autocorrelation of absolute returns above 0.1, and of returns under 0.1;
    - the sell-off (index drawdown beyond 15%);
    - the negative settle dropping two returns;
    - the source, the date filter and its errors;
    - the config;
    - the Yahoo path against SYN-OIL.
  - The Yahoo stand-in in `conftest.py` serves SYN-OIL's bars as CL=F.
  - The API, price-return, local-store and serve tests moved to synthetic tickers.
- **Counts:** 337 pytest tests, passing from the repo root and `tests/`; 21 unit tests; 10
  end-to-end tests (now on SYN-INDEX and SYN-OIL, with six tickers in Multi-ticker).
- **Eight seeded bugs,** each failing a test:
  - no volatility clustering;
  - normal shocks;
  - no negative settle;
  - one seed for all tickers (caught after the correlation test replaced a weaker one);
  - the cached frame handed out;
  - no sell-off;
  - the source reading Open;
  - an inclusive end date.
- **README:** a Synthetic data section replaces Demo data, plus the data sources, the layout,
  Package API, Tests, License and changelog.
- **Left for phase 4:**
  - the notebooks, which still name `demo`;
  - `docs/api.md` and `docs/dashboard.md`, whose examples and screenshots are of the demo data;
  - the README's notebook row.

## Phase 2: Download-all button ✅

- **The owner's choice:** one request to `POST /api/local/update-all`, not one per ticker. It is
  simpler, and the result lists every ticker at the end; there is no progress until then.
- **Built:**
  - `DownloadAll` in `dashboard/src/components/SavedData.tsx`, shown in the "Nothing saved yet"
    card, and in the panel under Yahoo Finance and Saved CSV;
  - the button names the count from the configured list ("Download all 12 default tickers");
  - while it runs, it says how long to expect;
  - afterwards it shows a summary line ("4 tickers: 2 saved, 1 updated, 1 failed.") and an
    **Each ticker** list, and every view refetches;
  - when every ticker fails (a 502), the list shows each error. For this, `ApiError` now keeps the
    server's raw detail.
- **Tests:**
  - unit: `describeSaveAll`, and `failuresOf` (21 unit tests);
  - end to end: a new test with the endpoint answered in the browser, since a real run would
    download from Yahoo into `data/local`. It checks that one click sends one request and that
    the result and failure are listed (10 end-to-end tests).
- **Checked:** screenshots before, during and after a run, on the empty saved set.
- **Docs:** `docs/dashboard.md` (Data choices), the README's Saved live data section, and the
  changelog.

## Phase 1: Save all ✅

- **Built as planned:**
  - `save_all` in `store.py`, exported from the package;
  - `POST /api/local/update-all`, whose body symbols are validated like the rest of the API's;
  - `scripts/update_local_data.py`, now a thin wrapper with the same output and exit code.
- **Docs:**
  - `docs/api.md`: the row, the endpoint map, and the Errors note (update-all, like
    multi-ticker, fails only when every symbol does);
  - the README's Saved live data section and Package API table, and a changelog line.
- **Tests:** five new, 329 in all, passing from the repo root and from `tests/`. The routes test
  failed until `docs/api.md` listed the endpoint.
- **Seven seeded bugs,** each failing a test:
  - stopping at the first failure;
  - no de-duplication;
  - ignoring the symbols argument;
  - a 502 on any failure;
  - the endpoint ignoring its body;
  - the script's exit code ignoring failures;
  - the progress callback never called.

**As planned:**

- **Package:** `save_all(symbols=None, start_date, directory, now, config_path, on_result)` in
  `store.py`.
  - It calls `save_local` for each symbol, by default every ticker in `configs/tickers.yaml`,
    in order and without duplicates.
  - A symbol that fails is listed under `failed` with its error, and the rest continue.
  - `on_result` reports each symbol as it finishes, for progress.
  - It returns `{'saved': [...], 'failed': [...]}`.
- **API:** `POST /api/local/update-all`.
  - The body is optional: `{"symbols": [...]}`; without it, the default list.
  - The `start_date` query works as for one symbol.
  - Partial failures are listed in the answer; if every symbol fails, the answer is 502.
- **Script:** `scripts/update_local_data.py` calls `save_all`, with the same output and exit
  code.
- **Tests:**
  - the default list is the config's, in order;
  - one failure does not stop the others;
  - a second run downloads only the overlap;
  - the endpoint's body, failures and limits;
  - the script, unchanged.
- **Docs:** the README (Saved live data, Package API) and `docs/api.md`. The routes test requires
  the new route there.
