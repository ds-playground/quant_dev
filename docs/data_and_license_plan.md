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
| **1** ✅ | Save all | `save_all` in `store.py`: `save_local` over the default list, one failure not stopping the rest; `POST /api/local/update-all`; the refresh script uses it; tests |
| **2** | Download-all button | the dashboard button, with each ticker's result |
| **3** | Synthetic data | a seeded synthetic series for the tests, end-to-end checks and screenshots; the `demo` source, `data/demo`, its script and config removed |
| **4** | Docs and notebooks | screenshots and examples retaken on the synthetic series; notebooks default to saved data, falling back to simulated |
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
