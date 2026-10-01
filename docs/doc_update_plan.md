# Documentation update plan (`dev/doc_update`)

## Picking this up

- **Branch:** `dev/doc_update`, cut from `master` at `4f64225` (the merge of the dashboard POC,
  PR #10, also kept as `archive/03_dashboard_poc`).
- **Working rules:**
  - One phase per request.
  - Each phase opens by showing the owner example diagrams and stopping for a choice. Only then
    are the contents written, committed and pushed, and the commit is recorded here.
  - Every command and request shown on a page is run against the code before it is committed.
- **Diagrams** are Mermaid blocks in the Markdown, which GitHub renders. The examples are rendered
  to PNG outside the repo, with `mermaid` from npm in Chromium, and the same render checks each
  committed block.

## Context

The dashboard POC is merged, and the repo is going public. The README described the four parts
but did not show how they work together. The HTTP API and the dashboard were each covered only by
a paragraph and by `/docs` (Swagger), with no page a newcomer could read on its own. The owner
asked for:

1. **README:** a diagram of the development workflow, in which the dashboard is used to develop
   and to present analyses on top of continuously developed Python packages.
2. **API page:** how to run the API on its own, a list of the endpoints, and how queries work.
3. **Dashboard page:** a screenshot of each tab, with an explanation of its analysis.

## Phases

| # | Phase | Deliverable |
|---|---|---|
| **1** ✅ | README workflow diagram | `## Development workflow` after `## Purpose`; `## API` renamed `## Package API`; this plan — **done, `1585e90`** |
| **2** | API page | `docs/api.md`: run the API alone, every endpoint, how a query works, worked examples; a test that every route is documented |
| **3** | Dashboard page | `docs/dashboard.md`: a screenshot and explanation per tab; `npm run screenshots` to regenerate the images in `docs/images/dashboard/` |
| **4** | Wrap-up | README layout, Tests and changelog; full checks; PR when asked |

## Phase 1: README workflow diagram ✅

- **Examples shown:**
  - A (a cycle);
  - B (layers: dashboard, API, packages, with notebooks on the side);
  - C (a research loop and a delivery loop joined by the packages).
- **The owner chose A, with these edits:**
  - the cycle starts from **"A trading idea"**;
  - the notebook step also **explores the methodology**;
  - the notebook-to-package arrow reads **"formulate"**;
  - the loop back reads **"a new idea"**.
- **Contents:**
  - the diagram, plus four numbered steps (notebook, package, API, dashboard) and the rule that
    analysis lives only in the packages;
  - the package function table is now headed `## Package API`, since the HTTP API gets its own
    page (no links pointed at the old anchor);
  - a changelog entry.

## Phase 2: API page (`docs/api.md`)

- **Examples first:**
  - the request flow: client → FastAPI validation → cache → `price_return` → JSON and Plotly
    figures;
  - the endpoint map, by group;
  - the rare-event sequence: the full table is built and cached once, and later bound changes
    only refilter.
- **Contents:**
  1. Run the API alone (`python -m src.api` or `uvicorn src.api.app:app --reload`, `--port`,
     `--host`, `/docs`, `/openapi.json`).
  2. A table of every route in `src/api/app.py`: method, path, purpose, inputs.
  3. How a query works:
     - the body is `Params`'s fields, and unknown fields give 422;
     - data sources, with `end_date` exclusive;
     - query parameters per endpoint;
     - response shapes;
     - errors;
     - caching.
  4. Worked examples (curl and Python), with trimmed real output on the demo data.
  5. A test that every `/api/*` route appears in `docs/api.md`.

## Phase 3: Dashboard page (`docs/dashboard.md`)

- **Examples first:**
  - the page map: the parameter panel → five tabs → the endpoints each calls;
  - the data-source choice: Demo, Yahoo, Saved CSV, Other ticker…;
  - a suggested reading path through the tabs.
- **Screenshots:**
  - `dashboard/e2e/screenshots.spec.ts`, run by `npm run screenshots` and not part of
    `npm run e2e`, captures the demo data (SPX);
  - one image per tab, the rare-event filters, dark theme and phone width;
  - 1280 px wide, compressed, and every image inspected.
- **Contents per tab:**
  - what question the analysis answers;
  - how to read each table and chart;
  - the controls;
  - the package functions behind it.

## Phase 4: Wrap-up

- **README:**
  - the layout lists `docs/api.md`, `docs/dashboard.md`, `docs/images/` and the screenshots
    spec;
  - the Tests section covers the routes-documented test;
  - the changelog.
- **Checks:** `pytest`, `npm run typecheck`, `npm test`, `npm run e2e`, `npm run screenshots`.
- **PR** into `master` only when the owner asks.
