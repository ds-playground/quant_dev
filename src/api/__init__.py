"""HTTP API for the dashboard: the `src.tools` packages behind JSON endpoints.

The API only wraps and serializes. Every number and chart comes from `src.tools`, so the
dashboard and the notebooks cannot drift apart. Run it with `python -m src.api` (Phase 5 of
`docs/dashboard_plan.md`) or `uvicorn src.api.app:app`; it needs the `api` extra.
"""
