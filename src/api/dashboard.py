"""Serving the built dashboard (dashboard/dist) from the API server, so one process runs both.

The dashboard is a single page: `index.html` plus hashed files under `assets/`. Any path that is
not an API route and not a file gets `index.html`. The build folder is looked up on each request,
so `npm run build` takes effect without restarting the server, and before the first build `/`
explains how to make one.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from src.tools.price_return.params import _repo_root

DIST = _repo_root() / 'dashboard' / 'dist'

# Hashed asset names change with their content, so browsers may keep them; index.html names
# the current ones, so it is always revalidated.
ASSET_CACHE = 'public, max-age=31536000, immutable'
INDEX_CACHE = 'no-cache'

NOT_BUILT = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard not built</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem;color:#0b0b0b;background:#f9f9f7}}
code,pre{{background:#eeede8;border-radius:4px;padding:2px 5px}}pre{{padding:10px 12px}}</style></head>
<body><h1>The API is running; the dashboard is not built yet</h1>
<p>Build it once (Node 20 or newer), from the repo root:</p>
<pre>cd dashboard
npm install
npm run build</pre>
<p>then reload this page. No restart is needed. Expected at <code>{dist}</code>.</p>
<p>The API itself is live: see <a href="/docs">/docs</a>.</p></body></html>"""


def is_built():
    return (DIST / 'index.html').is_file()


def add_dashboard_routes(app: FastAPI):
    """Register the dashboard's catch-all route. Call it after every API route."""

    @app.get('/{path:path}', include_in_schema=False)
    def dashboard(path: str):
        if path == 'api' or path.startswith('api/'):
            raise HTTPException(status_code=404, detail=f'No API route /{path}')
        dist = DIST.resolve()
        if not (dist / 'index.html').is_file():
            return HTMLResponse(NOT_BUILT.format(dist=dist), status_code=503)
        if path:
            target = (dist / path).resolve()
            if target.is_file() and target.is_relative_to(dist):
                cache = ASSET_CACHE if target.parent == dist / 'assets' else INDEX_CACHE
                return FileResponse(target, headers={'Cache-Control': cache})
            if '.' in Path(path).name:          # a file that is not there, not a page
                raise HTTPException(status_code=404, detail=f'No file /{path}')
        return FileResponse(dist / 'index.html', headers={'Cache-Control': INDEX_CACHE})
