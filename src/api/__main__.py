"""Run the dashboard and its API: `python -m src.api`, then open http://127.0.0.1:8000.

    python -m src.api                  # http://127.0.0.1:8000, this computer only
    python -m src.api --port 8080 --open
    python -m src.api --reload         # restart on Python changes (development)

The dashboard is served from dashboard/dist; build it once with `npm install && npm run build`
in dashboard/. Without a build the API still runs, and the page at / says how to build.
"""
import argparse
import sys
import webbrowser

LOCAL_HOSTS = {'127.0.0.1', 'localhost', '::1'}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m src.api',
                                     description='Run the price-return dashboard and its API.')
    parser.add_argument('--host', default='127.0.0.1',
                        help='address to listen on (default 127.0.0.1: this computer only)')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true', help='restart when Python files change')
    parser.add_argument('--open', action='store_true', help='open the dashboard in a browser')
    args = parser.parse_args(argv)

    import uvicorn

    from .dashboard import DIST, is_built

    url = f'http://{"127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host}:{args.port}'
    print(f'Dashboard and API: {url}   (API docs: {url}/docs; stop with Ctrl+C)')
    if not is_built():
        print(f'The dashboard is not built ({DIST} is missing). Build it once:\n'
              f'    cd dashboard && npm install && npm run build\n'
              f'The API runs meanwhile.')
    if args.host not in LOCAL_HOSTS:
        print(f'Warning: listening on {args.host}, so other machines can reach this server. '
              f'It has no login; use this only on a network you trust.')
    if args.open:
        webbrowser.open(url)
    uvicorn.run('src.api.app:app', host=args.host, port=args.port, reload=args.reload,
                log_level='info')
    return 0


if __name__ == '__main__':
    sys.exit(main())
