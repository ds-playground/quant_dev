"""Development: the API (restarting on Python changes) and the Vite dev server (hot reload),
together, until Ctrl+C. Works on Windows, macOS and Linux.

    python scripts/dev.py     # then open http://localhost:5173

Vite forwards /api to the API on :8000. For everyday use, build once and run
`python -m src.api` instead: one server, at http://127.0.0.1:8000.
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / 'dashboard'


# Each server runs in its own process group: `npm run dev` starts Vite as a child and uvicorn's
# reloader starts a worker, and stopping only the parent would leave those running.
WINDOWS = os.name == 'nt'
GROUP = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS
         else {'start_new_session': True})


def start(command, cwd):
    return subprocess.Popen(command, cwd=cwd, **GROUP)


def stop(process):
    """Stop a server and everything it started."""
    if process.poll() is not None:
        return
    if WINDOWS:
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if not WINDOWS:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    npm = shutil.which('npm')
    if npm is None:
        print('npm is not on PATH: install Node 20 or newer (https://nodejs.org).')
        return 1
    if not (DASHBOARD / 'node_modules').is_dir():
        print('Installing the dashboard packages (once)...')
        subprocess.run([npm, 'install'], cwd=DASHBOARD, check=True)

    api = start([sys.executable, '-m', 'uvicorn', 'src.api.app:app', '--reload',
                 '--host', '127.0.0.1', '--port', '8000'], ROOT)
    vite = start([npm, 'run', 'dev'], DASHBOARD)
    print('API on http://127.0.0.1:8000, dashboard on http://localhost:5173 (Ctrl+C stops both)')
    stopped_by_user = False
    try:
        while api.poll() is None and vite.poll() is None:     # stop when either one stops
            time.sleep(0.5)
    except KeyboardInterrupt:
        stopped_by_user = True
    finally:
        for process in (vite, api):
            stop(process)
    if not stopped_by_user:
        print('One of the servers stopped on its own; see its output above.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
