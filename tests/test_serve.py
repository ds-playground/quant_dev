"""Serving the built dashboard from the API server, and `python -m src.api`."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import src.api.__main__ as runner
from src.api import dashboard
from src.api.app import app

client = TestClient(app)


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A stand-in build: index.html and one hashed asset, plus a secret beside the folder."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>dashboard</title>")
    (dist / "assets" / "index-abc123.js").write_text("console.log('app')")
    (tmp_path / "secret.txt").write_text("not for the browser")
    monkeypatch.setattr(dashboard, "DIST", dist)
    return dist


def test_the_page_and_its_assets_are_served(built):
    page = client.get("/")
    assert page.status_code == 200 and "<title>dashboard</title>" in page.text
    assert page.headers["cache-control"] == "no-cache"
    asset = client.get("/assets/index-abc123.js")
    assert asset.status_code == 200 and asset.text == "console.log('app')"
    assert "immutable" in asset.headers["cache-control"]
    assert client.get("/api/health").json()["dashboard_built"] is True


def test_the_api_and_its_docs_come_first(built):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200 and "swagger" in client.get("/docs").text.lower()
    assert client.get("/openapi.json").json()["info"]["title"] == "quant_dev API"
    missing = client.get("/api/nope")
    assert missing.status_code == 404 and "dashboard" not in missing.text
    assert client.post("/api/overview", json={"data_source": "synthetic", "ticker": "SYN-INDEX"}).status_code == 200


def test_other_paths_get_the_page_but_missing_files_do_not(built):
    assert "<title>dashboard</title>" in client.get("/statistics").text
    assert client.get("/assets/gone-123.js").status_code == 404


@pytest.mark.parametrize("path", ["/..%2Fsecret.txt", "/assets/..%2F..%2Fsecret.txt",
                                  "/%2E%2E/secret.txt"])
def test_nothing_outside_the_build_is_served(built, path):
    response = client.get(path)
    assert "not for the browser" not in response.text


def test_before_a_build_the_page_says_how_to_build(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "DIST", tmp_path / "dist")
    page = client.get("/")
    assert page.status_code == 503 and "npm run build" in page.text and "/docs" in page.text
    assert client.get("/api/health").json()["dashboard_built"] is False
    assert client.get("/api/tickers").status_code == 200           # the API still works


def test_runner_defaults_to_this_computer_only(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append((app, kwargs)))
    assert runner.main([]) == 0
    assert calls == [("src.api.app:app", {"host": "127.0.0.1", "port": 8000, "reload": False,
                                          "log_level": "info"})]
    assert "http://127.0.0.1:8000" in capsys.readouterr().out


def test_runner_warns_when_reachable_from_other_machines(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: None)
    monkeypatch.setattr(dashboard, "DIST", tmp_path / "dist")
    runner.main(["--host", "0.0.0.0", "--port", "9000", "--reload"])
    out = capsys.readouterr().out
    assert "http://127.0.0.1:9000" in out and "no login" in out and "npm run build" in out


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX process groups")
def test_dev_launcher_stops_a_server_and_what_it_started(tmp_path):
    """`npm run dev` runs Vite as a child; stopping npm alone left Vite running."""
    import importlib.util
    import os
    import sys
    import time

    spec = importlib.util.spec_from_file_location(
        "dev", dashboard._repo_root() / "scripts" / "dev.py")
    dev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dev)

    child_pid = tmp_path / "child.pid"
    parent = dev.start([sys.executable, "-c",
                        "import subprocess, sys, time;"
                        f"c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
                        f"open({str(child_pid)!r}, 'w').write(str(c.pid)); time.sleep(60)"], tmp_path)
    for _ in range(100):
        if child_pid.exists() and child_pid.read_text():
            break
        time.sleep(0.05)
    child = int(child_pid.read_text())
    dev.stop(parent)
    assert parent.poll() is not None
    for _ in range(100):                     # the grandchild goes too
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child, 9)
        pytest.fail("the child process outlived its parent's stop")
