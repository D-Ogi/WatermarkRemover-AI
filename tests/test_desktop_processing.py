"""Regression checks for desktop worker completion and cancellation reports."""
import json
import sys
import threading
import time

import pytest

pytest.importorskip("webview")
import remwmgui


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Capture bridge events while using real child processes and no model downloads."""
    class Prepared:
        def start(self, **kwargs):
            return {"status": "started"}
        def status(self):
            return {"status": "ready"}
        def close(self):
            pass
    monkeypatch.setattr(remwmgui, "ModelPreparation", Prepared)
    monkeypatch.setattr(remwmgui, "CONFIG_FILE", str(tmp_path / "ui.yml"))
    instance = remwmgui.Api()
    instance.events = []
    instance._call_js = instance.events.append
    yield instance
    instance._close()


def completion(api):
    """Decode the externally visible completion payload sent to the page."""
    event = next(x for x in reversed(api.events) if x.startswith("processingComplete("))
    return json.loads(event.removeprefix("processingComplete(").removesuffix(")"))


@pytest.mark.parametrize("code", [0, 7])
def test_exit_status_is_not_always_success(api, code):
    api.is_running = True
    api._run_process([sys.executable, "-c", f"raise SystemExit({code})"])
    result = completion(api)
    assert result == {"success": code == 0, "cancelled": False, "exit_code": code}
    assert not api.is_running and api.process is None


def test_cancel_running_worker_reaps_child(api):
    api.is_running = True
    thread = threading.Thread(target=api._run_process,
                              args=([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"],))
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while api.process is None and time.monotonic() < deadline:
            time.sleep(.01)
        process = api.process
        assert process is not None
        api.stop_processing()
        thread.join(10)
        assert not thread.is_alive() and process.poll() is not None
        result = completion(api)
        assert result["cancelled"] and not result["success"]
        assert not api.is_running and api.process is None
    finally:
        api.stop_processing()
        thread.join(10)


def test_cancel_before_spawn_still_completes_page(api, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Cancelled job spawned a worker")
    monkeypatch.setattr(remwmgui.subprocess, "Popen", forbidden)
    api.is_running = True
    api.stop_processing()
    api._run_process([sys.executable, "-c", "pass"])
    assert 'processingComplete({success: false, cancelled: true})' in api.events
    assert not api.is_running and api.process is None
