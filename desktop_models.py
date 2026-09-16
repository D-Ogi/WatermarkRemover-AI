"""Keep model verification/downloads off the desktop bridge and UI threads."""

import json
import subprocess
import threading

from desktop_runtime import APP_ROOT, python_executable, worker_options


class ModelPreparation:
    """One preparation worker per window, with pollable progress and explicit retry."""

    def __init__(self):
        """Start idle; callers decide when to check or download model assets."""
        self._lock = threading.Lock()
        self._process = None
        self._running = False
        self._closed = False
        self._state = dict(status="unchecked", message="Checking model files...")

    def status(self):
        """Return a copy so bridge serialization never races with worker updates."""
        with self._lock:
            return dict(self._state, busy=self._running)

    def start(self, *, check=False):
        """Start checking or downloading; completed/failed jobs can be retried."""
        with self._lock:
            if self._closed:
                return {"error": "Window is closing"}
            if self._running:
                return {"error": "Model preparation is already running"}
            self._running = True
            self._state = dict(status="checking", message="Checking model files...")
        threading.Thread(target=self._run, args=(check,), daemon=True).start()
        return {"status": "started"}

    def _run(self, check):
        """Consume structured progress while keeping errors visible and retryable."""
        try:
            cmd = [python_executable(), str(APP_ROOT / "model_assets.py")]
            if check:
                cmd.append("--check")
            with self._lock:
                if self._closed:
                    return
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           **worker_options())
                self._process = process
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and "status" in event:
                    with self._lock:
                        self._state = event
            process.stdout.close()
            code = process.wait()
            with self._lock:
                if code or self._state.get("status") != "ready":
                    if self._state.get("status") not in {"missing", "error"}:
                        self._state = dict(status="error", message=f"Model preparation stopped (exit {code}). Please retry.")
        except Exception as exc:
            with self._lock:
                self._state = dict(status="error", message=str(exc))
        finally:
            with self._lock:
                self._process = None
                self._running = False

    def close(self):
        """Terminate only this window's worker; downloads remain safe to retry."""
        with self._lock:
            self._closed = True
            process = self._process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
