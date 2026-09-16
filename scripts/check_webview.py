"""Check that the installed desktop backend can call Python from JavaScript."""

import os
import threading
import time

import webview


class Bridge:
    """Minimal API used only by this installation check."""

    def ping(self):
        """Return a value for the browser to expose after resolving its API call."""
        return "bridge-ok"


completed = threading.Event()
finished = threading.Event()
window = webview.create_window(
    "Backend verification",
    html="""<html><body><script>
    window.addEventListener('pywebviewready', async () => {
      window.bridgeResult = await window.pywebview.api.ping();
    });
    </script></body></html>""",
    js_api=Bridge(),
    hidden=True,
)


def check_bridge():
    """Close only after the browser has received its Python API response."""
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if window.evaluate_js("window.bridgeResult || null") == "bridge-ok":
            completed.set()
            break
        time.sleep(0.1)
    window.destroy()


def watchdog():
    """Fail even if startup, a native JS call or window teardown hangs."""
    if not finished.wait(60):
        print("Backend bridge/startup/teardown timed out", flush=True)
        os._exit(1)


threading.Thread(target=watchdog, daemon=True).start()
webview.start(check_bridge)
finished.set()
if not completed.is_set():
    raise SystemExit("Backend closed before completing its bridge check")
print("Desktop backend JavaScript/Python bridge passed")
