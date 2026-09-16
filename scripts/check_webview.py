"""Check that the installed desktop backend can call Python from JavaScript."""

import os
import threading

import webview


class Bridge:
    """Minimal API used only by this installation check."""

    def ping(self):
        """Return a value the browser must send back through a second API call."""
        return "bridge-ok"

    def finish(self, value):
        """Record completion only after a browser-to-Python round trip."""
        if value == "bridge-ok":
            completed.set()
        window.destroy()


completed = threading.Event()
window = webview.create_window(
    "Backend verification",
    html="""<html><body><script>
    window.addEventListener('pywebviewready', async () => {
      const value = await window.pywebview.api.ping();
      await window.pywebview.api.finish(value);
    });
    </script></body></html>""",
    js_api=Bridge(),
    hidden=True,
)


def timeout():
    """Fail a hung backend without leaving the CI process/window alive."""
    if not completed.wait(45):
        print("Backend bridge did not respond within 45 seconds", flush=True)
        os._exit(1)


threading.Thread(target=timeout, daemon=True).start()
webview.start()
if not completed.is_set():
    raise SystemExit("Backend closed before completing its bridge check")
print("Desktop backend JavaScript/Python bridge passed")
