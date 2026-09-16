"""Exercise the actual local desktop page, assets, configuration and API bridge."""
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desktop_runtime import configure_runtime
configure_runtime()
logging.getLogger().addHandler(logging.StreamHandler())
import remwmgui

finished = threading.Event()
failures = []


def exercise(window, api):
    """Read real Alpine state and execute actions through JavaScript API promises."""
    original_config = dict(api.get_config())
    try:
        assert window.events.loaded.wait(60), "Desktop page did not finish loading"
        deadline = time.monotonic() + 20
        state = None
        while time.monotonic() < deadline:
            state = window.evaluate_js("""JSON.stringify({
                ready: !!window.appInstance && !window.appInstance.isLoading,
                api: !!window.pywebview?.api,
                version: window.appInstance?.appVersion,
                translations: Object.keys(window.appInstance?.t || {}).length,
                ram: window.appInstance?.systemInfo?.ram,
                models: window.appInstance?.models?.status,
                background: getComputedStyle(document.body).backgroundColor,
                origin: location.protocol,
                resources: performance.getEntriesByType('resource').map(x => x.name)
            })""")
            state = json.loads(state) if isinstance(state, str) else state
            if state and state.get('ready') and state.get('translations') and state.get('ram', 0) > 0:
                break
            time.sleep(.2)
        assert state and state['ready'] and state['api'], state
        assert state['origin'] == 'http:', state
        assert state['translations'] >= 10 and state['version'] != '0.0', state
        assert all(url.startswith(('http://127.0.0.1:', 'http://localhost:')) for url in state['resources']), state
        window.evaluate_js("document.querySelector('#models-button').click()")
        assert window.evaluate_js('window.appInstance.modelsOpen')
        window.evaluate_js("document.querySelector('#close-models').click()")
        assert not window.evaluate_js('window.appInstance.modelsOpen')
        window.evaluate_js("window.savedProbe = null; window.pywebview.api.save_config({theme:'korpo',lang:'en'}).then(() => window.pywebview.api.get_config()).then(x => window.savedProbe = x)")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if window.evaluate_js("window.savedProbe?.theme") == 'korpo':
                break
            time.sleep(.1)
        assert window.evaluate_js("window.savedProbe?.lang") == 'en'
        print('DESKTOP PASS', json.dumps(state), flush=True)
    except BaseException as exc:
        failures.append(repr(exc))
    finally:
        api.save_config(original_config)
        window.destroy()


def watchdog():
    """Bound native window startup and shutdown as well as the page checks."""
    if not finished.wait(90):
        print('Desktop check timed out', flush=True)
        os._exit(1)


threading.Thread(target=watchdog, daemon=True).start()
# WebKit can defer network page loading for an entirely hidden native window.
# Exercise its normal visible startup on the macOS CI desktop.
remwmgui.main(exercise, hidden=sys.platform != "darwin")
finished.set()
if failures:
    raise SystemExit('; '.join(failures))
