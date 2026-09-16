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



def check_sidebar_layout(window):
    """Measure actual rendered controls and tooltip bounds across all themes/languages."""
    window.resize(800, 600)
    window.evaluate_js("""window.sidebarProbe = null; (async () => {
        const app = window.appInstance, rows = [];
        for (const language of app.availableLanguages) {
            app.t = await loadLanguage(language.id);
            await Alpine.nextTick();
            for (const theme of app.availableThemes) {
                switchTheme(theme.id);
                const aside = document.querySelector('aside');
                void aside.offsetWidth;
                await document.fonts.ready;
                const bounds = aside.getBoundingClientRect();
                const outside = [...aside.querySelectorAll('button, input, select')]
                    .filter(node => node.getClientRects().length)
                    .filter(node => node.getBoundingClientRect().right > bounds.right + 1)
                    .map(node => node.tagName);
                // Invisible pseudo-elements still affect the scrollable width.
                // The same box must remain inside when its tooltip becomes visible.
                const hints = [...aside.querySelectorAll('[data-tooltip]')]
                    .filter(node => node.dataset.tooltip)
                    .map(node => {
                        const hint = getComputedStyle(node, '::after');
                        const host = node.getBoundingClientRect();
                        const right = hint.right === 'auto'
                            ? host.left + parseFloat(hint.left) + parseFloat(hint.width)
                            : host.right - parseFloat(hint.right);
                        return right <= bounds.right + 1;
                    });
                rows.push({theme: theme.id, language: language.id,
                    client: aside.clientWidth, scroll: aside.scrollWidth,
                    outside, hintsInside: hints.every(Boolean)});
            }
        }
        return rows;
    })().then(rows => { window.sidebarProbe = {rows}; })
        .catch(error => { window.sidebarProbe = {error: String(error)}; });""")
    deadline = time.monotonic() + 40
    probe = None
    while time.monotonic() < deadline:
        probe = window.evaluate_js('window.sidebarProbe')
        if probe:
            break
        time.sleep(.1)
    assert probe and 'rows' in probe, probe
    results = probe['rows']
    assert isinstance(results, list) and results, results
    for row in results:
        assert row['scroll'] <= row['client'] + 1, row
        assert not row['outside'] and row['hintsInside'], row
    print('SIDEBAR PASS', len(results), 'theme/language combinations', flush=True)


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
        check_sidebar_layout(window)
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
