"""Desktop diagnostics must not overwrite the user configuration on timeout."""
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_watchdog_leaves_user_configuration_untouched(monkeypatch, tmp_path):
    """Force the real watchdog branch after a simulated UI writes probe settings."""
    user_config = tmp_path / "ui.yml"
    original = b"theme: anime\nlang: jp\n"
    user_config.write_bytes(original)
    targets = []
    temporary_paths = []
    gui = SimpleNamespace(CONFIG_FILE=str(user_config))

    def main(*args, **kwargs):
        probe = Path(gui.CONFIG_FILE)
        temporary_paths.append(probe)
        assert probe != user_config
        probe.write_text("theme: korpo\nlang: en\n", encoding="utf-8")
        targets[0]()

    def exit_check(code):
        raise SystemExit(code)

    gui.main = main
    monkeypatch.setitem(sys.modules, "remwmgui", gui)
    monkeypatch.setitem(sys.modules, "desktop_runtime", SimpleNamespace(configure_runtime=lambda: None))
    monkeypatch.setattr("threading.Thread", lambda target, **kwargs: SimpleNamespace(start=lambda: targets.append(target)))
    monkeypatch.setattr("threading.Event.wait", lambda self, timeout=None: False)
    monkeypatch.setattr("os._exit", exit_check)
    try:
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_desktop.py"))
        assert error.value.code == 1
        assert user_config.read_bytes() == original
        assert temporary_paths
        assert not temporary_paths[0].parent.exists()
    finally:
        for probe in temporary_paths:
            probe.unlink(missing_ok=True)
            if probe.parent.exists():
                probe.parent.rmdir()
