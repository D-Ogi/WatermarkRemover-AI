"""Check configuration assets required before the UI can start."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "ui").rglob("*.json")), ids=lambda p: p.name)
def test_ui_json_is_parseable(path):
    """Each shipped JSON asset must parse as an object."""
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


def test_configured_languages_have_translation_files():
    """Every selectable locale must have a corresponding translation resource."""
    config = json.loads((ROOT / "ui/config.json").read_text(encoding="utf-8"))
    for language in config["languages"]:
        assert (ROOT / "ui/lang" / (language["id"] + ".json")).is_file()


@pytest.mark.parametrize("name", ["setup.sh", "run.sh"])
def test_shell_scripts_have_unix_line_endings(name):
    """Prevent CRLF shebang failures on fresh Linux and macOS checkouts."""
    content = (ROOT / name).read_bytes()
    assert content.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r\n" not in content
