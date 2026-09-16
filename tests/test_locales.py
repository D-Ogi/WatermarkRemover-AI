"""Validate the interface contract for every registered translation."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "ui"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
BASE = json.loads((ROOT / "lang/en.json").read_text(encoding="utf-8"))


def validate_translation(reference, translated, path=""):
    """Preserve lookup keys, value types and interpolation fields recursively."""
    assert type(translated) is type(reference), path
    if isinstance(reference, dict):
        assert translated.keys() == reference.keys(), path
        for key in reference:
            validate_translation(reference[key], translated[key], f"{path}.{key}")
    elif isinstance(reference, list):
        assert translated, path
        for index, item in enumerate(translated):
            validate_translation(reference[0], item, f"{path}[{index}]")
    else:
        assert isinstance(translated, str), path
        assert not reference.strip() or translated.strip(), path
        assert sorted(re.findall(r"\{[^{}]+\}", translated)) == sorted(re.findall(r"\{[^{}]+\}", reference)), path


@pytest.mark.parametrize("language", CONFIG["languages"], ids=lambda item: item["id"])
def test_registered_locale_preserves_ui_contract(language):
    """Missing labels or substitutions must fail before the desktop is packaged."""
    translated = json.loads((ROOT / "lang" / f"{language['id']}.json").read_text(encoding="utf-8"))
    validate_translation(BASE, translated, language["id"])


def test_language_ids_are_unique():
    ids = [item["id"] for item in CONFIG["languages"]]
    assert len(ids) == len(set(ids))
