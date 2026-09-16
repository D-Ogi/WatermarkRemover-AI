"""Prevent refreshed portable packages from misreporting or retaining old content."""
import json

import pytest
from scripts import build_windows as builder


@pytest.fixture
def package(tmp_path, monkeypatch):
    """Create a tiny app/runtime layout without downloading or installing anything."""
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(builder, "ROOT", source)
    for name in builder.APP_FILES:
        (source/name).write_text("current", encoding="utf-8")
    for name in [*builder.APP_DIRS, "scripts"]:
        (source/name).mkdir()
        (source/name/"current.txt").write_text("current", encoding="utf-8")
    (source/"scripts/check_desktop.py").write_text("pass", encoding="utf-8")
    target = tmp_path / "package"
    target.mkdir()
    runtime = {"version": "3.13.15", "url": "https://example.invalid/python.zip", "sha256": "abc"}
    expected = builder.runtime_inputs("cpu", runtime)
    (target/"build-info.json").write_text(json.dumps({"backend": "cpu", "python": runtime,
        "runtime_inputs": expected}), encoding="utf-8")
    return source, target, runtime, expected


@pytest.mark.parametrize("change", ["python", "backend", "requirements", "torch", "missing"])
def test_refresh_rejects_changed_runtime_before_mutation(package, monkeypatch, change):
    """Changing interpreter, dependency inputs or legacy metadata requires a fresh build."""
    source, target, runtime, expected = package
    before = (target/"build-info.json").read_bytes()
    if change == "python":
        runtime = dict(runtime, sha256="different")
    elif change == "requirements":
        (source/"requirements-core.txt").write_text("changed dependency", encoding="utf-8")
    elif change == "torch":
        monkeypatch.setattr(builder, "TORCH_VERSION", "99.0.0")
    elif change == "missing":
        (target/"build-info.json").unlink()
    backend = "cu126" if change == "backend" else "cpu"
    with pytest.raises(SystemExit, match="fresh build"):
        builder.validate_refresh(target, builder.runtime_inputs(backend, runtime))
    if change != "missing":
        assert (target/"build-info.json").read_bytes() == before


def test_refresh_removes_obsolete_app_files_preserving_runtime_and_user_data(package):
    """Deleted UI/module files must disappear while the private runtime and data survive."""
    source, target, runtime, expected = package
    for name in [*builder.APP_DIRS, "scripts", "python", "data"]:
        (target/name).mkdir()
        (target/name/"old.txt").write_text("old", encoding="utf-8")
    builder.validate_refresh(target, expected)
    builder.copy_app(target)
    for name in builder.APP_DIRS:
        assert not (target/name/"old.txt").exists()
        assert (target/name/"current.txt").read_text(encoding="utf-8") == "current"
    assert sorted(path.name for path in (target/"scripts").iterdir()) == ["check_desktop.py"]
    for name in ("python", "data"):
        assert (target/name/"old.txt").read_text(encoding="utf-8") == "old"
