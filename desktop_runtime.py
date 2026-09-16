"""Paths and subprocess settings shared by source and portable desktop launches."""

import logging
import os
from pathlib import Path
import subprocess
import sys

APP_ROOT = Path(__file__).resolve().parent


def data_dir():
    """Keep writable user state outside application files, or beside a portable build."""
    override = os.environ.get("WMR_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if (APP_ROOT / "portable.flag").exists():
        return APP_ROOT / "data"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "WatermarkRemover-AI"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "WatermarkRemover-AI"


def configure_runtime():
    """Prepare logs and portable caches before importing the graphical backend."""
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    if (APP_ROOT / "portable.flag").exists():
        os.environ.setdefault("TORCH_HOME", str(root / "models/torch"))
        os.environ.setdefault("HF_HOME", str(root / "models/huggingface"))
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, (root / "desktop.log").open("a", encoding="utf-8", buffering=1))
    logging.basicConfig(filename=root / "desktop.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")


def python_executable():
    """Workers need python.exe with real pipes even when the GUI uses pythonw.exe."""
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        return str(executable.with_name("python.exe"))
    return str(executable)


def worker_options(*, offline=False):
    """Launch bounded local workers without console windows or shell interpretation."""
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return dict(cwd=str(APP_ROOT), env=env, text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
