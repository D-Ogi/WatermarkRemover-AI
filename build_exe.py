#!/usr/bin/env python3
"""Build OpenWipe portable zip for Windows.

Users unzip and double-click OpenWipe.exe. No Python required.
Delete the folder when done — leaves nothing behind.

Requirements: pip install pyinstaller

Usage:
    python build_exe.py

Output: OpenWipe-portable-win64.zip
"""

import subprocess
import sys
import os
import shutil
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).parent
    print("=== OpenWipe Portable Build ===")

    try:
        import PyInstaller
        print(f"PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("ERROR: PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)

    # Clean previous builds
    for d in ["build", "dist"]:
        p = root / d
        if p.exists():
            shutil.rmtree(p)

    # Run PyInstaller
    cmd = [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", "openwipe.spec"]
    print(f"Building...")
    result = subprocess.run(cmd, cwd=root)

    if result.returncode != 0:
        print(f"Build failed (exit {result.returncode})")
        sys.exit(1)

    # Create zip
    dist_dir = root / "dist" / "OpenWipe"
    if not dist_dir.exists():
        print("ERROR: dist/OpenWipe not found")
        sys.exit(1)

    zip_path = root / "OpenWipe-portable-win64.zip"
    print(f"Creating {zip_path.name}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in dist_dir.rglob("*"):
            if f.is_file():
                arcname = f"OpenWipe/{f.relative_to(dist_dir)}"
                zf.write(f, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    exe_size_mb = (dist_dir / "OpenWipe.exe").stat().st_size / (1024 * 1024)

    print(f"\n=== BUILD SUCCESSFUL ===")
    print(f"Portable zip: {zip_path.name} ({size_mb:.1f} MB)")
    print(f"Uncompressed: {exe_size_mb:.1f} MB exe + dependencies")
    print(f"\nUsers: unzip → double-click OpenWipe.exe → done.")
    print(f"ML models download on first run (~500MB).")
    print(f"Delete the folder to uninstall — leaves nothing behind.")


if __name__ == "__main__":
    main()
