"""Build a versioned portable Windows folder and ZIP with an executable launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
APP_FILES = ["VERSION", "LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", "remwm.py",
             "remwmgui.py", "utils.py", "desktop_main.py", "desktop_runtime.py",
             "desktop_models.py", "model_assets.py", "requirements.txt", "requirements-core.txt"]
APP_DIRS = ["lama_inpaint", "ui", "models", "licenses", "docs"]


def run(*args):
    """Stop packaging at the first failed command rather than publishing a broken build."""
    subprocess.run([str(arg) for arg in args], check=True, cwd=ROOT)


def sha256(path):
    """Hash large archives without holding them in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def copy_app(target):
    """Copy an explicit public file list, excluding caches, private settings and models."""
    for name in APP_FILES:
        shutil.copy2(ROOT/name, target/name)
    for name in APP_DIRS:
        shutil.copytree(ROOT/name, target/name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target/"scripts").mkdir(exist_ok=True)
    shutil.copy2(ROOT/"scripts/check_desktop.py", target/"scripts/check_desktop.py")
    (target/"portable.flag").write_text("Store settings, logs and model caches in data/.\n")


def main():
    """Install a private runtime, compile a small native launcher and archive tested files."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cu126"], default="cpu")
    parser.add_argument("--refresh-app", action="store_true")
    parser.add_argument("--skip-archive", action="store_true")
    args=parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("Windows packaging must run on Windows")
    version=(ROOT/"VERSION").read_text().strip()
    if json.loads((ROOT/"ui/config.json").read_text(encoding="utf-8"))["version"] != version:
        raise SystemExit("VERSION and ui/config.json must agree")
    name=f"WatermarkRemover-AI-{version}-windows-x64-{args.backend}"
    target=args.output.resolve()/name
    runtime=json.loads((ROOT/"packaging/python-runtime.json").read_text())
    args.cache.mkdir(parents=True,exist_ok=True)
    if args.refresh_app:
        marker=target/"build-info.json"
        if not marker.exists() or json.loads(marker.read_text())["backend"] != args.backend:
            raise SystemExit("Only an existing matching build may be refreshed")
    else:
        target.mkdir(parents=True,exist_ok=False)
        archive=args.cache/Path(runtime["url"]).name
        if not archive.exists():
            with urlopen(runtime["url"],timeout=30) as source, archive.open("wb") as output:
                shutil.copyfileobj(source,output)
        if sha256(archive) != runtime["sha256"]:
            raise SystemExit("Embedded Python checksum mismatch; remove the incomplete cached archive and retry")
        python_dir=target/"python";python_dir.mkdir()
        with zipfile.ZipFile(archive) as source:
            for member in source.namelist():
                if not (python_dir/member).resolve().is_relative_to(python_dir):
                    raise SystemExit("Unsafe archive member")
            source.extractall(python_dir)
        major_minor="".join(runtime["version"].split(".")[:2])
        (python_dir/f"python{major_minor}._pth").write_text(f"python{major_minor}.zip\n.\n..\nLib/site-packages\nimport site\n")
        (python_dir/"Lib/site-packages").mkdir(parents=True)
        executable=python_dir/"python.exe"
        run(sys.executable,"-m","pip","--python",executable,"install","pip==26.2.1")
        run(executable,"-m","pip","--isolated","install","torch==2.14.0","torchvision==0.29.0",
            "--index-url",f"https://download.pytorch.org/whl/{args.backend}")
        run(executable,"-m","pip","--isolated","install","-r",ROOT/"requirements.txt")
    copy_app(target)
    executable=target/"python/python.exe"
    run(executable,"-m","pip","check")
    run(executable,"-c","import remwm, webview; from PySide6 import QtWebEngineWidgets")
    source=target/"Launcher.cs"
    source.write_text((ROOT/"packaging/Launcher.cs").read_text().replace("__VERSION__",version))
    compiler=Path(os.environ["WINDIR"])/"Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    run(compiler,"/nologo","/target:winexe","/platform:x64","/optimize+","/debug-",
        "/reference:System.Windows.Forms.dll",f"/out:{target/'WatermarkRemover-AI.exe'}",source)
    source.unlink()
    freeze=subprocess.check_output([str(executable),"-m","pip","freeze","--all"],text=True)
    (target/"installed-packages.txt").write_text(freeze,encoding="utf-8")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    (target/"build-info.json").write_text(json.dumps(dict(version=version,backend=args.backend,
        python=runtime,source_commit=commit,
        source_dirty=bool(subprocess.check_output(["git","diff","--ignore-space-at-eol","HEAD"],cwd=ROOT))),indent=2)+"\n")
    if not args.skip_archive:
        bundle=args.output.resolve()/(name+".zip")
        with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as output:
            for path in sorted(target.rglob("*")):
                relative=path.relative_to(target)
                if path.is_file() and relative.parts[0]!="data" and "__pycache__" not in relative.parts and relative.parts[:2] != ("python", "Scripts"):
                    output.write(path,Path(name)/relative)
        digest=sha256(bundle)
        bundle.with_suffix(".zip.sha256").write_text(f"{digest}  {bundle.name}\n")
        print("Built",bundle,"SHA256",digest,flush=True)
    print("Portable folder",target,flush=True)


if __name__ == "__main__":
    main()
