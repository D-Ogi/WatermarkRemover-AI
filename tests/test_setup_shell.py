"""Execute Linux setup with controlled command stubs; never install or download."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux installer branches"
)
REPO = Path(__file__).resolve().parents[1]


def run_setup(tmp_path, *, gpu, mirror, fail_install=False):
    """Record selected indexes and stage ordering from the actual shell script."""
    project = tmp_path / "project with spaces"
    project.mkdir()
    shutil.copyfile(REPO / "setup.sh", project / "setup.sh")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    program = tmp_path / "fake_python.py"
    program.write_text(
        r"""import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['WMR_TEST_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\n')
if args[:1] == ['-c'] and 'sys.version_info' in args[1]:
    print('3.12')
elif args[:2] == ['-m', 'venv']:
    activate = Path(args[2]) / 'bin' / 'activate'
    activate.parent.mkdir(parents=True)
    activate.write_text('# controlled environment\n')
elif 'requirements.txt' in args and os.environ.get('WMR_TEST_FAIL') == '1':
    sys.exit(17)
""",
        encoding="utf-8",
    )
    wrapper = '#!/bin/sh\nexec "$WMR_TEST_PYTHON" "$WMR_TEST_PROGRAM" "$@"\n'
    for name in ("python3.12", "python", "pip"):
        executable = bindir / name
        executable.write_text(wrapper, encoding="utf-8")
        executable.chmod(0o755)
    if gpu:
        executable = bindir / "nvidia-smi"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    env = os.environ.copy()
    env.update(
        PATH=f"{bindir}:/usr/bin:/bin",
        WMR_TEST_PYTHON=sys.executable,
        WMR_TEST_PROGRAM=str(program),
        WMR_TEST_LOG=str(log),
        WMR_TEST_FAIL="1" if fail_install else "0",
    )
    result = subprocess.run(
        ["/bin/bash", str(project / "setup.sh")],
        input=("y" if mirror else "n") + "n",
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    return result, commands


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("mirror", [False, True])
def test_torch_backend_index_is_unambiguous(tmp_path, gpu, mirror):
    """Verify CPU/CUDA index selection independently of the general package mirror."""
    result, commands = run_setup(tmp_path, gpu=gpu, mirror=mirror)
    assert result.returncode == 0, result.stdout + result.stderr
    torch_install = next(args for args in commands if "torch>=2.4.0" in args)
    expected = "https://download.pytorch.org/whl/" + ("cu124" if gpu else "cpu")
    assert torch_install[torch_install.index("--index-url") + 1] == expected
    assert "--extra-index-url" not in torch_install
    assert "-i" not in torch_install
    app_install = next(args for args in commands if "requirements.txt" in args)
    assert ("-i" in app_install) == mirror
    assert ["-m", "pip", "check"] in commands
    assert ["-m", "lama_inpaint", "download"] in commands


def test_install_failure_stops_before_model_preparation(tmp_path):
    """A failed dependency install must stop before download or success reporting."""
    result, commands = run_setup(tmp_path, gpu=False, mirror=False, fail_install=True)
    assert result.returncode == 17
    assert ["-m", "lama_inpaint", "download"] not in commands
    assert "Setup complete!" not in result.stdout
