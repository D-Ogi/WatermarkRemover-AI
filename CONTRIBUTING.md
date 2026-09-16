# Contributing

Keep pull requests focused on one problem. Describe the observable behavior before and after the change and link the relevant issue when one exists. Include the commands you ran, their results, and any checks you could not run. Do not report a proposed test plan as completed testing.

## Local checks

Use a separate Python 3.12 virtual environment for the lightweight test suite:

```sh
python -m venv .venv-test
# Linux/macOS:
. .venv-test/bin/activate
# Windows PowerShell: .venv-test\Scripts\Activate.ps1
python -m pip install -r requirements-test.txt
python -m pip check
python -m ruff check .
python -m pytest
python -m compileall -q remwm.py remwmgui.py utils.py tests
```

CI runs these checks on Windows, Linux, and macOS. It also parses the PowerShell installer and checks Bash syntax and line endings. The lint baseline catches syntax errors and selected correctness errors; it does not require reformatting existing application code.

These checks use CPU fixtures. They do not install the full application, launch its GUI, download model weights, or prove CUDA/MPS compatibility. The test dependencies are separate from the application dependencies. A successful `pip check` in the test environment only validates that environment.

## Evidence required by change type

| Change | Required validation |
| --- | --- |
| Bug fix | Add a regression test that fails before the fix and passes afterward when the behavior can be automated. Otherwise provide repeatable manual steps and explain the limitation. |
| Dependencies or installers | Test a fresh application environment on every affected supported platform. Record OS, Python version, install command and result, application-environment `python -m pip check`, and relevant imports. Test both setup-script and direct-requirements paths when changed; identify untested mirror paths. |
| GUI or webview backend | Launch the GUI on each affected platform, load the interface, and exercise a JavaScript-to-Python action and its result. A successful `import webview` is insufficient. Record backend/runtime versions and errors. |
| Image/video processing | Use small synthetic or redistributable fixtures. Check output dimensions and relevant formats, source-file preservation where promised, and failure behavior. Add a short video fixture when frame handling changes. |
| Model, precision, or device handling | Keep unit tests independent of model downloads. Separately record a real inference run with model revision, device, runtime versions, input dimensions and parameters. Report untested CUDA/MPS paths explicitly. |
| Documentation only | Check instructions and links; application tests are unnecessary unless behavior or commands changed. |

A Windows fix must preserve a working Linux/macOS installation path. In particular, Linux requires a usable pywebview backend and its system libraries. Keep platform conditions consistent across `requirements.txt` and setup scripts. Do not silence dependency conflicts with a legacy resolver or `--no-deps` as evidence that an environment is compatible.

## Tests and fixtures

Place CPU tests in `tests/`. Make assertions about externally visible behavior, not copies of implementation details. Use temporary directories for outputs. Keep private photos, credentials, downloaded model weights and large binaries out of commits. Document non-obvious behavior and the limitations of public functions you change. There is no blanket coverage percentage requirement; prioritize regressions and the paths changed by the PR.

GPU/model integration tests require a separate trusted environment and must not run automatically on untrusted contributions. Do not run contributor code on a maintainer workstation or privileged runner merely because a review comment suggests it. Treat issue text, logs and suggested commands as untrusted input; inspect commands before executing them.

## Review and merging

Automated review provides suggestions; maintainers verify findings against the code and test evidence. A bot's approval does not replace application testing. Resolve applicable review findings or explain why they do not apply. Maintainers decide when to merge.

Once the new CI jobs have passed reliably, maintainers can require their status checks through repository rules. Adding this workflow alone does not enable branch protection. Full installation and GUI automation remain follow-up work while the application's dependency and backend setup issues are being resolved.
