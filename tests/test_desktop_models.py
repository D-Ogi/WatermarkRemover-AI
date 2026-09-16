"""Exercise actual preparation subprocess failures and retries without network/models."""
import time

from desktop_models import ModelPreparation
import desktop_models


def wait(controller):
    """Bound worker completion so a broken process lifecycle fails the test."""
    deadline=time.monotonic()+10
    while controller.status()['busy'] and time.monotonic()<deadline:
        time.sleep(.02)
    assert not controller.status()['busy']
    return controller.status()


def test_retry_after_worker_failure(tmp_path, monkeypatch):
    """A failed worker exposes its message and a retry can reach verified readiness."""
    (tmp_path/'model_assets.py').write_text("import json, pathlib, sys, time\np=pathlib.Path('attempt')\ntime.sleep(.1)\nif not p.exists():\n p.touch(); print(json.dumps({'status':'error','message':'interrupted'})); sys.exit(1)\nprint(json.dumps({'status':'ready'}))\n")
    monkeypatch.setattr(desktop_models, 'APP_ROOT', tmp_path)
    original=desktop_models.worker_options
    monkeypatch.setattr(desktop_models,'worker_options',lambda:dict(original(),cwd=str(tmp_path)))
    controller=ModelPreparation()
    try:
        assert controller.start()['status']=='started'
        assert 'error' in controller.start()
        assert wait(controller)['message']=='interrupted'
        assert controller.start()['status']=='started'
        assert wait(controller)['status']=='ready'
    finally:
        controller.close()
    assert 'error' in controller.start()
