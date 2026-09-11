"""Detached Colab worker. The bootstrap needs only the runtime's standard library."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def bootstrap(root: Path) -> None:
    # Read credentials once, outside command arguments and Colab execution history.
    credentials = root / "credentials.json"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if credentials.exists():
        try:
            env.update(json.loads(credentials.read_text()))
        finally:
            credentials.unlink()
    from phantasm.storage import write_json_atomic

    state_path = root / "state.json"
    write_json_atomic(state_path, {"status": "setting_up", "pid": os.getpid()})
    try:
        uv = shutil.which("uv")
        if uv and not subprocess.check_output([uv, "--version"], text=True).startswith(
            "uv 0.12.9 "
        ):
            uv = None
        if uv is None:
            target = root / "uv-tool"
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--target", str(target), "uv==0.12.9"],
                check=True,
            )
            uv = str(target / "bin" / "uv")
            env["PYTHONPATH"] = str(target)
        venv = root / "venv"
        subprocess.run([uv, "venv", "--python", "3.11", str(venv)], check=True, env=env)
        python = str(venv / "bin" / "python")
        subprocess.run(
            [
                uv,
                "pip",
                "sync",
                "--python",
                python,
                "--require-hashes",
                str(root / "code/phantasm/resources/training.txt"),
            ],
            check=True,
            env=env,
        )
        env["PYTHONPATH"] = str(root / "code")
        result = subprocess.run(
            [python, "-u", "-m", "phantasm.remote_worker", "--work", str(root)], env=env
        )
        if result.returncode and json.loads(state_path.read_text()).get("status") not in (
            "failed",
            "transfer_failed",
        ):
            write_json_atomic(state_path, {"status": "failed", "exit_code": result.returncode})
    except Exception:
        write_json_atomic(state_path, {"status": "setup_failed"})
        raise RuntimeError("Remote environment setup failed; inspect the job log") from None


def work(root: Path) -> None:
    from phantasm.cli import main
    from phantasm.storage import write_json_atomic

    spec = json.loads((root / "spec.json").read_text())
    state_path = root / "state.json"
    write_json_atomic(state_path, {"status": "training", "pid": os.getpid()})
    sys.argv = ["phantasm", "train", *spec["train_args"]]
    try:
        main()
    except BaseException:
        write_json_atomic(state_path, {"status": "failed"})
        raise
    publish(root)


def publish(root: Path) -> None:
    from phantasm.artifacts import archive_run
    from phantasm.pixeldrain import Pixeldrain, sha256
    from phantasm.storage import write_json_atomic

    spec = json.loads((root / "spec.json").read_text())
    state_path = root / "state.json"
    write_json_atomic(state_path, {"status": "publishing"})
    try:
        name = f"phantasm-{spec['id']}-result.tar"
        bundle = archive_run(root / "output", root / name)
        receipt = {"name": name, "sha256": sha256(bundle), "size": bundle.stat().st_size}
        if spec["artifact_store"] == "pixeldrain":
            receipt = Pixeldrain().upload(bundle, name)
        write_json_atomic(root / "result.json", receipt)
        write_json_atomic(state_path, {"status": "completed", "result": receipt})
    except Exception:
        write_json_atomic(state_path, {"status": "transfer_failed"})
        raise RuntimeError(
            "Result upload failed; local output remains available for fetching"
        ) from None


if __name__ == "__main__":
    mode, directory = sys.argv[1:]
    (work if mode == "--work" else bootstrap)(Path(directory))
