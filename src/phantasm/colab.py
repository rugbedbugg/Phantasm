"""Phantasm-owned job lifecycle over the pinned Google Colab CLI."""

import base64
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests

from phantasm import __file__ as package_file
from phantasm.artifacts import extract_bundle, find_backup
from phantasm.credentials import resolve_secret
from phantasm.downloads import download_verified
from phantasm.pixeldrain import Pixeldrain, sha256
from phantasm.storage import write_json_atomic

COLAB_VERSION = "0.6.0"
JOBS = Path(".phantasm/jobs")


class Colab:
    def __init__(self):
        self.executable = shutil.which("colab")
        if not self.executable:
            raise RuntimeError("Run phantasm colab setup to install the Colab backend")
        result = subprocess.run([self.executable, "version"], capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != f"Version: {COLAB_VERSION}":
            raise RuntimeError(
                f"Phantasm requires google-colab-cli {COLAB_VERSION}; run phantasm colab setup"
            )

    def call(self, *args, capture=False):
        result = subprocess.run(
            [self.executable, *map(str, args)], capture_output=capture, text=True
        )
        if result.returncode:
            raise RuntimeError(
                "Colab operation failed; job state was retained. Check phantasm colab status/logs"
            )
        return result.stdout or ""

    def execute(self, session: str, code: str, timeout: int = 60) -> str:
        marker = "PHANTASM_OK_" + uuid.uuid4().hex
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", encoding="utf-8") as script:
            script.write(code + f"\nprint({marker!r})\n")
            script.flush()
            result = self.call(
                "exec", "-s", session, "--timeout", timeout, "-f", script.name, capture=True
            )
        # colab 0.6.0 may return exit 0 for a remote Python error.
        if marker not in result.splitlines():
            raise RuntimeError("Remote command did not complete; inspect phantasm colab logs")
        return "\n".join(line for line in result.splitlines() if line != marker)

    def _runtime(self, session):
        store = Path.home() / ".config/colab-cli/sessions.json"
        entry = None
        for attempt in range(3):
            try:
                entry = json.loads(store.read_text()).get(session)
                break
            except (OSError, ValueError):
                if attempt < 2:
                    time.sleep(0.1)  # The backend updates this file in place.
        if not isinstance(entry, dict) or not entry.get("url") or not entry.get("token"):
            raise RuntimeError("Colab session credentials are unavailable; check the job status")
        url = urlsplit(entry["url"])
        if (
            url.scheme != "https"
            or not (url.hostname or "").endswith(".colab.dev")
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.port not in (None, 443)
        ):
            raise RuntimeError("Unexpected Colab runtime address")
        return entry["url"].rstrip("/"), {
            "authuser": "0",
            "colab-runtime-proxy-token": entry["token"],
        }

    def upload(self, session, source, remote):
        # Jupyter's chunk protocol avoids the backend's whole-file base64 buffer.
        base, params = self._runtime(session)
        endpoint = base + "/api/contents/" + quote(remote.lstrip("/"), safe="/")
        source = Path(source)
        expected = sha256(source)
        for attempt in range(4):
            try:
                with source.open("rb") as stream:
                    index = 1
                    while True:
                        chunk = stream.read(4 * 1024 * 1024)
                        final = not chunk and index > 1
                        payload = {
                            "name": remote.rsplit("/", 1)[-1],
                            "path": remote,
                            "type": "file",
                            "format": "base64",
                            "content": base64.b64encode(chunk).decode("ascii"),
                            "chunk": -1 if final else index,
                        }
                        with requests.put(
                            endpoint,
                            params=params,
                            json=payload,
                            allow_redirects=False,
                            timeout=(30, 120),
                        ) as response:
                            if response.status_code in (429, 500, 502, 503, 504):
                                raise requests.ConnectionError("Transient upload failure")
                            if response.status_code not in (200, 201):
                                raise RuntimeError(f"Colab upload HTTP {response.status_code}")
                        if final:
                            break
                        index += 1
                break
            except requests.RequestException:
                if attempt == 3:
                    raise RuntimeError("Colab upload failed; retry the Phantasm command") from None
                time.sleep(2**attempt)  # Restart at chunk 1; replaying a chunk would append twice.
        self.execute(
            session,
            f"import hashlib\nfrom pathlib import Path\np=Path({remote!r})\nh=hashlib.sha256()\n"
            "with p.open('rb') as f:\n    for b in iter(lambda: f.read(1048576), b''):\n        h.update(b)\n"
            f"assert h.hexdigest() == {expected!r}, 'Uploaded file failed checksum verification'",
            timeout=300,
        )

    def download(self, session, remote, destination, checksum, size):
        # The backend's `download` buffers base64 for the entire file. Its
        # authenticated Jupyter /files endpoint streams large artifacts instead.
        base, params = self._runtime(session)
        endpoint = base + "/files/" + quote(remote.lstrip("/"), safe="/")
        return download_verified(
            lambda headers: requests.get(
                endpoint,
                params=params,
                headers=headers,
                stream=True,
                allow_redirects=False,
                timeout=(30, 120),
            ),
            Path(destination),
            size,
            checksum,
        )


def job_path(name: str, jobs_dir=JOBS) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Job names may contain only letters, digits, underscores and hyphens")
    return Path(jobs_dir).expanduser().resolve() / name


def read_job(name, jobs_dir=JOBS):
    path = job_path(name, jobs_dir)
    state = json.loads((path / "job.json").read_text())
    if not re.fullmatch(r"[0-9a-f]{32}", state.get("id", "")):
        raise ValueError("Invalid saved job ID")
    if state.get("remote") != f"/content/phantasm/{state['id']}":
        raise ValueError("Invalid remote job path")
    return path, state


def save_job(path, state):
    write_json_atomic(path / "job.json", state)


def train_arguments(args, remote: str, identifier: str) -> list[str]:
    result = ["--train-dataset", remote + "/inputs/train.jsonl", "--output-dir", remote + "/output"]
    if args.eval_dataset:
        result += ["--eval-dataset", remote + "/inputs/validation.jsonl"]
    for key in (
        "base_model",
        "loss",
        "max_seq_length",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "batch_size",
        "gradient_accumulation",
        "eval_steps",
        "weight_decay",
        "lr_scheduler",
        "early_stopping_patience",
        "learning_rate",
        "seed",
        "quant_method",
    ):
        result += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    # --epochs and --max-steps are mutually exclusive at run time; forwarding both
    # would record a configuration that contradicts what actually ran.
    if args.epochs is None:
        result += ["--max-steps", str(args.max_steps)]
    else:
        result += ["--epochs", str(args.epochs)]
    if args.warmup_steps is not None:
        result += ["--warmup-steps", str(args.warmup_steps)]
    if args.model_revision:
        result += ["--model-revision", args.model_revision]
    if args.artifact_store == "pixeldrain":
        result += ["--artifact-store", "pixeldrain", "--backup-job-id", identifier]
    return result


def make_bundle(path: Path, state: dict) -> Path:
    bundle = path / "submission.zip"
    code = Path(package_file).parent
    spec = {
        "id": state["id"],
        "artifact_store": state["artifact_store"],
        "train_args": state["train_args"],
    }
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(code.rglob("*")):
            if source.is_file() and (source.suffix == ".py" or source.name == "training.txt"):
                archive.write(source, "code/phantasm/" + source.relative_to(code).as_posix())
        for source in (path / "inputs").iterdir():
            archive.write(source, "inputs/" + source.name)
        if state.get("recovery"):
            restored = path / state["recovery"]
            for source in restored.rglob("*"):
                if source.is_file():
                    archive.write(source, "output/" + source.relative_to(restored).as_posix())
        archive.writestr("spec.json", json.dumps(spec))
    return bundle


def launch(path: Path, state: dict, backend=None):
    backend = backend or Colab()
    credentials = {
        key: resolve_secret(key)
        for key in ("PIXELDRAIN_API_KEY", "PIXELDRAIN_DOMAIN", "HF_TOKEN")
        if resolve_secret(key)
    }
    if state["artifact_store"] == "pixeldrain":
        Pixeldrain()  # Fail before allocating GPU resources if configuration is missing.
    bundle = make_bundle(path, state)
    for key in ("stopped", "downloaded", "result", "latest_backup", "backup_warning"):
        state.pop(key, None)
    state["status"] = "provisioning"
    save_job(path, state)
    if state["owned_session"]:
        backend.call("new", "-s", state["session"], "--gpu", state["gpu"])
    remote = state["remote"]
    backend.execute(
        state["session"],
        f"from pathlib import Path\nPath({remote!r}).mkdir(parents=True, exist_ok=True)",
    )
    backend.upload(state["session"], bundle, remote + "/submission.zip")
    if credentials:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as secret:
            json.dump(credentials, secret)
            secret.flush()
            backend.upload(state["session"], secret.name, remote + "/credentials.json")
    code = f"""import json, os, subprocess, sys, zipfile
from pathlib import Path
root = Path({remote!r})
if (root / "state.json").exists():
    old = json.loads((root / "state.json").read_text())
    if old.get("status") in ("setting_up", "training", "publishing"):
        raise RuntimeError("An existing worker may still be running; do not launch twice")
with zipfile.ZipFile(root / "submission.zip") as archive:
    for name in archive.namelist():
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError("Unsafe submission archive")
    archive.extractall(root)
secret = root / "credentials.json"
if secret.exists():
    secret.chmod(0o600)
env = os.environ.copy()
env["PYTHONPATH"] = str(root / "code")
with (root / "worker.log").open("ab") as log:
    process = subprocess.Popen([sys.executable, "-u", "-m", "phantasm.remote_worker", "--bootstrap", str(root)],
        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(root / "worker.pid").write_text(str(process.pid))
"""
    backend.execute(state["session"], code)
    state["status"] = "submitted"
    save_job(path, state)
    print(f"Submitted {path.name}. Status: phantasm colab status {path.name}")


def submit(args):
    from phantasm.training import prepare_run

    prepare_run(args)
    if args.croc_transfer or args.resume_from_checkpoint:
        raise ValueError(
            "For Colab use artifact storage and phantasm recover instead of local transfer/resume flags"
        )
    identifier = uuid.uuid4().hex
    name = args.job or f"run-{identifier[:8]}"
    path = job_path(name, args.jobs_dir)
    state = {
        "version": 1,
        "id": identifier,
        "name": name,
        "remote": f"/content/phantasm/{identifier}",
        "session": args.session or f"phantasm-{identifier[:12]}",
        "owned_session": not bool(args.session),
        "gpu": args.gpu,
        "artifact_store": args.artifact_store,
        "keep": args.keep,
        "train_args": train_arguments(args, f"/content/phantasm/{identifier}", identifier),
    }
    if args.dry_run:
        print(json.dumps(state, indent=2))
        return
    if path.exists():
        raise ValueError("Job already exists; use status/fetch/recover or choose a new name")
    if args.artifact_store == "pixeldrain":
        Pixeldrain()
    backend = Colab()
    (path / "inputs").mkdir(parents=True)
    shutil.copyfile(args.train_dataset, path / "inputs/train.jsonl")
    if args.eval_dataset:
        shutil.copyfile(args.eval_dataset, path / "inputs/validation.jsonl")
    save_job(path, state)
    launch(path, state, backend)
    if not args.detach:
        wait_for_job(path, state, backend)


def remote_status(path, state, backend):
    remote = state["remote"]
    code = f"""import json, os
from pathlib import Path
root = Path({remote!r})
value = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {{"status": "starting"}}
if (root / "latest-backup.json").exists():
    value["latest_backup"] = json.loads((root / "latest-backup.json").read_text())
if (root / "backup-warning.json").exists():
    value["backup_warning"] = json.loads((root / "backup-warning.json").read_text())
if value.get("status") in ("starting", "setting_up", "training", "publishing") and (root / "worker.pid").exists():
    try:
        os.kill(int((root / "worker.pid").read_text()), 0)
    except ProcessLookupError:
        value["status"] = "worker_lost"
print("PHANTASM_STATE=" + json.dumps(value))
"""
    result = backend.execute(state["session"], code)
    rows = [
        line.removeprefix("PHANTASM_STATE=")
        for line in result.splitlines()
        if line.startswith("PHANTASM_STATE=")
    ]
    if len(rows) != 1:
        raise RuntimeError("Colab returned no valid job status")
    state.update(json.loads(rows[0]))
    save_job(path, state)
    return state


def fetch(path, state, backend=None):
    if state.get("downloaded") and (path / "result/run.json").is_file():
        print(f"Results already verified: {path / 'result'}")
        return
    backend = backend or Colab()
    remote_status(path, state, backend)
    if state["status"] == "transfer_failed":
        # Fetch can retrieve the already-created bundle directly when API upload failed.
        name = f"phantasm-{state['id']}-result.tar"
        backend.execute(
            state["session"],
            "import subprocess, os\nsubprocess.run("
            + repr(
                [
                    state["remote"] + "/venv/bin/python",
                    "-c",
                    "from pathlib import Path; from phantasm.artifacts import archive_run; "
                    + f"archive_run(Path({state['remote'] + '/output'!r}), Path({state['remote'] + '/' + name!r}))",
                ]
            )
            + ", env={**os.environ, 'PYTHONPATH': "
            + repr(state["remote"] + "/code")
            + "}, check=True)",
            timeout=3600,
        )
        result = backend.execute(
            state["session"],
            f"import hashlib, json\nfrom pathlib import Path\np = Path({state['remote'] + '/' + name!r})\nh = hashlib.sha256()\nwith p.open('rb') as f:\n    for b in iter(lambda: f.read(1048576), b''):\n        h.update(b)\nprint('PHANTASM_RESULT=' + json.dumps({{'name': p.name, 'sha256': h.hexdigest(), 'size': p.stat().st_size}}))",
        )
        state["result"] = json.loads(
            next(
                line.removeprefix("PHANTASM_RESULT=")
                for line in result.splitlines()
                if line.startswith("PHANTASM_RESULT=")
            )
        )
    elif state["status"] != "completed":
        raise RuntimeError(f"Job is {state['status']}; no completed result is available")
    receipt = state["result"]
    archive = path / "result.tar"
    if receipt.get("id"):
        Pixeldrain().download(receipt["id"], archive, receipt["sha256"])
    else:
        temporary = archive.with_suffix(".part")
        backend.download(
            state["session"],
            state["remote"] + "/" + receipt["name"],
            temporary,
            receipt["sha256"],
            receipt.get("size"),
        )
        if sha256(temporary) != receipt["sha256"]:
            raise RuntimeError("Downloaded Colab artifact failed checksum verification")
        temporary.replace(archive)
    extract_bundle(archive, path / "result")
    state["downloaded"] = True
    state["status"] = "downloaded"
    save_job(path, state)
    print(f"Verified results: {path / 'result'}")
    if state["owned_session"] and not state["keep"]:
        backend.call("stop", "-s", state["session"])
        state["stopped"] = True
        save_job(path, state)


def wait_for_job(path, state, backend):
    previous = None
    while True:
        remote_status(path, state, backend)
        if state["status"] != previous:
            print(f"{path.name}: {state['status']}")
            previous = state["status"]
        if state["status"] in ("completed", "transfer_failed"):
            fetch(path, state, backend)
            return
        if state["status"] in ("failed", "setup_failed", "worker_lost"):
            raise RuntimeError(
                f"Job {state['status']}; inspect logs or use phantasm recover {path.name}"
            )
        time.sleep(5)


def recover(args):
    path, state = read_job(args.job, args.jobs_dir)
    if args.resume and (path / "result").exists():
        raise ValueError("This job already has fetched results; submit a new job to train again")
    if state["artifact_store"] != "pixeldrain":
        raise ValueError(
            "This job has no durable Pixeldrain backup; use colab fetch if its runtime still exists"
        )
    client = Pixeldrain()
    entry = find_backup(client, state["id"], checkpoint_only=args.resume)
    archive = client.download(entry["id"], path / (entry["id"] + ".tar"))
    destination = path / ("recovered-" + entry["id"])
    if not destination.exists():
        extract_bundle(archive, destination)
    print(f"Recovered and verified: {destination}")
    if not args.resume:
        return
    checkpoints = list((destination / "checkpoints").glob("checkpoint-*"))
    if not checkpoints:
        raise RuntimeError("Recovery archive contains no checkpoint")
    checkpoints.sort(key=lambda p: int(p.name.removeprefix("checkpoint-")), reverse=True)
    state["recovery"] = destination.name
    state["session"] = args.session or f"phantasm-{uuid.uuid4().hex[:12]}"
    state["owned_session"] = not bool(args.session)
    state["keep"] = args.keep
    arguments = state["train_args"]
    if "--resume-from-checkpoint" in arguments:
        index = arguments.index("--resume-from-checkpoint")
        del arguments[index : index + 2]
    arguments += [
        "--resume-from-checkpoint",
        state["remote"] + "/output/checkpoints/" + checkpoints[0].name,
    ]
    save_job(path, state)
    backend = Colab()
    launch(path, state, backend)
    if not args.detach:
        wait_for_job(path, state, backend)


def command(args):
    if args.action == "login":
        backend = Colab()
        try:
            backend.call("whoami")
        except RuntimeError:
            raise RuntimeError("Colab sign-in failed; rerun phantasm colab login") from None
        return
    if args.action == "setup":
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("Install uv first, then rerun phantasm colab setup")
        try:
            subprocess.run(
                [uv, "tool", "install", f"google-colab-cli=={COLAB_VERSION}"], check=True
            )
        except subprocess.CalledProcessError:
            raise RuntimeError("Colab backend installation failed") from None
        return
    path, state = read_job(args.job, args.jobs_dir)
    if args.action == "status" and state.get("stopped"):
        print(json.dumps(state, indent=2))
        return
    if args.action == "fetch" and state.get("downloaded"):
        fetch(path, state)
        return
    backend = Colab()
    if args.action == "status":
        print(json.dumps(remote_status(path, state, backend), indent=2))
    elif args.action == "fetch":
        fetch(path, state, backend)
    elif args.action == "stop":
        backend.call("stop", "-s", state["session"])
        state["stopped"] = True
        save_job(path, state)
    elif args.action == "logs":
        code = f"from pathlib import Path\np = Path({state['remote'] + '/worker.log'!r})\nwith p.open('rb') as f:\n    f.seek(max(0, p.stat().st_size - 65536))\n    print(f.read().decode('utf-8', errors='replace'))"
        print(backend.execute(state["session"], code))


def add_commands(subparsers):
    parser = subparsers.add_parser("colab", help="Set up and manage remote training jobs")
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("setup", help="Install the pinned remote backend").set_defaults(func=command)
    actions.add_parser("login", help="Sign in to Google without creating a runtime").set_defaults(
        func=command
    )
    for name in ("status", "logs", "fetch", "stop"):
        action = actions.add_parser(name)
        action.add_argument("job")
        action.add_argument("--jobs-dir", default=str(JOBS))
        action.set_defaults(func=command)
    parser = subparsers.add_parser(
        "recover", help="Recover a durable result or resume a backed-up checkpoint"
    )
    parser.add_argument("job")
    parser.add_argument("--jobs-dir", default=str(JOBS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session", help="Use an existing runtime for resumed training")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.set_defaults(func=recover)
