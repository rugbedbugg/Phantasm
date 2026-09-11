"""Job orchestration and durable checkpoint recovery without allocating cloud resources."""

import argparse
import io
import json
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phantasm.artifacts import archive_run, extract_bundle, find_backup, verify_resume
from phantasm.colab import Colab, read_job, submit
from phantasm.training import add_training_arguments


def arguments(tmp_path):
    parser = argparse.ArgumentParser()
    add_training_arguments(parser)
    dataset = tmp_path / "dataset with spaces.jsonl"
    dataset.write_text(
        json.dumps(
            {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}
        )
        + "\n"
    )
    return parser.parse_args(
        [
            "--train-dataset",
            str(dataset),
            "--backend",
            "colab",
            "--job",
            "example",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--detach",
        ]
    )


def test_colab_dry_run_never_allocates_or_uploads(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path)
    args.dry_run = True
    backend = Mock()
    monkeypatch.setattr("phantasm.colab.Colab", backend)
    submit(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["gpu"] == "T4"
    assert "--train-dataset" in plan["train_args"]
    assert "--loss" in plan["train_args"]
    backend.assert_not_called()
    assert not Path(args.jobs_dir).exists()


def test_submit_snapshots_data_and_bundles_worker_and_lock(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    backend = Mock()
    monkeypatch.setattr("phantasm.colab.Colab", lambda: backend)
    monkeypatch.setenv("HF_TOKEN", "private-hf-token")
    submit(args)
    path, state = read_job("example", args.jobs_dir)
    assert state["status"] == "submitted"
    assert (path / "inputs/train.jsonl").read_bytes() == Path(args.train_dataset).read_bytes()
    assert "private-hf-token" not in (path / "job.json").read_text()
    with zipfile.ZipFile(path / "submission.zip") as archive:
        assert "code/phantasm/resources/training.txt" in archive.namelist()
        assert "code/phantasm/remote_worker.py" in archive.namelist()
        assert all(
            "private-hf-token" not in archive.read(name).decode() for name in archive.namelist()
        )
    assert backend.call.call_args_list[0].args[0] == "new"
    for call in backend.execute.call_args_list:
        assert "private-hf-token" not in call.args[1]
        compile(call.args[1], "generated-colab.py", "exec")


def test_missing_pixeldrain_key_fails_before_provisioning(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.artifact_store = "pixeldrain"
    monkeypatch.delenv("PIXELDRAIN_API_KEY", raising=False)
    backend = Mock()
    monkeypatch.setattr("phantasm.colab.Colab", backend)
    with pytest.raises(ValueError, match="PIXELDRAIN"):
        submit(args)
    backend.assert_not_called()


def test_pixeldrain_domain_is_forwarded_to_worker(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.artifact_store = "pixeldrain"
    monkeypatch.setenv("PIXELDRAIN_API_KEY", "private-key")
    monkeypatch.setenv("PIXELDRAIN_DOMAIN", "pixeldrain.net")
    forwarded = {}
    backend = Mock()

    def upload(session, source, destination):
        if destination.endswith("/credentials.json"):
            forwarded.update(json.loads(Path(source).read_text()))

    backend.upload.side_effect = upload
    monkeypatch.setattr("phantasm.colab.Colab", lambda: backend)
    submit(args)
    assert forwarded["PIXELDRAIN_DOMAIN"] == "pixeldrain.net"
    assert forwarded["PIXELDRAIN_API_KEY"] == "private-key"
    path, _ = read_job("example", args.jobs_dir)
    assert "private-key" not in (path / "job.json").read_text()


def test_reused_session_is_not_provisioned_again(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.session = "existing-session"
    backend = Mock()
    monkeypatch.setattr("phantasm.colab.Colab", lambda: backend)
    submit(args)
    assert all(c.args[0] != "new" for c in backend.call.call_args_list)


def test_remote_python_error_with_zero_exit_is_not_success(monkeypatch):
    monkeypatch.setattr("phantasm.colab.shutil.which", lambda _: "/test/colab")
    monkeypatch.setattr(
        "phantasm.colab.subprocess.run",
        Mock(
            side_effect=[
                SimpleNamespace(returncode=0, stdout="Version: 0.6.0\n"),
                SimpleNamespace(returncode=0, stdout="Traceback: error\n"),
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="did not complete"):
        Colab().execute("session", "raise ValueError('failed')")


@pytest.mark.parametrize("name", ["../../outside", "/absolute", "gguf/../../escape"])
def test_archive_traversal_does_not_write_files(tmp_path, name):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo(name)
        member.size = 1
        stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        extract_bundle(archive, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_archive_links_are_rejected(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("gguf/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        stream.addfile(member)
    with pytest.raises(ValueError, match="Unsafe"):
        extract_bundle(archive, tmp_path / "restored")


def checkpoint_run(tmp_path):
    output = tmp_path / "output"
    checkpoint = output / "checkpoints/checkpoint-10"
    checkpoint.mkdir(parents=True)
    manifest = {
        "train": {"sha256": "abc"},
        "validation": None,
        "model_revision": "commit",
        "config": {
            "base_model": "model",
            "model_revision": None,
            "seed": 1,
            "max_seq_length": 2048,
            "learning_rate": 0.001,
        },
    }
    (output / "run.json").write_text(json.dumps(manifest))
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 10, "best_model_checkpoint": str(checkpoint)})
    )
    for name in ("optimizer.pt", "scheduler.pt", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"state")
    return output, checkpoint, manifest


def test_checkpoint_archive_roundtrip_and_resume_validation(tmp_path):
    output, checkpoint, manifest = checkpoint_run(tmp_path)
    bundle = archive_run(output, tmp_path / "checkpoint.tar", checkpoint)
    restored = extract_bundle(bundle, tmp_path / "restored")
    manifest["config"]["model_revision"] = "commit"
    verify_resume(restored / "checkpoints/checkpoint-10", manifest)
    manifest["train"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="dataset"):
        verify_resume(restored / "checkpoints/checkpoint-10", manifest)


def test_backup_lookup_prefers_result_then_latest_checkpoint():
    job = "a" * 32
    entries = [
        {"name": f"phantasm-{job}-checkpoint-000000010.tar", "id": "old"},
        {"name": f"phantasm-{job}-checkpoint-000000020.tar", "id": "new"},
        {"name": f"phantasm-{job}-result.tar", "id": "result"},
        {"name": "phantasm-other-result.tar", "id": "wrong"},
    ]
    client = Mock()
    client.files.return_value = entries
    assert find_backup(client, job)["id"] == "result"
    assert find_backup(client, job, checkpoint_only=True)["id"] == "new"


def test_remote_worker_uploads_result_and_records_receipt(tmp_path, monkeypatch):
    from phantasm.remote_worker import work

    output, _, _ = checkpoint_run(tmp_path)
    for directory in ("adapter", "gguf"):
        (output / directory).mkdir()
        (output / directory / "model").write_bytes(b"weights")
    identifier = "a" * 32
    (tmp_path / "spec.json").write_text(
        json.dumps({"id": identifier, "train_args": [], "artifact_store": "pixeldrain"})
    )
    monkeypatch.setattr("phantasm.cli.main", lambda: None)
    monkeypatch.setattr("sys.argv", [])
    client = Mock()
    client.upload.return_value = {"id": "remote-id", "sha256": "checksum"}
    monkeypatch.setattr("phantasm.pixeldrain.Pixeldrain", lambda: client)
    work(tmp_path)
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "completed"
    assert json.loads((tmp_path / "result.json").read_text())["id"] == "remote-id"
    client.upload.assert_called_once()


def test_fetch_verifies_before_stopping_owned_runtime(tmp_path, monkeypatch):
    from phantasm.colab import fetch
    from phantasm.pixeldrain import sha256

    output, _, _ = checkpoint_run(tmp_path)
    for name in ("adapter", "gguf"):
        (output / name).mkdir()
    bundle = archive_run(output, tmp_path / "source.tar")
    job = tmp_path / "job"
    job.mkdir()
    state = {
        "status": "completed",
        "session": "session",
        "remote": "/remote",
        "owned_session": True,
        "keep": False,
        "result": {"name": "result.tar", "sha256": sha256(bundle)},
    }
    backend = Mock()

    backend.download.side_effect = lambda session, remote, destination, *args: Path(
        destination
    ).write_bytes(bundle.read_bytes())
    monkeypatch.setattr("phantasm.colab.remote_status", lambda *args: state)
    fetch(job, state, backend)
    assert state["downloaded"] and state["stopped"]
    assert (job / "result/run.json").is_file()
    assert [c.args[0] for c in backend.call.call_args_list] == ["stop"]


def test_failed_fetch_never_stops_runtime(tmp_path, monkeypatch):
    from phantasm.colab import fetch

    state = {
        "status": "completed",
        "session": "session",
        "remote": "/remote",
        "owned_session": True,
        "keep": False,
        "result": {"name": "result.tar", "sha256": "0" * 64},
    }
    backend = Mock()
    backend.download.side_effect = lambda session, remote, destination, *args: Path(
        destination
    ).write_bytes(b"bad")
    monkeypatch.setattr("phantasm.colab.remote_status", lambda *args: state)
    with pytest.raises(RuntimeError, match="checksum"):
        fetch(tmp_path, state, backend)
    assert [c.args[0] for c in backend.call.call_args_list] == []
    assert not state.get("downloaded")


def test_recover_without_resume_needs_no_live_colab(tmp_path, monkeypatch):
    import shutil

    from phantasm.colab import recover, save_job

    output, checkpoint, _ = checkpoint_run(tmp_path)
    bundle = archive_run(output, tmp_path / "checkpoint.tar", checkpoint)
    job = tmp_path / "jobs/example"
    job.mkdir(parents=True)
    identifier = "a" * 32
    save_job(
        job,
        {
            "id": identifier,
            "artifact_store": "pixeldrain",
            "remote": f"/content/phantasm/{identifier}",
        },
    )
    client = Mock()
    client.files.return_value = [
        {"id": "backup", "name": f"phantasm-{identifier}-checkpoint-000000010.tar"}
    ]

    def download(identifier, destination):
        shutil.copyfile(bundle, destination)
        return destination

    client.download.side_effect = download
    monkeypatch.setattr("phantasm.colab.Pixeldrain", lambda: client)
    backend = Mock()
    monkeypatch.setattr("phantasm.colab.Colab", backend)
    recover(SimpleNamespace(job="example", jobs_dir=tmp_path / "jobs", resume=False))
    assert (job / "recovered-backup/checkpoints/checkpoint-10/optimizer.pt").is_file()
    backend.assert_not_called()


def test_bootstrap_consumes_credentials_without_putting_them_in_commands(tmp_path, monkeypatch):
    from phantasm.remote_worker import bootstrap

    (tmp_path / "credentials.json").write_text(json.dumps({"PIXELDRAIN_API_KEY": "private-key"}))
    monkeypatch.setattr("phantasm.remote_worker.shutil.which", lambda _: "/uv")
    monkeypatch.setattr(
        "phantasm.remote_worker.subprocess.check_output", lambda *a, **kw: "uv 0.12.9 test"
    )
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "--work" in argv:
            (tmp_path / "state.json").write_text(json.dumps({"status": "completed"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("phantasm.remote_worker.subprocess.run", run)
    bootstrap(tmp_path)
    assert not (tmp_path / "credentials.json").exists()
    assert all("private-key" not in repr(argv) for argv, _ in calls)
    assert calls[-1][1]["env"]["PIXELDRAIN_API_KEY"] == "private-key"
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "completed"


def test_checkpoint_backup_keeps_best_and_reports_failed_upload(tmp_path, monkeypatch):
    import sys

    from phantasm.artifacts import backup_callback

    output, checkpoint, _ = checkpoint_run(tmp_path)
    best = output / "checkpoints/checkpoint-5"
    best.mkdir()
    (best / "adapter_model.safetensors").write_bytes(b"best weights")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 10, "best_model_checkpoint": str(best)})
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))
    client = Mock()
    client.upload.side_effect = RuntimeError("unavailable")
    monkeypatch.setattr("phantasm.artifacts.Pixeldrain", lambda: client)
    callback = backup_callback(output, "a" * 32)
    args, state, control = (
        SimpleNamespace(output_dir=checkpoint.parent),
        SimpleNamespace(global_step=10),
        object(),
    )
    assert callback.on_save(args, state, control) is control
    assert (tmp_path / "backup-warning.json").exists()
    assert (checkpoint / "optimizer.pt").exists()

    def upload(path, name):
        with tarfile.open(path) as archive:
            assert "checkpoints/checkpoint-5/adapter_model.safetensors" in archive.getnames()
            assert "checkpoints/checkpoint-10/optimizer.pt" in archive.getnames()
        return {"id": "backed-up"}

    client.upload.side_effect = upload
    callback.on_save(args, state, control)
    assert not (tmp_path / "backup-warning.json").exists()
    assert json.loads((tmp_path / "latest-backup.json").read_text())["id"] == "backed-up"
    assert not list(tmp_path.glob("phantasm-*.tar"))


def test_launch_resets_stale_runtime_state(tmp_path, monkeypatch):
    from phantasm.colab import launch

    args = arguments(tmp_path)
    backend = Mock()
    monkeypatch.setattr("phantasm.colab.Colab", lambda: backend)
    submit(args)
    path, state = read_job("example", args.jobs_dir)
    state.update(stopped=True, downloaded=True, result={"id": "old"}, backup_warning={"step": 1})
    launch(path, state, backend)
    saved = read_job("example", args.jobs_dir)[1]
    assert saved["status"] == "submitted"
    assert all(key not in saved for key in ("stopped", "downloaded", "result", "backup_warning"))


def test_login_authenticates_without_job_or_runtime(tmp_path, monkeypatch):
    from phantasm.cli import main

    backend = Mock()
    monkeypatch.setattr("phantasm.colab.Colab", lambda: backend)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["phantasm", "colab", "login"])
    main()
    backend.call.assert_called_once_with("whoami")
    assert not (tmp_path / ".phantasm").exists()


def test_login_failure_exits_nonzero(monkeypatch, capsys):
    from phantasm.cli import main

    backend = Mock()
    backend.call.side_effect = RuntimeError("backend failure")
    monkeypatch.setattr("phantasm.colab.Colab", lambda: backend)
    monkeypatch.setattr("sys.argv", ["phantasm", "colab", "login"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "rerun phantasm colab login" in capsys.readouterr().err


def test_result_archive_excludes_export_intermediates(tmp_path):
    output, _, _ = checkpoint_run(tmp_path)
    (output / "adapter").mkdir()
    (output / "adapter/adapter_model.safetensors").write_bytes(b"adapter")
    (output / "gguf").mkdir()
    (output / "gguf/model.Q4_K_M.gguf").write_bytes(b"gguf")
    (output / "gguf/model-00001.safetensors").write_bytes(b"large intermediate")
    (output / "gguf/.cache").mkdir()
    (output / "gguf/.cache/download.lock").write_bytes(b"cache")
    bundle = archive_run(output, tmp_path / "result.tar")
    with tarfile.open(bundle) as archive:
        names = archive.getnames()
    assert "gguf/model.Q4_K_M.gguf" in names
    assert "adapter/adapter_model.safetensors" in names
    assert all("model-00001" not in name and ".cache" not in name for name in names)


def runtime_store(tmp_path, monkeypatch, url="https://runtime.prod.colab.dev"):
    monkeypatch.setattr("phantasm.colab.Path.home", lambda: tmp_path)
    directory = tmp_path / ".config/colab-cli"
    directory.mkdir(parents=True)
    (directory / "sessions.json").write_text(
        json.dumps({"session": {"url": url, "token": "runtime-secret"}})
    )
    return Colab.__new__(Colab)


def test_colab_download_streams_and_resumes_without_cli_buffering(tmp_path, monkeypatch):
    import hashlib

    import requests

    backend = runtime_store(tmp_path, monkeypatch)
    content = b"abcdefgh"
    checksum = hashlib.sha256(content).hexdigest()
    destination = tmp_path / "result.tar"
    (tmp_path / f".result.tar.{checksum}.part").write_bytes(b"abc")
    response = requests.Response()
    response.status_code = 206
    response.headers["Content-Range"] = "bytes 3-7/8"
    response.raw = io.BytesIO(b"defgh")
    request = Mock(return_value=response)
    monkeypatch.setattr("phantasm.colab.requests.get", request)
    backend.download("session", "/content/file with spaces.tar", destination, checksum, 8)
    assert destination.read_bytes() == content
    args, kwargs = request.call_args
    assert args[0] == "https://runtime.prod.colab.dev/files/content/file%20with%20spaces.tar"
    assert kwargs["stream"] is True
    assert kwargs["allow_redirects"] is False
    assert kwargs["headers"] == {"Range": "bytes=3-"}


@pytest.mark.parametrize(
    "url",
    [
        "https://colab.dev.attacker.test",
        "http://runtime.prod.colab.dev",
        "https://user@runtime.prod.colab.dev",
    ],
)
def test_colab_download_rejects_untrusted_runtime_address(tmp_path, monkeypatch, url):
    backend = runtime_store(tmp_path, monkeypatch, url)
    request = Mock()
    monkeypatch.setattr("phantasm.colab.requests.get", request)
    with pytest.raises(RuntimeError, match="runtime address"):
        backend.download("session", "/content/model", tmp_path / "out", "0" * 64, 1)
    request.assert_not_called()


def test_colab_download_never_echoes_runtime_token_on_failure(tmp_path, monkeypatch):
    import requests

    backend = runtime_store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "phantasm.colab.requests.get", Mock(side_effect=requests.ConnectionError("runtime-secret"))
    )
    monkeypatch.setattr("phantasm.downloads.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError) as error:
        backend.download("session", "/content/model", tmp_path / "out", "0" * 64, 1)
    assert "runtime-secret" not in str(error.value)


def test_colab_upload_bounds_memory_and_restarts_after_failed_chunk(tmp_path, monkeypatch):
    import base64

    import requests

    backend = runtime_store(tmp_path, monkeypatch)
    backend.execute = Mock()
    source = tmp_path / "source.bin"
    source.write_bytes(b"a" * (4 * 1024 * 1024) + b"ending")
    stored = bytearray()
    indices = []
    failed = False

    def put(url, **kwargs):
        nonlocal failed
        payload = kwargs["json"]
        index = payload["chunk"]
        indices.append(index)
        data = base64.b64decode(payload["content"])
        assert len(data) <= 4 * 1024 * 1024
        assert kwargs["allow_redirects"] is False
        if index == 1:
            stored.clear()
        stored.extend(data)
        status = 200
        if index == 2 and not failed:
            failed = True
            status = 503
        response = requests.Response()
        response.status_code = status
        response.raw = io.BytesIO(b"")
        return response

    monkeypatch.setattr("phantasm.colab.requests.put", put)
    monkeypatch.setattr("phantasm.colab.time.sleep", lambda _: None)
    backend.upload("session", source, "/content/source.bin")
    assert indices == [1, 2, 1, 2, -1]
    assert stored == source.read_bytes()
    backend.execute.assert_called_once()


def test_step_and_epoch_budgets_are_never_both_forwarded(tmp_path):
    """A remote config that names both would contradict what actually ran."""
    from phantasm.colab import train_arguments

    args = arguments(tmp_path)
    steps = train_arguments(args, "/remote", "id")
    assert "--max-steps" in steps and "--epochs" not in steps
    assert steps[steps.index("--max-steps") + 1] == str(args.max_steps)

    args.epochs = 2.0
    epochs = train_arguments(args, "/remote", "id")
    assert "--epochs" in epochs and "--max-steps" not in epochs
    assert epochs[epochs.index("--epochs") + 1] == "2.0"
