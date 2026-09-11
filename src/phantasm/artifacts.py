"""Checkpoint and result bundles with bounded, traversal-safe extraction."""

import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from phantasm.pixeldrain import Pixeldrain
from phantasm.storage import write_json_atomic


def archive_run(output: Path, destination: Path, checkpoint: Path | None = None) -> Path:
    roots = (
        [output / "run.json", checkpoint]
        if checkpoint
        else [output / "run.json", output / "adapter", output / "gguf"]
    )
    if checkpoint:
        trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
        best = trainer_state.get("best_model_checkpoint")
        if best:
            best = Path(best).resolve()
            if not best.is_relative_to(output.resolve() / "checkpoints"):
                raise ValueError("Best checkpoint is outside this run")
            if best != checkpoint.resolve():
                roots.append(best)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    try:
        with tarfile.open(temporary, "w") as archive:
            for root in roots:
                if root is None or not root.exists():
                    raise ValueError("Run is missing files required for recovery")
                # Unsloth also leaves full-precision weights and download caches
                # here. Transfer the GGUF exports, not those large intermediates.
                pattern = "*.gguf" if not checkpoint and root == output / "gguf" else "*"
                paths = [root, *root.rglob(pattern)] if root.is_dir() else [root]
                for path in paths:
                    if path.is_symlink() or not (path.is_dir() or path.is_file()):
                        raise ValueError("Artifact bundles cannot contain links or special files")
                    archive.add(path, arcname=str(path.relative_to(output)), recursive=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract_bundle(archive_path: Path, destination: Path) -> Path:
    if destination.exists():
        raise ValueError("Recovery destination already exists; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=destination.parent) as temporary:
        root = Path(temporary) / "output"
        root.mkdir()
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            seen = set()
            for member in members:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or "\\" in member.name
                    or not name.parts
                    or name.parts[0] not in ("run.json", "checkpoints", "adapter", "gguf")
                    or not (member.isfile() or member.isdir())
                    or member.name in seen
                ):
                    raise ValueError("Unsafe artifact archive member")
                seen.add(member.name)
            if sum(member.size for member in members) > shutil.disk_usage(root).free:
                raise ValueError("Not enough disk space to restore artifact")
            for member in members:
                path = root.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, path.open("xb") as target:
                        shutil.copyfileobj(source, target)
        if not (root / "run.json").is_file():
            raise ValueError("Artifact has no run manifest")
        root.replace(destination)
    return destination


def find_backup(client: Pixeldrain, job_id: str, checkpoint_only: bool = False) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid job ID")
    pattern = re.compile(rf"phantasm-{job_id}-(result|checkpoint-(\d+))\.tar")
    matches = []
    for entry in client.files():
        match = pattern.fullmatch(entry.get("name", ""))
        if match and (not checkpoint_only or match[2]):
            priority = int(match[2]) if match[2] else float("inf")
            matches.append((priority, entry.get("date_upload", ""), entry))
    if not matches:
        raise RuntimeError("No uploaded backup exists for this job")
    return max(matches, key=lambda item: item[:2])[2]


def backup_callback(output: Path, job_id: str):
    from transformers import TrainerCallback

    class Backup(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            name = f"phantasm-{job_id}-checkpoint-{state.global_step:09d}.tar"
            bundle = output.parent / name
            try:
                archive_run(output, bundle, checkpoint)
                receipt = Pixeldrain().upload(bundle, name)
                write_json_atomic(output.parent / "latest-backup.json", receipt)
                (output.parent / "backup-warning.json").unlink(missing_ok=True)
                print(f"Checkpoint {state.global_step} backed up to Pixeldrain")
            except (OSError, ValueError, RuntimeError):
                print(
                    f"Warning: checkpoint {state.global_step} backup failed; local checkpoint retained"
                )
                write_json_atomic(
                    output.parent / "backup-warning.json", {"step": state.global_step}
                )
            finally:
                bundle.unlink(missing_ok=True)
            return control

    return Backup()


def verify_resume(checkpoint: Path, manifest: dict) -> None:
    if (
        not re.fullmatch(r"checkpoint-\d+", checkpoint.name)
        or not (checkpoint / "trainer_state.json").is_file()
    ):
        raise ValueError("Resume requires a complete Trainer checkpoint directory")
    previous = json.loads((checkpoint.parent.parent / "run.json").read_text())
    for split in ("train", "validation"):
        old, new = previous.get(split), manifest.get(split)
        if (old or {}).get("sha256") != (new or {}).get("sha256"):
            raise ValueError("Recovery dataset differs from the original training run")
    for key in ("base_model", "model_revision", "seed", "max_seq_length", "learning_rate"):
        old_value = previous["config"].get(key)
        if key == "model_revision":
            old_value = old_value or previous.get("model_revision")
        if old_value != manifest["config"].get(key):
            raise ValueError(f"Recovery setting differs from original run: {key}")
    if not (checkpoint / "optimizer.pt").is_file() or not (checkpoint / "scheduler.pt").is_file():
        raise ValueError("Checkpoint is missing optimizer/scheduler state; cannot resume training")
