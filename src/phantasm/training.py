"""Packaged Unsloth training entry point; GPU imports occur only when training starts."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

from phantasm.storage import write_json_atomic
from phantasm.transfer import transfer_gguf


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, help="Training ShareGPT JSONL")
    parser.add_argument("--backend", choices=["local", "colab"], default="local")
    parser.add_argument("--artifact-store", choices=["colab", "pixeldrain"], default="colab")
    parser.add_argument("--job", help="Local name for the remote job")
    parser.add_argument("--jobs-dir", default=".phantasm/jobs")
    parser.add_argument("--session", help="Reuse an existing Colab session")
    parser.add_argument("--gpu", choices=["T4", "L4", "G4", "H100", "A100"], default="T4")
    parser.add_argument("--detach", action="store_true", help="Submit without waiting locally")
    parser.add_argument(
        "--keep", action="store_true", help="Keep an owned runtime after verified retrieval"
    )
    parser.add_argument("--resume-from-checkpoint", help="Resume a local Trainer checkpoint")
    parser.add_argument("--backup-job-id", help=argparse.SUPPRESS)

    parser.add_argument("--validation-dataset", help="Disjoint validation ShareGPT JSONL")
    parser.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--model-revision", help="Hugging Face model commit for reproducible runs")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-dir", default="phantasm_model", help="New or empty run directory")
    parser.add_argument("--quant-method", choices=["q4_k_m", "q8_0", "f16"], default="q4_k_m")
    parser.add_argument(
        "--croc-transfer",
        action="store_true",
        help="Transfer the exported GGUF using installed croc",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/configuration without GPU dependencies",
    )


def read_dataset(path: str) -> tuple[list[dict], dict]:
    target = Path(path).expanduser().resolve()
    raw = target.read_bytes()
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        convo = row.get("conversations") if isinstance(row, dict) else None
        if not isinstance(convo, list) or len(convo) < 2:
            raise ValueError(f"{target.name}:{line_number}: expected a conversation")
        for turn in convo:
            if (
                not isinstance(turn, dict)
                or turn.get("from") not in ("system", "human", "gpt")
                or not isinstance(turn.get("value"), str)
                or not turn["value"].strip()
            ):
                raise ValueError(f"{target.name}:{line_number}: invalid role/content")
        dialogue = convo[1:] if convo[0]["from"] == "system" else convo
        if (
            not dialogue
            or dialogue[0]["from"] != "human"
            or dialogue[-1]["from"] != "gpt"
            or any(t["from"] == "system" for t in dialogue)
        ):
            raise ValueError(f"{target.name}:{line_number}: require user-to-assistant dialogue")
        rows.append({"conversations": convo})
    if not rows:
        raise ValueError(f"Dataset is empty: {target}")
    return rows, {
        "path": str(target),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "samples": len(rows),
    }


def prepare_run(args: argparse.Namespace) -> tuple[list, list, dict]:
    if min(args.max_seq_length, args.max_steps, args.eval_steps) < 1:
        raise ValueError("Sequence length, training steps and evaluation steps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    train_rows, train_info = read_dataset(args.dataset)
    val_rows, val_info = (
        read_dataset(args.validation_dataset) if args.validation_dataset else ([], None)
    )
    signatures = {json.dumps(row, sort_keys=True) for row in train_rows}
    if any(json.dumps(row, sort_keys=True) in signatures for row in val_rows):
        raise ValueError(
            "Training and validation contain identical samples; regenerate disjoint splits"
        )
    return (
        train_rows,
        val_rows,
        {
            "config": {
                key: value for key, value in vars(args).items() if key not in ("func", "command")
            },
            "train": train_info,
            "validation": val_info,
            "python": platform.python_version(),
        },
    )


def trainer_settings(args: argparse.Namespace, has_validation: bool) -> dict:
    evaluate = has_validation
    return {
        "max_length": args.max_seq_length,
        "dataset_text_field": "text",
        "packing": False,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": min(10, args.max_steps // 10),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": min(args.eval_steps, args.max_steps),
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "output_dir": str(Path(args.output_dir) / "checkpoints"),
        "report_to": "none",
        "eval_strategy": "steps" if evaluate else "no",
        "eval_steps": min(args.eval_steps, args.max_steps),
        "save_strategy": "steps",
        "save_steps": min(args.eval_steps, args.max_steps),
        "save_total_limit": 2,
        "load_best_model_at_end": evaluate,
        "metric_for_best_model": "eval_loss" if evaluate else None,
        "greater_is_better": False if evaluate else None,
    }


def collect_gguf_exports(directory: Path, quant_method: str) -> list[Path]:
    # The pinned Unsloth exporter writes into a sibling '<directory>_gguf'.
    # Keep compatibility with exporters that write directly into directory.
    sibling = directory.with_name(directory.name + "_gguf")
    sources = sorted(sibling.glob("*.gguf")) if sibling.is_dir() else []
    if sources:
        # A quantized export can leave a full-precision conversion intermediate.
        requested = [p for p in sources if f".{quant_method.upper()}" in p.name.upper()]
        if not requested:
            raise RuntimeError(f"Export produced no {quant_method} GGUF files in {sibling}")
        if any(not p.stat().st_size or p.is_symlink() for p in requested):
            raise RuntimeError("Export produced an empty or linked GGUF file")
        if any((directory / p.name).exists() for p in requested):
            raise ValueError("GGUF destination already exists; refusing to overwrite it")
        directory.mkdir(parents=True, exist_ok=True)
        for source in requested:
            source.replace(directory / source.name)
    ggufs = sorted(directory.glob("*.gguf"))
    if not ggufs or any(not p.stat().st_size or p.is_symlink() for p in ggufs):
        raise RuntimeError(f"Export produced no valid GGUF files in {directory}")
    return ggufs


def train(args: argparse.Namespace) -> None:
    if args.backend == "colab":
        from phantasm.colab import submit

        submit(args)
        return
    train_rows, val_rows, manifest = prepare_run(args)
    checkpoint = (
        Path(args.resume_from_checkpoint).expanduser().resolve()
        if args.resume_from_checkpoint
        else None
    )
    if checkpoint:
        from phantasm.artifacts import verify_resume

        previous = json.loads((checkpoint.parent.parent / "run.json").read_text())
        if not args.model_revision:
            args.model_revision = previous.get("model_revision")
            manifest["config"]["model_revision"] = args.model_revision
        verify_resume(checkpoint, manifest)
    if args.artifact_store == "pixeldrain":
        import re

        if not args.backup_job_id or not re.fullmatch(r"[0-9a-f]{32}", args.backup_job_id):
            raise ValueError("Pixeldrain checkpoint backup requires a Colab job ID")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    output = Path(args.output_dir).expanduser().resolve()
    if (
        output.exists()
        and (not output.is_dir() or any(output.iterdir()))
        and (checkpoint is None or checkpoint.parent.parent != output)
    ):
        raise ValueError("Choose a new or empty output directory to preserve previous runs")
    # Unsloth must patch its integrations before importing TRL/Transformers.
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: skip
        from unsloth.chat_templates import standardize_sharegpt  # isort: skip
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        raise RuntimeError(
            "Install the locked GPU environment described in docs/TRAINING.md"
        ) from None
    args.output_dir = str(output)
    manifest["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("unsloth", "unsloth-zoo", "torch", "transformers", "trl", "datasets")
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "run.json", manifest)
    model_options = {"revision": args.model_revision} if args.model_revision else {}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        **model_options,
    )
    manifest["model_revision"] = getattr(model.config, "_commit_hash", None)
    write_json_atomic(output / "run.json", manifest)
    if not tokenizer.chat_template:
        raise ValueError("The selected model must provide a chat template")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    def prepare(rows):
        dataset = standardize_sharegpt(Dataset.from_list(rows))

        def render(examples):
            texts = [
                tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                for c in examples["conversations"]
            ]
            lengths = [
                len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts
            ]
            if any(length > args.max_seq_length for length in lengths):
                raise ValueError(
                    "A sample exceeds max-seq-length; reduce the formatting window or increase the limit"
                )
            return {"text": texts}

        return dataset.map(render, batched=True)

    train_dataset = prepare(train_rows)
    eval_dataset = prepare(val_rows) if val_rows else None
    settings = trainer_settings(args, bool(val_rows))
    settings.update(bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported())
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**settings),
    )
    if args.artifact_store == "pixeldrain":
        from phantasm.artifacts import backup_callback

        trainer.add_callback(backup_callback(output, args.backup_job_id))
    result = (
        trainer.train(resume_from_checkpoint=str(checkpoint)) if checkpoint else trainer.train()
    )
    metrics = {"training": result.metrics}
    if val_rows:
        metrics["validation"] = trainer.evaluate()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    manifest.update(
        metrics=metrics,
        best_checkpoint=trainer.state.best_model_checkpoint,
        model_revision=getattr(model.config, "_commit_hash", None),
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    )
    write_json_atomic(output / "run.json", manifest)
    gguf_dir = output / "gguf"
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=args.quant_method)
    ggufs = collect_gguf_exports(gguf_dir, args.quant_method)
    manifest["gguf_files"] = [str(p.relative_to(output)) for p in ggufs]
    write_json_atomic(output / "run.json", manifest)
    print(f"Training and export complete: {output}")
    if args.croc_transfer:
        if len(ggufs) != 1:
            raise ValueError(
                "Multiple GGUF files were exported; transfer the required files explicitly with croc"
            )
        transfer_gguf(ggufs[0])
