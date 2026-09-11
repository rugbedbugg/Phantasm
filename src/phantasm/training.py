"""Packaged Unsloth training entry point; GPU imports occur only when training starts.

Training optimizes a single objective: given conversation context, produce the
*target persona's* next message. Loss is therefore masked to the assistant spans
of each sample by default (``--loss response_only``); the prompt tokens that
describe the context receive no gradient. ``--loss full`` restores plain
full-sequence language modelling.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from typing import Any

from phantasm.dataset import read_sharegpt
from phantasm.storage import write_json_atomic
from phantasm.transfer import transfer_gguf

#: Configuration keys echoed in the pre-flight summary, in display order.
SUMMARY_KEYS = (
    "base_model",
    "model_revision",
    "loss",
    "max_seq_length",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "learning_rate",
    "batch_size",
    "gradient_accumulation",
    "epochs",
    "max_steps",
    "eval_steps",
    "early_stopping_patience",
    "seed",
    "quant_method",
    "output_dir",
)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Register every training option, including the 0.1 flag spellings."""
    parser.add_argument(
        "--train-dataset",
        "--dataset",
        dest="train_dataset",
        required=True,
        help="Training ShareGPT JSONL (--dataset is accepted as the old spelling)",
    )
    parser.add_argument(
        "--eval-dataset",
        "--validation-dataset",
        dest="eval_dataset",
        help="Disjoint validation ShareGPT JSONL used for eval loss and best-model selection",
    )
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

    parser.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--model-revision", help="Hugging Face model commit for reproducible runs")
    parser.add_argument(
        "--loss",
        choices=["response_only", "full"],
        default="response_only",
        help="Apply loss to target responses only (default) or to the whole sequence",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument(
        "--epochs", type=float, help="Train for epochs instead of a fixed --max-steps"
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--eval-steps", type=int, default=10, help="Evaluation/checkpoint interval")
    parser.add_argument(
        "--warmup-steps", type=int, help="Default: 10%% of --max-steps, capped at 10"
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-scheduler",
        default="linear",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after N evaluations without improvement (needs --eval-dataset; 0 disables)",
    )
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
    """Strictly validate a ShareGPT dataset before it reaches a GPU."""
    return read_sharegpt(path, strict=True)


def validate_configuration(args: argparse.Namespace) -> None:
    """Reject impossible hyperparameters before anything expensive happens."""
    if min(args.max_seq_length, args.eval_steps) < 1:
        raise ValueError("Sequence length and evaluation steps must be positive")
    if args.epochs is None and args.max_steps < 1:
        raise ValueError("Supply a positive --max-steps or --epochs")
    if args.epochs is not None and (not math.isfinite(args.epochs) or args.epochs <= 0):
        raise ValueError("--epochs must be positive and finite")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    if min(args.lora_r, args.lora_alpha, args.batch_size, args.gradient_accumulation) < 1:
        raise ValueError("LoRA rank/alpha, batch size and accumulation must be positive")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must be at least 0 and below 1")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative")
    if args.early_stopping_patience and not args.eval_dataset:
        raise ValueError("Early stopping needs --eval-dataset")


def prepare_run(args: argparse.Namespace) -> tuple[list, list, dict]:
    """Validate configuration and datasets, returning rows and the run manifest."""
    validate_configuration(args)
    train_rows, train_info = read_dataset(args.train_dataset)
    val_rows, val_info = read_dataset(args.eval_dataset) if args.eval_dataset else ([], None)
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


def summarize(manifest: dict[str, Any]) -> str:
    """Compact configuration summary printed before training begins."""
    config = manifest["config"]
    lines = ["Phantasm training configuration", "─" * 46]
    lines += [
        f"{'train samples':<26}{manifest['train']['samples']}",
        f"{'eval samples':<26}"
        + str(manifest["validation"]["samples"] if manifest["validation"] else 0),
    ]
    lines += [
        f"{key.replace('_', ' '):<26}{config[key]}"
        for key in SUMMARY_KEYS
        if config.get(key) is not None
    ]
    effective = config["batch_size"] * config["gradient_accumulation"]
    lines.append(f"{'effective batch':<26}{effective}")
    return "\n".join(lines)


def trainer_settings(args: argparse.Namespace, has_validation: bool) -> dict:
    """Build the ``SFTConfig`` keyword arguments for one run."""
    evaluate = has_validation
    interval = min(args.eval_steps, args.max_steps) if args.epochs is None else args.eval_steps
    warmup = (
        args.warmup_steps
        if args.warmup_steps is not None
        else min(10, max(args.max_steps // 10, 0) if args.epochs is None else 10)
    )
    settings = {
        "max_length": args.max_seq_length,
        "dataset_text_field": "text",
        "packing": False,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "warmup_steps": warmup,
        "learning_rate": args.learning_rate,
        "logging_steps": interval,
        "optim": "adamw_8bit",
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler,
        "seed": args.seed,
        "output_dir": str(Path(args.output_dir) / "checkpoints"),
        "report_to": "none",
        "eval_strategy": "steps" if evaluate else "no",
        "eval_steps": interval,
        "save_strategy": "steps",
        "save_steps": interval,
        "save_total_limit": 2,
        "load_best_model_at_end": evaluate,
        "metric_for_best_model": "eval_loss" if evaluate else None,
        "greater_is_better": False if evaluate else None,
    }
    if args.epochs is None:
        settings["max_steps"] = args.max_steps
    else:
        settings["num_train_epochs"] = args.epochs
    return settings


def export_name(directory: Path, quant_method: str) -> str:
    """Canonical GGUF filename for a run: ``<run directory>.<QUANT>.gguf``.

    The converter names its output after the base architecture, so a fine-tuned
    persona is written as, for example,
    ``Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf``. That is indistinguishable from a
    stock upstream release once the file leaves its directory, which invites
    deleting a trained model by mistake. Naming the export after the run makes
    what it is obvious from the filename alone.
    """
    run = directory.parent.name or "phantasm"
    return f"{run}.{quant_method.upper()}.gguf"


def collect_gguf_exports(directory: Path, quant_method: str) -> list[Path]:
    """Move, verify and rename the export, leaving exactly one GGUF in place."""
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
    if len(ggufs) > 1:
        names = ", ".join(p.name for p in ggufs)
        raise RuntimeError(
            f"Export produced {len(ggufs)} GGUF files ({names}); a run must yield one "
            "model. Re-export a single quantization, or move the extra files aside."
        )
    target = directory / export_name(directory, quant_method)
    if ggufs[0] != target:
        if target.exists():
            raise ValueError(f"{target.name} already exists; refusing to overwrite it")
        ggufs[0].replace(target)
    return [target]


def assistant_messages(conversation: list[dict[str, Any]]) -> list[str]:
    """Assistant/target contents of a conversation in ShareGPT or role/content form."""
    roles = {"gpt": "assistant", "human": "user", "system": "system"}
    answers = []
    for turn in conversation:
        role = turn.get("role") or roles.get(turn.get("from"))
        if role == "assistant":
            answers.append(turn.get("content") or turn.get("value") or "")
    return answers


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
    print(summarize(manifest))
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
        from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only  # isort: skip
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
    write_json_atomic(output / "training_config.json", manifest["config"])
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

    def render_conversation(conversation, add_generation_prompt=False):
        return tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    markers = None
    if args.loss == "response_only":
        from phantasm.chat_masking import derive_response_markers, verify_masking

        markers = derive_response_markers(render_conversation)
        manifest["loss_markers"] = {"instruction": markers[0], "response": markers[1]}
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    def prepare(rows):
        dataset = standardize_sharegpt(Dataset.from_list(rows))

        def render(examples):
            texts = [render_conversation(c) for c in examples["conversations"]]
            lengths = [
                len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts
            ]
            if any(length > args.max_seq_length for length in lengths):
                raise ValueError(
                    "A sample exceeds max-seq-length; reduce the formatting window or increase the limit"
                )
            if markers and texts:
                # Fail before the GPU run if masking would drop a target response.
                verify_masking(texts[0], *markers, assistant_messages(examples["conversations"][0]))
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
    if markers:
        trainer = train_on_responses_only(
            trainer, instruction_part=markers[0], response_part=markers[1]
        )
    if args.early_stopping_patience:
        from transformers import EarlyStoppingCallback

        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
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
