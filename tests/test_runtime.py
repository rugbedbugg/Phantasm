"""GPU-free contracts for chat history, training orchestration and secure transfers."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phantasm.credentials import read_secret
from phantasm.inference import chat_loop, trim_history
from phantasm.training import add_training_arguments, prepare_run, train, trainer_settings
from phantasm.transfer import transfer_gguf


def test_trimming_preserves_system_latest_user_and_whole_exchanges():
    history = [
        {"role": r, "content": str(i)}
        for i, r in enumerate(["system", "user", "assistant", "user", "assistant", "user"])
    ]
    assert trim_history(history, len, 4) == [history[0], *history[3:]]
    assert len(history) == 6
    with pytest.raises(ValueError, match="exceed"):
        trim_history([history[0], history[-1]], len, 1)


def test_chat_failure_rolls_back_and_counts_formatted_prompt(monkeypatch):
    inputs = iter(["first", "failed", "third", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    llm = Mock()
    llm.tokenize.side_effect = lambda text, **kw: list(text)
    llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": "reply"}}]},
        RuntimeError("generation failed"),
        {"choices": [{"message": {"content": "done"}}]},
    ]
    formatter = Mock(
        side_effect=lambda messages: SimpleNamespace(
            prompt="template:" + json.dumps(messages), added_special=True
        )
    )
    chat_loop(llm, formatter, "system", 1000, 20, 0.7, 0.9)
    last = llm.create_chat_completion.call_args.kwargs["messages"]
    assert [m["content"] for m in last] == ["system", "first", "reply", "third"]
    assert llm.tokenize.call_args.kwargs == {"add_bos": False, "special": True}


def test_overlong_chat_input_does_not_generate(monkeypatch):
    inputs = iter(["too long", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    llm = Mock()
    llm.tokenize.return_value = list(range(100))
    formatter = Mock(return_value=SimpleNamespace(prompt="p", added_special=True))
    chat_loop(llm, formatter, "", 100, 20, 0.7, 0.9)
    llm.create_chat_completion.assert_not_called()


@pytest.mark.parametrize(
    "secret", ["legitimate-secret", "quote'; echo unexpected; '", "$(touch unwanted)"]
)
def test_transfer_treats_secrets_and_filenames_as_data(tmp_path, monkeypatch, capsys, secret):
    target = tmp_path / "- model; echo unwanted.gguf"
    target.write_bytes(b"GGUF")
    monkeypatch.setenv("CROC_SECRET", secret)
    monkeypatch.setattr("phantasm.transfer.shutil.which", lambda _: "/usr/bin/croc")
    run = Mock()
    monkeypatch.setattr("phantasm.transfer.subprocess.run", run)
    transfer_gguf(target)
    argv = run.call_args.args[0]
    kwargs = run.call_args.kwargs
    assert argv[-1] == str(target.resolve())
    assert secret not in argv
    assert kwargs["env"]["CROC_SECRET"] == secret
    assert not kwargs.get("shell", False)
    assert kwargs["check"] is True
    assert kwargs["stdout"] == subprocess.DEVNULL and kwargs["stderr"] == subprocess.DEVNULL
    assert secret not in capsys.readouterr().out


def test_transfer_failure_is_not_success_and_hides_subprocess_details(
    tmp_path, monkeypatch, capsys
):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"GGUF")
    secret = "private-secret"
    monkeypatch.setenv("CROC_SECRET", secret)
    monkeypatch.setattr("phantasm.transfer.shutil.which", lambda _: "/usr/bin/croc")
    monkeypatch.setattr(
        "phantasm.transfer.subprocess.run",
        Mock(side_effect=subprocess.CalledProcessError(1, secret)),
    )
    with pytest.raises(RuntimeError) as exc:
        transfer_gguf(target)
    assert secret not in str(exc.value)
    assert "Transfer complete" not in capsys.readouterr().out


def test_missing_croc_does_not_install_anything(tmp_path, monkeypatch):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"GGUF")
    monkeypatch.setattr("phantasm.transfer.shutil.which", lambda _: None)
    run = Mock()
    monkeypatch.setattr("phantasm.transfer.subprocess.run", run)
    with pytest.raises(RuntimeError, match="Install croc"):
        transfer_gguf(target)
    run.assert_not_called()


def test_credentials_require_env_without_tty(monkeypatch):
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(ValueError, match="DISCORD_TOKEN"):
        read_secret("DISCORD_TOKEN", "Token: ")
    monkeypatch.setenv("DISCORD_TOKEN", "token-from-env")
    assert read_secret("DISCORD_TOKEN", "Token: ") == "token-from-env"


def training_args(tmp_path, validation=True):
    def write(name, text):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "conversations": [
                        {"from": "human", "value": text},
                        {"from": "gpt", "value": "reply"},
                    ]
                }
            )
            + "\n"
        )
        return str(path)

    parser = argparse.ArgumentParser()
    add_training_arguments(parser)
    args = [
        "--train-dataset",
        write("train.jsonl", "train question"),
        "--output-dir",
        str(tmp_path / "run"),
    ]
    if validation:
        args += ["--eval-dataset", write("val.jsonl", "validation question")]
    return parser.parse_args(args)


def test_training_preflight_fingerprints_and_disjointness(tmp_path):
    args = training_args(tmp_path)
    _, _, manifest = prepare_run(args)
    assert len(manifest["train"]["sha256"]) == 64
    assert manifest["validation"]["samples"] == 1
    Path(args.eval_dataset).write_text(Path(args.train_dataset).read_text())
    with pytest.raises(ValueError, match="identical"):
        prepare_run(args)


def test_empty_validation_is_rejected(tmp_path):
    args = training_args(tmp_path)
    Path(args.eval_dataset).write_text("")
    with pytest.raises(ValueError, match="empty"):
        prepare_run(args)


def test_validation_selects_best_checkpoint_and_no_validation_disables_it(tmp_path):
    args = training_args(tmp_path)
    settings = trainer_settings(args, True)
    assert settings["load_best_model_at_end"]
    assert settings["metric_for_best_model"] == "eval_loss"
    assert settings["eval_steps"] == settings["save_steps"]
    assert not trainer_settings(args, False)["load_best_model_at_end"]


def test_dry_run_needs_no_gpu_and_creates_no_output(tmp_path, capsys):
    args = training_args(tmp_path)
    args.dry_run = True
    train(args)
    printed = capsys.readouterr().out
    assert "Phantasm training configuration" in printed
    manifest = json.loads(printed[printed.index("{") :])
    assert manifest["train"]["samples"] == 1
    assert not Path(args.output_dir).exists()


def fake_gpu_stack(monkeypatch, *, recorder=None):
    """Install a minimal stand-in for the Unsloth/TRL/datasets stack."""

    class Dataset:
        def __init__(self, rows):
            self.rows = rows

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def map(self, func, batched):
            assert func({"conversations": [r["conversations"] for r in self.rows]})["text"]
            return self

    def apply_chat_template(conversation, tokenize=False, add_generation_prompt=False):
        roles = {"human": "user", "gpt": "assistant", "system": "system"}
        text = "<|begin_of_text|>"
        for turn in conversation:
            role = turn.get("role") or roles[turn["from"]]
            content = turn.get("content", turn.get("value"))
            text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        if add_generation_prompt:
            text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return text

    tokenizer = Mock(chat_template="template")
    tokenizer.apply_chat_template.side_effect = apply_chat_template
    tokenizer.return_value = {"input_ids": [1, 2, 3]}
    model = Mock()
    model.config._commit_hash = "model-commit"

    def export(path, tok, quantization_method):
        Path(path).mkdir()
        (Path(path) / "model.gguf").write_bytes(b"GGUF")

    model.save_pretrained_gguf.side_effect = export
    fast = Mock()
    fast.from_pretrained.return_value = (model, tokenizer)
    fast.get_peft_model.return_value = model
    trainer = Mock()
    trainer.train.return_value.metrics = {"train_loss": 0.5}
    trainer.evaluate.return_value = {"eval_loss": 0.6}
    trainer.state.best_model_checkpoint = "checkpoint-10"
    factory = Mock(return_value=trainer)
    masking = Mock(side_effect=lambda trainer, **kwargs: trainer)
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=fast, is_bfloat16_supported=lambda: False),
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth.chat_templates",
        SimpleNamespace(standardize_sharegpt=lambda x: x, train_on_responses_only=masking),
    )
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=Dataset))
    monkeypatch.setitem(
        sys.modules, "trl", SimpleNamespace(SFTConfig=lambda **kw: kw, SFTTrainer=factory)
    )
    monkeypatch.setattr("phantasm.training.importlib.metadata.version", lambda _: "test-version")
    return SimpleNamespace(
        factory=factory, trainer=trainer, masking=masking, tokenizer=tokenizer, peft=fast
    )


def test_training_orchestration_with_fake_gpu_stack(tmp_path, monkeypatch):
    args = training_args(tmp_path)
    stack = fake_gpu_stack(monkeypatch)
    train(args)
    assert stack.factory.call_args.kwargs["processing_class"] is stack.tokenizer
    assert stack.factory.call_args.kwargs["eval_dataset"] is not None
    assert stack.trainer.save_model.called and stack.tokenizer.save_pretrained.called
    manifest = json.loads((Path(args.output_dir) / "run.json").read_text())
    assert manifest["metrics"]["validation"]["eval_loss"] == 0.6
    assert manifest["best_checkpoint"] == "checkpoint-10"
    assert manifest["gguf_files"] == ["gguf/model.gguf"]
    config = json.loads((Path(args.output_dir) / "training_config.json").read_text())
    assert config["seed"] == args.seed and config["loss"] == "response_only"
    assert config["train_dataset"] == args.train_dataset


def test_response_only_loss_masks_context_with_derived_markers(tmp_path, monkeypatch):
    args = training_args(tmp_path)
    stack = fake_gpu_stack(monkeypatch)
    train(args)
    kwargs = stack.masking.call_args.kwargs
    assert kwargs["instruction_part"] == "<|start_header_id|>user<|end_header_id|>\n\n"
    assert kwargs["response_part"] == "<|start_header_id|>assistant<|end_header_id|>\n\n"
    manifest = json.loads((Path(args.output_dir) / "run.json").read_text())
    assert manifest["loss_markers"]["response"].startswith("<|start_header_id|>assistant")


def test_full_loss_skips_masking(tmp_path, monkeypatch):
    args = training_args(tmp_path)
    args.loss = "full"
    stack = fake_gpu_stack(monkeypatch)
    train(args)
    stack.masking.assert_not_called()


def test_lora_and_batch_options_reach_the_trainer(tmp_path, monkeypatch):
    args = training_args(tmp_path)
    args.lora_r, args.lora_alpha, args.lora_dropout = 32, 64, 0.05
    args.batch_size, args.gradient_accumulation, args.epochs = 1, 8, 2.0
    stack = fake_gpu_stack(monkeypatch)
    train(args)
    peft = stack.peft.get_peft_model.call_args.kwargs
    assert (peft["r"], peft["lora_alpha"], peft["lora_dropout"]) == (32, 64, 0.05)
    settings = stack.factory.call_args.kwargs["args"]
    assert settings["per_device_train_batch_size"] == 1
    assert settings["gradient_accumulation_steps"] == 8
    assert settings["num_train_epochs"] == 2.0
    assert "max_steps" not in settings


def test_early_stopping_is_registered_when_requested(tmp_path, monkeypatch):
    args = training_args(tmp_path)
    args.early_stopping_patience = 2
    callback = Mock()
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(EarlyStoppingCallback=callback)
    )
    stack = fake_gpu_stack(monkeypatch)
    train(args)
    callback.assert_called_once_with(early_stopping_patience=2)
    assert stack.trainer.add_callback.called


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("lora_r", 0, "LoRA rank"),
        ("lora_dropout", 1.0, "lora-dropout"),
        ("batch_size", 0, "batch size"),
        ("epochs", 0.0, "epochs must be positive"),
        ("learning_rate", 0.0, "Learning rate"),
        ("early_stopping_patience", -1, "cannot be negative"),
    ],
)
def test_invalid_hyperparameters_are_rejected(tmp_path, field, value, message):
    args = training_args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        prepare_run(args)


def test_early_stopping_requires_validation_data(tmp_path):
    args = training_args(tmp_path, validation=False)
    args.early_stopping_patience = 1
    with pytest.raises(ValueError, match="needs --eval-dataset"):
        prepare_run(args)


def test_training_dry_run_through_actual_cli(tmp_path, monkeypatch, capsys):
    from phantasm.cli import main

    args = training_args(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phantasm",
            "train",
            "--dataset",
            args.train_dataset,
            "--validation-dataset",
            args.eval_dataset,
            "--dry-run",
        ],
    )
    main()
    printed = capsys.readouterr().out
    manifest = json.loads(printed[printed.index("{") :])
    assert "func" not in manifest["config"]
    assert manifest["train"]["samples"] == 1


def test_collect_export_from_unsloth_sibling_keeps_requested_quantization(tmp_path):
    from phantasm.training import collect_gguf_exports

    requested = tmp_path / "gguf"
    requested.mkdir()
    (requested / "model.safetensors").write_bytes(b"merge")
    actual = tmp_path / "gguf_gguf"
    actual.mkdir()
    quantized = actual / "model.Q4_K_M.gguf"
    quantized.write_bytes(b"quantized")
    (actual / "model.F16.gguf").write_bytes(b"intermediate")
    result = collect_gguf_exports(requested, "q4_k_m")
    assert result == [requested / quantized.name]
    assert result[0].read_bytes() == b"quantized"
    assert (actual / "model.F16.gguf").exists()
    assert not quantized.exists()


def test_collect_export_rejects_incomplete_quantization(tmp_path):
    from phantasm.training import collect_gguf_exports

    actual = tmp_path / "gguf_gguf"
    actual.mkdir()
    (actual / "model.F16.gguf").write_bytes(b"intermediate")
    with pytest.raises(RuntimeError, match="no q4_k_m"):
        collect_gguf_exports(tmp_path / "gguf", "q4_k_m")
