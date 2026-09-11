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
        "--dataset",
        write("train.jsonl", "train question"),
        "--output-dir",
        str(tmp_path / "run"),
    ]
    if validation:
        args += ["--validation-dataset", write("val.jsonl", "validation question")]
    return parser.parse_args(args)


def test_training_preflight_fingerprints_and_disjointness(tmp_path):
    args = training_args(tmp_path)
    _, _, manifest = prepare_run(args)
    assert len(manifest["train"]["sha256"]) == 64
    assert manifest["validation"]["samples"] == 1
    Path(args.validation_dataset).write_text(Path(args.dataset).read_text())
    with pytest.raises(ValueError, match="identical"):
        prepare_run(args)


def test_empty_validation_is_rejected(tmp_path):
    args = training_args(tmp_path)
    Path(args.validation_dataset).write_text("")
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
    assert json.loads(capsys.readouterr().out)["train"]["samples"] == 1
    assert not Path(args.output_dir).exists()


def test_training_orchestration_with_fake_gpu_stack(tmp_path, monkeypatch):
    args = training_args(tmp_path)

    class Dataset:
        def __init__(self, rows):
            self.rows = rows

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def map(self, func, batched):
            assert func({"conversations": [r["conversations"] for r in self.rows]})["text"]
            return self

    tokenizer = Mock(chat_template="template")
    tokenizer.apply_chat_template.return_value = "formatted text"
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
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=fast, is_bfloat16_supported=lambda: False),
    )
    monkeypatch.setitem(
        sys.modules, "unsloth.chat_templates", SimpleNamespace(standardize_sharegpt=lambda x: x)
    )
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=Dataset))
    monkeypatch.setitem(
        sys.modules, "trl", SimpleNamespace(SFTConfig=lambda **kw: kw, SFTTrainer=factory)
    )
    monkeypatch.setattr("phantasm.training.importlib.metadata.version", lambda _: "test-version")
    train(args)
    assert factory.call_args.kwargs["processing_class"] is tokenizer
    assert factory.call_args.kwargs["eval_dataset"] is not None
    assert trainer.save_model.called and tokenizer.save_pretrained.called
    manifest = json.loads((Path(args.output_dir) / "run.json").read_text())
    assert manifest["metrics"]["validation"]["eval_loss"] == 0.6
    assert manifest["best_checkpoint"] == "checkpoint-10"
    assert manifest["gguf_files"] == ["gguf/model.gguf"]


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
            args.dataset,
            "--validation-dataset",
            args.validation_dataset,
            "--dry-run",
        ],
    )
    main()
    manifest = json.loads(capsys.readouterr().out)
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
