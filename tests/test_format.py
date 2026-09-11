"""End-to-end dataset building through the library and the real CLI."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from phantasm.cli import build_parser
from phantasm.formatter import FormatConfig, format_dataset
from phantasm.training import read_dataset

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def transcript(sessions=8, *, turns=4, others=False):
    messages = []
    index = 0
    for block in range(sessions):
        moment = START + timedelta(days=block)
        for position in range(turns):
            participant = "self" if position % 2 == 0 else "target"
            if others and position == 0 and block % 3 == 0:
                participant = "other"
            index += 1
            messages.append(
                {
                    "id": str(index),
                    "participant": participant,
                    "author_id": participant,
                    "display_name": participant,
                    "content": f"{participant} message {index}",
                    "timestamp": (moment + timedelta(minutes=position)).isoformat(),
                }
            )
    return messages


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_format_dataset_writes_three_splits_and_a_manifest(tmp_path):
    prefix = str(tmp_path / "dataset")
    manifest = format_dataset(transcript(), prefix, FormatConfig())
    for name in ("train", "val", "test"):
        assert Path(f"{prefix}_{name}_sharegpt.jsonl").is_file()
    saved = json.loads(Path(f"{prefix}_split_manifest.json").read_text())
    assert saved == manifest
    assert manifest["splits"]["train"]["samples"] > 0
    assert manifest["config"]["session_gap_seconds"] == 1800


def test_generated_samples_load_through_the_training_reader(tmp_path):
    prefix = str(tmp_path / "dataset")
    format_dataset(transcript(others=True), prefix, FormatConfig(speaker_labels="others"))
    rows, info = read_dataset(f"{prefix}_train_sharegpt.jsonl")
    assert info["samples"] == len(rows)
    for row in rows:
        assert row["conversations"][-1]["from"] == "gpt"


def test_only_target_turns_become_assistant_answers(tmp_path):
    prefix = str(tmp_path / "dataset")
    format_dataset(transcript(others=True), prefix, FormatConfig())
    for name in ("train", "val", "test"):
        for row in read_jsonl(f"{prefix}_{name}_sharegpt.jsonl"):
            for turn in row["conversations"]:
                if turn["from"] == "gpt":
                    assert turn["value"].startswith("target ")


def test_source_messages_never_appear_in_two_splits(tmp_path):
    prefix = str(tmp_path / "dataset")
    format_dataset(transcript(sessions=12), prefix, FormatConfig(val_ratio=0.25, test_ratio=0.25))

    def texts(name):
        return {
            turn["value"]
            for row in read_jsonl(f"{prefix}_{name}_sharegpt.jsonl")
            for turn in row["conversations"]
        }

    train, val, test = (texts(name) for name in ("train", "val", "test"))
    assert train and val and test
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)


def test_disabled_splits_are_written_empty_to_avoid_stale_reuse(tmp_path):
    prefix = str(tmp_path / "dataset")
    format_dataset(transcript(), prefix, FormatConfig(val_ratio=0.25, test_ratio=0.25))
    assert Path(f"{prefix}_val_sharegpt.jsonl").read_text()
    format_dataset(transcript(), prefix, FormatConfig(val_ratio=0, test_ratio=0))
    assert Path(f"{prefix}_val_sharegpt.jsonl").read_text() == ""
    assert Path(f"{prefix}_test_sharegpt.jsonl").read_text() == ""


def test_system_prompt_is_applied_to_every_sample(tmp_path):
    prefix = str(tmp_path / "dataset")
    format_dataset(transcript(), prefix, FormatConfig(system_prompt="be brief"))
    rows = read_jsonl(f"{prefix}_train_sharegpt.jsonl")
    assert all(row["conversations"][0] == {"from": "system", "value": "be brief"} for row in rows)


def test_gap_configuration_is_validated():
    with pytest.raises(ValueError, match="turn-gap-seconds <= session-gap-seconds"):
        FormatConfig(turn_gap_seconds=600, session_gap_seconds=300)
    with pytest.raises(ValueError, match="speaker-labels"):
        FormatConfig(speaker_labels="loud")


def test_filters_are_reported_in_the_manifest(tmp_path):
    messages = transcript(sessions=4)
    messages.append({**messages[0], "id": "bot", "content": "[deleted]"})
    manifest = format_dataset(messages, str(tmp_path / "dataset"), FormatConfig())
    assert manifest["filtering"]["by_reason"] == {"deleted_placeholder": 1}


def run_cli(monkeypatch, *argv):
    from phantasm.cli import main

    monkeypatch.setattr("sys.argv", ["phantasm", *argv])
    main()


def test_cli_format_accepts_old_and_new_flag_spellings(tmp_path, monkeypatch, capsys):
    parsed = tmp_path / "parsed.json"
    parsed.write_text(json.dumps(transcript()))
    prefix = str(tmp_path / "dataset")
    run_cli(
        monkeypatch,
        "format",
        "-i",
        str(parsed),
        "-p",
        prefix,
        "--max-gap",
        "300",
        "--val-split",
        "0.2",
    )
    assert "train" in capsys.readouterr().out
    manifest = json.loads(Path(f"{prefix}_split_manifest.json").read_text())
    assert manifest["config"]["turn_gap_seconds"] == 300
    assert manifest["config"]["val_ratio"] == 0.2


def test_cli_reports_user_errors_without_a_traceback(tmp_path, monkeypatch, capsys):
    parsed = tmp_path / "parsed.json"
    parsed.write_text(json.dumps(transcript(sessions=1)))
    monkeypatch.setattr(
        "sys.argv",
        ["phantasm", "format", "-i", str(parsed), "-p", str(tmp_path / "d"), "--val-ratio", "0.5"],
    )
    from phantasm.cli import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Error: " in capsys.readouterr().err


def test_parse_command_exposes_self_and_target_options():
    args = build_parser().parse_args(["parse", "raw.json", "--self", "A", "--target", "B"])
    assert (args.self_ref, args.target_ref, args.others) == ("A", "B", "context")
    legacy = build_parser().parse_args(["parse", "raw.json", "--user-id", "1", "--target-id", "2"])
    assert (legacy.self_ref, legacy.target_ref) == ("1", "2")


def test_full_cli_pipeline_on_the_bundled_example(tmp_path, monkeypatch, capsys):
    example = Path(__file__).resolve().parents[1] / "examples/demo.json"
    if not example.is_file():  # pragma: no cover - source distributions ship the example
        pytest.skip("bundled example is unavailable")
    parsed = tmp_path / "parsed.json"
    prefix = str(tmp_path / "dataset")
    run_cli(
        monkeypatch,
        "parse",
        str(example),
        "--self",
        "user",
        "--target",
        "persona",
        "-o",
        str(parsed),
    )
    run_cli(monkeypatch, "format", "-i", str(parsed), "-p", prefix)
    run_cli(monkeypatch, "inspect", str(parsed))
    run_cli(monkeypatch, "audit", f"{prefix}_train_sharegpt.jsonl")
    output = capsys.readouterr().out
    assert "Usable training samples" in output
    assert "Phantasm privacy audit" in output


def test_format_report_shows_what_was_filtered(tmp_path, monkeypatch, capsys):
    parsed = tmp_path / "parsed.json"
    messages = transcript(sessions=4)
    messages.append({**messages[0], "id": "x", "content": "[deleted]"})
    parsed.write_text(json.dumps(messages))
    run_cli(monkeypatch, "format", "-i", str(parsed), "-p", str(tmp_path / "dataset"))
    output = capsys.readouterr().out
    assert "removed 1" in output
    assert "deleted placeholder" in output
