"""Statistics reported by ``phantasm inspect``."""

import json

import pytest

from phantasm.inspection import inspect_parsed, inspect_path, length_stats, render


def message(index, participant, minute, content, **extra):
    return {
        "id": str(index),
        "participant": participant,
        "author_id": participant,
        "content": content,
        "timestamp": f"2026-01-01T10:{minute:02d}:00Z",
        **extra,
    }


TRANSCRIPT = [
    message(1, "self", 0, "hey"),
    message(2, "target", 1, "hello there friend"),
    message(3, "other", 2, "hi everyone"),
    message(4, "target", 3, "look at https://example.invalid ```code``` 😀"),
    message(5, "self", 4, "", attachments=["https://example.invalid/a.png"]),
    message(6, "self", 5, "still there?"),
    message(7, "target", 6, "hello there friend"),
]


def test_counts_per_participant_and_invalid_rows():
    stats = inspect_parsed({"messages": [*TRANSCRIPT, None, {"role": "moderator", "content": "x"}]})
    assert stats["messages"]["target"] == 3
    assert stats["messages"]["self"] == 2
    assert stats["messages"]["other"] == 1
    assert stats["messages"]["empty"] == 1
    assert stats["messages"]["invalid"] == 2


def test_turn_session_and_sample_estimates():
    stats = inspect_parsed(TRANSCRIPT, session_gap_seconds=60)
    assert stats["turns"]["total"] == 6
    assert stats["turns"]["target"] == 3
    assert stats["turns"]["sessions"] >= 1
    assert stats["usable_training_samples"] >= 1


def test_markers_duplicates_and_timestamp_coverage():
    stats = inspect_parsed(TRANSCRIPT)
    assert stats["markers"] == {"urls": 1, "code_blocks": 1, "emoji": 1}
    assert stats["duplicates"]["exact_duplicates"] == 1
    assert stats["attachments"] == 1
    assert stats["timestamps"]["coverage"] == 1.0
    assert stats["timestamps"]["first"].startswith("2026-01-01T10:00")


def test_missing_timestamps_lower_coverage():
    rows = [dict(item) for item in TRANSCRIPT[:2]]
    rows[1]["timestamp"] = "broken"
    assert inspect_parsed(rows)["timestamps"]["coverage"] == 0.5


def test_length_statistics_are_reported_for_responses():
    stats = length_stats(["one", "two words here", "a"])
    assert stats["count"] == 3
    assert stats["median_words"] == 1
    assert stats["max_words"] == 3
    assert length_stats([]) == {"count": 0}


def test_sharegpt_dataset_statistics(tmp_path):
    path = tmp_path / "dataset_train_sharegpt.jsonl"
    row = {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo 😀"}]}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n" + "{bad\n")
    stats = inspect_path(str(path))
    assert stats["kind"] == "ShareGPT dataset"
    assert stats["samples"] == 2
    assert stats["duplicate_samples"] == 1
    assert stats["invalid_rows"] == 1
    assert stats["markers"]["emoji"] == 2
    assert "invalid JSON" in stats["problems"][0]


def test_empty_dataset_reports_zero_rather_than_failing(tmp_path):
    path = tmp_path / "empty_sharegpt.jsonl"
    path.write_text("")
    stats = inspect_path(str(path))
    assert stats["samples"] == 0
    assert stats["target_responses"] == {"count": 0}
    assert "Samples" in render(stats)


def test_rendered_report_is_plain_text():
    rendered = render(inspect_parsed(TRANSCRIPT))
    assert "Phantasm inspection" in rendered
    assert "Usable training samples" in rendered
    assert "\x1b[" not in rendered


def test_unusable_input_is_rejected():
    with pytest.raises(ValueError, match="message list"):
        inspect_parsed({"messages": "not a list"})
