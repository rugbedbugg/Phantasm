"""Unit tests for Phantasm parsing, grouping, formatting, and edge-case handling."""

import json
from pathlib import Path

import pytest

from phantasm.formatter import (
    build_conversations,
    clean_text,
    export_sharegpt,
    group_consecutive_messages,
    parse_iso,
)
from phantasm.parser import parse_export
from phantasm.scraper import _mask_token


def test_clean_text_basic():
    raw = "hello   world\n\n\n\nhey"
    assert clean_text(raw) == "hello world\n\nhey"


def test_clean_text_sanitization():
    raw_with_nulls = "hello\x00\x07\x1bworld"
    assert clean_text(raw_with_nulls) == "helloworld"
    assert clean_text(None) == ""
    assert clean_text(12345) == "12345"


def test_clean_text_keeps_unicode_emoji_and_multiline_shape():
    raw = "первый  ✨\n\n\n\nsecond 😀 line"
    assert clean_text(raw) == "первый ✨\n\nsecond 😀 line"


def test_parse_iso_edge_cases():
    assert parse_iso("2026-09-10T10:00:00Z") is not None
    assert parse_iso("2026-09-10T10:00:00+00:00") is not None
    assert parse_iso("invalid-timestamp") is None
    assert parse_iso("") is None
    assert parse_iso(None) is None


def test_group_consecutive_messages():
    messages = [
        {"role": "you", "content": "hey", "timestamp": "2026-09-10T10:00:00Z"},
        {"role": "you", "content": "how are you?", "timestamp": "2026-09-10T10:01:00Z"},
        {"role": "them", "content": "good!", "timestamp": "2026-09-10T10:02:00Z"},
        {"role": "them", "content": "and you?", "timestamp": "2026-09-10T10:02:30Z"},
    ]
    grouped = group_consecutive_messages(messages, max_gap_seconds=300)
    assert len(grouped) == 2
    assert grouped[0]["role"] == "you"
    assert grouped[0]["content"] == "hey\nhow are you?"
    assert grouped[1]["role"] == "them"
    assert grouped[1]["content"] == "good!\nand you?"


def test_unknown_timestamps_no_longer_merge_into_one_turn():
    """Phantasm 0.2 corrects 0.1's behaviour of merging unknown timestamps.

    An unparseable timestamp says nothing about adjacency, so treating it as a
    zero-second gap invented a burst that may never have happened.
    """
    messages = [
        {"role": "you", "content": ""},
        None,
        {"role": "you", "content": "hello", "timestamp": "corrupt"},
        {"role": "you", "content": "world", "timestamp": None},
    ]
    grouped = group_consecutive_messages(messages, max_gap_seconds=300)
    assert [turn["content"] for turn in grouped] == ["hello", "world"]


def test_build_conversations():
    turns = [
        {"role": "you", "participant": "self", "content": "hello"},
        {"role": "them", "participant": "target", "content": "hi there"},
        {"role": "you", "participant": "self", "content": "what's up"},
        {"role": "them", "participant": "target", "content": "not much"},
    ]
    samples = build_conversations(turns, window_size=4)
    assert len(samples) == 2
    assert samples[0][-1]["content"] == "hi there"
    assert samples[1][-1]["content"] == "not much"


def test_build_conversations_empty_and_boundaries():
    assert build_conversations([]) == []
    assert build_conversations([{"role": "you", "participant": "self", "content": "hi"}]) == []
    assert build_conversations([{"role": "them", "participant": "target", "content": "hi"}]) == []


def test_export_sharegpt(tmp_path: Path):
    conversations = [
        [
            {"role": "you", "participant": "self", "content": "hello"},
            {"role": "them", "participant": "target", "content": "hi there"},
        ]
    ]
    sharegpt_path = tmp_path / "train_sharegpt.jsonl"
    export_sharegpt(conversations, str(sharegpt_path), system_prompt="Test persona")
    lines = sharegpt_path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert len(data["conversations"]) == 3
    assert data["conversations"][0]["from"] == "system"
    assert data["conversations"][1]["from"] == "human"
    assert data["conversations"][2]["from"] == "gpt"


def test_export_sharegpt_preserves_unicode_and_multiline_content(tmp_path: Path):
    conversations = [
        [
            {"participant": "self", "content": "ça va ?"},
            {"participant": "target", "content": "oui 😀\nvraiment"},
        ]
    ]
    path = tmp_path / "unicode.jsonl"
    export_sharegpt(conversations, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["conversations"][1]["value"] == "oui 😀\nvraiment"
    assert "😀" in path.read_text(encoding="utf-8")


def test_export_sharegpt_labels_other_speakers_only(tmp_path: Path):
    conversations = [
        [
            {"participant": "other", "content": "hey all", "author_label": "guest"},
            {"participant": "self", "content": "hi", "author_label": "me"},
            {"participant": "target", "content": "hello", "author_label": "persona"},
        ]
    ]
    path = tmp_path / "labelled.jsonl"
    export_sharegpt(conversations, str(path), speaker_labels="others")
    values = [turn["value"] for turn in json.loads(path.read_text())["conversations"]]
    assert values == ["guest: hey all", "hi", "hello"]


def test_parse_export_with_utf8_bom(tmp_path: Path):
    raw_json = json.dumps(
        {
            "channel": {"name": "test-channel"},
            "messages": [
                {
                    "id": "1",
                    "timestamp": "2026-09-10T10:00:00Z",
                    "author": {"username": "me"},
                    "content": "first message",
                },
                {
                    "id": "2",
                    "timestamp": "2026-09-10T10:01:00Z",
                    "author": {"username": "friend"},
                    "content": "second message",
                },
            ],
        }
    )
    bom_file = tmp_path / "export_with_bom.json"
    bom_file.write_bytes(b"\xef\xbb\xbf" + raw_json.encode("utf-8"))

    out_file = tmp_path / "parsed_bom.json"
    parsed = parse_export(str(bom_file), your_username="me", output_path=str(out_file))

    assert parsed["total_messages"] == 2
    assert parsed["messages"][0]["role"] == "you"
    assert parsed["messages"][0]["participant"] == "self"
    assert parsed["messages"][1]["participant"] == "target"
    assert "friend" in parsed["participants"]


def test_parse_export_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_export("/non/existent/file.json", your_username="me")


def test_parse_export_rejects_malformed_json(tmp_path: Path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError):
        parse_export(str(broken), "me", str(tmp_path / "out.json"))


def test_token_masking():
    token = "mfa.AbCdEf1234567890XYZ"
    log_msg = f"Failed connecting with token {token} on endpoint"
    masked = _mask_token(log_msg, token)
    assert token not in masked
    assert "[REDACTED_TOKEN]" in masked
