"""Participant identity resolution and group-chat contamination prevention."""

import json
from pathlib import Path

import pytest

from phantasm.formatter import build_conversations, group_consecutive_messages
from phantasm.parser import assign_participants, normalize_messages, parse_export


def message(identifier, author_id, username, text="text", minute=0, **extra):
    return {
        "id": str(identifier),
        "author": {"id": author_id, "username": username},
        "timestamp": f"2026-01-01T10:{minute:02d}:00Z",
        "content": text,
        **extra,
    }


def write(tmp_path: Path, messages, name="raw.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(messages), encoding="utf-8")
    return str(path)


def parse(tmp_path: Path, messages, **kwargs):
    return parse_export(
        write(tmp_path, messages), output_path=str(tmp_path / "parsed.json"), **kwargs
    )


def test_two_person_dm_assigns_self_and_target(tmp_path):
    parsed = parse(
        tmp_path,
        [message(1, "100", "me"), message(2, "200", "friend", minute=1)],
        self_ref="100",
        target_ref="200",
    )
    assert [m["participant"] for m in parsed["messages"]] == ["self", "target"]
    assert [m["role"] for m in parsed["messages"]] == ["you", "them"]
    assert parsed["counts"] == {"self": 1, "target": 1, "other": 0}


def test_third_participant_never_becomes_the_persona(tmp_path):
    """0.1 folded every non-self speaker into the persona; 0.2 keeps them separate."""
    messages = [
        message(1, "100", "me", "hey"),
        message(2, "300", "stranger", "unrelated opinion", minute=1),
        message(3, "200", "friend", "actual reply", minute=2),
    ]
    parsed = parse(tmp_path, messages, self_ref="100", target_ref="200")
    by_participant = {m["participant"]: m["content"] for m in parsed["messages"]}
    assert by_participant["other"] == "unrelated opinion"
    assert by_participant["target"] == "actual reply"
    assert parsed["counts"]["other"] == 1
    # The stranger's words must never be emitted as a target response.
    samples = build_conversations(group_consecutive_messages(parsed["messages"]))
    answers = {turn["content"] for sample in samples for turn in sample if turn["role"] == "them"}
    assert answers == {"actual reply"}


def test_others_drop_excludes_and_breaks_context(tmp_path):
    messages = [
        message(1, "100", "me"),
        message(2, "300", "stranger", minute=1),
        message(3, "200", "friend", minute=2),
    ]
    parsed = parse(tmp_path, messages, self_ref="100", target_ref="200", others="drop")
    assert [m["author_id"] for m in parsed["messages"]] == ["100", "200"]
    assert parsed["dropped_other_messages"] == 1
    # Removing a speaker must not splice the surrounding turns into one exchange.
    assert build_conversations(group_consecutive_messages(parsed["messages"])) == []


def test_others_error_refuses_group_transcripts(tmp_path):
    messages = [
        message(1, "100", "me"),
        message(2, "300", "x", minute=1),
        message(3, "200", "f", minute=2),
    ]
    with pytest.raises(ValueError, match="Additional participants"):
        parse(tmp_path, messages, self_ref="100", target_ref="200", others="error")


def test_ignore_others_flag_remains_supported(tmp_path):
    messages = [
        message(1, "100", "me"),
        message(2, "300", "x", minute=1),
        message(3, "200", "f", minute=2),
    ]
    parsed = parse(tmp_path, messages, self_ref="100", target_ref="200", ignore_others=True)
    assert parsed["others_policy"] == "drop"
    assert parsed["counts"]["other"] == 0


def test_usernames_resolve_and_survive_renames(tmp_path):
    messages = [
        message(1, "100", "old-name"),
        message(2, "100", "new-name", minute=1),
        message(3, "200", "friend", minute=2),
    ]
    parsed = parse(tmp_path, messages, self_ref="new-name", target_ref="friend")
    assert [m["participant"] for m in parsed["messages"]] == ["self", "self", "target"]


def test_case_only_username_collision_is_rejected(tmp_path):
    messages = [
        message(1, "100", "alice"),
        message(2, "200", "ALICE", minute=1),
        message(3, "300", "bob", minute=2),
    ]
    with pytest.raises(ValueError, match="differ only by case"):
        parse(tmp_path, messages, self_ref="bob", target_ref="Alice")


def test_exact_case_match_wins_over_case_insensitive_match(tmp_path):
    messages = [
        message(1, "100", "Alice"),
        message(2, "200", "alice", minute=1),
        message(3, "300", "bob", minute=2),
    ]
    parsed = parse(tmp_path, messages, self_ref="bob", target_ref="Alice")
    assert parsed["target"] == {"author_id": "100", "name": "Alice"}
    assert [m["participant"] for m in parsed["messages"]] == ["target", "other", "self"]


def test_missing_identity_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing its author identity"):
        normalize_messages([{"id": "1", "content": "who said this?"}])


def test_messages_without_identity_but_also_without_text_are_skipped():
    assert normalize_messages([{"id": "1", "content": "   "}]) == []


def test_target_identity_not_present_is_reported(tmp_path):
    messages = [message(1, "100", "me"), message(2, "200", "friend", minute=1)]
    with pytest.raises(ValueError, match="--target 'ghost' is not present"):
        parse(tmp_path, messages, self_ref="100", target_ref="ghost")


def test_self_identity_not_present_is_reported(tmp_path):
    messages = [message(1, "100", "me"), message(2, "200", "friend", minute=1)]
    with pytest.raises(ValueError, match="--self 'ghost' is not present"):
        parse(tmp_path, messages, self_ref="ghost", target_ref="200")


def test_self_may_be_omitted_in_a_group_transcript(tmp_path):
    messages = [
        message(1, "300", "guest"),
        message(2, "400", "other-guest", minute=1),
        message(3, "200", "friend", minute=2),
    ]
    parsed = parse(tmp_path, messages, target_ref="200")
    assert parsed["self"] is None
    assert parsed["counts"] == {"self": 0, "target": 1, "other": 2}


def test_target_must_be_supplied_for_group_transcripts(tmp_path):
    messages = [
        message(1, "100", "me"),
        message(2, "300", "x", minute=1),
        message(3, "200", "f", minute=2),
    ]
    with pytest.raises(ValueError, match="cannot be inferred"):
        parse(tmp_path, messages, self_ref="100")


def test_self_and_target_must_differ(tmp_path):
    messages = [message(1, "100", "me"), message(2, "200", "friend", minute=1)]
    with pytest.raises(ValueError, match="must name different speakers"):
        parse(tmp_path, messages, self_ref="100", target_ref="me")


def test_target_without_messages_after_filtering_is_reported():
    records = normalize_messages([{"id": "1", "author": {"id": "1"}, "content": "hi"}])
    with pytest.raises(ValueError, match="not present"):
        assign_participants(records, self_ref="1", target_ref="2")


def test_parsed_output_keeps_author_metadata(tmp_path):
    messages = [
        message(1, "100", "me", attachments=[{"url": "https://example.invalid/a.png"}]),
        message(2, "200", "friend", minute=1, reply_to="1"),
    ]
    parsed = parse(tmp_path, messages, self_ref="100", target_ref="200")
    first, second = parsed["messages"]
    assert first["author_id"] == "100" and first["username"] == "me"
    assert first["attachments"] == ["https://example.invalid/a.png"]
    assert second["reply_to"] == "1"
