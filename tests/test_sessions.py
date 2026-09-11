"""Burst aggregation, sessionization, window building and leakage-safe splitting."""

from datetime import datetime, timedelta, timezone

import pytest

from phantasm.sessions import (
    aggregate_turns,
    build_windows,
    sessionize,
    split_sessions,
    validate_ratios,
    windows_for_splits,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def message(index, participant, seconds, author=None, **extra):
    return {
        "id": str(index),
        "participant": participant,
        "author_id": author or participant,
        "content": f"text-{index}",
        "timestamp": (START + timedelta(seconds=seconds)).isoformat(),
        **extra,
    }


def alternating(count, *, session_every=4, session_gap=7200):
    """Messages forming ``count // session_every`` well-separated sessions."""
    messages = []
    for index in range(count):
        block, position = divmod(index, session_every)
        messages.append(
            message(
                index,
                "self" if position % 2 == 0 else "target",
                block * session_gap + position * 60,
            )
        )
    return messages


# --------------------------------------------------------------------------
# Burst aggregation
# --------------------------------------------------------------------------


def test_same_speaker_burst_becomes_one_turn():
    turns = aggregate_turns(
        [message(1, "self", 0), message(2, "self", 30), message(3, "target", 60)],
        turn_gap_seconds=300,
    )
    assert [turn["content"] for turn in turns] == ["text-1\ntext-2", "text-3"]
    assert turns[0]["source_ids"] == ["1", "2"]


def test_turn_gap_boundary_is_inclusive():
    exactly_at_gap = aggregate_turns([message(1, "self", 0), message(2, "self", 300)], 300)
    just_past_gap = aggregate_turns([message(1, "self", 0), message(2, "self", 301)], 300)
    assert len(exactly_at_gap) == 1
    assert len(just_past_gap) == 2


def test_different_speakers_never_merge():
    turns = aggregate_turns([message(1, "self", 0), message(2, "other", 1, author="guest")], 300)
    assert len(turns) == 2


def test_unknown_timestamps_stay_separate_unless_requested():
    messages = [
        message(1, "self", 0, timestamp=None),
        message(2, "self", 0, timestamp="garbage"),
    ]
    assert len(aggregate_turns(messages, 300)) == 2
    merged = aggregate_turns(messages, 300, merge_unknown_timestamps=True)
    assert len(merged) == 1
    assert merged[0]["timestamp"] is None


def test_out_of_order_messages_are_rejected():
    with pytest.raises(ValueError, match="chronological"):
        aggregate_turns([message(1, "self", 300), message(2, "self", 0)], 300)


def test_duplicate_message_ids_are_rejected():
    duplicate = message(1, "self", 0)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_turns([duplicate, duplicate], 300)


def test_unknown_participant_is_rejected():
    with pytest.raises(ValueError, match="unknown participant"):
        aggregate_turns([{"id": "1", "participant": "moderator", "content": "hi"}], 300)


def test_turn_gap_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        aggregate_turns([], 0)


# --------------------------------------------------------------------------
# Sessionization
# --------------------------------------------------------------------------


def test_long_silence_starts_a_new_session():
    turns = aggregate_turns([message(1, "self", 0), message(2, "target", 4 * 86400)], 300)
    sessions = sessionize(turns, 1800)
    assert len(sessions) == 2
    assert [turn["session_id"] for turn in turns] == [0, 1]


def test_session_gap_boundary_is_inclusive():
    turns = aggregate_turns([message(1, "self", 0), message(2, "target", 1800)], 300)
    assert len(sessionize(turns, 1800)) == 1
    turns = aggregate_turns([message(1, "self", 0), message(2, "target", 1801)], 300)
    assert len(sessionize(turns, 1800)) == 2


def test_crossing_known_and_unknown_time_starts_a_session():
    turns = aggregate_turns([message(1, "self", 0), message(2, "target", 0, timestamp=None)], 300)
    assert len(sessionize(turns, 1800)) == 2


def test_consecutive_unknown_timestamps_stay_in_one_session():
    messages = [
        message(1, "self", 0, timestamp=None),
        message(2, "target", 0, timestamp=None),
    ]
    turns = aggregate_turns(messages, 300)
    assert len(sessionize(turns, 1800)) == 1
    assert len(build_windows(sessionize(turns, 1800)[0])) == 1


def test_changed_upstream_boundary_starts_a_session():
    turns = aggregate_turns(
        [message(1, "self", 0, session_id=0), message(2, "target", 60, session_id=1)], 300
    )
    assert len(sessionize(turns, 1800)) == 2


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------


def test_windows_end_on_target_and_start_on_context():
    turns = aggregate_turns(alternating(4), 300)
    session = sessionize(turns, 1800)[0]
    windows = build_windows(session, window_size=6)
    assert [[turn["participant"] for turn in window] for window in windows] == [
        ["self", "target"],
        ["self", "target", "self", "target"],
    ]


def test_window_size_limits_context():
    turns = aggregate_turns(alternating(8, session_every=8), 300)
    session = sessionize(turns, 1800)[0]
    assert max(len(window) for window in build_windows(session, window_size=2)) == 2


def test_window_must_be_at_least_two_turns():
    with pytest.raises(ValueError, match="at least 2"):
        build_windows([], window_size=1)


def test_other_participants_can_open_a_window_but_never_close_one():
    messages = [
        message(1, "other", 0, author="guest"),
        message(2, "target", 60),
    ]
    session = sessionize(aggregate_turns(messages, 300), 1800)[0]
    windows = build_windows(session)
    assert [turn["participant"] for turn in windows[0]] == ["other", "target"]


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def sessions_for(count):
    turns = aggregate_turns(alternating(count * 4), 300)
    return sessionize(turns, 1800)


def test_chronological_split_holds_out_the_newest_sessions():
    sessions = sessions_for(10)
    splits = split_sessions(sessions, val_ratio=0.1, test_ratio=0.1)
    assert [len(splits[name]) for name in ("train", "val", "test")] == [8, 1, 1]
    assert splits["test"][0][0]["timestamp"] > splits["train"][-1][0]["timestamp"]


def test_splits_never_share_sessions_or_message_ids():
    splits = split_sessions(sessions_for(12), val_ratio=0.25, test_ratio=0.25)
    samples, summaries = windows_for_splits(splits, window_size=6)

    def identifiers(name):
        return {i for sample in samples[name] for turn in sample for i in turn["source_ids"]}

    def session_ids(name):
        return {turn["session_id"] for sample in samples[name] for turn in sample}

    train, val, test = (identifiers(name) for name in ("train", "val", "test"))
    assert train and val and test
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert session_ids("train").isdisjoint(session_ids("val"))
    assert session_ids("train").isdisjoint(session_ids("test"))
    assert session_ids("val").isdisjoint(session_ids("test"))
    for sample in samples["train"] + samples["val"] + samples["test"]:
        assert len({turn["session_id"] for turn in sample}) == 1
    assert summaries["train"].sessions == len(splits["train"])


def test_disabled_splits_produce_empty_files_worth_of_samples():
    splits = split_sessions(sessions_for(4), val_ratio=0, test_ratio=0)
    samples, _ = windows_for_splits(splits, 6)
    assert samples["val"] == [] and samples["test"] == []
    assert len(splits["train"]) == 4


def test_random_split_requires_a_seed_and_is_reproducible():
    sessions = sessions_for(10)
    with pytest.raises(ValueError, match="explicit --split-seed"):
        split_sessions(sessions, val_ratio=0.1, test_ratio=0.1, strategy="random")
    first = split_sessions(sessions, val_ratio=0.2, test_ratio=0.2, strategy="random", seed=7)
    second = split_sessions(sessions, val_ratio=0.2, test_ratio=0.2, strategy="random", seed=7)
    assert [s[0]["source_ids"] for s in first["test"]] == [
        s[0]["source_ids"] for s in second["test"]
    ]


@pytest.mark.parametrize(
    "val,test",
    [(-0.1, 0.1), (0.6, 0.5), (float("nan"), 0.1), (0.1, float("inf")), (1.0, 0.0)],
)
def test_invalid_ratios_are_rejected(val, test):
    with pytest.raises(ValueError):
        validate_ratios(val, test)


def test_empty_input_is_reported_clearly():
    with pytest.raises(ValueError, match="No usable conversation sessions"):
        split_sessions([], val_ratio=0.1, test_ratio=0.1)


def test_tiny_dataset_cannot_fill_every_split():
    with pytest.raises(ValueError, match="cannot fill the requested splits"):
        split_sessions(sessions_for(2), val_ratio=0.1, test_ratio=0.1)


def test_duplicate_samples_are_removed_from_training_not_from_held_out():
    repeated = []
    for block in range(4):
        base = block * 7200
        repeated += [
            message(f"{block}a", "self", base, author="self"),
            message(f"{block}b", "target", base + 60, author="target"),
        ]
    for entry in repeated:
        entry["content"] = "same"
    turns = aggregate_turns(repeated, 300)
    splits = split_sessions(sessionize(turns, 1800), val_ratio=0.25, test_ratio=0.25)
    samples, summaries = windows_for_splits(splits, 6)
    assert summaries["test"].samples == 1
    assert summaries["train"].samples == 0
    assert summaries["train"].duplicates_removed == 2
    keep, _ = windows_for_splits(splits, 6, drop_duplicate_samples=False)
    assert len(keep["train"]) == 2
