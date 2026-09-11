"""Turn aggregation, sessionization, windowing, and leakage-safe splitting.

The dataset pipeline is four explicit stages:

.. code-block:: text

    messages -> turns (burst aggregation) -> sessions -> windows per split

Burst aggregation merges a speaker's consecutive messages that arrive within
``turn_gap_seconds`` of one another. Sessionization is a separate concept: a
silence longer than ``session_gap_seconds`` ends the conversation, so Monday's
exchange never becomes one training sample with Friday's.

Splits are assigned to whole sessions *before* any sliding window is generated,
which is what keeps the same source message out of two splits.
"""

import math
import random
from dataclasses import dataclass
from typing import Any

from phantasm.parser import LEGACY_ROLE, OTHER, SELF, TARGET
from phantasm.text import clean_text, parse_iso

#: Values accepted in the ``participant`` field, including the legacy ``role`` spellings.
PARTICIPANT_ALIASES = {
    SELF: SELF,
    TARGET: TARGET,
    OTHER: OTHER,
    "you": SELF,
    "them": TARGET,
    "user": SELF,
    "assistant": TARGET,
    "human": SELF,
    "gpt": TARGET,
}

SPLIT_NAMES = ("train", "val", "test")


def participant_of(message: dict[str, Any], index: int) -> str:
    """Read the participant label, accepting the legacy ``role`` field."""
    raw = message.get("participant") or message.get("role")
    if not isinstance(raw, str):
        raise ValueError(f"Message {index}: participant must be a string")
    participant = PARTICIPANT_ALIASES.get(raw.strip().casefold())
    if participant is None:
        raise ValueError(f"Message {index}: unknown participant {raw!r}")
    return participant


def aggregate_turns(
    messages: list[dict[str, Any]],
    turn_gap_seconds: int = 300,
    *,
    merge_unknown_timestamps: bool = False,
) -> list[dict[str, Any]]:
    """Merge each speaker's consecutive message bursts into single turns.

    Messages merge only when the same speaker sent them within
    ``turn_gap_seconds``. An unknown or unparseable timestamp never counts as a
    zero-second gap: such messages stay separate turns unless
    ``merge_unknown_timestamps`` is explicitly enabled, because an unknown
    temporal relationship is not evidence of adjacency.
    """
    if turn_gap_seconds < 1:
        raise ValueError("turn-gap-seconds must be at least 1")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    turns: list[dict[str, Any]] = []
    previous_ts = None
    last_known_ts = None
    previous_boundary = None
    seen_ids: set[str] = set()
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = clean_text(msg.get("content"))
        if not content:
            continue
        participant = participant_of(msg, index)
        source_id = str(msg["id"]) if msg.get("id") is not None else f"row:{index}"
        if source_id in seen_ids:
            raise ValueError(f"Duplicate message ID: {source_id}")
        seen_ids.add(source_id)
        ts = parse_iso(msg.get("timestamp"))
        if ts and last_known_ts and ts < last_known_ts:
            raise ValueError("Messages must be in chronological order")
        gap = (ts - previous_ts).total_seconds() if ts and previous_ts else None
        boundary = (msg.get("channel_id"), msg.get("session_id"))
        author = str(msg.get("author_id") or msg.get("username") or participant)
        mergeable_gap = gap is not None and gap <= turn_gap_seconds
        if gap is None and ts is None and previous_ts is None and merge_unknown_timestamps:
            mergeable_gap = True
        if (
            turns
            and boundary == previous_boundary
            and turns[-1]["participant"] == participant
            and turns[-1]["author_id"] == author
            and mergeable_gap
        ):
            turns[-1]["content"] += "\n" + content
            turns[-1]["source_ids"].append(source_id)
            turns[-1]["end_timestamp"] = ts.isoformat() if ts else None
        else:
            turns.append(
                {
                    "participant": participant,
                    "role": LEGACY_ROLE[participant],
                    "content": content,
                    "author_id": author,
                    "author_label": str(msg.get("display_name") or msg.get("username") or author),
                    "channel_id": msg.get("channel_id"),
                    "source_session": msg.get("session_id"),
                    "source_ids": [source_id],
                    "timestamp": ts.isoformat() if ts else None,
                    "end_timestamp": ts.isoformat() if ts else None,
                    "session_id": 0,
                }
            )
        previous_ts = ts
        last_known_ts = ts or last_known_ts
        previous_boundary = boundary
    return turns


def sessionize(
    turns: list[dict[str, Any]], session_gap_seconds: int = 1800
) -> list[list[dict[str, Any]]]:
    """Split a turn stream into conversation sessions and stamp ``session_id``.

    A session ends when the silence between turns exceeds
    ``session_gap_seconds``, when the channel or upstream boundary changes, or
    when the transcript crosses between known and unknown timestamps. Runs of
    turns with unknown timestamps keep their input order inside one session:
    input order is the only ordering signal available, and fabricating a gap
    would be worse than preserving it.
    """
    if session_gap_seconds < 1:
        raise ValueError("session-gap-seconds must be at least 1")
    sessions: list[list[dict[str, Any]]] = []
    previous = None
    index = -1
    for turn in turns:
        start = parse_iso(turn.get("timestamp"))
        boundary = (turn.get("channel_id"), turn.get("source_session"))
        if previous is None:
            new_session = True
        else:
            previous_end = parse_iso(previous.get("end_timestamp"))
            gap = (start - previous_end).total_seconds() if start and previous_end else None
            new_session = (
                boundary != (previous.get("channel_id"), previous.get("source_session"))
                or (start is None) != (previous_end is None)
                or (gap is not None and gap > session_gap_seconds)
            )
        if new_session:
            index += 1
            sessions.append([])
        turn["session_id"] = index
        sessions[index].append(turn)
        previous = turn
    return sessions


def build_windows(
    session_turns: list[dict[str, Any]], window_size: int = 6
) -> list[list[dict[str, Any]]]:
    """Build context windows inside one session, each ending on a target turn.

    A window starts on a context turn (``self`` or ``other``) and ends on the
    ``target`` turn it should teach the model to produce. Only ``target`` turns
    become training answers, so nobody else's messages can be learned as the
    persona.
    """
    if window_size < 2:
        raise ValueError("window must be at least 2")
    samples = []
    for position, turn in enumerate(session_turns):
        if turn["participant"] != TARGET:
            continue
        history = session_turns[max(0, position - window_size + 1) : position + 1]
        while history and history[0]["participant"] == TARGET:
            history = history[1:]
        if len(history) >= 2:
            samples.append(history)
    return samples


def validate_ratios(val_ratio: float, test_ratio: float) -> None:
    """Reject ratios that are negative, non-finite, or leave no training data."""
    for name, value in (("val", val_ratio), ("test", test_ratio)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name}-ratio must be a number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name}-ratio must be a finite value of at least 0")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val-ratio plus test-ratio must be below 1 to leave training data")


def split_sessions(
    sessions: list[list[dict[str, Any]]],
    *,
    val_ratio: float = 0.075,
    test_ratio: float = 0.075,
    strategy: str = "chronological",
    seed: int | None = None,
) -> dict[str, list[list[dict[str, Any]]]]:
    """Assign whole sessions to train/validation/test.

    ``chronological`` (the default) keeps the oldest sessions for training and
    holds out the most recent ones, which is the honest arrangement for temporal
    data. ``random`` requires an explicit ``seed`` so runs stay reproducible.
    """
    validate_ratios(val_ratio, test_ratio)
    if strategy not in ("chronological", "random"):
        raise ValueError("split strategy must be 'chronological' or 'random'")
    ordered = list(sessions)
    if strategy == "random":
        if seed is None:
            raise ValueError("random splitting requires an explicit --split-seed")
        random.Random(seed).shuffle(ordered)
    total = len(ordered)
    if not total:
        raise ValueError(
            "No usable conversation sessions; check participant selection, filters, and gaps"
        )
    val_count = max(1, int(total * val_ratio)) if val_ratio else 0
    test_count = max(1, int(total * test_ratio)) if test_ratio else 0
    if val_count + test_count >= total:
        raise ValueError(
            f"{total} usable session(s) cannot fill the requested splits; supply more data "
            "or lower --val-ratio/--test-ratio"
        )
    cut = total - val_count - test_count
    return {
        "train": ordered[:cut],
        "val": ordered[cut : cut + val_count],
        "test": ordered[cut + val_count :],
    }


@dataclass
class SplitSummary:
    """Per-split counts used for reporting and reproducibility manifests."""

    sessions: int
    samples: int
    messages: int
    duplicates_removed: int


def _signature(sample: list[dict[str, Any]]) -> tuple:
    return tuple((turn["participant"], turn["content"]) for turn in sample)


def windows_for_splits(
    splits: dict[str, list[list[dict[str, Any]]]],
    window_size: int = 6,
    *,
    drop_duplicate_samples: bool = True,
) -> tuple[dict[str, list[list[dict[str, Any]]]], dict[str, SplitSummary]]:
    """Generate sliding windows independently inside each split.

    Exact duplicate samples are removed within a split, and a training sample
    identical to a held-out sample is dropped from training so evaluation stays
    honest. Held-out copies are kept.
    """
    samples: dict[str, list[list[dict[str, Any]]]] = {}
    summaries: dict[str, SplitSummary] = {}
    # Held-out splits are processed first so training loses the shared copy.
    for name in ("test", "val", "train"):
        sessions = splits.get(name, [])
        built = [sample for session in sessions for sample in build_windows(session, window_size)]
        duplicates = 0
        if drop_duplicate_samples:
            taken = {_signature(existing) for other in samples.values() for existing in other}
            unique = []
            for sample in built:
                signature = _signature(sample)
                if signature in taken:
                    duplicates += 1
                    continue
                taken.add(signature)
                unique.append(sample)
            built = unique
        samples[name] = built
        summaries[name] = SplitSummary(
            sessions=len(sessions),
            samples=len(built),
            messages=len({i for sample in built for turn in sample for i in turn["source_ids"]}),
            duplicates_removed=duplicates,
        )
    return {name: samples[name] for name in SPLIT_NAMES}, {
        name: summaries[name] for name in SPLIT_NAMES
    }
