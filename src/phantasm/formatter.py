"""Formatting parsed conversation turns into disjoint ShareGPT datasets."""

import json
import re
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Any

from phantasm.storage import write_text_atomic


def parse_iso(ts: str) -> datetime | None:
    """Normalize ISO timestamps to UTC; timestamps without offsets mean UTC."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (
            parsed.replace(tzinfo=timezone.utc)
            if parsed.tzinfo is None
            else parsed.astimezone(timezone.utc)
        )
    except ValueError:
        return None


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(text))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def group_consecutive_messages(
    messages: list[dict[str, Any]], max_gap_seconds: int = 300, session_gap_seconds: int = 1800
) -> list[dict[str, Any]]:
    """Keep source IDs and session boundaries while merging same-author bursts.

    Unknown timestamps can merge only with other unknown timestamps. Crossing
    between known and unknown time starts a session. Out-of-order data is rejected.
    """
    if max_gap_seconds < 1 or session_gap_seconds < max_gap_seconds:
        raise ValueError("Require 1 <= max-gap <= session-gap")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    grouped: list[dict[str, Any]] = []
    previous_ts = None
    last_known_ts = None
    previous_boundary = None
    session = 0
    seen_ids: set[str] = set()
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = clean_text(msg.get("content"))
        if not content:
            continue
        if not isinstance(msg.get("role"), str):
            raise ValueError(f"Message {index}: role must be a string")
        role = {"user": "you", "assistant": "them"}.get(msg.get("role"), msg.get("role"))
        if role not in ("you", "them"):
            raise ValueError(f"Message {index}: unknown role {role!r}")
        source_id = str(msg["id"]) if msg.get("id") is not None else f"row:{index}"
        if source_id in seen_ids:
            raise ValueError(f"Duplicate message ID: {source_id}")
        seen_ids.add(source_id)
        ts = parse_iso(msg.get("timestamp"))
        if ts and last_known_ts and ts < last_known_ts:
            raise ValueError("Messages must be in chronological order")
        gap = (ts - previous_ts).total_seconds() if ts and previous_ts else None
        boundary = (msg.get("channel_id"), msg.get("session_id"))
        if grouped and (
            boundary != previous_boundary
            or (ts is None) != (previous_ts is None)
            or (gap is not None and gap > session_gap_seconds)
        ):
            session += 1
        author = str(msg.get("author_id") or msg.get("username") or role)
        merge = (
            grouped
            and grouped[-1]["session_id"] == session
            and grouped[-1]["role"] == role
            and grouped[-1]["author_id"] == author
            and (gap is None or gap <= max_gap_seconds)
        )
        if merge:
            grouped[-1]["content"] += "\n" + content
            grouped[-1]["source_ids"].append(source_id)
            grouped[-1]["end_timestamp"] = ts.isoformat() if ts else None
        else:
            grouped.append(
                {
                    "role": role,
                    "content": content,
                    "author_id": author,
                    "session_id": session,
                    "source_ids": [source_id],
                    "timestamp": ts.isoformat() if ts else None,
                    "end_timestamp": ts.isoformat() if ts else None,
                }
            )
        previous_ts = ts
        last_known_ts = ts or last_known_ts
        previous_boundary = boundary
    return grouped


def build_conversations(
    turns: list[dict[str, Any]], window_size: int = 6
) -> list[list[dict[str, Any]]]:
    """Build windows inside sessions, starting with the user and ending with the target."""
    if window_size < 2:
        raise ValueError("window must be at least 2")
    samples = []
    for _, session in groupby(turns, key=lambda turn: turn.get("session_id", 0)):
        session_turns = list(session)
        for i, turn in enumerate(session_turns):
            if turn.get("role") != "them":
                continue
            history = session_turns[max(0, i - window_size + 1) : i + 1]
            while history and history[0].get("role") != "you":
                history = history[1:]
            if len(history) >= 2:
                samples.append(history)
    return samples


def split_conversations(
    turns: list[dict[str, Any]], val_split: float, window_size: int = 6
) -> tuple[list, list]:
    """Split whole usable sessions chronologically, before creating any windows."""
    if not 0 <= val_split < 1:
        raise ValueError("val-split must be >= 0 and < 1")
    sessions = [list(group) for _, group in groupby(turns, key=lambda t: t.get("session_id", 0))]
    sessions = [session for session in sessions if build_conversations(session, window_size)]
    if not sessions:
        raise ValueError(
            "No valid conversations; check participant selection and session boundaries"
        )
    if val_split and len(sessions) < 2:
        raise ValueError(
            "Validation needs at least two conversation sessions; supply more data or use --val-split 0"
        )
    val_count = max(1, int(len(sessions) * val_split)) if val_split else 0
    cut = len(sessions) - val_count
    train = [
        sample for session in sessions[:cut] for sample in build_conversations(session, window_size)
    ]
    val = [
        sample for session in sessions[cut:] for sample in build_conversations(session, window_size)
    ]
    return train, val


def export_sharegpt(conversations: list, output_path: str, system_prompt: str = "") -> None:
    lines = []
    for convo in conversations:
        dialogue = [{"from": "system", "value": system_prompt}] if system_prompt else []
        dialogue.extend(
            {"from": "human" if t["role"] == "you" else "gpt", "value": t["content"]} for t in convo
        )
        lines.append(json.dumps({"conversations": dialogue}, ensure_ascii=False) + "\n")
    write_text_atomic(Path(output_path), "".join(lines))
