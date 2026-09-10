"""Formatting parsed conversation turns into ShareGPT fine-tuning datasets."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_iso(ts: str) -> datetime | None:
    """Parse ISO timestamp with timezone support."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def clean_text(text: Any) -> str:
    """Normalize whitespace and strip control characters in chat content."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Remove non-printable control chars except tabs/newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def group_consecutive_messages(
    messages: list[dict[str, Any]],
    max_gap_seconds: int = 300,
) -> list[dict[str, Any]]:
    """Merge rapid consecutive messages from the same author into unified turns."""
    gap_limit = max(int(max_gap_seconds), 1)
    grouped: list[dict[str, Any]] = []
    current_turn: dict[str, Any] | None = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        content = clean_text(msg.get("content", ""))
        if not content:
            continue

        role = msg.get("role")
        if role not in ("you", "them"):
            role = "you" if str(role).lower() in ("you", "user") else "them"

        raw_ts = msg.get("timestamp")
        ts = parse_iso(str(raw_ts)) if raw_ts else None

        if current_turn is None:
            current_turn = {
                "role": role,
                "content_lines": [content],
                "last_timestamp": ts,
            }
            continue

        last_ts = current_turn.get("last_timestamp")
        time_gap = (ts - last_ts).total_seconds() if (ts and last_ts) else 0

        if current_turn["role"] == role and time_gap <= gap_limit:
            current_turn["content_lines"].append(content)
            current_turn["last_timestamp"] = ts
        else:
            grouped.append(
                {
                    "role": current_turn["role"],
                    "content": "\n".join(current_turn["content_lines"]),
                }
            )
            current_turn = {
                "role": role,
                "content_lines": [content],
                "last_timestamp": ts,
            }

    if current_turn:
        grouped.append(
            {
                "role": current_turn["role"],
                "content": "\n".join(current_turn["content_lines"]),
            }
        )

    return grouped


def build_conversations(
    turns: list[dict[str, Any]],
    window_size: int = 6,
) -> list[list[dict[str, Any]]]:
    """Slice grouped turns into sliding conversation contexts ending with target persona."""
    win = max(int(window_size), 2)
    samples: list[list[dict[str, Any]]] = []

    for i in range(len(turns)):
        if turns[i].get("role") != "them":
            continue

        start_idx = max(0, i - win + 1)
        history = turns[start_idx : i + 1]

        while history and history[0].get("role") != "you":
            history = history[1:]

        if len(history) < 2:
            continue

        samples.append(history)

    return samples


def export_sharegpt(
    conversations: list[list[dict[str, Any]]], output_path: str, system_prompt: str = ""
) -> None:
    """Export conversation samples safely to ShareGPT fine-tuning JSONL format."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        for convo in conversations:
            dialogue = []
            if system_prompt:
                dialogue.append({"from": "system", "value": system_prompt})
            for turn in convo:
                from_role = "human" if turn.get("role") == "you" else "gpt"
                dialogue.append({"from": from_role, "value": turn.get("content", "")})
            f.write(json.dumps({"conversations": dialogue}, ensure_ascii=False) + "\n")

    temp_file.replace(out_file)
