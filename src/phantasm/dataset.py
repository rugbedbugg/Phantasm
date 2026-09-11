"""Reading and validating ShareGPT JSONL datasets.

Training, auditing, inspection, and evaluation all read datasets through this
module so a file that trains is exactly the file that is audited and scored.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

ROLES = ("system", "human", "gpt")


def describe_problem(row: Any) -> str | None:
    """Return why a row is not a usable ShareGPT sample, or ``None`` if it is."""
    if not isinstance(row, dict):
        return "expected a JSON object"
    convo = row.get("conversations")
    if not isinstance(convo, list) or len(convo) < 2:
        return "expected a conversation"
    for turn in convo:
        if (
            not isinstance(turn, dict)
            or turn.get("from") not in ROLES
            or not isinstance(turn.get("value"), str)
            or not turn["value"].strip()
        ):
            return "invalid role/content"
    dialogue = convo[1:] if convo[0]["from"] == "system" else convo
    if (
        not dialogue
        or dialogue[0]["from"] != "human"
        or dialogue[-1]["from"] != "gpt"
        or any(turn["from"] == "system" for turn in dialogue)
    ):
        return "require user-to-assistant dialogue"
    return None


def read_sharegpt(path: str, *, strict: bool = True) -> tuple[list[dict], dict]:
    """Load a ShareGPT JSONL file.

    With ``strict`` the first malformed row raises :class:`ValueError` naming the
    line. Otherwise malformed rows are skipped and counted in the returned info,
    which is what auditing and inspection need for files of unknown quality.
    """
    target = Path(path).expanduser().resolve()
    raw = target.read_bytes()
    rows: list[dict] = []
    problems: list[str] = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            message = f"{target.name}:{line_number}: invalid JSON ({exc.msg})"
            if strict:
                raise ValueError(message) from None
            problems.append(message)
            continue
        problem = describe_problem(row)
        if problem:
            message = f"{target.name}:{line_number}: {problem}"
            if strict:
                raise ValueError(message)
            problems.append(message)
            continue
        rows.append({"conversations": row["conversations"]})
    if strict and not rows:
        raise ValueError(f"Dataset is empty: {target}")
    return rows, {
        "path": str(target),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "samples": len(rows),
        "invalid_rows": len(problems),
        "problems": problems[:20],
    }


def split_sample(row: dict) -> tuple[list[dict], str]:
    """Return the context turns and the final target response of a sample."""
    convo = row["conversations"]
    return convo[:-1], convo[-1]["value"]


def target_responses(rows: list[dict]) -> list[str]:
    """Every assistant/target message in a dataset, in order."""
    return [turn["value"] for row in rows for turn in row["conversations"] if turn["from"] == "gpt"]
