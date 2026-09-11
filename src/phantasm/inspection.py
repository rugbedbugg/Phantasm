"""Dataset and transcript statistics behind ``phantasm inspect``.

The report answers the questions that decide whether a persona run is worth
starting: how much of the data is actually the target speaking, how it is
distributed across sessions, and how many training samples it can support.
"""

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from phantasm.dataset import read_sharegpt, target_responses
from phantasm.parser import OTHER, SELF, TARGET
from phantasm.sessions import aggregate_turns, build_windows, participant_of, sessionize
from phantasm.text import (
    clean_text,
    count_code_blocks,
    count_emoji,
    count_urls,
    normalized_for_duplicates,
    parse_iso,
    word_tokens,
)


def length_stats(texts: list[str]) -> dict[str, Any]:
    """Word/character length summary for a set of responses."""
    words = [len(word_tokens(text)) for text in texts]
    characters = [len(text) for text in texts]
    if not words:
        return {"count": 0}
    ordered = sorted(words)
    return {
        "count": len(words),
        "mean_words": round(statistics.fmean(words), 2),
        "median_words": statistics.median(words),
        "p90_words": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max_words": ordered[-1],
        "mean_characters": round(statistics.fmean(characters), 2),
        "single_word_share": round(sum(1 for w in words if w <= 1) / len(words), 3),
    }


def content_markers(texts: list[str]) -> dict[str, int]:
    return {
        "urls": sum(count_urls(text) for text in texts),
        "code_blocks": sum(count_code_blocks(text) for text in texts),
        "emoji": sum(count_emoji(text) for text in texts),
    }


def duplicate_counts(texts: list[str]) -> dict[str, int]:
    """Exact and case/whitespace-insensitive duplicate counts."""
    exact = Counter(texts)
    near = Counter(normalized_for_duplicates(text) for text in texts)
    return {
        "exact_duplicates": sum(count - 1 for count in exact.values() if count > 1),
        "near_duplicates": sum(count - 1 for count in near.values() if count > 1),
        "distinct": len(near),
    }


def inspect_parsed(
    data: Any,
    *,
    window_size: int = 6,
    turn_gap_seconds: int = 300,
    session_gap_seconds: int = 1800,
) -> dict[str, Any]:
    """Summarize a parsed transcript produced by ``phantasm parse``."""
    messages = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(messages, list):
        raise ValueError("Expected a message list or an object containing messages")
    counts = Counter()
    invalid = 0
    empty = 0
    timestamped = 0
    attachments = 0
    stamps = []
    usable: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            invalid += 1
            continue
        attachments += len(message.get("attachments") or [])
        try:
            participant = participant_of(message, index)
        except ValueError:
            invalid += 1
            continue
        if not clean_text(message.get("content")):
            empty += 1
            continue
        counts[participant] += 1
        moment = parse_iso(message.get("timestamp"))
        if moment:
            timestamped += 1
            stamps.append(moment)
        usable.append(message)
    turns = aggregate_turns(usable, turn_gap_seconds)
    sessions = sessionize(turns, session_gap_seconds)
    windows = [sample for session in sessions for sample in build_windows(session, window_size)]
    target_texts = [turn["content"] for turn in turns if turn["participant"] == TARGET]
    total = counts[SELF] + counts[TARGET] + counts[OTHER]
    return {
        "kind": "parsed transcript",
        "messages": {
            "total": total,
            "self": counts[SELF],
            "target": counts[TARGET],
            "other": counts[OTHER],
            "empty": empty,
            "invalid": invalid,
        },
        "turns": {
            "total": len(turns),
            "target": sum(1 for turn in turns if turn["participant"] == TARGET),
            "sessions": len(sessions),
            "usable_sessions": sum(1 for s in sessions if build_windows(s, window_size)),
        },
        "target_responses": length_stats(target_texts),
        "markers": content_markers(target_texts),
        "attachments": attachments,
        "duplicates": duplicate_counts(target_texts),
        "timestamps": {
            "coverage": round(timestamped / total, 3) if total else 0.0,
            "first": stamps[0].isoformat() if stamps else None,
            "last": stamps[-1].isoformat() if stamps else None,
        },
        "usable_training_samples": len(windows),
    }


def inspect_sharegpt_rows(rows: list[dict], info: dict[str, Any]) -> dict[str, Any]:
    """Summarize an already-formatted ShareGPT dataset."""
    responses = target_responses(rows)
    contexts = [len(row["conversations"]) for row in rows]
    signatures = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    duplicate_samples = sum(count - 1 for count in Counter(signatures).values() if count > 1)
    return {
        "kind": "ShareGPT dataset",
        "samples": len(rows),
        "invalid_rows": info.get("invalid_rows", 0),
        "problems": info.get("problems", []),
        "duplicate_samples": duplicate_samples,
        "turns_per_sample": {
            "mean": round(statistics.fmean(contexts), 2) if contexts else 0,
            "max": max(contexts) if contexts else 0,
        },
        "target_responses": length_stats(responses),
        "markers": content_markers(responses),
        "duplicates": duplicate_counts(responses),
        "sha256": info.get("sha256"),
    }


def inspect_path(
    path: str,
    *,
    window_size: int = 6,
    turn_gap_seconds: int = 300,
    session_gap_seconds: int = 1800,
) -> dict[str, Any]:
    """Inspect a parsed transcript or a ShareGPT dataset, detected by content."""
    target = Path(path).expanduser()
    if target.suffix.lower() == ".jsonl":
        rows, info = read_sharegpt(str(target), strict=False)
        return inspect_sharegpt_rows(rows, info)
    data = json.loads(target.read_text(encoding="utf-8-sig"))
    return inspect_parsed(
        data,
        window_size=window_size,
        turn_gap_seconds=turn_gap_seconds,
        session_gap_seconds=session_gap_seconds,
    )


def _row(label: str, value: Any) -> str:
    return f"{label:<32}{value}"


def render(stats: dict[str, Any]) -> str:
    """Format statistics for a terminal."""
    lines = ["Phantasm inspection", "─" * 46, _row("Source", stats["kind"]), ""]
    if stats["kind"] == "parsed transcript":
        messages, turns = stats["messages"], stats["turns"]
        lines += [
            _row("Total messages", messages["total"]),
            _row("  target", messages["target"]),
            _row("  self", messages["self"]),
            _row("  other participants", messages["other"]),
            _row("  empty / invalid", f"{messages['empty']} / {messages['invalid']}"),
            "",
            _row("Turns", turns["total"]),
            _row("  target turns", turns["target"]),
            _row("Sessions", turns["sessions"]),
            _row("  usable sessions", turns["usable_sessions"]),
            _row("Timestamp coverage", f"{stats['timestamps']['coverage'] * 100:.1f}%"),
            _row(
                "First / last message",
                f"{stats['timestamps']['first']} .. {stats['timestamps']['last']}",
            ),
            _row("Attachments (all messages)", stats["attachments"]),
            _row("Usable training samples", stats["usable_training_samples"]),
        ]
    else:
        lines += [
            _row("Samples", stats["samples"]),
            _row("Invalid rows", stats["invalid_rows"]),
            _row("Duplicate samples", stats["duplicate_samples"]),
            _row("Mean turns per sample", stats["turns_per_sample"]["mean"]),
            _row("SHA-256", (stats.get("sha256") or "")[:16]),
        ]
    responses = stats["target_responses"]
    lines += ["", _row("Target responses", responses.get("count", 0))]
    if responses.get("count"):
        lines += [
            _row(
                "  mean / median words", f"{responses['mean_words']} / {responses['median_words']}"
            ),
            _row("  p90 / max words", f"{responses['p90_words']} / {responses['max_words']}"),
            _row("  mean characters", responses["mean_characters"]),
            _row("  one-word share", f"{responses['single_word_share'] * 100:.1f}%"),
            _row(
                "  URLs / code blocks / emoji",
                f"{stats['markers']['urls']} / {stats['markers']['code_blocks']} / "
                f"{stats['markers']['emoji']}",
            ),
            _row(
                "  exact / near duplicates",
                f"{stats['duplicates']['exact_duplicates']} / "
                f"{stats['duplicates']['near_duplicates']}",
            ),
        ]
    problems = stats.get("problems") or []
    if problems:
        lines += ["", "Problems:", *(f"  {problem}" for problem in problems)]
    return "\n".join(lines)


def add_arguments(parser: Any) -> None:
    """Register ``phantasm inspect`` options."""
    parser.add_argument("input", help="Parsed transcript JSON or ShareGPT JSONL")
    parser.add_argument("-w", "--window", type=int, default=6, help="Sliding window context size")
    parser.add_argument(
        "-g",
        "--turn-gap",
        "--max-gap",
        dest="turn_gap",
        type=int,
        default=300,
        help="Seconds within which one speaker's messages form a single turn",
    )
    parser.add_argument(
        "--session-gap", type=int, default=1800, help="Inactivity seconds separating sessions"
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def command(args: Any) -> None:
    """Print transcript or dataset statistics."""
    stats = inspect_path(
        args.input,
        window_size=args.window,
        turn_gap_seconds=args.turn_gap,
        session_gap_seconds=args.session_gap,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False) if args.json else render(stats))
