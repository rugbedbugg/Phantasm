"""ShareGPT formatting and the end-to-end dataset build.

The public entry point is :func:`format_dataset`, which runs the documented
pipeline in order: quality filtering, burst aggregation, sessionization, session
level splitting, and only then sliding-window generation inside each split.

The 0.1 helpers (:func:`group_consecutive_messages`, :func:`build_conversations`,
:func:`split_conversations`) remain available as thin compatibility wrappers.
"""

import json
from dataclasses import asdict, dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any

from phantasm.filtering import FilterConfig, filter_messages
from phantasm.parser import TARGET
from phantasm.sessions import (
    SPLIT_NAMES,
    aggregate_turns,
    build_windows,
    sessionize,
    split_sessions,
    windows_for_splits,
)
from phantasm.storage import write_json_atomic, write_text_atomic
from phantasm.text import clean_text, parse_iso  # noqa: F401  (0.1 import location)

SPEAKER_LABEL_MODES = ("none", "others", "all")


@dataclass
class FormatConfig:
    """Every knob the dataset build exposes, with conservative defaults."""

    window_size: int = 6
    turn_gap_seconds: int = 300
    session_gap_seconds: int = 1800
    val_ratio: float = 0.075
    test_ratio: float = 0.075
    strategy: str = "chronological"
    seed: int | None = None
    system_prompt: str = ""
    speaker_labels: str = "none"
    merge_unknown_timestamps: bool = False
    drop_duplicate_samples: bool = True
    filters: FilterConfig = field(default_factory=FilterConfig)

    def __post_init__(self) -> None:
        if self.turn_gap_seconds < 1 or self.session_gap_seconds < self.turn_gap_seconds:
            raise ValueError("Require 1 <= turn-gap-seconds <= session-gap-seconds")
        if self.speaker_labels not in SPEAKER_LABEL_MODES:
            raise ValueError(f"--speaker-labels must be one of {', '.join(SPEAKER_LABEL_MODES)}")


def render_turn(turn: dict[str, Any], speaker_labels: str = "none") -> str:
    """Render one turn's text, optionally prefixing the speaker of context turns."""
    content = turn["content"]
    if turn["participant"] == TARGET or speaker_labels == "none":
        return content
    if speaker_labels == "others" and turn["participant"] != "other":
        return content
    label = str(turn.get("author_label") or turn.get("author_id") or "").strip()
    return f"{label}: {content}" if label else content


def export_sharegpt(
    conversations: list, output_path: str, system_prompt: str = "", speaker_labels: str = "none"
) -> None:
    """Write ShareGPT JSONL; only ``target`` turns are emitted as ``gpt`` answers."""
    lines = []
    for convo in conversations:
        dialogue = [{"from": "system", "value": system_prompt}] if system_prompt else []
        for turn in convo:
            participant = turn.get("participant") or (
                TARGET if turn.get("role") == "them" else "self"
            )
            dialogue.append(
                {
                    "from": "gpt" if participant == TARGET else "human",
                    "value": render_turn({**turn, "participant": participant}, speaker_labels),
                }
            )
        lines.append(json.dumps({"conversations": dialogue}, ensure_ascii=False) + "\n")
    write_text_atomic(Path(output_path), "".join(lines))


def format_dataset(
    messages: list[dict[str, Any]], output_prefix: str, config: FormatConfig | None = None
) -> dict[str, Any]:
    """Build leakage-safe train/validation/test ShareGPT files and a manifest.

    Returns the manifest that is also written to ``<prefix>_split_manifest.json``.
    """
    config = config or FormatConfig()
    kept, filter_report = filter_messages(messages, config.filters)
    turns = aggregate_turns(
        kept,
        config.turn_gap_seconds,
        merge_unknown_timestamps=config.merge_unknown_timestamps,
    )
    sessions = sessionize(turns, config.session_gap_seconds)
    usable = [s for s in sessions if build_windows(s, config.window_size)]
    splits = split_sessions(
        usable,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        strategy=config.strategy,
        seed=config.seed,
    )
    samples, summaries = windows_for_splits(
        splits, config.window_size, drop_duplicate_samples=config.drop_duplicate_samples
    )
    outputs = {}
    for name in SPLIT_NAMES:
        path = f"{output_prefix}_{name}_sharegpt.jsonl"
        # Disabled splits are written empty so a stale file is never reused.
        export_sharegpt(samples[name], path, config.system_prompt, config.speaker_labels)
        outputs[name] = path
    manifest = {
        "config": {**asdict(config), "output_prefix": output_prefix},
        "filtering": filter_report.as_dict(),
        "turns": len(turns),
        "sessions": len(sessions),
        "usable_sessions": len(usable),
        "splits": {name: asdict(summaries[name]) for name in SPLIT_NAMES},
        "outputs": outputs,
    }
    write_json_atomic(Path(f"{output_prefix}_split_manifest.json"), manifest)
    return manifest


def render_format_report(manifest: dict[str, Any]) -> str:
    """Human-readable summary of a dataset build."""
    filtering = manifest["filtering"]
    lines = [f"Kept {filtering['kept']} messages; removed {filtering['removed']}"]
    lines.extend(
        f"  {reason.replace('_', ' '):<22} {count}"
        for reason, count in filtering["by_reason"].items()
    )
    lines.append(
        f"{manifest['turns']} turns in {manifest['sessions']} sessions "
        f"({manifest['usable_sessions']} usable)"
    )
    for name in SPLIT_NAMES:
        split = manifest["splits"][name]
        lines.append(
            f"  {name:<5} {split['samples']:>7} samples  {split['sessions']:>5} sessions  "
            f"{split['messages']:>7} source messages -> {manifest['outputs'][name]}"
        )
    removed = sum(manifest["splits"][name]["duplicates_removed"] for name in SPLIT_NAMES)
    if removed:
        lines.append(f"  removed {removed} duplicate sample(s)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Compatibility helpers for code written against Phantasm 0.1.
# --------------------------------------------------------------------------


def group_consecutive_messages(
    messages: list[dict[str, Any]], max_gap_seconds: int = 300, session_gap_seconds: int = 1800
) -> list[dict[str, Any]]:
    """Aggregate bursts and sessionize in one call, returning a flat turn list."""
    if max_gap_seconds < 1 or session_gap_seconds < max_gap_seconds:
        raise ValueError("Require 1 <= max-gap <= session-gap")
    turns = aggregate_turns(messages, max_gap_seconds)
    sessionize(turns, session_gap_seconds)
    return turns


def build_conversations(
    turns: list[dict[str, Any]], window_size: int = 6
) -> list[list[dict[str, Any]]]:
    """Build windows inside every session of a flat turn list."""
    return [
        sample
        for _, session in groupby(turns, key=lambda turn: turn.get("session_id", 0))
        for sample in build_windows(list(session), window_size)
    ]


def split_conversations(
    turns: list[dict[str, Any]], val_split: float, window_size: int = 6
) -> tuple[list, list]:
    """Two-way session split kept for 0.1 callers; prefer :func:`format_dataset`."""
    if not 0 <= val_split < 1:
        raise ValueError("val-split must be >= 0 and < 1")
    sessions = [
        list(group) for _, group in groupby(turns, key=lambda turn: turn.get("session_id", 0))
    ]
    usable = [session for session in sessions if build_windows(session, window_size)]
    if not usable:
        raise ValueError(
            "No valid conversations; check participant selection and session boundaries"
        )
    if val_split and len(usable) < 2:
        raise ValueError(
            "Validation needs at least two conversation sessions; supply more data "
            "or use --val-ratio 0"
        )
    splits = split_sessions(usable, val_ratio=val_split, test_ratio=0.0)
    samples, _ = windows_for_splits(splits, window_size, drop_duplicate_samples=False)
    return samples["train"], samples["val"]
