"""Configurable data-quality filtering for parsed messages.

Defaults are deliberately conservative: only content that is almost never a
genuine persona utterance is removed unless the user opts in. Every removal is
counted by reason so the caller can show exactly what was dropped and why.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from phantasm.text import (
    CODE_BLOCK_PATTERN,
    URL_PATTERN,
    clean_text,
    normalized_for_duplicates,
)

DELETED_PLACEHOLDER = re.compile(
    r"^(?:\[|<|\()?\s*(?:original\s+)?(?:message\s+)?"
    r"(?:deleted|removed|redacted)(?:\s+message)?"
    r"(?:\s+by\s+.+?)?\s*(?:\]|>|\))?$|^this message was deleted\.?$",
    re.IGNORECASE,
)
COMMAND_PATTERN = re.compile(r"^[!/$.;>~%&+\-]{1,2}[A-Za-z][\w-]*\b")

#: Removal reasons in report order.
REASONS = (
    "empty",
    "bot",
    "deleted_placeholder",
    "command",
    "url_only",
    "oversized",
    "code_heavy",
    "duplicate_message",
)


@dataclass
class FilterConfig:
    """Which quality filters run. ``True`` defaults are safe for any transcript."""

    drop_bots: bool = True
    drop_deleted_placeholders: bool = True
    drop_commands: bool = False
    drop_url_only: bool = False
    drop_duplicate_messages: bool = False
    max_message_chars: int | None = None
    max_code_blocks: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_message_chars", "max_code_blocks"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive when supplied")


@dataclass
class FilterReport:
    """Counts of kept and removed messages, grouped by reason."""

    kept: int = 0
    removed: Counter = field(default_factory=Counter)

    @property
    def total_removed(self) -> int:
        return sum(self.removed.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "kept": self.kept,
            "removed": self.total_removed,
            "by_reason": {
                reason: self.removed[reason] for reason in REASONS if self.removed[reason]
            },
        }

    def render(self) -> str:
        if not self.total_removed:
            return f"Kept all {self.kept} messages; no quality filter matched."
        lines = [f"Kept {self.kept} messages; removed {self.total_removed}:"]
        lines.extend(
            f"  {reason.replace('_', ' '):<22} {self.removed[reason]}"
            for reason in REASONS
            if self.removed[reason]
        )
        return "\n".join(lines)


def classify(message: dict[str, Any], config: FilterConfig, seen: set[str]) -> str | None:
    """Return the removal reason for a message, or ``None`` to keep it."""
    content = clean_text(message.get("content"))
    if not content:
        return "empty"
    if config.drop_bots and message.get("bot"):
        return "bot"
    if config.drop_deleted_placeholders and DELETED_PLACEHOLDER.match(content.strip()):
        return "deleted_placeholder"
    if config.drop_commands and COMMAND_PATTERN.match(content):
        return "command"
    if config.drop_url_only and not URL_PATTERN.sub(" ", content).strip():
        return "url_only"
    if config.max_message_chars is not None and len(content) > config.max_message_chars:
        return "oversized"
    if (
        config.max_code_blocks is not None
        and len(CODE_BLOCK_PATTERN.findall(content)) > config.max_code_blocks
    ):
        return "code_heavy"
    if config.drop_duplicate_messages:
        key = normalized_for_duplicates(content)
        if key in seen:
            return "duplicate_message"
        seen.add(key)
    return None


def filter_messages(
    messages: list[dict[str, Any]], config: FilterConfig | None = None
) -> tuple[list[dict[str, Any]], FilterReport]:
    """Apply the configured filters, preserving order and message metadata."""
    config = config or FilterConfig()
    report = FilterReport()
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            report.removed["empty"] += 1
            continue
        reason = classify(message, config, seen)
        if reason:
            report.removed[reason] += 1
            continue
        kept.append(message)
    report.kept = len(kept)
    return kept, report
