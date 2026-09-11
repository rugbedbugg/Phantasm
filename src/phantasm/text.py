"""Shared text normalization, timestamp parsing, and lightweight tokenization.

Every stage of the pipeline reads text through these helpers so that statistics,
audits, and evaluation measure exactly the strings that training sees.
"""

import re
from datetime import datetime, timezone
from typing import Any

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
CUSTOM_EMOJI_PATTERN = re.compile(r"<a?:\w+:\d+>")
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+")
PUNCTUATION_MARKS = ".,!?;:'\"()[]{}-_/\\*~`@#&%+=<>|^$…"

# Unicode ranges that cover pictographs, symbols, dingbats and regional indicators.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "\U00002600-\U000027bf"
    "\U00002b00-\U00002bff"
    "\U0000fe0f"
    "\U0001f1e6-\U0001f1ff"
    "]"
)


def parse_iso(ts: Any) -> datetime | None:
    """Normalize ISO timestamps to UTC; timestamps without offsets mean UTC."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


def clean_text(text: Any) -> str:
    """Strip control characters and collapse redundant whitespace without altering wording."""
    if text is None:
        return ""
    text = CONTROL_CHARACTERS.sub("", str(text))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def word_tokens(text: str) -> list[str]:
    """Lowercased word-like tokens; punctuation and separators are dropped."""
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text or "")]


def sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation and newlines, ignoring empty fragments."""
    return [part.strip() for part in SENTENCE_BOUNDARY.split(text or "") if part.strip()]


def count_emoji(text: str) -> int:
    """Count Unicode pictographs plus Discord-style custom emoji references."""
    return len(EMOJI_PATTERN.findall(text or "")) + len(CUSTOM_EMOJI_PATTERN.findall(text or ""))


def count_urls(text: str) -> int:
    return len(URL_PATTERN.findall(text or ""))


def count_code_blocks(text: str) -> int:
    return len(CODE_BLOCK_PATTERN.findall(text or ""))


def strip_code_blocks(text: str) -> str:
    return CODE_BLOCK_PATTERN.sub(" ", text or "")


def normalized_for_duplicates(text: str) -> str:
    """Casefolded, whitespace-collapsed form used to detect near-exact duplicates."""
    return " ".join((text or "").casefold().split())
