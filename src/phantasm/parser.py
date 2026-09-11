"""Parse chat exports into messages labelled with explicit participant identities.

Every message is assigned exactly one participant role:

``self``
    The person whose side of the conversation supplies context.
``target``
    The persona being reconstructed. Only these messages ever become training
    targets.
``other``
    Anybody else in the conversation. They are context at most; their messages
    can never be emitted as target responses.

Immutable author IDs are preferred. Usernames stay supported for exports that do
not carry IDs, and resolve case-insensitively when that is unambiguous.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phantasm.storage import write_json_atomic
from phantasm.text import clean_text

SELF = "self"
TARGET = "target"
OTHER = "other"

#: Legacy ``role`` values kept in parsed output for tools written against 0.1.
LEGACY_ROLE = {SELF: "you", TARGET: "them", OTHER: "other"}

#: How messages from participants that are neither ``self`` nor ``target`` are handled.
OTHERS_POLICIES = ("context", "drop", "error")


@dataclass
class Participant:
    """A distinct speaker discovered in an export."""

    key: str
    author_id: str | None = None
    names: list[str] = field(default_factory=list)
    messages: int = 0

    @property
    def label(self) -> str:
        return self.names[0] if self.names else (self.author_id or self.key)


def _identity_key(author_id: str, username: str) -> str:
    return author_id if author_id else f"name:{username.casefold()}"


def normalize_messages(raw: Any, *, channel_id: str | None = None) -> list[dict[str, Any]]:
    """Flatten raw export rows into records with cleaned text and author identity.

    Rows without text are skipped: they carry no persona signal. A row that has
    text but no author identity is an error, because guessing its speaker would
    risk attributing somebody else's words to the persona.
    """
    if not isinstance(raw, list):
        raise ValueError("Expected a message list or an object containing messages")
    records: list[dict[str, Any]] = []
    for index, msg in enumerate(raw):
        if not isinstance(msg, dict):
            continue
        content = clean_text(msg.get("content"))
        if not content:
            continue
        author = msg.get("author") if isinstance(msg.get("author"), dict) else {}
        username = str(
            author.get("username") or author.get("name") or msg.get("username") or ""
        ).strip()
        display_name = str(
            author.get("global_name")
            or author.get("display_name")
            or msg.get("display_name")
            or username
        ).strip()
        author_id = str(author.get("id") or msg.get("author_id") or "").strip()
        if not author_id and not username:
            raise ValueError(f"Message {index}: a nonempty message is missing its author identity")
        attachments = msg.get("attachments") or []
        if not isinstance(attachments, list):
            raise ValueError(f"Message {index}: attachments must be a list")
        reference = msg.get("message_reference") or msg.get("reference") or {}
        reply_to = msg.get("reply_to")
        if not reply_to and isinstance(reference, dict):
            reply_to = reference.get("message_id") or reference.get("messageId")
        records.append(
            {
                "id": msg.get("id"),
                "timestamp": msg.get("timestamp"),
                "author_id": author_id or None,
                "username": username,
                "display_name": display_name or username,
                "bot": bool(author.get("bot") or msg.get("bot")),
                "content": content,
                "channel_id": msg.get("channel_id") or channel_id,
                "attachments": [
                    a if isinstance(a, str) else a["url"]
                    for a in attachments
                    if isinstance(a, str) or (isinstance(a, dict) and a.get("url"))
                ],
                "reply_to": reply_to,
                "identity_key": _identity_key(author_id, username),
            }
        )
    return records


def collect_participants(records: list[dict[str, Any]]) -> dict[str, Participant]:
    """Group records into distinct speakers, keyed by author ID when available."""
    participants: dict[str, Participant] = {}
    for record in records:
        key = record["identity_key"]
        participant = participants.setdefault(
            key, Participant(key=key, author_id=record["author_id"])
        )
        participant.messages += 1
        name = record["username"] or record["display_name"]
        if name and name not in participant.names:
            participant.names.append(name)
    return participants


def resolve_identity(
    participants: dict[str, Participant], reference: str, role: str
) -> Participant:
    """Match a user-supplied ID or username against the export's speakers.

    Exact author IDs win, then exact usernames, then a case-insensitive username
    match. Anything that matches two different speakers is rejected rather than
    guessed, since picking the wrong one silently corrupts the persona.
    """
    wanted = (reference or "").strip()
    if not wanted:
        raise ValueError(f"--{role} requires a nonempty author ID or username")
    by_id = [p for p in participants.values() if p.author_id and p.author_id == wanted]
    if len(by_id) == 1:
        return by_id[0]
    exact = [p for p in participants.values() if wanted in p.names]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(
            f"--{role} {wanted!r} matches {len(exact)} different speakers; use author IDs"
        )
    folded = wanted.casefold()
    loose = [p for p in participants.values() if any(n.casefold() == folded for n in p.names)]
    if len(loose) == 1:
        return loose[0]
    if len(loose) > 1:
        raise ValueError(
            f"--{role} {wanted!r} matches {len(loose)} speakers whose names differ only by case; "
            "use author IDs"
        )
    known = ", ".join(sorted(p.label for p in participants.values())) or "none"
    raise ValueError(f"--{role} {wanted!r} is not present in the export; speakers: {known}")


def assign_participants(
    records: list[dict[str, Any]],
    *,
    self_ref: str | None = None,
    target_ref: str | None = None,
    others: str = "context",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label every record ``self``/``target``/``other`` and apply the others policy.

    Returns the labelled messages plus a summary describing the resolved
    identities and how many messages each participant contributed.
    """
    if others not in OTHERS_POLICIES:
        raise ValueError(f"others must be one of {', '.join(OTHERS_POLICIES)}")
    participants = collect_participants(records)
    if not participants:
        raise ValueError("The export contains no usable messages")
    target = resolve_identity(participants, target_ref, "target") if target_ref else None
    me = resolve_identity(participants, self_ref, "self") if self_ref else None
    if target is None:
        remaining = [p for p in participants.values() if me is None or p.key != me.key]
        if len(participants) != 2 or len(remaining) != 1:
            raise ValueError(
                "Supply --target: the persona cannot be inferred outside a two-person export"
            )
        target = remaining[0]
    if me is not None and me.key == target.key:
        raise ValueError("--self and --target must name different speakers")
    if me is None and len(participants) == 2:
        me = next(p for p in participants.values() if p.key != target.key)

    messages: list[dict[str, Any]] = []
    counts = {SELF: 0, TARGET: 0, OTHER: 0}
    dropped = 0
    boundary = 0
    for record in records:
        key = record["identity_key"]
        if key == target.key:
            participant = TARGET
        elif me is not None and key == me.key:
            participant = SELF
        else:
            participant = OTHER
        if participant == OTHER and others != "context":
            if others == "error":
                raise ValueError(
                    "Additional participants found; use --others context to keep them as "
                    "context or --others drop to exclude them"
                )
            # Dropping a speaker removes context, so the surrounding messages
            # must not be treated as a continuous exchange.
            dropped += 1
            boundary += 1
            continue
        counts[participant] += 1
        entry = {k: v for k, v in record.items() if k != "identity_key"}
        entry.update(participant=participant, role=LEGACY_ROLE[participant], session_id=boundary)
        messages.append(entry)
    if not counts[TARGET]:
        raise ValueError("The target persona has no messages after parsing")
    summary = {
        "self": None if me is None else {"author_id": me.author_id, "name": me.label},
        "target": {"author_id": target.author_id, "name": target.label},
        "others_policy": others,
        "counts": counts,
        "dropped_other_messages": dropped,
        "speakers": sorted(p.label for p in participants.values()),
    }
    return messages, summary


def parse_export(
    input_path: str,
    self_ref: str | None = None,
    output_path: str = "parsed.json",
    *,
    target_ref: str | None = None,
    others: str = "context",
    your_username: str | None = None,
    user_id: str | None = None,
    target_id: str | None = None,
    ignore_others: bool = False,
) -> dict[str, Any]:
    """Parse an export file into participant-labelled messages.

    ``self_ref``/``target_ref`` accept either an immutable author ID or a
    username. The deprecated ``your_username``/``user_id``/``target_id``
    arguments remain accepted so existing callers keep working.
    """
    self_ref = self_ref or user_id or your_username
    target_ref = target_ref or target_id
    if ignore_others:
        others = "drop"
    data = json.loads(Path(input_path).read_text(encoding="utf-8-sig"))
    raw = data.get("messages") if isinstance(data, dict) else data
    channel = data.get("channel", {}) if isinstance(data, dict) else {}
    records = normalize_messages(
        raw, channel_id=data.get("channel_id") if isinstance(data, dict) else None
    )
    if not records:
        raise ValueError("The export contains no usable messages")
    messages, summary = assign_participants(
        records, self_ref=self_ref, target_ref=target_ref, others=others
    )
    result = {
        "channel": channel.get("name", "DM") if isinstance(channel, dict) else str(channel),
        "participants": sorted({m["username"] for m in messages if m["username"]}),
        "total_messages": len(messages),
        **summary,
        "messages": messages,
    }
    write_json_atomic(Path(output_path), result)
    print(
        f"Parsed {len(messages)} messages "
        f"(self {summary['counts'][SELF]}, target {summary['counts'][TARGET]}, "
        f"other {summary['counts'][OTHER]}) -> {output_path}"
    )
    return result
