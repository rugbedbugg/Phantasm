"""Parse Discord exports with explicit participant identities."""

import json
from pathlib import Path
from typing import Any

from phantasm.formatter import clean_text
from phantasm.storage import write_json_atomic


def parse_export(
    input_path: str,
    your_username: str | None = None,
    output_path: str = "parsed.json",
    *,
    user_id: str | None = None,
    target_id: str | None = None,
    ignore_others: bool = False,
) -> dict[str, Any]:
    """Prefer immutable author IDs; legacy usernames work only for two-person exports."""
    data = json.loads(Path(input_path).read_text(encoding="utf-8-sig"))
    messages = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(messages, list):
        raise ValueError("Expected a message list or an object containing messages")
    if bool(user_id) != bool(target_id):
        raise ValueError("Supply both --user-id and --target-id")
    if user_id and (user_id == target_id or your_username):
        raise ValueError("Use distinct user/target IDs, without a username")
    if not user_id and not (your_username and your_username.strip()):
        raise ValueError("Supply --user-id and --target-id (or a legacy username)")
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = clean_text(msg.get("content"))
        if not content:
            continue
        author = msg.get("author") if isinstance(msg.get("author"), dict) else {}
        username = str(
            author.get("username") or author.get("name") or msg.get("username") or ""
        ).strip()
        author_id = str(author.get("id") or msg.get("author_id") or "")
        if not author_id and not username:
            raise ValueError("A nonempty message is missing its author identity")
        normalized.append((msg, username, author_id, content))
    if not user_id:
        identities = {(aid or name.lower()) for _, name, aid, _ in normalized}
        own = {
            aid or name.lower()
            for _, name, aid, _ in normalized
            if name.lower() == your_username.strip().lower()
        }
        if len(identities) != 2 or len(own) != 1:
            raise ValueError(
                "Username mode requires exactly two participants and an unambiguous user; use explicit IDs"
            )
        legacy_user = next(iter(own))
    else:
        present = {aid for _, _, aid, _ in normalized}
        if not {user_id, target_id} <= present:
            raise ValueError("Both selected participant IDs must have text messages in the export")
    structured = []
    boundary = 0
    for msg, username, author_id, content in normalized:
        if user_id:
            if author_id not in (user_id, target_id):
                if not ignore_others:
                    raise ValueError(
                        "Additional participants found; use --ignore-others to exclude them"
                    )
                boundary += 1
                continue
            role = "you" if author_id == user_id else "them"
        else:
            role = "you" if (author_id or username.lower()) == legacy_user else "them"
        attachments = msg.get("attachments") or []
        if not isinstance(attachments, list):
            raise ValueError("attachments must be a list")
        ref = msg.get("message_reference") or msg.get("reference") or {}
        reply = msg.get("reply_to")
        if not reply and isinstance(ref, dict):
            reply = ref.get("message_id") or ref.get("messageId")
        structured.append(
            {
                "id": msg.get("id"),
                "timestamp": msg.get("timestamp"),
                "role": role,
                "author_id": author_id or None,
                "username": username,
                "display_name": msg.get("display_name", username),
                "content": content,
                "session_id": boundary,
                "channel_id": msg.get("channel_id")
                or (data.get("channel_id") if isinstance(data, dict) else None),
                "attachments": [
                    a if isinstance(a, str) else a["url"]
                    for a in attachments
                    if isinstance(a, str) or (isinstance(a, dict) and a.get("url"))
                ],
                "reply_to": reply,
            }
        )
    channel = data.get("channel", {}) if isinstance(data, dict) else {}
    result = {
        "channel": channel.get("name", "DM") if isinstance(channel, dict) else str(channel),
        "participants": sorted({m["username"] for m in structured}),
        "total_messages": len(structured),
        "messages": structured,
    }
    write_json_atomic(Path(output_path), result)
    print(f"Parsed {len(structured)} messages -> {output_path}")
    return result
