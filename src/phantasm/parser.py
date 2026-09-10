"""Parser for raw Discord exports into standardized conversational turns."""

import json
import re
import sys
from pathlib import Path
from typing import Any


def parse_export(
    input_path: str,
    your_username: str,
    output_path: str = "parsed.json",
) -> dict[str, Any]:
    """Parse raw Discord message exports into normalized participant roles ('you' vs 'them')."""
    inp_file = Path(input_path)
    if not inp_file.is_file():
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    try:
        # utf-8-sig transparently handles and removes any UTF-8 Byte Order Mark (BOM)
        raw = inp_file.read_text(encoding="utf-8-sig")
        data = json.loads(raw)
    except json.JSONDecodeError as json_err:
        print(f"Error: Failed to parse JSON in '{input_path}': {json_err}")
        sys.exit(1)
    except Exception as exc:
        print(f"Error reading '{input_path}': {exc}")
        sys.exit(1)

    if isinstance(data, dict):
        messages = data.get("messages", [])
    elif isinstance(data, list):
        messages = data
    else:
        print(f"Error: Expected JSON object or list in '{input_path}', got {type(data).__name__}")
        sys.exit(1)

    structured = []
    target_username = str(your_username).strip().lower()

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        author = msg.get("author")
        if isinstance(author, dict):
            username = author.get("username", "") or author.get("name", "")
        else:
            username = msg.get("username", "")

        username = str(username).strip()
        raw_content = msg.get("content", "")
        content = str(raw_content) if raw_content is not None else ""
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content).strip()

        if not content:
            continue

        role = "you" if username.lower() == target_username else "them"

        attachments = []
        for a in msg.get("attachments", []):
            if isinstance(a, str):
                attachments.append(a)
            elif isinstance(a, dict) and a.get("url"):
                attachments.append(a["url"])

        reply_to = msg.get("reply_to")
        if not reply_to and msg.get("reference"):
            ref = msg.get("reference", {})
            if isinstance(ref, dict):
                reply_to = ref.get("messageId") or ref.get("message_id")

        structured.append(
            {
                "id": msg.get("id"),
                "timestamp": msg.get("timestamp"),
                "role": role,
                "username": username,
                "display_name": msg.get("display_name", username),
                "content": content,
                "attachments": [a for a in attachments if a],
                "reply_to": reply_to,
            }
        )

    channel_name = "DM"
    if isinstance(data, dict) and isinstance(data.get("channel"), dict):
        channel_name = data["channel"].get("name", "DM")

    output_data = {
        "channel": channel_name,
        "participants": sorted(list({m["username"] for m in structured if m.get("username")})),
        "total_messages": len(structured),
        "messages": structured,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_file.replace(out_file)

    print(f"Parsed {len(structured)} messages -> {output_path}")
    return output_data
