"""Discord channel message extractor."""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
MAX_RETRIES = 5


def _mask_token(text: str, token: str) -> str:
    """Mask token from error and log messages to avoid credential leakage."""
    if not token or len(token) < 8:
        return text
    return text.replace(token, "[REDACTED_TOKEN]")


def fetch_messages(
    channel_id: str,
    token: str,
    output_path: str = "raw_export.json",
    limit_per_batch: int = 100,
    delay_between_requests: float = 0.5,
) -> dict[str, Any]:
    """Fetch all messages from a Discord channel or DM and save safely as JSON."""
    clean_channel = str(channel_id).strip()
    clean_token = str(token).strip()

    if not clean_channel or not clean_channel.isdigit():
        print(f"Error: Invalid channel ID '{clean_channel}'. Channel IDs must be numeric digits.")
        sys.exit(1)

    if not clean_token or len(clean_token) < 10:
        print("Error: Invalid or empty Discord token provided.")
        sys.exit(1)

    headers = {
        "Authorization": clean_token,
        "Content-Type": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    base_url = f"https://discord.com/api/v9/channels/{clean_channel}/messages"

    all_messages: list[dict[str, Any]] = []
    before: str | None = None

    print(f"Fetching messages from Discord channel {clean_channel}...")

    retry_count = 0
    while True:
        params: dict[str, Any] = {"limit": min(max(limit_per_batch, 1), 100)}
        if before:
            params["before"] = before

        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as net_err:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                safe_err = _mask_token(str(net_err), clean_token)
                print(f"\nNetwork error after {MAX_RETRIES} retries: {safe_err}")
                break
            wait_sec = 2.0**retry_count
            print(
                f"\nNetwork interruption. Retrying in {wait_sec:.1f}s ({retry_count}/{MAX_RETRIES})..."
            )
            time.sleep(wait_sec)
            continue

        retry_count = 0

        if resp.status_code == 429:
            retry_after = resp.json().get("retry_after", 1.0)
            print(f"Rate limited by Discord. Waiting {retry_after}s...")
            time.sleep(float(retry_after))
            continue

        if resp.status_code == 401:
            print("Error 401: Unauthorized. Please check your Discord token.")
            break

        if resp.status_code == 403:
            print("Error 403: Forbidden. You do not have permission to read this channel.")
            break

        if resp.status_code == 404:
            print(f"Error 404: Channel ID {clean_channel} not found.")
            break

        if resp.status_code != 200:
            safe_resp = _mask_token(resp.text[:200], clean_token)
            print(f"Error {resp.status_code}: {safe_resp}")
            break

        try:
            batch = resp.json()
        except Exception:
            print("\nError: Received non-JSON response from Discord API.")
            break

        if not batch or not isinstance(batch, list):
            break

        all_messages.extend(batch)
        before = batch[-1]["id"]
        print(f"  Fetched {len(all_messages)} messages so far...", end="\r")
        time.sleep(delay_between_requests)

    print(f"\nFetch complete. Total raw messages: {len(all_messages)}")

    structured = []
    for msg in reversed(all_messages):
        if not isinstance(msg, dict):
            continue
        author = msg.get("author", {}) if isinstance(msg.get("author"), dict) else {}
        ref = msg.get("message_reference")

        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        # Sanitize non-printable characters except newlines/tabs
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content).strip()

        structured.append(
            {
                "id": msg.get("id"),
                "timestamp": msg.get("timestamp"),
                "username": author.get("username", ""),
                "display_name": author.get("global_name") or author.get("username", ""),
                "content": content,
                "attachments": [
                    a.get("url")
                    for a in msg.get("attachments", [])
                    if isinstance(a, dict) and a.get("url")
                ],
                "embeds": [
                    e.get("url") or e.get("title")
                    for e in msg.get("embeds", [])
                    if isinstance(e, dict)
                ],
                "reply_to": ref.get("message_id") if isinstance(ref, dict) else None,
                "type": msg.get("type", 0),
            }
        )

    output_data = {
        "channel_id": clean_channel,
        "total_messages": len(structured),
        "messages": structured,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_file.replace(out_file)
    print(f"Saved -> {output_path}")
    return output_data
