"""Discord extraction with resumable checkpoints and explicit failure semantics."""

import json
import math
import time
from pathlib import Path
from typing import Any

import requests

from phantasm.storage import write_json_atomic

MAX_RETRIES = 5


class ScrapeError(RuntimeError):
    """An incomplete download; the last successful export remains untouched."""


def _mask_token(text: str, token: str) -> str:
    return text.replace(token, "[REDACTED_TOKEN]") if token else text


def _checkpoint(path: Path, channel: str, messages: list, before: str | None) -> None:
    write_json_atomic(
        path,
        {
            "version": 1,
            "channel_id": channel,
            "before": before,
            "complete": False,
            "messages": messages,
        },
    )


def fetch_messages(
    channel_id: str,
    token: str,
    output_path: str = "raw_export.json",
    limit_per_batch: int = 100,
    delay_between_requests: float = 0.5,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    channel, token = str(channel_id).strip(), str(token).strip()
    if not channel.isascii() or not channel.isdigit():
        raise ValueError("Channel ID must contain ASCII digits")
    if len(token) < 10 or any(not 32 <= ord(character) <= 126 for character in token):
        raise ValueError("Invalid or empty Discord token")
    if not math.isfinite(delay_between_requests) or delay_between_requests < 0:
        raise ValueError("Request delay must be finite and nonnegative")
    output = Path(output_path)
    checkpoint = Path(f"{output_path}.checkpoint.json")
    messages: list[dict[str, Any]] = []
    before = None
    if resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("version") != 1 or state.get("channel_id") != channel:
            raise ValueError("Checkpoint version or channel does not match this request")
        messages, before = state.get("messages"), state.get("before")
        if not isinstance(messages, list) or any(
            not isinstance(m, dict) or not str(m.get("id", "")).isdigit() for m in messages
        ):
            raise ValueError("Malformed checkpoint messages")
        if before != (messages[-1]["id"] if messages else None):
            raise ValueError("Checkpoint cursor does not match its messages")
    seen = {str(m["id"]) for m in messages}
    if len(seen) != len(messages):
        raise ValueError("Duplicate messages in checkpoint")
    failures = 0
    print(f"Fetching messages from Discord channel {channel}...")
    while True:
        params = {"limit": min(max(limit_per_batch, 1), 100)}
        if before:
            params["before"] = before
        try:
            response = requests.get(
                f"https://discord.com/api/v10/channels/{channel}/messages",
                headers={"Authorization": token},
                params=params,
                timeout=30,
            )
        except (requests.ConnectionError, requests.Timeout):
            failures += 1
            if failures > MAX_RETRIES:
                raise ScrapeError(f"Network retries exhausted; resume from {checkpoint}") from None
            time.sleep(2**failures)
            continue
        except requests.RequestException:
            raise ScrapeError(
                "Request failed; check authorization and connection settings"
            ) from None
        if response.status_code == 429 or 500 <= response.status_code < 600:
            failures += 1
            if failures > MAX_RETRIES:
                raise ScrapeError(
                    f"HTTP {response.status_code}: retries exhausted; export unchanged"
                )
            wait = float(2**failures)
            if response.status_code == 429:
                try:
                    wait = float(response.json()["retry_after"])
                except (ValueError, TypeError, KeyError):
                    raise ScrapeError("Invalid rate-limit response; export unchanged") from None
                if not math.isfinite(wait) or not 0 <= wait <= 300:
                    raise ScrapeError(
                        "Rate-limit delay outside supported 0–300 seconds; retry later"
                    )
            time.sleep(wait)
            continue
        if response.status_code != 200:
            # Do not log provider response bodies, which may reflect credentials.
            raise ScrapeError(f"HTTP {response.status_code}: download failed; export unchanged")
        try:
            batch = response.json()
        except ValueError:
            raise ScrapeError("Non-JSON response; export unchanged") from None
        if not isinstance(batch, list):
            raise ScrapeError("Expected a message list; export unchanged")
        if not batch:
            break
        cursor = int(before) if before else None
        for msg in batch:
            if not isinstance(msg, dict) or not str(msg.get("id", "")).isdigit():
                raise ScrapeError("Malformed message ID; export unchanged")
            message_id = str(msg["id"])
            if message_id in seen or (cursor is not None and int(message_id) >= cursor):
                raise ScrapeError("Pagination did not advance; export unchanged")
            cursor = int(message_id)
            seen.add(message_id)
        messages.extend(batch)
        before = batch[-1]["id"]
        _checkpoint(checkpoint, channel, messages, before)
        failures = 0
        print(f"Checkpointed {len(messages)} messages")
        time.sleep(delay_between_requests)
    # Preserve raw author IDs, reply references and metadata for participant selection.
    result = {
        "channel_id": channel,
        "complete": True,
        "total_messages": len(messages),
        "messages": list(reversed(messages)),
    }
    write_json_atomic(output, result)
    checkpoint.unlink(missing_ok=True)
    print(f"Saved complete export ({len(messages)} messages) -> {output}")
    return result
