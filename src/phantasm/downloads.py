"""Bounded-memory, resumable downloads shared by artifact providers."""

import hashlib
import re
import time
from pathlib import Path

import requests

RETRIES = 3
CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(request, destination: Path, size: int, checksum: str) -> Path:
    if (
        not isinstance(checksum, str)
        or not re.fullmatch(r"[a-f0-9]{64}", checksum)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise RuntimeError("Invalid artifact metadata")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == size and sha256(destination) == checksum:
            return destination
        raise ValueError("Destination exists with different content; choose another path")
    # Include content hash so a partial file can never be reused for another artifact.
    part = destination.with_name(f".{destination.name}.{checksum}.part")
    if part.exists() and part.stat().st_size > size:
        part.unlink()
    for attempt in range(RETRIES + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset == size:
            if sha256(part) == checksum:
                part.replace(destination)
                return destination
            part.unlink()
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        print(f"Downloading artifact: {offset:,}/{size:,} bytes", flush=True)
        next_progress = offset + max(CHUNK, size // 20)
        try:
            with request(headers) as response:
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.ConnectionError("Transient download failure")
                if response.status_code == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                    )
                    if (
                        not match
                        or int(match[1]) != offset
                        or int(match[2]) != size - 1
                        or int(match[3]) != size
                    ):
                        raise RuntimeError("Invalid resumed download range")
                elif response.status_code == 200:
                    offset = 0  # Range ignored: replace the partial data, never append.
                else:
                    raise RuntimeError("Unexpected artifact download response")
                with part.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(CHUNK):
                        if offset + len(chunk) > size:
                            raise RuntimeError("Artifact exceeded its advertised size")
                        stream.write(chunk)
                        offset += len(chunk)
                        if offset >= next_progress:
                            print(f"Downloading artifact: {offset:,}/{size:,} bytes", flush=True)
                            next_progress = offset + max(CHUNK, size // 20)
        except requests.RequestException:
            pass  # Keep partial bytes and retry from the exact saved offset.
        if part.exists() and part.stat().st_size == size and sha256(part) == checksum:
            part.replace(destination)
            return destination
        if attempt < RETRIES:
            time.sleep(2**attempt)
    raise RuntimeError("Download incomplete or checksum mismatch; rerun to recover")
