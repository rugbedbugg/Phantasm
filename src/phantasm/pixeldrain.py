"""Retried Pixeldrain uploads and range-resumed, verified downloads."""

import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

from phantasm.credentials import resolve_secret, resolve_setting
from phantasm.downloads import download_verified, sha256

DOMAINS = frozenset(
    (
        "pixeldrain.com",
        "pixeldrain.net",
        "pixeldra.in",
        "pixeldrain.nl",
        "pixeldrain.biz",
        "pixeldrain.tech",
        "pixeldrain.dev",
    )
)
RETRIES = 3
ERROR_HINTS = {
    "email_address_not_verified": "verify your Pixeldrain account email before uploading",
    "authentication_failed": "the Pixeldrain API key is invalid, revoked, or expired",
    "authentication_required": "set a Pixeldrain API key",
    "user_out_of_space": "your Pixeldrain account has no remaining storage space",
    "file_too_large": "the artifact exceeds your Pixeldrain plan's file size limit",
}


def file_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid Pixeldrain file ID")
    return value


class Pixeldrain:
    def __init__(self, key: str | None = None):
        domain = resolve_setting("PIXELDRAIN_DOMAIN", "pixeldrain.com")
        if domain not in DOMAINS:
            raise ValueError("PIXELDRAIN_DOMAIN must be an official Pixeldrain hostname")
        self.api = f"https://{domain}/api"
        self.key = key if key is not None else resolve_secret("PIXELDRAIN_API_KEY")
        if not self.key or any(ord(c) < 33 or ord(c) > 126 for c in self.key):
            raise ValueError(
                "Set PIXELDRAIN_API_KEY, or save a key with: phantasm credentials set pixeldrain"
            )

    def _request(self, method, endpoint, **kwargs):
        for attempt in range(RETRIES + 1):
            try:
                response = requests.request(
                    method,
                    self.api + endpoint,
                    auth=("", self.key),
                    timeout=(30, 120),
                    allow_redirects=False,
                    **kwargs,
                )
            except requests.RequestException:
                if attempt == RETRIES:
                    raise RuntimeError(
                        "Pixeldrain connection failed; retry the Phantasm command"
                    ) from None
            else:
                if response.status_code not in (429, 500, 502, 503, 504):
                    if response.status_code >= 400 or 300 <= response.status_code < 400:
                        code = response.status_code
                        hint = "check account access/limits"
                        try:
                            error = response.json()
                            if isinstance(error, dict) and isinstance(error.get("value"), str):
                                hint = ERROR_HINTS.get(error["value"], hint)
                        except ValueError:
                            pass
                        response.close()
                        raise RuntimeError(f"Pixeldrain HTTP {code}; {hint}")
                    return response
                response.close()
                if attempt == RETRIES:
                    raise RuntimeError("Pixeldrain retry limit reached")
            # Rewind uploads on every retry; PUT uploads restart, not byte-resume.
            body = kwargs.get("data")
            if hasattr(body, "seek"):
                body.seek(0)
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _json(self, response):
        try:
            data = response.json()
            if not isinstance(data, dict) or data.get("success") is False:
                raise ValueError
            return data
        except ValueError:
            raise RuntimeError("Invalid Pixeldrain response") from None
        finally:
            response.close()

    def upload(self, path: Path, name: str | None = None) -> dict:
        path = path.resolve()
        expected = sha256(path)
        size = path.stat().st_size
        if not size:
            raise ValueError("Cannot upload an empty artifact")
        with path.open("rb") as stream:
            data = self._json(
                self._request("PUT", "/file/" + quote(name or path.name, safe=""), data=stream)
            )
        identifier = file_id(data.get("id"))
        if data.get("hash_sha256") != expected or data.get("size") != size:
            raise RuntimeError("Uploaded artifact did not match its local checksum/size")
        return {
            "id": identifier,
            "name": data.get("name", name or path.name),
            "sha256": expected,
            "size": size,
        }

    def files(self) -> list[dict]:
        data = self._json(self._request("GET", "/user/files"))
        if not isinstance(data.get("files"), list):
            raise RuntimeError("Invalid Pixeldrain file listing")
        return data["files"]

    def info(self, identifier: str) -> dict:
        return self._json(self._request("GET", f"/file/{file_id(identifier)}/info"))

    def download(
        self, identifier: str, destination: Path, expected_hash: str | None = None
    ) -> Path:
        info = self.info(identifier)
        checksum, size = info.get("hash_sha256"), info.get("size")
        if (
            not isinstance(checksum, str)
            or not re.fullmatch(r"[a-f0-9]{64}", checksum)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise RuntimeError("Invalid artifact metadata")
        if expected_hash and expected_hash != checksum:
            raise RuntimeError("Artifact metadata differs from the saved receipt")
        return download_verified(
            lambda headers: self._request(
                "GET", f"/file/{file_id(identifier)}", stream=True, headers=headers
            ),
            destination,
            size,
            checksum,
        )
