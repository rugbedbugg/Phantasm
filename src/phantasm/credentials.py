"""Resolve credentials without command-line arguments or logs.

Secrets are never accepted as command-line arguments, never printed, and never
written into job metadata. They are resolved in a fixed order:

1. the process environment,
2. an optional local credential store written by ``phantasm credentials set``,
3. a hidden terminal prompt, when attached to a TTY.

Only the credentials in :data:`STORABLE` may be written to the store. The
Discord token and the croc codephrase are deliberately excluded so their
handling stays exactly as it was: environment variable or hidden prompt only.
"""

import getpass
import json
import os
import stat
import sys
from pathlib import Path

from phantasm.storage import write_text_atomic

#: Credentials that may be saved locally, and the friendly name each accepts.
STORABLE = {
    "PIXELDRAIN_API_KEY": "pixeldrain",
    "HF_TOKEN": "huggingface",
}

#: Non-secret settings that travel with the stored credentials.
STORABLE_SETTINGS = {"PIXELDRAIN_DOMAIN": "pixeldrain-domain"}

_ALIASES = {alias: name for name, alias in {**STORABLE, **STORABLE_SETTINGS}.items()}


def resolve_name(reference: str) -> str:
    """Map ``pixeldrain`` or ``PIXELDRAIN_API_KEY`` onto the variable name."""
    wanted = (reference or "").strip()
    if wanted in STORABLE or wanted in STORABLE_SETTINGS:
        return wanted
    name = _ALIASES.get(wanted.lower())
    if name:
        return name
    known = ", ".join(sorted(_ALIASES))
    raise ValueError(f"Unknown credential {reference!r}; choose one of: {known}")


def store_path() -> Path:
    """Location of the credential store, honouring XDG and a test override."""
    override = os.environ.get("PHANTASM_CONFIG_DIR")
    if override:
        return Path(override).expanduser() / "credentials.json"
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "phantasm" / "credentials.json"


def load_store() -> dict[str, str]:
    """Read the credential store, refusing a file others can read."""
    path = store_path()
    if not path.is_file():
        return {}
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError(f"{path} is readable by other users; run: chmod 600 {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise ValueError(f"{path} is not valid JSON; fix or delete it") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def save_store(values: dict[str, str]) -> Path:
    """Write the credential store with owner-only permissions."""
    path = store_path()
    write_text_atomic(path, json.dumps(values, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)
    path.parent.chmod(0o700)
    return path


def store_secret(name: str, value: str) -> Path:
    """Save one credential, leaving any others in the store untouched."""
    name = resolve_name(name)
    value = (value or "").strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{name} must not contain control characters")
    values = load_store()
    values[name] = value
    return save_store(values)


def clear_secret(name: str) -> bool:
    """Remove one credential; returns whether anything was stored."""
    name = resolve_name(name)
    values = load_store()
    if name not in values:
        return False
    del values[name]
    save_store(values)
    return True


def stored_value(name: str) -> str:
    """The stored value for a storable credential, or an empty string."""
    if name not in STORABLE and name not in STORABLE_SETTINGS:
        return ""
    return load_store().get(name, "").strip()


def resolve_setting(name: str, default: str) -> str:
    """Resolve a non-secret setting.

    An environment variable that is present but empty is returned as-is: an
    explicitly blank setting is a misconfiguration the caller should reject,
    not something to paper over with the default.
    """
    if name in os.environ:
        return os.environ[name]
    return stored_value(name) or default


def resolve_secret(name: str) -> str:
    """Environment first, then the local store. Never prompts."""
    return os.environ.get(name, "").strip() or stored_value(name)


def describe_sources() -> list[tuple[str, str, str]]:
    """``(variable, friendly name, source)`` for every supported credential.

    ``source`` is ``environment``, ``stored`` or ``not set``. Values are never
    returned, so this is safe to print.
    """
    rows = []
    for name, alias in sorted({**STORABLE, **STORABLE_SETTINGS}.items()):
        if os.environ.get(name, "").strip():
            source = "environment"
        elif stored_value(name):
            source = "stored"
        else:
            source = "not set"
        rows.append((name, alias, source))
    return rows


def prompt_value(name: str, *, hidden: bool = True) -> str:
    """Read one value from an interactive terminal, hidden by default."""
    if not sys.stdin.isatty():
        raise ValueError(
            f"Saving {name} needs an interactive terminal; "
            "credentials are never accepted as command-line arguments"
        )
    value = (getpass.getpass(f"{name}: ") if hidden else input(f"{name}: ")).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def read_secret(variable: str, prompt: str) -> str:
    """Resolve a credential from the environment, the store, or a hidden prompt."""
    secret = os.environ.get(variable, "").strip() or stored_value(variable)
    if not secret:
        if not sys.stdin.isatty():
            hint = (
                f"Set {variable} for noninteractive use"
                if variable not in STORABLE
                else f"Set {variable}, or save it with: phantasm credentials set {STORABLE[variable]}"
            )
            raise ValueError(hint)
        secret = getpass.getpass(prompt).strip()
    if not secret:
        raise ValueError(f"{variable} must not be empty")
    return secret
