"""Optional GGUF transfer without shell execution or credential logging."""

import os
import shutil
import subprocess
from pathlib import Path

from phantasm.credentials import read_secret


def transfer_gguf(path: Path) -> None:
    target = path.expanduser().resolve()
    if not target.is_file() or target.suffix.lower() != ".gguf" or not target.stat().st_size:
        raise ValueError("Transfer requires an explicit nonempty GGUF file")
    croc = shutil.which("croc")
    if not croc:
        raise RuntimeError("Install croc before requesting a transfer")
    secret = read_secret("CROC_SECRET", "Croc codephrase: ")
    if len(secret) < 6 or "\x00" in secret:
        raise ValueError("CROC_SECRET must contain at least six characters and no NUL")
    env = os.environ.copy()
    env["CROC_SECRET"] = secret
    print("Transferring GGUF; receiver must use the same CROC_SECRET.")
    try:
        subprocess.run(
            [croc, "--quiet", "--disable-clipboard", "--ignore-stdin", "send", str(target)],
            env=env,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError("Croc transfer failed; verify the receiver and retry") from None
    print("Transfer complete.")
