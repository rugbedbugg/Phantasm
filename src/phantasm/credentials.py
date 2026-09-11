"""Resolve credentials without command-line arguments or logs."""

import getpass
import os
import sys


def read_secret(variable: str, prompt: str) -> str:
    secret = os.environ.get(variable, "").strip()
    if not secret:
        if not sys.stdin.isatty():
            raise ValueError(f"Set {variable} for noninteractive use")
        secret = getpass.getpass(prompt).strip()
    if not secret:
        raise ValueError(f"{variable} must not be empty")
    return secret
