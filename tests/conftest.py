"""Shared test isolation.

The credential store introduced in 0.2 reads a real file under the developer's
config directory. Without isolation a test could read, or overwrite, actual
secrets, and results would depend on whether the machine happens to have a key
saved. Every test therefore runs against an empty store in a temporary
directory.
"""

import pytest

CREDENTIAL_VARIABLES = (
    "PIXELDRAIN_API_KEY",
    "PIXELDRAIN_DOMAIN",
    "HF_TOKEN",
    "DISCORD_TOKEN",
    "CROC_SECRET",
)


@pytest.fixture(autouse=True)
def isolate_credentials(tmp_path_factory, monkeypatch):
    """Point the credential store at an empty temporary directory."""
    monkeypatch.setenv("PHANTASM_CONFIG_DIR", str(tmp_path_factory.mktemp("phantasm-config")))
    for name in CREDENTIAL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
