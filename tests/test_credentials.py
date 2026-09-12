"""Credential resolution, the local store, and its permission guarantees."""

import json
import stat
import sys
from pathlib import Path

import pytest

from phantasm.credentials import (
    STORABLE,
    clear_secret,
    describe_sources,
    load_store,
    prompt_value,
    read_secret,
    resolve_name,
    resolve_secret,
    store_path,
    store_secret,
    stored_value,
)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PHANTASM_CONFIG_DIR", str(tmp_path / "config"))
    for name in ("PIXELDRAIN_API_KEY", "PIXELDRAIN_DOMAIN", "HF_TOKEN", "DISCORD_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "config"


def test_friendly_names_and_variable_names_both_resolve():
    assert resolve_name("pixeldrain") == "PIXELDRAIN_API_KEY"
    assert resolve_name("PIXELDRAIN_API_KEY") == "PIXELDRAIN_API_KEY"
    assert resolve_name("huggingface") == "HF_TOKEN"
    assert resolve_name("pixeldrain-domain") == "PIXELDRAIN_DOMAIN"
    with pytest.raises(ValueError, match="Unknown credential"):
        resolve_name("aws")


def test_stored_key_is_resolved_when_environment_is_empty():
    store_secret("pixeldrain", "stored-key")
    assert resolve_secret("PIXELDRAIN_API_KEY") == "stored-key"
    assert stored_value("PIXELDRAIN_API_KEY") == "stored-key"


def test_environment_wins_over_the_store(monkeypatch):
    store_secret("pixeldrain", "stored-key")
    monkeypatch.setenv("PIXELDRAIN_API_KEY", "environment-key")
    assert resolve_secret("PIXELDRAIN_API_KEY") == "environment-key"


def test_store_is_owner_readable_only():
    path = store_secret("pixeldrain", "secret-value")
    assert path == store_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_world_readable_store_is_refused():
    path = store_secret("pixeldrain", "secret-value")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="readable by other users"):
        load_store()


def test_saving_one_credential_keeps_the_others():
    store_secret("pixeldrain", "key-one")
    store_secret("huggingface", "key-two")
    assert load_store() == {"PIXELDRAIN_API_KEY": "key-one", "HF_TOKEN": "key-two"}


def test_clearing_removes_only_the_named_credential():
    store_secret("pixeldrain", "key-one")
    store_secret("huggingface", "key-two")
    assert clear_secret("pixeldrain") is True
    assert clear_secret("pixeldrain") is False
    assert load_store() == {"HF_TOKEN": "key-two"}


def test_empty_and_control_character_values_are_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        store_secret("pixeldrain", "   ")
    with pytest.raises(ValueError, match="control characters"):
        store_secret("pixeldrain", "key\nwith-newline")


def test_corrupt_store_reports_the_path():
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="not valid JSON"):
        load_store()


def test_missing_store_is_not_an_error():
    assert load_store() == {}
    assert resolve_secret("PIXELDRAIN_API_KEY") == ""


def test_discord_token_is_never_read_from_the_store():
    """Discord token handling stays environment-or-prompt only, by design."""
    from phantasm.credentials import save_store

    save_store({"DISCORD_TOKEN": "should-be-ignored"})
    assert stored_value("DISCORD_TOKEN") == ""
    assert "DISCORD_TOKEN" not in STORABLE


def test_noninteractive_discord_message_is_unchanged(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(ValueError, match="Set DISCORD_TOKEN for noninteractive use"):
        read_secret("DISCORD_TOKEN", "Token: ")


def test_noninteractive_pixeldrain_message_points_at_the_store(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(ValueError, match="phantasm credentials set pixeldrain"):
        read_secret("PIXELDRAIN_API_KEY", "Key: ")


def test_read_secret_uses_a_stored_value_without_prompting(monkeypatch):
    store_secret("pixeldrain", "stored-key")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert read_secret("PIXELDRAIN_API_KEY", "Key: ") == "stored-key"


def test_prompting_requires_a_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(ValueError, match="interactive terminal"):
        prompt_value("PIXELDRAIN_API_KEY")


def test_prompted_value_is_hidden_and_stored(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("phantasm.credentials.getpass.getpass", lambda prompt: " typed-key ")
    assert prompt_value("PIXELDRAIN_API_KEY") == "typed-key"


def test_listing_reports_sources_but_never_values(monkeypatch, capsys):
    store_secret("pixeldrain", "super-secret-value")
    monkeypatch.setenv("HF_TOKEN", "environment-secret")
    rows = dict((name, source) for name, _, source in describe_sources())
    assert rows["PIXELDRAIN_API_KEY"] == "stored"
    assert rows["HF_TOKEN"] == "environment"
    assert rows["PIXELDRAIN_DOMAIN"] == "not set"

    from phantasm.cli import main

    monkeypatch.setattr(sys, "argv", ["phantasm", "credentials", "list"])
    main()
    printed = capsys.readouterr().out
    assert "super-secret-value" not in printed
    assert "environment-secret" not in printed
    assert "stored" in printed and "environment" in printed


def test_cli_set_requires_a_name(monkeypatch):
    from phantasm.cli import main

    monkeypatch.setattr(sys, "argv", ["phantasm", "credentials", "set"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_cli_set_stores_a_hidden_prompt_value(monkeypatch, capsys):
    from phantasm.cli import main

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("phantasm.credentials.getpass.getpass", lambda prompt: "typed-key")
    monkeypatch.setattr(sys, "argv", ["phantasm", "credentials", "set", "pixeldrain"])
    main()
    assert "typed-key" not in capsys.readouterr().out
    assert json.loads(store_path().read_text())["PIXELDRAIN_API_KEY"] == "typed-key"


def test_pixeldrain_client_accepts_a_stored_key():
    from phantasm.pixeldrain import Pixeldrain

    with pytest.raises(ValueError, match="phantasm credentials set pixeldrain"):
        Pixeldrain()
    store_secret("pixeldrain", "stored-key")
    assert Pixeldrain().key == "stored-key"


def test_stored_domain_is_validated():
    from phantasm.pixeldrain import Pixeldrain

    store_secret("pixeldrain", "stored-key")
    store_secret("pixeldrain-domain", "pixeldrain.net")
    assert Pixeldrain().api == "https://pixeldrain.net/api"
    store_secret("pixeldrain-domain", "evil.example")
    with pytest.raises(ValueError, match="official Pixeldrain hostname"):
        Pixeldrain()


def test_explicitly_blank_domain_is_still_a_misconfiguration(monkeypatch):
    """An empty PIXELDRAIN_DOMAIN must be rejected, not quietly defaulted."""
    from phantasm.pixeldrain import Pixeldrain

    monkeypatch.setenv("PIXELDRAIN_DOMAIN", "")
    with pytest.raises(ValueError, match="official Pixeldrain hostname"):
        Pixeldrain("key")


def test_colab_forwards_a_stored_key_to_the_runtime(monkeypatch):
    from phantasm.credentials import resolve_secret

    store_secret("pixeldrain", "stored-key")
    store_secret("huggingface", "stored-hf")
    forwarded = {
        key: resolve_secret(key)
        for key in ("PIXELDRAIN_API_KEY", "PIXELDRAIN_DOMAIN", "HF_TOKEN")
        if resolve_secret(key)
    }
    assert forwarded == {"PIXELDRAIN_API_KEY": "stored-key", "HF_TOKEN": "stored-hf"}


def test_tests_never_see_the_real_credential_store(monkeypatch):
    """conftest must isolate the store; a real saved key must not leak into tests."""
    import os

    from phantasm.credentials import store_path

    configured = os.environ.get("PHANTASM_CONFIG_DIR", "")
    real = Path("~/.config/phantasm/credentials.json").expanduser()
    assert configured, "conftest must redirect the store away from the real config directory"
    assert store_path() != real
    assert str(store_path()).startswith(configured)
    assert load_store() == {}
