"""Regression tests for data loss, leakage, identity and CLI failure handling."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from phantasm.cli import main
from phantasm.formatter import build_conversations, group_consecutive_messages, split_conversations
from phantasm.parser import parse_export
from phantasm.scraper import MAX_RETRIES, ScrapeError, fetch_messages


def message(index, role, minute):
    return {
        "id": str(index),
        "role": role,
        "content": f"text-{index}",
        "timestamp": (
            datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)
        ).isoformat(),
    }


def test_session_split_never_reuses_sources():
    messages = [
        message(i, "you" if i % 2 == 0 else "them", i // 4 * 120 + i % 4) for i in range(12)
    ]
    turns = group_consecutive_messages(messages)
    train, val = split_conversations(turns, 0.34)

    def ids(samples):
        return {i for sample in samples for turn in sample for i in turn["source_ids"]}

    assert train and val
    assert ids(train).isdisjoint(ids(val))
    for sample in train + val:
        assert len({t["session_id"] for t in sample}) == 1


def test_session_boundary_does_not_connect_unrelated_turns():
    turns = group_consecutive_messages([message(0, "you", 0), message(1, "them", 120)])
    assert build_conversations(turns) == []


def test_mixed_offsets_are_normalized_and_reverse_order_rejected():
    first, second = message(1, "you", 0), message(2, "you", 1)
    second["timestamp"] = second["timestamp"].replace("+00:00", "")
    assert group_consecutive_messages([first, second])[0]["content"] == "text-1\ntext-2"
    with pytest.raises(ValueError, match="chronological"):
        group_consecutive_messages([second, first])


def test_unknown_timestamp_breaks_known_session():
    messages = [message(1, "you", 0), message(2, "them", 1)]
    messages[1]["timestamp"] = "unknown"
    assert build_conversations(group_consecutive_messages(messages)) == []


def test_duplicate_sources_are_rejected():
    msg = message(1, "you", 0)
    with pytest.raises(ValueError, match="Duplicate"):
        group_consecutive_messages([msg, msg])


def test_validation_requires_independent_sessions():
    turns = group_consecutive_messages([message(1, "you", 0), message(2, "them", 1)])
    with pytest.raises(ValueError, match="two conversation sessions"):
        split_conversations(turns, 0.05)
    assert split_conversations(turns, 0)[1] == []


def test_zero_ratios_clear_previous_held_out_output(tmp_path, monkeypatch):
    inp = tmp_path / "parsed.json"
    inp.write_text(
        json.dumps(
            [message(i, "you" if i % 2 == 0 else "them", i // 2 * 120 + i % 2) for i in range(8)]
        )
    )
    prefix = tmp_path / "dataset"
    from phantasm.cli import main

    def run(*extra):
        monkeypatch.setattr(
            sys, "argv", ["phantasm", "format", "-i", str(inp), "-p", str(prefix), *extra]
        )
        main()

    run("--val-ratio", "0.25", "--test-ratio", "0.25")
    assert (tmp_path / "dataset_val_sharegpt.jsonl").read_text()
    assert (tmp_path / "dataset_test_sharegpt.jsonl").read_text()
    run("--val-ratio", "0", "--test-ratio", "0")
    assert (tmp_path / "dataset_val_sharegpt.jsonl").read_text() == ""
    assert (tmp_path / "dataset_test_sharegpt.jsonl").read_text() == ""


def author_message(author_id, username, index):
    return {
        "id": str(index),
        "author": {"id": author_id, "username": username},
        "content": f"text-{index}",
    }


def test_group_chat_keeps_other_speakers_out_of_the_persona(tmp_path):
    """0.1 refused group exports outright; 0.2 keeps third parties as context.

    The safety property is unchanged and stronger: another participant's text is
    never emitted as a target response.
    """
    source = tmp_path / "raw.json"
    source.write_text(
        json.dumps(
            [
                author_message("1", "me", 1),
                author_message("3", "other", 2),
                author_message("2", "friend", 3),
            ]
        )
    )
    parsed = parse_export(
        str(source), self_ref="1", target_ref="2", output_path=str(tmp_path / "out.json")
    )
    assert [m["participant"] for m in parsed["messages"]] == ["self", "other", "target"]
    samples = build_conversations(group_consecutive_messages(parsed["messages"]))
    assert all(turn["role"] != "them" or turn["author_id"] == "2" for s in samples for turn in s)
    with pytest.raises(ValueError, match="Additional"):
        parse_export(
            str(source),
            self_ref="1",
            target_ref="2",
            others="error",
            output_path=str(tmp_path / "out.json"),
        )
    with pytest.raises(ValueError, match="cannot be inferred"):
        parse_export(str(source), "me", str(tmp_path / "legacy.json"))


def test_ids_survive_username_changes(tmp_path):
    source = tmp_path / "raw.json"
    source.write_text(
        json.dumps(
            [
                author_message("1", "old-name", 1),
                author_message("1", "new-name", 2),
                author_message("2", "friend", 3),
            ]
        )
    )
    parsed = parse_export(
        str(source), user_id="1", target_id="2", output_path=str(tmp_path / "out.json")
    )
    assert [m["role"] for m in parsed["messages"]] == ["you", "you", "them"]
    # These synthetic messages carry no timestamps, so 0.2 refuses to merge the
    # two self messages into one burst; only the identity mapping is asserted.
    assert [t["author_id"] for t in group_consecutive_messages(parsed["messages"])] == [
        "1",
        "1",
        "2",
    ]


def response(status, body=None):
    return Mock(status_code=status, json=Mock(return_value=body))


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_http_errors_preserve_previous_export(tmp_path, monkeypatch, status):
    output = tmp_path / "export.json"
    output.write_text("previous export")
    monkeypatch.setattr("phantasm.scraper.requests.get", Mock(return_value=response(status)))
    monkeypatch.setattr("phantasm.scraper.time.sleep", Mock())
    with pytest.raises(ScrapeError):
        fetch_messages("123", "synthetic-token", str(output))
    assert output.read_text() == "previous export"


def test_checkpoint_resume_retains_author_ids_and_order(tmp_path, monkeypatch):
    output = tmp_path / "raw.json"
    output.write_text("previous export")
    page = [author_message("2", "friend", 20), author_message("1", "me", 10)]
    get = Mock(side_effect=[response(200, page), response(403)])
    monkeypatch.setattr("phantasm.scraper.requests.get", get)
    monkeypatch.setattr("phantasm.scraper.time.sleep", Mock())
    with pytest.raises(ScrapeError):
        fetch_messages("123", "synthetic-token", str(output))
    assert output.read_text() == "previous export"
    checkpoint = Path(str(output) + ".checkpoint.json")
    assert "synthetic-token" not in checkpoint.read_text()
    get.side_effect = [response(200, [author_message("1", "me", 5)]), response(200, [])]
    result = fetch_messages("123", "synthetic-token", str(output), resume=True)
    assert [m["id"] for m in result["messages"]] == ["5", "10", "20"]
    assert result["messages"][0]["author"]["id"] == "1"
    assert get.call_args_list[-2].kwargs["params"]["before"] == "10"
    assert not checkpoint.exists()


def test_repeated_rate_limits_are_bounded(tmp_path, monkeypatch):
    get = Mock(return_value=response(429, {"retry_after": 0}))
    monkeypatch.setattr("phantasm.scraper.requests.get", get)
    monkeypatch.setattr("phantasm.scraper.time.sleep", Mock())
    with pytest.raises(ScrapeError, match="retries exhausted"):
        fetch_messages("123", "synthetic-token", str(tmp_path / "out.json"))
    assert get.call_count == MAX_RETRIES + 1


def test_repeated_page_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "phantasm.scraper.requests.get", Mock(return_value=response(200, [{"id": "1"}]))
    )
    monkeypatch.setattr("phantasm.scraper.time.sleep", Mock())
    with pytest.raises(ScrapeError, match="Pagination"):
        fetch_messages("123", "synthetic-token", str(tmp_path / "out.json"))
    assert not (tmp_path / "out.json").exists()


def test_cli_failure_has_nonzero_exit_and_no_token(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DISCORD_TOKEN", "synthetic-token")
    monkeypatch.setattr("sys.argv", ["phantasm", "scrape", "123", "-o", str(tmp_path / "out.json")])
    monkeypatch.setattr("phantasm.scraper.requests.get", Mock(return_value=response(401)))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "synthetic-token" not in capsys.readouterr().err


def test_old_positional_token_is_not_echoed(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["phantasm", "scrape", "123", "private-token"])
    with pytest.raises(SystemExit):
        main()
    # No longer accepting the token must not expose it through argparse's error.
    assert "private-token" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "secret", ["synthetic-private-token\nextra", "synthetic-token\tmore", "synthetic-token-\u2603"]
)
def test_malformed_header_credentials_never_reach_requests(monkeypatch, capsys, tmp_path, secret):
    monkeypatch.setenv("DISCORD_TOKEN", secret)
    monkeypatch.setattr("sys.argv", ["phantasm", "scrape", "123", "-o", str(tmp_path / "out.json")])
    get = Mock()
    monkeypatch.setattr("phantasm.scraper.requests.get", get)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "synthetic" not in capsys.readouterr().err
    get.assert_not_called()


def test_request_exception_details_are_not_logged(monkeypatch, capsys, tmp_path):
    import requests

    monkeypatch.setenv("DISCORD_TOKEN", "synthetic-private-token")
    monkeypatch.setattr("sys.argv", ["phantasm", "scrape", "123", "-o", str(tmp_path / "out.json")])
    monkeypatch.setattr(
        "phantasm.scraper.requests.get",
        Mock(side_effect=requests.exceptions.InvalidHeader("synthetic-private-token")),
    )
    with pytest.raises(SystemExit):
        main()
    assert "synthetic-private-token" not in capsys.readouterr().err
