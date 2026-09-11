"""Heuristic privacy audit: detection, redaction and reporting."""

import json

import pytest

from phantasm.audit import audit_dataset, audit_rows, redact, render, scan_text


def sample(*values):
    turns = []
    for index, value in enumerate(values):
        turns.append({"from": "human" if index % 2 == 0 else "gpt", "value": value})
    return {"conversations": turns}


def rules_for(text):
    return {finding.rule for finding in scan_text(text, 1, "gpt")}


def test_detects_emails_phones_and_addresses():
    assert "email" in rules_for("write to someone@example.invalid")
    assert "phone_number" in rules_for("call +1 415 555 0199 tonight")
    assert "ip_address" in rules_for("the box is at 10.0.12.44")


def test_detects_credentials_and_secrets():
    assert "api_credential" in rules_for("token sk-abcdefghijklmnopqrstuvwxyz01")
    assert "api_credential" in rules_for("ghp_abcdefghijklmnopqrstuvwxyz0123456789")
    assert "credential_assignment" in rules_for("password: hunter2hunter2")
    assert "private_key" in rules_for("-----BEGIN RSA PRIVATE KEY-----")


def test_detects_paths_urls_and_long_identifiers():
    assert "filesystem_path" in rules_for("see /home/alice/taxes.pdf")
    assert "url" in rules_for("https://example.invalid/page")
    assert "long_identifier" in rules_for("order 123456789012345")


def test_ordinary_chat_is_not_flagged():
    assert rules_for("hey, are we still on for 7?") == set()
    assert rules_for("I scored 42 points in 3 games") == set()


def test_matched_values_are_redacted_and_never_echoed():
    secret = "sk-abcdefghijklmnopqrstuvwxyz01"
    report = audit_rows([sample("key?", f"use {secret}")])
    rendered = render(report)
    assert secret not in rendered
    assert secret not in json.dumps(report.as_dict())
    assert report.findings[0].redacted.startswith("sk-")
    assert "*" in report.findings[0].redacted


def test_redaction_keeps_only_a_short_prefix():
    assert redact("ab") == "**"
    assert redact("abcdefghijkl").startswith("abc")
    assert "abcdefghijkl" not in redact("abcdefghijkl")


def test_locations_are_reported_by_sample_number():
    report = audit_rows([sample("a", "fine"), sample("b", "mail me at x@example.invalid")])
    assert [f.sample for f in report.findings] == [2]
    assert "sample 2" in render(report)


def test_duplicate_long_and_repeated_samples_are_counted():
    duplicate = sample("hi", "hello")
    report = audit_rows(
        [duplicate, duplicate, sample("hi", "x" * 40)] + [sample("q", "same")] * 5,
        long_response_chars=20,
        repeated_string_threshold=5,
    )
    assert report.duplicate_samples == 5
    assert report.long_responses == 1
    assert report.repeated_strings == 1


def test_severity_counts_and_category_summary():
    report = audit_rows([sample("a", "mail x@example.invalid and see https://example.invalid")])
    assert report.severity_counts["medium"] == 1
    assert report.severity_counts["low"] == 1
    categories = report.as_dict()["categories"]
    assert categories["email"]["severity"] == "medium"
    assert categories["url"]["description"] == "URL"


def test_audit_tolerates_malformed_rows(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(sample("hi", "hello")) + "\n{oops\n")
    report = audit_dataset(str(path))
    assert report.samples == 1
    assert report.invalid_rows == 1
    assert "Unreadable rows" in render(report)


def test_empty_dataset_audits_cleanly(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    report = audit_dataset(str(path))
    assert report.samples == 0
    assert "No heuristic pattern matched." in render(report)


def test_report_states_that_it_is_heuristic():
    assert "cannot" in render(audit_rows([sample("a", "b")])).lower()


@pytest.mark.parametrize("text", ["ünïcodé ✨ message", "line one\nline two", "emoji only 😀😀"])
def test_unicode_and_multiline_content_is_handled(text):
    report = audit_rows([sample("q", text)])
    assert report.samples == 1
