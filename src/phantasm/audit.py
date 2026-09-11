"""Local heuristic privacy and data-quality auditing for training datasets.

This is a *heuristic* scan, not a privacy guarantee. It runs entirely offline:
nothing is uploaded, no model is consulted, and matched secrets are redacted
before anything is printed. Treat a clean report as "no obvious problems found",
never as proof that a dataset is safe to publish.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from phantasm.dataset import read_sharegpt
from phantasm.text import normalized_for_duplicates

#: Severity ordering used for report grouping.
SEVERITIES = ("high", "medium", "low")


@dataclass(frozen=True)
class Rule:
    """One heuristic detector."""

    name: str
    severity: str
    pattern: re.Pattern
    description: str


RULES: tuple[Rule, ...] = (
    Rule(
        "api_credential",
        "high",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}"
            r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
            r"\.[A-Za-z0-9_-]{10,}|[A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,})\b"
        ),
        "Potential API credential",
    ),
    Rule(
        "credential_assignment",
        "high",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|token|credential)s?\b\s*"
            r"[:=]\s*\S{6,}"
        ),
        "Potential credential assignment",
    ),
    Rule(
        "private_key",
        "high",
        re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
        "Potential private key block",
    ),
    Rule("email", "medium", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), "Potential email"),
    Rule(
        "phone_number",
        "medium",
        re.compile(
            r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]?\d{3,4}(?![\w.])"
        ),
        "Potential phone number",
    ),
    Rule(
        "ip_address",
        "medium",
        re.compile(
            r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.])"
        ),
        "Potential IP address",
    ),
    Rule(
        "filesystem_path",
        "low",
        re.compile(r"(?:/(?:home|Users|root|var|etc)/[\w.@-]+|[A-Za-z]:\\Users\\[\w.@-]+)"),
        "Potential filesystem path",
    ),
    Rule(
        "long_identifier",
        "low",
        re.compile(r"(?<!\d)\d{12,}(?!\d)"),
        "Very long numeric identifier",
    ),
    Rule("url", "low", re.compile(r"(?:https?://|www\.)[^\s<>\"']+"), "URL"),
)

DEFAULT_LONG_RESPONSE_CHARS = 1500


@dataclass
class Finding:
    """A single heuristic hit, with the matched value already redacted."""

    rule: str
    severity: str
    description: str
    sample: int
    role: str
    redacted: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "sample": self.sample,
            "role": self.role,
            "redacted": self.redacted,
        }


@dataclass
class AuditReport:
    """Aggregated audit results for one dataset."""

    samples: int = 0
    invalid_rows: int = 0
    duplicate_samples: int = 0
    long_responses: int = 0
    repeated_strings: int = 0
    findings: list[Finding] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    severity_counts: Counter = field(default_factory=Counter)
    descriptions: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "invalid_rows": self.invalid_rows,
            "duplicate_samples": self.duplicate_samples,
            "long_responses": self.long_responses,
            "repeated_strings": self.repeated_strings,
            "categories": {
                name: {
                    "count": count,
                    "severity": next(r.severity for r in RULES if r.name == name),
                    "description": self.descriptions.get(name, name),
                }
                for name, count in self.counts.most_common()
            },
            "severity_counts": dict(self.severity_counts),
            "findings": [finding.as_dict() for finding in self.findings],
        }


def redact(value: str, keep: int = 3) -> str:
    """Mask a matched value so a report never reprints the secret itself."""
    text = value.strip()
    if len(text) <= keep:
        return "*" * len(text)
    visible = text[:keep]
    return f"{visible}{'*' * min(len(text) - keep, 12)} ({len(text)} chars)"


def scan_text(text: str, sample: int, role: str) -> list[Finding]:
    """Apply every rule to one message, returning redacted findings."""
    found = []
    for rule in RULES:
        for match in rule.pattern.finditer(text or ""):
            found.append(
                Finding(
                    rule=rule.name,
                    severity=rule.severity,
                    description=rule.description,
                    sample=sample,
                    role=role,
                    redacted=redact(match.group(0)),
                )
            )
    return found


def audit_rows(
    rows: list[dict],
    *,
    info: dict[str, Any] | None = None,
    long_response_chars: int = DEFAULT_LONG_RESPONSE_CHARS,
    repeated_string_threshold: int = 5,
) -> AuditReport:
    """Audit parsed ShareGPT rows without transmitting or printing raw content."""
    report = AuditReport(samples=len(rows), invalid_rows=(info or {}).get("invalid_rows", 0))
    signatures = Counter()
    responses = Counter()
    for index, row in enumerate(rows, 1):
        signatures[json.dumps(row, sort_keys=True, ensure_ascii=False)] += 1
        for turn in row["conversations"]:
            value = turn["value"]
            report.findings.extend(scan_text(value, index, turn["from"]))
            if turn["from"] == "gpt":
                responses[normalized_for_duplicates(value)] += 1
                if len(value) > long_response_chars:
                    report.long_responses += 1
    report.duplicate_samples = sum(count - 1 for count in signatures.values() if count > 1)
    report.repeated_strings = sum(
        1 for count in responses.values() if count >= repeated_string_threshold
    )
    for finding in report.findings:
        report.counts[finding.rule] += 1
        report.severity_counts[finding.severity] += 1
        report.descriptions[finding.rule] = finding.description
    return report


def audit_dataset(path: str, **kwargs: Any) -> AuditReport:
    """Read a ShareGPT dataset tolerantly and audit it."""
    rows, info = read_sharegpt(path, strict=False)
    return audit_rows(rows, info=info, **kwargs)


def render(report: AuditReport, *, locations: int = 10) -> str:
    """Format an audit report for a terminal, without echoing matched values."""
    lines = [
        "Phantasm privacy audit",
        "─" * 46,
        f"{'Samples scanned':<32}{report.samples}",
    ]
    if report.invalid_rows:
        lines.append(f"{'Unreadable rows':<32}{report.invalid_rows}")
    lines.append("")
    if report.counts:
        for name, count in report.counts.most_common():
            lines.append(f"{report.descriptions[name]:<32}{count}")
    else:
        lines.append("No heuristic pattern matched.")
    lines += [
        f"{'Duplicate training samples':<32}{report.duplicate_samples}",
        f"{'Very long responses':<32}{report.long_responses}",
        f"{'Repeated response strings':<32}{report.repeated_strings}",
        "",
        "Severity: " + ", ".join(f"{name} {report.severity_counts[name]}" for name in SEVERITIES),
    ]
    flagged = [f for f in report.findings if f.severity in ("high", "medium")]
    if flagged:
        lines += ["", f"First {min(locations, len(flagged))} location(s):"]
        for finding in flagged[:locations]:
            lines.append(
                f"  sample {finding.sample:<6} {finding.role:<6} "
                f"{finding.description}: {finding.redacted}"
            )
    lines += [
        "",
        "Heuristic scan only. It runs locally and never uploads data; it cannot",
        "prove that a dataset is free of personal or secret information.",
    ]
    return "\n".join(lines)


def add_arguments(parser: Any) -> None:
    """Register ``phantasm audit`` options."""
    parser.add_argument("input", help="ShareGPT JSONL dataset to audit")
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=DEFAULT_LONG_RESPONSE_CHARS,
        help="Responses longer than this are reported as suspiciously long",
    )
    parser.add_argument(
        "--repeat-threshold",
        type=int,
        default=5,
        help="Identical responses occurring this often are reported as repeated strings",
    )
    parser.add_argument(
        "--locations", type=int, default=10, help="How many redacted locations to list"
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def command(args: Any) -> None:
    """Run the local heuristic audit and print a redacted report."""
    if min(args.max_response_chars, args.repeat_threshold) < 1 or args.locations < 0:
        raise ValueError("Audit thresholds must be positive and --locations non-negative")
    report = audit_dataset(
        args.input,
        long_response_chars=args.max_response_chars,
        repeated_string_threshold=args.repeat_threshold,
    )
    print(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False)
        if args.json
        else render(report, locations=args.locations)
    )
