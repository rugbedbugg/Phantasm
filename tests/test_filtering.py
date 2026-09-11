"""Data-quality filtering defaults, opt-in rules, and removal accounting."""

import pytest

from phantasm.filtering import FilterConfig, filter_messages


def messages(*contents, **extra):
    return [
        {"id": str(i), "participant": "self", "content": c, **extra} for i, c in enumerate(contents)
    ]


def test_defaults_keep_ordinary_conversation():
    kept, report = filter_messages(messages("hey", "how's it going", "!!! ok"))
    assert len(kept) == 3
    assert report.total_removed == 0
    assert "no quality filter matched" in report.render()


def test_empty_and_control_only_messages_are_removed():
    kept, report = filter_messages(messages("", "   ", "\x00\x07", "real"))
    assert [m["content"] for m in kept] == ["real"]
    assert report.removed["empty"] == 3


def test_non_dict_rows_are_counted_not_crashed():
    kept, report = filter_messages([None, "text", {"participant": "self", "content": "ok"}])
    assert len(kept) == 1
    assert report.removed["empty"] == 2


def test_bot_messages_are_removed_by_default():
    rows = messages("beep boop")
    rows[0]["bot"] = True
    kept, report = filter_messages(rows)
    assert kept == []
    assert report.removed["bot"] == 1
    kept, _ = filter_messages(rows, FilterConfig(drop_bots=False))
    assert len(kept) == 1


@pytest.mark.parametrize(
    "text", ["[deleted]", "<Message deleted>", "This message was deleted", "(removed)"]
)
def test_deleted_placeholders_are_removed_by_default(text):
    kept, report = filter_messages(messages(text))
    assert kept == []
    assert report.removed["deleted_placeholder"] == 1


def test_deleted_placeholder_does_not_match_ordinary_sentences():
    kept, _ = filter_messages(messages("the file was deleted from my laptop"))
    assert len(kept) == 1


def test_commands_and_link_only_messages_are_opt_in():
    rows = messages("!play something", "https://example.invalid/page", "real text")
    kept, _ = filter_messages(rows)
    assert len(kept) == 3
    kept, report = filter_messages(rows, FilterConfig(drop_commands=True, drop_url_only=True))
    assert [m["content"] for m in kept] == ["real text"]
    assert report.removed["command"] == 1 and report.removed["url_only"] == 1


def test_oversized_and_code_heavy_messages_are_opt_in():
    rows = messages("x" * 50, "```a```\n```b```")
    kept, report = filter_messages(rows, FilterConfig(max_message_chars=40, max_code_blocks=1))
    assert kept == []
    assert report.removed["oversized"] == 1 and report.removed["code_heavy"] == 1


def test_duplicate_messages_are_opt_in_and_case_insensitive():
    rows = messages("lol", "LOL ", "lmao")
    assert len(filter_messages(rows)[0]) == 3
    kept, report = filter_messages(rows, FilterConfig(drop_duplicate_messages=True))
    assert [m["content"] for m in kept] == ["lol", "lmao"]
    assert report.removed["duplicate_message"] == 1


def test_report_lists_every_reason_and_total():
    rows = messages("", "!cmd")
    rows[1]["bot"] = True
    _, report = filter_messages(rows, FilterConfig(drop_commands=True))
    rendered = report.render()
    assert "removed 2" in rendered
    assert report.as_dict()["by_reason"] == {"empty": 1, "bot": 1}


def test_negative_thresholds_are_rejected():
    with pytest.raises(ValueError, match="max-message-chars"):
        FilterConfig(max_message_chars=0)
