"""Response-only loss: marker derivation and preprocessing checks without a GPU."""

import pytest

from phantasm.chat_masking import (
    align_to_delimiter,
    derive_response_markers,
    longest_common_suffix,
    masked_preview,
    verify_masking,
)
from phantasm.training import assistant_messages

LLAMA_HEADERS = {"user": "user", "assistant": "assistant", "system": "system"}


def llama3(conversation, add_generation_prompt=False):
    """A faithful reproduction of the Llama 3.1 chat template's text output."""
    text = "<|begin_of_text|>"
    for turn in conversation:
        text += (
            f"<|start_header_id|>{LLAMA_HEADERS[turn['role']]}<|end_header_id|>\n\n"
            f"{turn['content']}<|eot_id|>"
        )
    if add_generation_prompt:
        text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return text


def chatml(conversation, add_generation_prompt=False):
    text = "".join(
        f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n" for turn in conversation
    )
    return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def gemma(conversation, add_generation_prompt=False):
    roles = {"user": "user", "assistant": "model", "system": "user"}
    text = "<bos>" + "".join(
        f"<start_of_turn>{roles[turn['role']]}\n{turn['content']}<end_of_turn>\n"
        for turn in conversation
    )
    return text + ("<start_of_turn>model\n" if add_generation_prompt else "")


def no_generation_prompt(conversation, add_generation_prompt=False):
    """A template that ignores ``add_generation_prompt``, forcing the fallback path."""
    return "<s>" + "".join(
        f"[{turn['role'].upper()}] {turn['content']} [/{turn['role'].upper()}]"
        for turn in conversation
    )


def test_longest_common_suffix():
    assert longest_common_suffix(["abcxyz", "defxyz"]) == "xyz"
    assert longest_common_suffix(["abc", "def"]) == ""
    assert longest_common_suffix([]) == ""


def test_align_to_delimiter_trims_a_partial_token():
    assert (
        align_to_delimiter(
            "><|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        == "<|start_header_id|>user<|end_header_id|>\n\n"
    )
    assert align_to_delimiter("no-delimiter-here", "<|x|>") == "no-delimiter-here"


def test_llama3_markers_match_the_documented_strings():
    instruction, response = derive_response_markers(llama3)
    assert instruction == "<|start_header_id|>user<|end_header_id|>\n\n"
    assert response == "<|start_header_id|>assistant<|end_header_id|>\n\n"


@pytest.mark.parametrize("template", [llama3, chatml, gemma, no_generation_prompt])
def test_markers_are_derived_for_several_template_families(template):
    instruction, response = derive_response_markers(template)
    assert instruction and response
    assert instruction not in response and response not in instruction
    rendered = template(
        [
            {"role": "user", "content": "hey"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "see you"},
        ]
    )
    assert rendered.count(instruction) == 2
    assert rendered.count(response) == 2


@pytest.mark.parametrize("template", [llama3, chatml, gemma])
def test_masking_keeps_only_target_responses(template):
    instruction, response = derive_response_markers(template)
    conversation = [
        {"role": "user", "content": "context one"},
        {"role": "assistant", "content": "persona one"},
        {"role": "user", "content": "context two"},
        {"role": "assistant", "content": "persona two"},
    ]
    rendered = template(conversation)
    spans = masked_preview(rendered, instruction, response)
    kept = "".join(rendered[start:end] for start, end in spans)
    assert "persona one" in kept and "persona two" in kept
    assert "context one" not in kept and "context two" not in kept
    verify_masking(rendered, instruction, response, ["persona one", "persona two"])


def test_masking_with_a_system_prompt_still_excludes_context():
    instruction, response = derive_response_markers(llama3)
    conversation = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "context"},
        {"role": "assistant", "content": "reply"},
    ]
    rendered = llama3(conversation)
    kept = "".join(
        rendered[start:end] for start, end in masked_preview(rendered, instruction, response)
    )
    assert "reply" in kept
    assert "be brief" not in kept and "context" not in kept


def test_verification_fails_when_answers_do_not_line_up():
    instruction, response = derive_response_markers(llama3)
    rendered = llama3([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    with pytest.raises(ValueError, match="assistant span"):
        verify_masking(rendered, instruction, response, ["a", "b"])
    with pytest.raises(ValueError, match="exclude a target response"):
        verify_masking(rendered, instruction, response, ["missing answer"])


def test_templates_that_hide_content_are_rejected():
    with pytest.raises(ValueError, match="dropped message content"):
        derive_response_markers(lambda conversation, add_generation_prompt=False: "static text")


def test_empty_template_output_is_rejected():
    with pytest.raises(ValueError, match="produced no text"):
        derive_response_markers(lambda conversation, add_generation_prompt=False: "")


def test_indistinguishable_roles_are_rejected():
    def same_marker(conversation, add_generation_prompt=False):
        text = "".join(f"<|turn|>{turn['content']}" for turn in conversation)
        return text + ("<|turn|>" if add_generation_prompt else "")

    with pytest.raises(ValueError, match="not distinguishable"):
        derive_response_markers(same_marker)


def test_assistant_messages_accepts_both_dataset_shapes():
    assert assistant_messages([{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]) == [
        "a"
    ]
    assert assistant_messages(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    ) == ["a"]
