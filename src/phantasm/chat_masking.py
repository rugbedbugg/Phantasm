"""Derive the chat-template markers that restrict training loss to target responses.

Phantasm learns ``conversation context -> target persona's response``. Computing
loss over the context as well teaches the model to imitate everybody in the
transcript, so training masks the prompt and keeps loss on the assistant spans
only.

Unsloth's :func:`train_on_responses_only` performs the masking; it needs the two
literal strings a chat template emits before a user turn and before an assistant
turn. Hardcoding Llama-3 headers would silently mislabel any other template, so
the markers are derived from the tokenizer's own chat template at run time and
verified before training starts:

* the **response** marker is exactly what the template appends when asked for a
  generation prompt;
* the **instruction** marker is the text every user turn is introduced by, cut
  back to the same structural delimiter the response marker starts with.
"""

from collections.abc import Callable

#: Sentinel contents that cannot appear in a template's own markup. The two in
#: each pair end with different characters so a shared suffix can never leak
#: into a derived marker.
_USER_SENTINELS = ("\x01phantasm-user-a", "\x01phantasm-user-b")
_ASSISTANT_SENTINELS = ("\x02phantasm-reply-a", "\x02phantasm-reply-b")

#: ``render(conversation, add_generation_prompt=False) -> str``
Renderer = Callable[..., str]


def longest_common_suffix(values: list[str]) -> str:
    """The longest string that ends every input."""
    if not values:
        return ""
    shortest = min(len(value) for value in values)
    length = 0
    while length < shortest and len({value[-length - 1] for value in values}) == 1:
        length += 1
    return values[0][len(values[0]) - length :] if length else ""


def _prefixes(rendered: str, sentinels: tuple[str, ...]) -> list[str]:
    prefixes = []
    for sentinel in sentinels:
        position = rendered.find(sentinel)
        if position < 0:
            raise ValueError("The chat template dropped message content; cannot mask responses")
        if rendered.find(sentinel, position + 1) >= 0:
            raise ValueError("The chat template repeats message content; cannot mask responses")
        prefixes.append(rendered[:position])
    return prefixes


def align_to_delimiter(marker: str, reference: str) -> str:
    """Trim ``marker`` so it starts at the delimiter ``reference`` also starts with.

    A raw common suffix can begin mid-token, for example at the ``>`` shared by
    ``<|begin_of_text|>`` and ``<|eot_id|>``. The assistant marker always begins
    with the template's turn delimiter (``<|start_header_id|>``, ``<|im_start|>``,
    ``<start_of_turn>``, ``[``...), so the longest prefix of the assistant marker
    that occurs in ``marker`` locates the same delimiter for the user turn.
    """
    delimiter = ""
    for length in range(1, len(reference) + 1):
        if reference[:length] in marker:
            delimiter = reference[:length]
        else:
            break
    # A whitespace-only delimiter is ordinary spacing, not a turn delimiter, and
    # trimming to it would destroy the marker.
    if not delimiter.strip():
        return marker
    trimmed = marker[marker.find(delimiter) :]
    return trimmed if trimmed.strip() else marker


def derive_response_markers(render: Renderer) -> tuple[str, str]:
    """Return ``(instruction_part, response_part)`` for the given chat template.

    ``render`` receives a conversation of ``{"role", "content"}`` messages plus an
    ``add_generation_prompt`` keyword, exactly like
    ``tokenizer.apply_chat_template(..., tokenize=False)``.
    """
    conversation = [
        {"role": "user", "content": _USER_SENTINELS[0]},
        {"role": "assistant", "content": _ASSISTANT_SENTINELS[0]},
        {"role": "user", "content": _USER_SENTINELS[1]},
        {"role": "assistant", "content": _ASSISTANT_SENTINELS[1]},
    ]
    rendered = render(conversation, add_generation_prompt=False)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("The chat template produced no text")
    opening = [{"role": "user", "content": _USER_SENTINELS[0]}]
    plain = render(opening, add_generation_prompt=False)
    prompted = render(opening, add_generation_prompt=True)
    if isinstance(prompted, str) and isinstance(plain, str) and prompted.startswith(plain):
        response = prompted[len(plain) :]
    else:
        response = ""
    if not response.strip():
        response = longest_common_suffix(_prefixes(rendered, _ASSISTANT_SENTINELS))
    instruction = align_to_delimiter(
        longest_common_suffix(_prefixes(rendered, _USER_SENTINELS)), response
    )
    if not instruction.strip() or not response.strip():
        raise ValueError(
            "Could not derive user/assistant markers from the chat template; "
            "train with --loss full or choose a model with a standard chat template"
        )
    if instruction in response or response in instruction:
        raise ValueError(
            "The chat template's user and assistant markers are not distinguishable; "
            "train with --loss full"
        )
    if rendered.count(instruction) != len(_USER_SENTINELS) or rendered.count(response) != len(
        _ASSISTANT_SENTINELS
    ):
        raise ValueError(
            "The derived chat markers do not appear once per turn; train with --loss full"
        )
    return instruction, response


def masked_preview(rendered: str, instruction: str, response: str) -> list[tuple[int, int]]:
    """Spans of ``rendered`` that keep training loss under response-only masking.

    Each span starts after an assistant marker and ends at the next user marker
    or the end of the text, mirroring what the trainer's collator keeps. This
    lets preprocessing be checked without a GPU.
    """
    spans: list[tuple[int, int]] = []
    position = rendered.find(response)
    while position >= 0:
        start = position + len(response)
        following = rendered.find(instruction, start)
        spans.append((start, following if following >= 0 else len(rendered)))
        position = rendered.find(response, start)
    return spans


def verify_masking(rendered: str, instruction: str, response: str, answers: list[str]) -> None:
    """Fail loudly if the markers would not put loss on the expected answers.

    ``answers`` are the assistant messages of the rendered conversation, in
    order. Each must fall inside its own unmasked span, and no context turn may
    appear inside one.
    """
    spans = masked_preview(rendered, instruction, response)
    if len(spans) != len(answers):
        raise ValueError(
            f"Response masking found {len(spans)} assistant span(s) for {len(answers)} answer(s)"
        )
    for (start, end), answer in zip(spans, answers, strict=True):
        if answer.strip() and answer not in rendered[start:end]:
            raise ValueError("Response masking would exclude a target response")
