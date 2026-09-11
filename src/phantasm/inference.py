"""Local chat with exact prompt budgeting and transactional history updates."""

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


def trim_history(messages: list[dict[str, str]], count_tokens: Callable, budget: int) -> list:
    """Remove oldest complete exchanges, preserving the system and newest user message."""
    candidate = list(messages)
    first = 1 if candidate and candidate[0]["role"] == "system" else 0
    while count_tokens(candidate) > budget:
        if len(candidate) - first < 3:
            raise ValueError("Your message and system prompt exceed the context budget")
        if [m["role"] for m in candidate[first : first + 2]] != ["user", "assistant"]:
            raise ValueError("History must contain complete user/assistant exchanges")
        del candidate[first : first + 2]
    return candidate


def chat_loop(
    llm: Any,
    formatter: Any,
    system_prompt: str,
    n_ctx: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    history = [{"role": "system", "content": system_prompt}] if system_prompt else []

    def count(messages):
        rendered = formatter(messages=messages)
        return len(
            llm.tokenize(
                rendered.prompt.encode("utf-8"), add_bos=not rendered.added_special, special=True
            )
        )

    print("Phantasm ready. Type /quit to exit or /reset to clear history.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("/quit", "quit", "exit"):
            break
        if user_input == "/reset":
            history = [{"role": "system", "content": system_prompt}] if system_prompt else []
            continue
        if not user_input:
            continue
        try:
            candidate = trim_history(
                history + [{"role": "user", "content": user_input}], count, n_ctx - max_tokens
            )
            response = llm.create_chat_completion(
                messages=candidate, max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )
            reply = response["choices"][0]["message"]["content"]
            if not isinstance(reply, str) or not reply.strip():
                raise ValueError("Model returned an empty text response")
            history = candidate + [{"role": "assistant", "content": reply.strip()}]
            print(f"Phantasm: {reply.strip()}\n")
        except Exception as exc:
            print(f"Inference error: {exc}")


def run_llama_cpp(
    model_path: str,
    system_prompt: str = "",
    n_ctx: int = 2048,
    n_threads: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 150,
    n_gpu_layers: int = 0,
) -> None:
    target = Path(model_path).expanduser().resolve()
    if not target.is_file() or not target.stat().st_size:
        raise ValueError(f"Model must be a nonempty GGUF file: {target}")
    if not 0 < max_tokens < n_ctx or n_threads < 1 or n_gpu_layers < -1:
        raise ValueError("Require 0 < max-tokens < context-size, threads >= 1 and gpu-layers >= -1")
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling parameters")
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    except ImportError:
        raise RuntimeError(
            "Install local inference with: uv sync --locked --extra inference"
        ) from None
    llm = Llama(
        model_path=str(target),
        n_ctx=n_ctx,
        n_threads=min(n_threads, os.cpu_count() or 1),
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    try:
        template = llm.metadata.get("tokenizer.chat_template")
        if not template:
            raise ValueError(
                "GGUF must include tokenizer.chat_template; re-export with its tokenizer"
            )

        def special(token_id):
            return llm.detokenize([token_id], special=True).decode("utf-8") if token_id >= 0 else ""

        formatter = Jinja2ChatFormatter(
            template=template,
            eos_token=special(llm.token_eos()),
            bos_token=special(llm.token_bos()),
            stop_token_ids=[llm.token_eos()],
        )
        # Count and generate with the same formatter, including template overhead.
        llm.chat_handler = formatter.to_chat_handler()
        chat_loop(llm, formatter, system_prompt, n_ctx, max_tokens, temperature, top_p)
    finally:
        llm.close()
