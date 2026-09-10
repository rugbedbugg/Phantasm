"""Interactive chat interface for Phantasm models via local llama.cpp."""

import os
import sys
from pathlib import Path
from typing import Any


def run_llama_cpp(
    model_path: str,
    system_prompt: str = "",
    n_ctx: int = 2048,
    n_threads: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 150,
) -> None:
    """Run local inference loop safely using llama-cpp-python."""
    target_path = Path(model_path).expanduser().resolve()
    if not target_path.is_file():
        print(f"Error: Model file '{model_path}' does not exist.")
        print(f"Checked path: {target_path}")
        sys.exit(1)

    if target_path.stat().st_size == 0:
        print(f"Error: Model file '{model_path}' is empty (0 bytes).")
        sys.exit(1)

    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: llama-cpp-python is not installed.")
        print("Install with: pip install 'phantasm[inference]' or pip install llama-cpp-python")
        sys.exit(1)

    threads = max(1, min(n_threads, os.cpu_count() or 4))
    print(f"Loading local model from {target_path} (threads={threads}, ctx={n_ctx})...")

    try:
        llm = Llama(
            model_path=str(target_path),
            n_ctx=n_ctx,
            n_threads=threads,
            verbose=False,
        )
    except Exception as exc:
        print(f"Error initializing llama-cpp model: {exc}")
        print("Verify that the GGUF file is intact and you have sufficient RAM.")
        sys.exit(1)

    history: list[dict[str, Any]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    print("Phantasm local inference ready. Type /quit or exit to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "exit", "quit"):
            break

        history.append({"role": "user", "content": user_input})

        try:
            response = llm.create_chat_completion(
                messages=history,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            reply = response["choices"][0]["message"]["content"].strip()
            print(f"Phantasm: {reply}\n")
            history.append({"role": "assistant", "content": reply})
        except Exception as exc:
            print(f"\nInference error: {exc}\n")
