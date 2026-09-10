"""Standalone script for model inference via llama.cpp."""

import argparse

from phantasm.inference import run_llama_cpp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with fine-tuned Phantasm model.")
    parser.add_argument("--model", type=str, required=True, help="Path to local GGUF model file")
    parser.add_argument("--system-prompt", type=str, default="", help="Optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens to generate")

    args = parser.parse_args()
    run_llama_cpp(
        model_path=args.model,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
