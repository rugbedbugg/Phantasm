"""Unified command line interface for Phantasm."""

import argparse
import json
import sys
from pathlib import Path

from phantasm import __version__
from phantasm.credentials import read_secret
from phantasm.formatter import (
    export_sharegpt,
    group_consecutive_messages,
    split_conversations,
)
from phantasm.inference import run_llama_cpp
from phantasm.parser import parse_export
from phantasm.scraper import fetch_messages


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        if message.startswith("unrecognized arguments:"):
            message = "Unrecognized arguments; use --help. Credentials belong in environment variables or hidden prompts."
        super().error(message)


def cmd_scrape(args: argparse.Namespace) -> None:
    fetch_messages(
        channel_id=args.channel_id,
        token=read_secret("DISCORD_TOKEN", "Discord token: "),
        resume=args.resume,
        output_path=args.output,
    )


def cmd_parse(args: argparse.Namespace) -> None:
    parse_export(
        input_path=args.input,
        your_username=args.username,
        user_id=args.user_id,
        target_id=args.target_id,
        ignore_others=args.ignore_others,
        output_path=args.output,
    )


def cmd_format(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    raw_msgs = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(raw_msgs, list):
        raise ValueError("Expected a message list or an object containing messages")
    print(f"Loaded {len(raw_msgs)} messages from {args.input}")

    grouped_turns = group_consecutive_messages(
        raw_msgs, max_gap_seconds=args.max_gap, session_gap_seconds=args.session_gap
    )
    train_convos, val_convos = split_conversations(grouped_turns, args.val_split, args.window)
    train_out = f"{args.output_prefix}_train_sharegpt.jsonl"
    val_out = f"{args.output_prefix}_val_sharegpt.jsonl"
    export_sharegpt(train_convos, train_out, args.system_prompt)
    # Always replace validation output, including an empty split, to prevent stale data.
    export_sharegpt(val_convos, val_out, args.system_prompt)
    print(f"Exported {len(train_convos)} training and {len(val_convos)} validation samples")


def cmd_chat(args: argparse.Namespace) -> None:
    run_llama_cpp(
        model_path=args.model,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n_ctx=args.context_size,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
    )


def main() -> None:
    parser = SafeArgumentParser(
        prog="phantasm",
        description="Phantasm: Pipeline for fine-tuning LLMs on conversational logs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Scrape
    p_scrape = subparsers.add_parser("scrape", help="Scrape messages from Discord channel")
    p_scrape.add_argument("channel_id", help="Target channel ID (digits)")
    p_scrape.add_argument("--resume", action="store_true", help="Resume the output checkpoint")
    p_scrape.add_argument("-o", "--output", default="raw_export.json", help="Output JSON path")
    p_scrape.set_defaults(func=cmd_scrape)

    # Parse
    p_parse = subparsers.add_parser("parse", help="Parse raw export into participant turns")
    p_parse.add_argument("input", help="Input raw_export.json")
    p_parse.add_argument("username", nargs="?", help="Legacy username for two-person exports")
    p_parse.add_argument("--user-id", help="Your immutable Discord author ID")
    p_parse.add_argument("--target-id", help="Target persona author ID")
    p_parse.add_argument(
        "--ignore-others", action="store_true", help="Exclude other authors and break context"
    )
    p_parse.add_argument("-o", "--output", default="parsed.json", help="Output parsed JSON path")
    p_parse.set_defaults(func=cmd_parse)

    # Format
    p_format = subparsers.add_parser("format", help="Format parsed messages into training datasets")
    p_format.add_argument("-i", "--input", default="parsed.json", help="Input parsed JSON path")
    p_format.add_argument(
        "-p", "--output-prefix", default="dataset", help="Output prefix for JSONL"
    )
    p_format.add_argument(
        "-w", "--window", type=int, default=6, help="Sliding window context size (min 2)"
    )
    p_format.add_argument(
        "-g", "--max-gap", type=int, default=300, help="Max gap in seconds to group turns"
    )
    p_format.add_argument(
        "--session-gap", type=int, default=1800, help="Inactivity seconds separating sessions"
    )
    p_format.add_argument("-s", "--system-prompt", default="", help="Optional system prompt")
    p_format.add_argument(
        "-v", "--val-split", type=float, default=0.05, help="Validation split ratio (0.0 - 1.0)"
    )
    p_format.set_defaults(func=cmd_format)

    # Chat / Infer
    p_chat = subparsers.add_parser("chat", help="Chat with fine-tuned model via llama.cpp")
    p_chat.add_argument("-m", "--model", required=True, help="Path to local GGUF model file")
    p_chat.add_argument("-s", "--system-prompt", default="", help="Optional system prompt")
    p_chat.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
    p_chat.add_argument(
        "-k", "--max-tokens", type=int, default=150, help="Maximum generated tokens"
    )
    p_chat.add_argument("--context-size", type=int, default=2048)
    p_chat.add_argument("--threads", type=int, default=4)
    p_chat.add_argument("--gpu-layers", type=int, default=0, help="Offloaded layers; -1 means all")
    p_chat.set_defaults(func=cmd_chat)

    from phantasm.training import add_training_arguments, train

    p_train = subparsers.add_parser("train", help="Fine-tune and export a GGUF model")
    add_training_arguments(p_train)
    p_train.set_defaults(func=train)

    from phantasm.colab import add_commands

    add_commands(subparsers)

    try:
        args = parser.parse_args()
        args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
