"""Unified command line interface for Phantasm."""

import argparse
import json
import sys
from pathlib import Path

from phantasm import __version__
from phantasm.formatter import (
    build_conversations,
    export_sharegpt,
    group_consecutive_messages,
)
from phantasm.inference import run_llama_cpp
from phantasm.parser import parse_export
from phantasm.scraper import fetch_messages


def cmd_scrape(args: argparse.Namespace) -> None:
    fetch_messages(
        channel_id=args.channel_id,
        token=args.token,
        output_path=args.output,
    )


def cmd_parse(args: argparse.Namespace) -> None:
    parse_export(
        input_path=args.input,
        your_username=args.username,
        output_path=args.output,
    )


def cmd_format(args: argparse.Namespace) -> None:
    input_file = Path(args.input)
    if not input_file.is_file():
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    if not 0.0 <= args.val_split < 1.0:
        print(f"Error: --val-split must be between 0.0 and 1.0, got {args.val_split}")
        sys.exit(1)

    try:
        data = json.loads(input_file.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"Error reading JSON from '{args.input}': {exc}")
        sys.exit(1)

    raw_msgs = (
        data.get("messages", [])
        if isinstance(data, dict)
        else (data if isinstance(data, list) else [])
    )
    print(f"Loaded {len(raw_msgs)} messages from {args.input}")

    grouped_turns = group_consecutive_messages(raw_msgs, max_gap_seconds=args.max_gap)
    print(f"Aggregated into {len(grouped_turns)} conversational turns")

    conversations = build_conversations(grouped_turns, window_size=args.window)
    print(f"Generated {len(conversations)} training samples (window_size={args.window})")

    if not conversations:
        print("Warning: No valid training conversations generated with the current parameters.")
        return

    val_count = int(len(conversations) * args.val_split)
    train_convos = conversations[:-val_count] if val_count > 0 else conversations
    val_convos = conversations[-val_count:] if val_count > 0 else []

    print(f"Train samples: {len(train_convos)}, Validation samples: {len(val_convos)}")

    train_out = f"{args.output_prefix}_train_sharegpt.jsonl"
    export_sharegpt(train_convos, train_out, args.system_prompt)
    if val_convos:
        export_sharegpt(val_convos, f"{args.output_prefix}_val_sharegpt.jsonl", args.system_prompt)
    print(f"Exported ShareGPT JSONL files: {args.output_prefix}_[train/val]_sharegpt.jsonl")


def cmd_chat(args: argparse.Namespace) -> None:
    run_llama_cpp(
        model_path=args.model,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phantasm",
        description="Phantasm: Pipeline for fine-tuning LLMs on conversational logs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Scrape
    p_scrape = subparsers.add_parser("scrape", help="Scrape messages from Discord channel")
    p_scrape.add_argument("channel_id", help="Target channel ID (digits)")
    p_scrape.add_argument("token", help="Discord user token")
    p_scrape.add_argument("-o", "--output", default="raw_export.json", help="Output JSON path")
    p_scrape.set_defaults(func=cmd_scrape)

    # Parse
    p_parse = subparsers.add_parser("parse", help="Parse raw export into participant turns")
    p_parse.add_argument("input", help="Input raw_export.json")
    p_parse.add_argument("username", help="Your Discord username (to distinguish roles)")
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
    p_chat.set_defaults(func=cmd_chat)

    try:
        args = parser.parse_args()
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
