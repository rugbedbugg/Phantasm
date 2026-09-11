"""Unified command line interface for Phantasm.

Commands only parse arguments and dispatch; the work lives in the modules that
own it, so the library stays usable without argparse. Library code raises
exceptions and this entry point turns them into short messages and exit codes.
"""

import argparse
import json
import sys
from pathlib import Path

from phantasm import __version__, audit, evaluate, inspection
from phantasm.credentials import (
    STORABLE,
    clear_secret,
    describe_sources,
    prompt_value,
    read_secret,
    resolve_name,
    store_path,
    store_secret,
)
from phantasm.filtering import FilterConfig
from phantasm.formatter import FormatConfig, format_dataset, render_format_report
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
        self_ref=args.self_ref or args.username,
        target_ref=args.target_ref,
        others=args.others,
        ignore_others=args.ignore_others,
        output_path=args.output,
    )


def cmd_format(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    raw_msgs = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(raw_msgs, list):
        raise ValueError("Expected a message list or an object containing messages")
    print(f"Loaded {len(raw_msgs)} messages from {args.input}")
    config = FormatConfig(
        window_size=args.window,
        turn_gap_seconds=args.turn_gap,
        session_gap_seconds=args.session_gap,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        strategy=args.split_strategy,
        seed=args.split_seed,
        system_prompt=args.system_prompt,
        speaker_labels=args.speaker_labels,
        merge_unknown_timestamps=args.merge_unknown_timestamps,
        drop_duplicate_samples=not args.keep_duplicate_samples,
        filters=FilterConfig(
            drop_bots=not args.keep_bots,
            drop_deleted_placeholders=not args.keep_deleted,
            drop_commands=args.drop_commands,
            drop_url_only=args.drop_url_only,
            drop_duplicate_messages=args.drop_duplicate_messages,
            max_message_chars=args.max_message_chars,
            max_code_blocks=args.max_code_blocks,
        ),
    )
    manifest = format_dataset(raw_msgs, args.output_prefix, config)
    print(render_format_report(manifest))


def cmd_credentials(args: argparse.Namespace) -> None:
    """Save, list or remove locally stored credentials. Values are never shown."""
    if args.action == "list":
        print(f"{'Credential':<22}{'Name':<18}{'Source'}")
        for name, alias, source in describe_sources():
            print(f"{name:<22}{alias:<18}{source}")
        print(f"\nStore: {store_path()}")
        return
    if not args.name:
        raise ValueError(f"credentials {args.action} requires a name, for example: pixeldrain")
    name = resolve_name(args.name)
    if args.action == "clear":
        print(f"Removed {name}." if clear_secret(name) else f"{name} was not stored.")
        return
    # Secrets are typed at a hidden prompt, never passed as command-line arguments.
    path = store_secret(name, prompt_value(name, hidden=name in STORABLE))
    print(f"Saved {name} to {path} (owner-readable only).")


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


def add_parse_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input raw_export.json")
    parser.add_argument("username", nargs="?", help="Legacy positional spelling of --self")
    parser.add_argument(
        "--self",
        "--user-id",
        dest="self_ref",
        help="Your author ID or username (--user-id is accepted as the old spelling)",
    )
    parser.add_argument(
        "--target",
        "--target-id",
        dest="target_ref",
        help="Author ID or username of the persona to reconstruct",
    )
    parser.add_argument(
        "--others",
        choices=["context", "drop", "error"],
        default="context",
        help="Keep other participants as context (default), exclude them, or refuse to parse",
    )
    parser.add_argument(
        "--ignore-others",
        action="store_true",
        help="Deprecated alias for --others drop",
    )
    parser.add_argument("-o", "--output", default="parsed.json", help="Output parsed JSON path")


def add_format_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", default="parsed.json", help="Input parsed JSON path")
    parser.add_argument("-p", "--output-prefix", default="dataset", help="Output prefix for JSONL")
    parser.add_argument(
        "-w", "--window", type=int, default=6, help="Sliding window context size (min 2)"
    )
    parser.add_argument(
        "-g",
        "--turn-gap",
        "--max-gap",
        dest="turn_gap",
        type=int,
        default=300,
        help="Seconds within which one speaker's messages merge into a single turn",
    )
    parser.add_argument(
        "--session-gap", type=int, default=1800, help="Inactivity seconds separating sessions"
    )
    parser.add_argument("-s", "--system-prompt", default="", help="Optional system prompt")
    parser.add_argument(
        "-v",
        "--val-ratio",
        "--val-split",
        dest="val_ratio",
        type=float,
        default=0.075,
        help="Share of usable sessions held out for validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.075,
        help="Share of usable sessions held out for the persona evaluation test split",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["chronological", "random"],
        default="chronological",
        help="Hold out the most recent sessions (default) or a seeded random selection",
    )
    parser.add_argument("--split-seed", type=int, help="Required seed for --split-strategy random")
    parser.add_argument(
        "--speaker-labels",
        choices=["none", "others", "all"],
        default="none",
        help="Prefix context turns with the speaker's name (recommended for group chats)",
    )
    parser.add_argument(
        "--merge-unknown-timestamps",
        action="store_true",
        help="Allow messages without usable timestamps to merge into one turn",
    )
    parser.add_argument(
        "--keep-duplicate-samples",
        action="store_true",
        help="Keep exact duplicate training samples instead of removing them",
    )
    quality = parser.add_argument_group("data quality filters")
    quality.add_argument("--keep-bots", action="store_true", help="Keep messages flagged as bots")
    quality.add_argument(
        "--keep-deleted", action="store_true", help="Keep deleted-message placeholders"
    )
    quality.add_argument(
        "--drop-commands", action="store_true", help="Remove bot-command style messages"
    )
    quality.add_argument(
        "--drop-url-only", action="store_true", help="Remove messages that are only links"
    )
    quality.add_argument(
        "--drop-duplicate-messages",
        action="store_true",
        help="Keep only the first occurrence of each repeated message",
    )
    quality.add_argument(
        "--max-message-chars", type=int, help="Remove messages longer than this many characters"
    )
    quality.add_argument(
        "--max-code-blocks", type=int, help="Remove messages with more code blocks than this"
    )


def add_chat_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model", required=True, help="Path to local GGUF model file")
    parser.add_argument("-s", "--system-prompt", default="", help="Optional system prompt")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "-k", "--max-tokens", type=int, default=150, help="Maximum generated tokens"
    )
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-layers", type=int, default=0, help="Offloaded layers; -1 means all")


def build_parser() -> argparse.ArgumentParser:
    """Assemble the full command tree."""
    parser = SafeArgumentParser(
        prog="phantasm",
        description="Phantasm: conversational persona reconstruction, fine-tuning and evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_scrape = subparsers.add_parser("scrape", help="Scrape messages from Discord channel")
    p_scrape.add_argument("channel_id", help="Target channel ID (digits)")
    p_scrape.add_argument("--resume", action="store_true", help="Resume the output checkpoint")
    p_scrape.add_argument("-o", "--output", default="raw_export.json", help="Output JSON path")
    p_scrape.set_defaults(func=cmd_scrape)

    p_parse = subparsers.add_parser("parse", help="Resolve participants in a raw export")
    add_parse_arguments(p_parse)
    p_parse.set_defaults(func=cmd_parse)

    p_format = subparsers.add_parser("format", help="Build train/validation/test datasets")
    add_format_arguments(p_format)
    p_format.set_defaults(func=cmd_format)

    p_inspect = subparsers.add_parser("inspect", help="Report transcript or dataset statistics")
    inspection.add_arguments(p_inspect)
    p_inspect.set_defaults(func=inspection.command)

    p_audit = subparsers.add_parser("audit", help="Heuristic local privacy and quality audit")
    audit.add_arguments(p_audit)
    p_audit.set_defaults(func=audit.command)

    p_evaluate = subparsers.add_parser("evaluate", help="Measure persona fidelity on held-out data")
    evaluate.add_arguments(p_evaluate)
    p_evaluate.set_defaults(func=evaluate.command)

    p_credentials = subparsers.add_parser(
        "credentials", help="Save a Pixeldrain or Hugging Face credential for Phantasm to use"
    )
    p_credentials.add_argument("action", choices=["set", "list", "clear"])
    p_credentials.add_argument(
        "name",
        nargs="?",
        help="pixeldrain, huggingface, pixeldrain-domain, or the variable name",
    )
    p_credentials.set_defaults(func=cmd_credentials)

    p_chat = subparsers.add_parser("chat", help="Chat with fine-tuned model via llama.cpp")
    add_chat_arguments(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    from phantasm.training import add_training_arguments, train

    p_train = subparsers.add_parser("train", help="Fine-tune and export a GGUF model")
    add_training_arguments(p_train)
    p_train.set_defaults(func=train)

    from phantasm.colab import add_commands

    add_commands(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
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
