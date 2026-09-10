"""Standalone script to format training datasets."""

import argparse

from phantasm.cli import cmd_format

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format parsed Discord messages into fine-tuning datasets."
    )
    parser.add_argument("--input", default="parsed.json", help="Path to parsed.json")
    parser.add_argument("--output-prefix", default="dataset", help="Output file prefix")
    parser.add_argument(
        "--window", type=int, default=6, help="Conversation window size (number of turns)"
    )
    parser.add_argument(
        "--max-gap", type=int, default=300, help="Max gap in seconds to group consecutive messages"
    )
    parser.add_argument("--system-prompt", default="", help="Optional system prompt")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio")

    args = parser.parse_args()
    cmd_format(args)
