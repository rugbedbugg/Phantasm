"""Standalone script to scrape Discord messages."""

import sys

from phantasm.scraper import fetch_messages

if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv or len(sys.argv) < 3:
        print("usage: python scrape_discord.py <channel_id> <token> [output.json]")
        sys.exit(0 if ("-h" in sys.argv or "--help" in sys.argv) else 1)

    channel = sys.argv[1]
    tok = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "raw_export.json"
    fetch_messages(channel, tok, out)
