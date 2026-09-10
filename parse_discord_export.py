"""Standalone script to parse raw Discord exports."""

import sys

from phantasm.parser import parse_export

if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv or len(sys.argv) < 3:
        print("usage: python parse_discord_export.py <export.json> <your_username> [output.json]")
        sys.exit(0 if ("-h" in sys.argv or "--help" in sys.argv) else 1)

    inp = sys.argv[1]
    usr = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "parsed.json"
    parse_export(inp, usr, out)
