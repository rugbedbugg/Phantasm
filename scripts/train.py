"""Compatibility wrapper for the packaged phantasm train command."""

import sys

from phantasm.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "train")
    main()
