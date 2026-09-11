"""Compatibility entry point; accepts the same options as phantasm chat."""

import sys

from phantasm.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "chat")
    main()
