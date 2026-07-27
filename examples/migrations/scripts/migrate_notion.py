#!/usr/bin/env python3
"""
Migrate a Notion workspace export into Memanto.

Configure:
    ZIP_PATH = "/path/to/notion_export.zip"   # set this

Run:
    python scripts/migrate_notion.py [--dry-run] [--agent <id>]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── configure ────────────────────────────────────────────────────────────────
ZIP_PATH = "/path/to/notion_export.zip"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    """
    Migrate a Notion export ZIP into Memanto using the command-line interface.
    
    Returns:
        int: The Memanto command's exit code, or `1` when the export file is missing.
    """
    parser = argparse.ArgumentParser(description="Migrate Notion export to Memanto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--file", default=ZIP_PATH, help="Path to Notion export ZIP")
    args = parser.parse_args()

    if not Path(args.file).is_file():
        print(f"ZIP not found: {args.file}", file=sys.stderr)
        print("Set ZIP_PATH at the top of this script or pass --file.", file=sys.stderr)
        print("Export from: Notion settings → Settings & Members → Settings → Export content", file=sys.stderr)
        return 1

    cmd = ["memanto", "migrate", "notion", "--file", args.file]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.agent:
        cmd += ["--agent", args.agent]

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
