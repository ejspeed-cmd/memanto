#!/usr/bin/env python3
"""
Migrate a Gemini Takeout export into Memanto.

Configure:
    ZIP_PATH = "/path/to/takeout-*.zip"   # set this

Run:
    python scripts/migrate_gemini.py [--dry-run] [--agent <id>]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── configure ────────────────────────────────────────────────────────────────
ZIP_PATH = str(Path(__file__).parent.parent / "sample_data" / "gemini_export.zip")
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate Gemini Takeout export to Memanto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent", default=None)
    args = parser.parse_args()

    if not Path(ZIP_PATH).exists():
        print(f"ZIP not found: {ZIP_PATH}", file=sys.stderr)
        print("Set ZIP_PATH at the top of this script to your Gemini Takeout zip.", file=sys.stderr)
        return 1

    cmd = ["memanto", "migrate", "conversations", ZIP_PATH, "--source", "gemini"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.agent:
        cmd += ["--agent", args.agent]

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
