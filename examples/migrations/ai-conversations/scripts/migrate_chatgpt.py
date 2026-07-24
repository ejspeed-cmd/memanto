#!/usr/bin/env python3
"""
Migrate a ChatGPT conversation export into Memanto.

Configure:
    ZIP_PATH = "/path/to/chatgpt_export.zip"   # set this

Run:
    python scripts/migrate_chatgpt.py [--dry-run] [--agent <id>]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── configure ────────────────────────────────────────────────────────────────
ZIP_PATH = str(Path(__file__).parent.parent / "sample_data" / "chatgpt_export.zip")
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate ChatGPT export to Memanto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent", default=None)
    args = parser.parse_args()

    if not Path(ZIP_PATH).exists():
        print(f"ZIP not found: {ZIP_PATH}", file=sys.stderr)
        print("Set ZIP_PATH at the top of this script to your ChatGPT export zip.", file=sys.stderr)
        return 1

    cmd = ["memanto", "migrate", "conversations", ZIP_PATH, "--source", "chatgpt"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.agent:
        cmd += ["--agent", args.agent]

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
