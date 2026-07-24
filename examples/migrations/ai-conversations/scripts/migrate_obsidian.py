#!/usr/bin/env python3
"""
Migrate an Obsidian vault into Memanto.

Configure:
    VAULT_PATH = "/path/to/your/vault"   # set this

Run:
    python scripts/migrate_obsidian.py [--dry-run] [--agent <id>]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── configure ────────────────────────────────────────────────────────────────
VAULT_PATH = "/path/to/your/obsidian/vault"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate Obsidian vault to Memanto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent", default=None)
    args = parser.parse_args()

    if not Path(VAULT_PATH).is_dir():
        print(f"Vault directory not found: {VAULT_PATH}", file=sys.stderr)
        print("Set VAULT_PATH at the top of this script to your Obsidian vault directory.", file=sys.stderr)
        return 1

    cmd = ["memanto", "migrate", "obsidian", "--file", VAULT_PATH]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.agent:
        cmd += ["--agent", args.agent]

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
