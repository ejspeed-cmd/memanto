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
    """Migrate an Obsidian vault to Memanto using the command-line arguments.
    
    Returns:
    	int: The exit code from the migration command, or `1` if the vault directory is invalid.
    """
    parser = argparse.ArgumentParser(description="Migrate Obsidian vault to Memanto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--vault", default=VAULT_PATH, help="Path to Obsidian vault directory")
    args = parser.parse_args()

    if not Path(args.vault).is_dir():
        print(f"Vault directory not found: {args.vault}", file=sys.stderr)
        print("Set VAULT_PATH at the top of this script or pass --vault.", file=sys.stderr)
        return 1

    cmd = ["memanto", "migrate", "obsidian", args.vault]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.agent:
        cmd += ["--agent", args.agent]

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
