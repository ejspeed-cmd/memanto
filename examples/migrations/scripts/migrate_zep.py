#!/usr/bin/env python3
"""
Migrate Zep Cloud graph edge facts into Memanto.

Requires:
    ZEP_API_KEY env var (exits when neither --api-key nor ZEP_API_KEY is provided)

Run:
    python scripts/migrate_zep.py [--dry-run] [--agent <id>]
"""

import argparse
import os
import subprocess
import sys


def main() -> int:
    """
    Run the Zep Cloud memory migration through Memanto.
    
    Returns:
        int: The subprocess exit code, or `1` when no Zep API key is provided.
    """
    parser = argparse.ArgumentParser(description="Migrate Zep Cloud memories to Memanto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--api-key", default=None, help="Zep API key (overrides ZEP_API_KEY env)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ZEP_API_KEY", "")
    if not api_key:
        print("ZEP_API_KEY is not set. Export it or pass --api-key.", file=sys.stderr)
        return 1

    cmd = ["memanto", "migrate", "zep"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.agent:
        cmd += ["--agent", args.agent]

    env = {**os.environ, "ZEP_API_KEY": api_key}
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
