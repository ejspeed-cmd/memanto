#!/usr/bin/env python3
"""
Dump a LangGraph store and migrate it into Memanto in one step.

Uses an InMemoryStore seeded with demo data when LANGGRAPH_POSTGRES_URI is
absent. Set LANGGRAPH_POSTGRES_URI to connect to a real Postgres-backed store.

Run:
    python scripts/migrate_langgraph.py [--dry-run] [--agent <id>]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump LangGraph store and migrate to Memanto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent", default=None)
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        dump_cmd = [sys.executable, str(_SCRIPTS / "dump_langgraph.py"), "--output", tmp_path]
        result = subprocess.run(dump_cmd)
        if result.returncode != 0:
            print("LangGraph dump failed.", file=sys.stderr)
            return 1

        migrate_cmd = ["memanto", "migrate", "langgraph", "--file", tmp_path]
        if args.dry_run:
            migrate_cmd.append("--dry-run")
        if args.agent:
            migrate_cmd += ["--agent", args.agent]

        return subprocess.run(migrate_cmd).returncode
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
