#!/usr/bin/env python3
"""
Connect to a running ChromaDB instance and migrate a collection into Memanto.

Requires:
    CHROMA_COLLECTION env var  — collection name to migrate
    CHROMA_HOST env var        — ChromaDB host (default: localhost)
    CHROMA_PORT env var        — ChromaDB port (default: 8000)

Run:
    python scripts/migrate_chroma.py [--dry-run] [--agent <id>]
"""

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate a Chroma collection to Memanto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--collection", default=None, help="Collection name (overrides CHROMA_COLLECTION env)")
    parser.add_argument("--host", default=None, help="ChromaDB host (overrides CHROMA_HOST env)")
    parser.add_argument("--port", default=None, help="ChromaDB port (overrides CHROMA_PORT env)")
    args = parser.parse_args()

    collection = args.collection or os.environ.get("CHROMA_COLLECTION", "")
    if not collection:
        print("CHROMA_COLLECTION is not set. Export it or pass --collection.", file=sys.stderr)
        return 1

    host = args.host or os.environ.get("CHROMA_HOST", "localhost")
    port = args.port or os.environ.get("CHROMA_PORT", "8000")

    cmd = [
        "memanto", "migrate", "chroma",
        "--collection", collection,
        "--host", host,
        "--port", str(port),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.agent:
        cmd += ["--agent", args.agent]

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
