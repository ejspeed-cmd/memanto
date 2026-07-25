#!/usr/bin/env python3
"""
Standalone showcase runner for the ai-conversations migration example.

Runs `memanto migrate conversations --dry-run` for chatgpt, claude and gemini
sources against the sample_data/ ZIPs, then prints a per-source summary table.
Also runs `memanto migrate langgraph --dry-run` if scripts/langgraph_seed.json
exists.

No live Memanto server or API key is needed when only --dry-run is used.

Usage:
    python migrate.py
    python migrate.py --live --agent my-agent-id     # live migration
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).parent
_SAMPLE = _HERE / "sample_data"
_SCRIPTS = _HERE / "scripts"


def _run(cmd: list[str], extra_env: dict | None = None) -> subprocess.CompletedProcess:
    import os
    env = {**os.environ, **(extra_env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _parse_summary(stdout: str) -> dict:
    summary = {"source_records": 0, "mapped": 0, "skipped": 0, "types": {}}
    for line in stdout.splitlines():
        stripped = line.strip()
        if "Source records:" in stripped:
            try:
                summary["source_records"] = int(stripped.split("Source records:")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "Mapped memories:" in stripped:
            try:
                parts = stripped.split("Mapped memories:")[-1].strip().split()
                summary["mapped"] = int(parts[0])
                if "(skipped" in stripped:
                    summary["skipped"] = int(stripped.split("(skipped")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "Type breakdown:" in stripped:
            raw = stripped.split("Type breakdown:")[-1].strip()
            for part in raw.split(","):
                part = part.strip()
                if ":" in part:
                    k, v = part.split(":", 1)
                    try:
                        summary["types"][k.strip()] = int(v.strip())
                    except ValueError:
                        pass
    return summary


def _run_conversation(source: str, agent: str | None, dry_run: bool) -> dict | None:
    zip_path = _SAMPLE / f"{source}_export.zip"
    if not zip_path.exists():
        print(f"  [skip] {zip_path} not found")
        return None

    cmd = [
        sys.executable, "-m", "memanto.cli.main",
        "migrate", "conversations", str(zip_path),
        "--source", source,
    ]
    if dry_run:
        cmd.append("--dry-run")
    if agent:
        cmd += ["--agent", agent]

    result = _run(cmd)
    combined = result.stdout + result.stderr
    summary = _parse_summary(combined)
    summary["exit_code"] = result.returncode
    summary["error"] = None

    if result.returncode != 0 and not dry_run:
        for line in combined.splitlines():
            if "error" in line.lower() or "failed" in line.lower():
                summary["error"] = line.strip()
                break

    return summary


def _run_langgraph(agent: str | None, dry_run: bool) -> dict | None:
    seed = _SCRIPTS / "langgraph_seed.json"
    if not seed.exists():
        return None

    cmd = [
        sys.executable, "-m", "memanto.cli.main",
        "migrate", "langgraph", "--file", str(seed),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if agent:
        cmd += ["--agent", agent]

    result = _run(cmd)
    combined = result.stdout + result.stderr
    summary = _parse_summary(combined)
    summary["exit_code"] = result.returncode
    summary["error"] = None
    return summary


def _print_table(rows: list[tuple]) -> None:
    headers = ["source", "records", "mapped", "skipped", "types", "status"]
    widths = [12, 9, 8, 9, 28, 8]

    def fmt(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    print()
    print(fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt(row))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Showcase migration runner")
    parser.add_argument("--agent", default=None, help="Target agent ID (omit for dry-run)")
    parser.add_argument("--live", action="store_true", help="Run live migration (requires --agent and MOORCHEH_API_KEY)")
    args = parser.parse_args()

    dry_run = not args.live
    agent = args.agent

    if args.live and not agent:
        print("--live requires --agent <id>", file=sys.stderr)
        return 1

    mode = "dry-run preview" if dry_run else f"live migration → {agent}"
    print(f"\nai-conversations migration showcase  [{mode}]")
    print("=" * 60)

    rows = []
    sources = ["chatgpt", "claude", "gemini"]

    for source in sources:
        print(f"  running {source}...")
        s = _run_conversation(source, agent, dry_run)
        if s is None:
            rows.append((source, "—", "—", "—", "sample zip missing", "SKIP"))
            continue
        types_str = ", ".join(f"{k}:{v}" for k, v in s["types"].items()) or "auto"
        status = "OK" if s["exit_code"] == 0 else "FAIL"
        rows.append((source, s["source_records"], s["mapped"], s["skipped"], types_str, status))

    lg = _run_langgraph(agent, dry_run)
    if lg is not None:
        print("  running langgraph...")
        types_str = ", ".join(f"{k}:{v}" for k, v in lg["types"].items()) or "auto"
        status = "OK" if lg["exit_code"] == 0 else "FAIL"
        rows.append(("langgraph", lg["source_records"], lg["mapped"], lg["skipped"], types_str, status))

    _print_table(rows)

    ok = all(r[5] in ("OK", "SKIP") for r in rows)
    if not ok:
        failed = [r[0] for r in rows if r[5] == "FAIL"]
        print(f"Failed sources: {', '.join(failed)}", file=sys.stderr)
        return 1

    if dry_run:
        print("Dry-run complete. Pass --live --agent <id> to write memories to Memanto.")
    else:
        print("Migration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
