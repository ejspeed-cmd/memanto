"""Tests for the ai-conversations migration showcase runner (migrate.py).

migrate.py is a standalone script (not part of the memanto package), so it is
loaded directly from its file path via importlib, mirroring the pattern used
by other examples/*/tests suites in this repository.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "migrate.py"


def _load_migrate():
    spec = importlib.util.spec_from_file_location("_ai_conversations_migrate", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ai_conversations_migrate"] = module
    spec.loader.exec_module(module)
    return module


migrate = _load_migrate()


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr)


class TestParseSummary:
    def test_parses_source_records(self) -> None:
        stdout = "│ Source records: 5                                            │"
        summary = migrate._parse_summary(stdout)
        assert summary["source_records"] == 5

    def test_parses_mapped_and_skipped(self) -> None:
        stdout = "│ Mapped memories: 5  (skipped 2 empty)                        │"
        summary = migrate._parse_summary(stdout)
        assert summary["mapped"] == 5
        assert summary["skipped"] == 2

    def test_mapped_without_skipped_leaves_default_zero(self) -> None:
        stdout = "Mapped memories: 3"
        summary = migrate._parse_summary(stdout)
        assert summary["mapped"] == 3
        assert summary["skipped"] == 0

    def test_parses_type_breakdown(self) -> None:
        stdout = "Type breakdown: auto: 5"
        summary = migrate._parse_summary(stdout)
        assert summary["types"] == {"auto": 5}

    def test_parses_multiple_types_in_breakdown(self) -> None:
        stdout = "Type breakdown: fact: 3, event: 2"
        summary = migrate._parse_summary(stdout)
        assert summary["types"] == {"fact": 3, "event": 2}

    def test_full_multiline_output(self) -> None:
        stdout = (
            "Source records: 5\n"
            "Mapped memories: 5  (skipped 0 empty)\n"
            "Type breakdown: auto: 5\n"
        )
        summary = migrate._parse_summary(stdout)
        assert summary == {
            "source_records": 5,
            "mapped": 5,
            "skipped": 0,
            "types": {"auto": 5},
        }

    def test_empty_stdout_returns_defaults(self) -> None:
        summary = migrate._parse_summary("")
        assert summary == {
            "source_records": 0,
            "mapped": 0,
            "skipped": 0,
            "types": {},
        }

    def test_malformed_source_records_is_ignored(self) -> None:
        # Non-numeric value after the label must not raise.
        stdout = "Source records: not-a-number"
        summary = migrate._parse_summary(stdout)
        assert summary["source_records"] == 0

    def test_malformed_type_breakdown_entry_is_skipped(self) -> None:
        stdout = "Type breakdown: auto: five, fact: 2"
        summary = migrate._parse_summary(stdout)
        # "auto: five" fails int() conversion and is skipped; "fact: 2" is kept.
        assert summary["types"] == {"fact": 2}

    def test_type_breakdown_entry_without_colon_is_ignored(self) -> None:
        stdout = "Type breakdown: auto5"
        summary = migrate._parse_summary(stdout)
        assert summary["types"] == {}


class TestRun:
    def test_run_captures_stdout_and_returncode(self) -> None:
        result = migrate._run([sys.executable, "-c", "print('hello')"])
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_run_merges_extra_env(self) -> None:
        result = migrate._run(
            [sys.executable, "-c", "import os; print(os.environ.get('MY_TEST_VAR', ''))"],
            extra_env={"MY_TEST_VAR": "showcase-value"},
        )
        assert "showcase-value" in result.stdout

    def test_run_captures_nonzero_exit_code(self) -> None:
        result = migrate._run([sys.executable, "-c", "import sys; sys.exit(3)"])
        assert result.returncode == 3


class TestRunConversation:
    def test_missing_zip_returns_none(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setattr(migrate, "_SAMPLE", tmp_path)
        result = migrate._run_conversation("chatgpt", agent=None, dry_run=True)
        assert result is None
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_builds_expected_command_and_summary(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(migrate, "_SAMPLE", tmp_path)
        zip_path = tmp_path / "chatgpt_export.zip"
        zip_path.write_bytes(b"fake-zip-contents")

        captured_cmd = {}

        def fake_run(cmd, extra_env=None):
            captured_cmd["cmd"] = cmd
            return _completed(
                stdout="Source records: 5\nMapped memories: 5  (skipped 0 empty)\nType breakdown: auto: 5\n",
                returncode=0,
            )

        monkeypatch.setattr(migrate, "_run", fake_run)

        result = migrate._run_conversation("chatgpt", agent="agent-1", dry_run=True)

        assert result is not None
        assert result["source_records"] == 5
        assert result["mapped"] == 5
        assert result["exit_code"] == 0
        assert result["error"] is None

        cmd = captured_cmd["cmd"]
        assert cmd[0] == sys.executable
        assert "migrate" in cmd
        assert "conversations" in cmd
        assert str(zip_path) in cmd
        assert "--source" in cmd
        assert "chatgpt" in cmd
        assert "--dry-run" in cmd
        assert "--agent" in cmd
        assert "agent-1" in cmd

    def test_omits_dry_run_flag_when_live(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(migrate, "_SAMPLE", tmp_path)
        zip_path = tmp_path / "claude_export.zip"
        zip_path.write_bytes(b"fake")

        captured_cmd = {}

        def fake_run(cmd, extra_env=None):
            captured_cmd["cmd"] = cmd
            return _completed(returncode=0)

        monkeypatch.setattr(migrate, "_run", fake_run)

        migrate._run_conversation("claude", agent="agent-1", dry_run=False)
        assert "--dry-run" not in captured_cmd["cmd"]

    def test_omits_agent_flag_when_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(migrate, "_SAMPLE", tmp_path)
        zip_path = tmp_path / "gemini_export.zip"
        zip_path.write_bytes(b"fake")

        captured_cmd = {}

        def fake_run(cmd, extra_env=None):
            captured_cmd["cmd"] = cmd
            return _completed(returncode=0)

        monkeypatch.setattr(migrate, "_run", fake_run)

        migrate._run_conversation("gemini", agent=None, dry_run=True)
        assert "--agent" not in captured_cmd["cmd"]

    def test_extracts_error_line_on_live_failure(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(migrate, "_SAMPLE", tmp_path)
        zip_path = tmp_path / "chatgpt_export.zip"
        zip_path.write_bytes(b"fake")

        def fake_run(cmd, extra_env=None):
            return _completed(
                stdout="doing work\n",
                stderr="Error: something went wrong\n",
                returncode=1,
            )

        monkeypatch.setattr(migrate, "_run", fake_run)

        result = migrate._run_conversation("chatgpt", agent="agent-1", dry_run=False)
        assert result["exit_code"] == 1
        assert result["error"] == "Error: something went wrong"

    def test_no_error_extraction_during_dry_run_failure(self, tmp_path, monkeypatch) -> None:
        # Error extraction only happens when dry_run is False.
        monkeypatch.setattr(migrate, "_SAMPLE", tmp_path)
        zip_path = tmp_path / "chatgpt_export.zip"
        zip_path.write_bytes(b"fake")

        def fake_run(cmd, extra_env=None):
            return _completed(stderr="Error: failed\n", returncode=1)

        monkeypatch.setattr(migrate, "_run", fake_run)

        result = migrate._run_conversation("chatgpt", agent=None, dry_run=True)
        assert result["exit_code"] == 1
        assert result["error"] is None


class TestRunLanggraph:
    def test_missing_seed_returns_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(migrate, "_SCRIPTS", tmp_path)
        result = migrate._run_langgraph(agent=None, dry_run=True)
        assert result is None

    def test_builds_expected_command(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(migrate, "_SCRIPTS", tmp_path)
        seed = tmp_path / "langgraph_seed.json"
        seed.write_text("{}", encoding="utf-8")

        captured_cmd = {}

        def fake_run(cmd, extra_env=None):
            captured_cmd["cmd"] = cmd
            return _completed(
                stdout="Source records: 4\nMapped memories: 4  (skipped 0 empty)\n",
                returncode=0,
            )

        monkeypatch.setattr(migrate, "_run", fake_run)

        result = migrate._run_langgraph(agent="agent-1", dry_run=True)

        assert result["source_records"] == 4
        assert result["mapped"] == 4
        cmd = captured_cmd["cmd"]
        assert "langgraph" in cmd
        assert "--file" in cmd
        assert str(seed) in cmd
        assert "--dry-run" in cmd
        assert "--agent" in cmd
        assert "agent-1" in cmd


class TestPrintTable:
    def test_prints_headers_and_rows(self, capsys) -> None:
        rows = [("chatgpt", 5, 5, 0, "auto:5", "OK")]
        migrate._print_table(rows)
        out = capsys.readouterr().out
        assert "source" in out
        assert "chatgpt" in out
        assert "OK" in out

    def test_handles_empty_rows(self, capsys) -> None:
        migrate._print_table([])
        out = capsys.readouterr().out
        assert "source" in out  # header still printed


class TestMain:
    def test_live_without_agent_fails_fast(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "argv", ["migrate.py", "--live"])
        rc = migrate.main()
        assert rc == 1
        captured = capsys.readouterr()
        assert "--live requires --agent" in captured.err

    def test_dry_run_all_ok_returns_zero(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "argv", ["migrate.py"])

        def fake_run_conversation(source, agent, dry_run):
            return {
                "source_records": 5,
                "mapped": 5,
                "skipped": 0,
                "types": {"auto": 5},
                "exit_code": 0,
                "error": None,
            }

        monkeypatch.setattr(migrate, "_run_conversation", fake_run_conversation)
        monkeypatch.setattr(migrate, "_run_langgraph", lambda agent, dry_run: None)

        rc = migrate.main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Dry-run complete" in captured.out

    def test_dry_run_missing_sample_reports_skip(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "argv", ["migrate.py"])
        monkeypatch.setattr(migrate, "_run_conversation", lambda source, agent, dry_run: None)
        monkeypatch.setattr(migrate, "_run_langgraph", lambda agent, dry_run: None)

        rc = migrate.main()
        # All sources skipped is still considered a passing run.
        assert rc == 0
        captured = capsys.readouterr()
        assert "SKIP" in captured.out

    def test_failed_source_returns_one_and_reports_failure(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "argv", ["migrate.py"])

        def fake_run_conversation(source, agent, dry_run):
            exit_code = 1 if source == "claude" else 0
            return {
                "source_records": 1,
                "mapped": 1,
                "skipped": 0,
                "types": {"auto": 1},
                "exit_code": exit_code,
                "error": "boom" if exit_code else None,
            }

        monkeypatch.setattr(migrate, "_run_conversation", fake_run_conversation)
        monkeypatch.setattr(migrate, "_run_langgraph", lambda agent, dry_run: None)

        rc = migrate.main()
        assert rc == 1
        captured = capsys.readouterr()
        assert "Failed sources: claude" in captured.err

    def test_live_mode_with_agent_prints_completion_message(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "argv", ["migrate.py", "--live", "--agent", "agent-1"])

        def fake_run_conversation(source, agent, dry_run):
            assert dry_run is False
            assert agent == "agent-1"
            return {
                "source_records": 1,
                "mapped": 1,
                "skipped": 0,
                "types": {"auto": 1},
                "exit_code": 0,
                "error": None,
            }

        monkeypatch.setattr(migrate, "_run_conversation", fake_run_conversation)
        monkeypatch.setattr(migrate, "_run_langgraph", lambda agent, dry_run: None)

        rc = migrate.main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Migration complete." in captured.out
        assert "live migration" in captured.out

    def test_langgraph_row_included_when_present(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "argv", ["migrate.py"])
        monkeypatch.setattr(migrate, "_run_conversation", lambda source, agent, dry_run: None)

        def fake_run_langgraph(agent, dry_run):
            return {
                "source_records": 4,
                "mapped": 4,
                "skipped": 0,
                "types": {"auto": 4},
                "exit_code": 0,
                "error": None,
            }

        monkeypatch.setattr(migrate, "_run_langgraph", fake_run_langgraph)

        rc = migrate.main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "langgraph" in captured.out