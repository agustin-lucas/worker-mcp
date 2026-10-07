"""Real DeepSeek Harness *binary* integration test (no model call needed).

This drives the actual ``dsh --profile headless --json`` process through the
whole worker stack -- detached turn-runner, lock handoff, JSONL parsing, run
records -- using a task that fails before any model request. That is enough to
validate, against the real binary:

* the exact ``--json`` event stream (``session`` / ``status`` / ``turn_end`` /
  ``final``) and its field names;
* that the session id is captured while the turn is still running;
* that a failed in-turn result is recorded as ``failed`` with the structured
  DeepSeek error, not as a crash;
* that the next turn genuinely adopts that session, because the Harness only
  reports ``MISSING_CREDENTIAL`` when ``--session-id`` resolved to a real
  persisted session (an unknown id fails differently).

Set ``WORKER_MCP_TEST_DSH_COMMAND_JSON`` to the command array for the server
under test. Without it, ``dsh`` on ``PATH`` is tried and the module skips when
it is absent or unusable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from conftest import wait_for_session, wait_idle, wait_until, write_plan
from worker_mcp.config import ensure_state_dirs, load_config
from worker_mcp.registry import Registry
from worker_mcp.service import WorkerService
from worker_mcp.state import StateLayout

pytestmark = pytest.mark.real_binary


def dsh_command() -> list:
    raw = os.environ.get("WORKER_MCP_TEST_DSH_COMMAND_JSON")
    if raw:
        return json.loads(raw)
    found = shutil.which("dsh")
    return [found] if found else []


@pytest.fixture(autouse=True)
def dsh_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve a real dsh, isolate its DSH_HOME, and skip if unusable."""

    command = dsh_command()
    if not command:
        pytest.skip("no dsh command available (set WORKER_MCP_TEST_DSH_COMMAND_JSON)")

    home = tmp_path / "dsh-home"
    home.mkdir()
    # This module deliberately makes no model call: it exercises the real CLI,
    # event stream, session capture and failure recording for free. Strip the
    # credential so a configured developer machine does not silently start
    # spending money here; tests/test_real_harness.py is the paid path.
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.setenv("WORKER_MCP_DSH_COMMAND_JSON", json.dumps(command))

    probe = subprocess.run(
        command + ["--profile", "headless", "--help"],
        capture_output=True,
        text=True,
        timeout=240,
        env={**os.environ, "DSH_HOME": str(home)},
    )
    if probe.returncode != 0 or "--json" not in probe.stdout:
        pytest.skip(
            f"dsh at {command!r} does not expose the headless --json profile "
            f"(exit={probe.returncode}, stderr={probe.stderr[:200]!r})"
        )
    return home


@pytest.fixture
def real_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkerService:
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("WORKER_MCP_STATE_DIR", str(state))
    config = load_config()
    ensure_state_dirs(config)
    return WorkerService(config, Registry(StateLayout(config.state_dir)))


class TestRealHarnessBinary:
    def test_event_stream_and_failure_recording(
        self, real_service: WorkerService, tmp_path: Path, dsh_env: Path
    ) -> None:
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path, "# Smoke\n\nCreate hello.txt.\n")

        started = time.monotonic()
        result = real_service.start(name="real", cwd=str(worktree), plan_path=str(plan))
        elapsed = time.monotonic() - started
        assert result["code"] == "STARTED"
        assert elapsed < 10.0, "worker_start must not wait for the Harness"

        # The real turn may already have failed by the time we look, so poll for
        # the recorded identity rather than racing on the transient state.
        session_id = wait_for_session(real_service, "real", timeout=180)
        # The real Harness mints session-<uuid> identities.
        assert session_id.startswith("session-")

        wait_idle(real_service, "real", timeout=180)
        run_dir = (
            Path(real_service.config.state_dir) / "workers" / "real" / "runs" / result["run_id"]
        )
        events = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        types = [event["type"] for event in events]
        assert types[0] == "session"
        assert "final" in types
        assert events[0]["sessionId"] == session_id
        assert events[0]["cwd"] == str(worktree)

        outcome = real_service.result(name="real")
        assert outcome["session_id"] == session_id
        assert outcome["exit_code"] is not None
        # Either the model ran (credentials present) or the Harness reported a
        # structured in-turn error. Both are valid; a crash is not.
        if outcome["status"] == "failed":
            assert outcome["error"]
            turn_end = [e for e in events if e.get("phase") == "turn_end"]
            assert turn_end, "a failed real turn must still report turn_end"
            assert turn_end[0]["reason"]["kind"] in {"error", "cancelled"}
        else:
            assert outcome["status"] == "completed"

    def test_next_turn_adopts_the_recorded_session(
        self, real_service: WorkerService, tmp_path: Path, dsh_env: Path
    ) -> None:
        """A wrong session id would be rejected differently from a missing key."""

        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path, "# Smoke\n\nDo nothing.\n")
        real_service.start(name="real", cwd=str(worktree), plan_path=str(plan))
        session_id = wait_for_session(real_service, "real", timeout=180)
        wait_idle(real_service, "real", timeout=180)

        assert real_service.message(name="real", message="continue")["code"] == "STARTED"

        # Wait for the second run to reach a terminal state.
        from worker_mcp.models import RUN_TERMINAL_STATUSES

        assert wait_until(
            lambda: (
                len(real_service.registry.layout.list_run_ids("real")) == 2
                and (latest := real_service.registry.latest_run("real")) is not None
                and latest.status in RUN_TERMINAL_STATUSES
            ),
            timeout=180,
            interval=0.2,
        ), "the second real turn never finished"

        run = real_service.registry.latest_run("real")
        assert run is not None
        # The runner reused the persisted identity for the second turn.
        assert run.session_id_before == session_id
        assert run.session_id_after == session_id

        error = run.error or ""
        assert "does not exist" not in error, (
            "the Harness refused the session id we recorded: " + error
        )


def service_lock_free(state_dir: Path, worker: str) -> bool:
    """Whether no process currently holds the worker's lock."""

    from worker_mcp.locking import is_worker_locked

    return not is_worker_locked(state_dir / "workers" / worker / "lock")


class TestRealHarnessOverStdioMcp:
    def test_full_mcp_boundary_against_the_real_harness(
        self, tmp_path: Path, dsh_env: Path
    ) -> None:
        """The whole stack, over the real stdio transport, against real dsh."""

        import anyio
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        from test_mcp import EXPECTED_TOOLS, SERVER_SCRIPT

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path, "# Smoke\n\nCreate hello.txt.\n")

        env = {
            "WORKER_MCP_STATE_DIR": str(state_dir),
            "WORKER_MCP_DSH_COMMAND_JSON": os.environ["WORKER_MCP_DSH_COMMAND_JSON"],
            "DSH_HOME": str(dsh_env),
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
        }

        async def scenario() -> tuple:
            params = StdioServerParameters(command=str(SERVER_SCRIPT), args=[], env=env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    assert EXPECTED_TOOLS <= {tool.name for tool in listed.tools}

                    start = await session.call_tool(
                        "worker_start",
                        {"name": "stdio-real", "cwd": str(worktree), "plan_path": str(plan)},
                    )
                    assert start.structured_content["code"] == "STARTED"

                    # Poll over the protocol until the real Harness turn ends.
                    deadline = time.monotonic() + 300
                    status = None
                    while time.monotonic() < deadline:
                        probe = await session.call_tool("worker_status", {"name": "stdio-real"})
                        status = probe.structured_content
                        if status["session_id_available"] and status["state"] != "running":
                            break
                        await anyio.sleep(0.5)

                    result = await session.call_tool("worker_result", {"name": "stdio-real"})
                    return status, result.structured_content

        status, result = anyio.run(scenario)
        assert status is not None
        assert status["session_id_available"], "the real Harness session was not captured"
        assert status["session_id"].startswith("session-")
        assert result["run_id"]
        # Without credentials the turn is recorded as failed with the Harness's
        # own structured error; with credentials it completes.
        assert result["status"] in {"completed", "failed"}
        if result["status"] == "failed":
            assert result["error"]
        # The turn is over, so the worker must be available for another message.
        assert service_lock_free(state_dir, "stdio-real")
