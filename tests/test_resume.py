"""Session continuity: a follow-up turn resumes the same DeepSeek session.

Covers plan tests 4 and 7 (the interrupted half of test 7 lives in
``test_interrupt.py``).
"""

from __future__ import annotations

from pathlib import Path

from conftest import harness_count, read_invocations, wait_until, write_plan
from worker_mcp.service import WorkerService


def wait_idle(service: WorkerService, name: str, timeout: float = 30.0) -> None:
    assert wait_until(lambda: service.status(name=name)["state"] != "running", timeout=timeout), (
        f"worker {name} never went idle"
    )


class TestResume:
    def test_second_turn_passes_session_id_and_same_cwd(
        self, service: WorkerService, tmp_path: Path, fake_harness, harness_log: Path
    ) -> None:
        """Plan test 4."""

        fake_harness(SLEEP=0.2, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_idle(service, "w1")

        assert service.message(name="w1", message="continue the work")["code"] == "STARTED"
        wait_idle(service, "w1")
        assert wait_until(lambda: harness_count(harness_log) == 2)

        invocations = read_invocations(harness_log)
        first, second = invocations[0], invocations[1]

        # First turn starts a fresh session: no --session-id.
        assert first["session_id"] is None
        assert "--json" in first["argv"]
        assert "--profile" in first["argv"]

        # Second turn resumes the persisted session in the identical cwd.
        assert second["session_id"] == "session-abc"
        assert "--session-id" in second["argv"]
        assert second["cwd"] == first["cwd"]
        assert second["task"] == "continue the work"

    def test_worker_session_id_is_stable_across_turns(
        self, service: WorkerService, tmp_path: Path, fake_harness, harness_log: Path
    ) -> None:
        fake_harness(SLEEP=0.1, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_idle(service, "w1")

        for index in range(2):
            assert service.message(name="w1", message=f"round {index}")["code"] == "STARTED"
            wait_idle(service, "w1")

        assert wait_until(lambda: harness_count(harness_log) == 3)
        session_ids = [entry["session_id"] for entry in read_invocations(harness_log)]
        assert session_ids == [None, "session-abc", "session-abc"]

    def test_result_reports_final_text_and_status(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=0.1, SESSION_ID="session-abc", FINAL_TEXT="all done here")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_idle(service, "w1")

        result = service.result(name="w1")
        assert result["status"] == "completed"
        assert result["final_text"] == "all done here"
        assert result["exit_code"] == 0
        assert result["interrupted"] is False
        assert result["session_id"] == "session-abc"

    def test_failed_turn_reports_error(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=0.1, EXIT_CODE=2, FINAL_TEXT="")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_idle(service, "w1")

        result = service.result(name="w1")
        assert result["status"] == "failed"
        assert result["exit_code"] == 2
        assert result["error"]
        assert service.status(name="w1")["state"] == "failed"

    def test_backend_error_event_fails_the_run_even_on_exit_zero(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=0, EMIT_ERROR=True, EXIT_CODE=0, ERROR_MESSAGE="no credentials")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_idle(service, "w1")

        result = service.result(name="w1")
        assert result["status"] == "failed"
        assert "no credentials" in result["error"]

    def test_missing_backend_executable_fails_cleanly(
        self, service: WorkerService, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv(
            "WORKER_MCP_DSH_COMMAND_JSON",
            '["/nonexistent/dsh-binary-that-cannot-exist"]',
        )
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        # worker_start itself succeeds: the detached runner discovers the
        # missing executable and records a failed turn.
        assert service.start(name="w1", cwd=str(worktree), plan_path=str(plan))["code"] == "STARTED"
        wait_idle(service, "w1")

        result = service.result(name="w1")
        assert result["status"] == "failed"
        assert "backend_executable_missing" in result["error"]

    def test_turn_continues_after_the_mcp_process_exits(
        self, service: WorkerService, tmp_path: Path, fake_harness, harness_log: Path
    ) -> None:
        """Restart safety: a detached turn is not tied to the caller's lifetime."""

        import multiprocessing

        fake_harness(SLEEP=2.5, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)

        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        child = context.Process(
            target=_start_then_exit,
            args=(str(tmp_path / "state"), str(worktree), str(plan), queue),
        )
        child.start()
        child.join(timeout=30)
        assert queue.get(timeout=10)["code"] == "STARTED"
        # The "MCP process" is gone; the turn must still finish.
        assert wait_until(lambda: service.status(name="w1")["state"] != "running", timeout=30)
        assert service.result(name="w1")["status"] == "completed"


def _start_then_exit(state_dir: str, worktree: str, plan: str, queue) -> None:
    import os

    os.environ["WORKER_MCP_STATE_DIR"] = state_dir
    from worker_mcp.config import ensure_state_dirs, load_config
    from worker_mcp.service import WorkerService as Service

    config = load_config()
    ensure_state_dirs(config)
    queue.put(Service(config).start(name="w1", cwd=worktree, plan_path=plan))
    # Process exits immediately; the detached runner keeps going.
