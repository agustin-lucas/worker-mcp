"""Interruption: process-tree termination, lock-release guarantee, continuity.

Covers plan tests 5, 6, 7, 8.
"""

from __future__ import annotations

import time
from pathlib import Path

from conftest import (
    harness_count,
    pid_alive,
    read_invocations,
    wait_for_run_field,
    wait_for_session,
    wait_idle,
    wait_running,
    wait_until,
    write_plan,
)
from worker_mcp.service import WorkerService  # noqa: E402  (used in annotations)


class TestInterrupt:
    def test_interrupt_terminates_the_process_tree(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """Plan test 5: the Harness and the child it spawned both die."""

        child_marker = tmp_path / "child.pid"
        fake_harness(
            SLEEP=120,
            SESSION_ID="session-abc",
            SPAWN_CHILD=True,
            CHILD_PID_FILE=str(child_marker),
        )
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_running(service, "w1")

        assert wait_until(child_marker.exists, timeout=15)
        child_pid = int(child_marker.read_text(encoding="utf-8").strip())
        harness_pid = service.registry.latest_run("w1").backend_pid
        assert harness_pid is not None
        assert pid_alive(harness_pid)
        assert pid_alive(child_pid)

        result = service.interrupt(name="w1")

        assert result["ok"] is True
        assert result["code"] == "INTERRUPTED"
        assert not pid_alive(harness_pid), "Harness process survived the interrupt"
        assert not pid_alive(child_pid), "process spawned by the Harness survived"

        run = service.registry.get_run("w1", result["run_id"])
        assert run.status == "interrupted"
        assert run.exit_code is not None
        # The lock is released, so the worker is genuinely available again.
        assert not service.registry.is_busy("w1")
        assert service.status(name="w1")["state"] == "interrupted"

    def test_interrupt_escalates_to_sigkill(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """A backend that ignores SIGTERM is still stopped."""

        fake_harness(SLEEP=120, SESSION_ID="session-abc", IGNORE_SIGTERM=True)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_running(service, "w1")
        harness_pid = wait_for_run_field(service.registry, "w1", "backend_pid").backend_pid
        assert harness_pid is not None
        # Wait until the Harness is past interpreter start-up, so its SIGTERM
        # handler is definitely installed and the escalation path is exercised.
        wait_for_session(service, "w1")

        started = time.monotonic()
        result = service.interrupt(name="w1")
        elapsed = time.monotonic() - started

        assert result["code"] == "INTERRUPTED"
        assert not pid_alive(harness_pid)
        # SIGTERM was ignored, so the grace period must have been consumed.
        assert service.config.interrupt_grace_seconds < 5, "test expects a short grace period"
        assert elapsed >= service.config.interrupt_grace_seconds * 0.5, (
            f"interrupt returned in {elapsed:.2f}s without honouring the grace period"
        )
        assert not service.registry.is_busy("w1")

    def test_message_immediately_after_interrupt_succeeds(
        self, service: WorkerService, tmp_path: Path, fake_harness, harness_log: Path
    ) -> None:
        """Plan test 6: no race-generated BUSY after interrupt returns."""

        fake_harness(SLEEP=120, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_running(service, "w1")
        # Steer a turn that is a little way in, as a controller actually would.
        wait_for_session(service, "w1")

        assert service.message(name="w1", message="steer")["code"] == "BUSY"
        assert service.interrupt(name="w1")["ok"] is True
        assert service.message(name="w1", message="steer now")["code"] == "STARTED"
        assert service.status(name="w1")["state"] == "running"
        wait_for_session(service, "w1")
        service.interrupt(name="w1")

    def test_interrupted_worker_resumes_the_same_session(
        self, service: WorkerService, tmp_path: Path, fake_harness, harness_log: Path
    ) -> None:
        """Plan test 7."""

        fake_harness(SLEEP=120, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_running(service, "w1")
        assert wait_for_session(service, "w1") == "session-abc"

        assert service.interrupt(name="w1")["code"] == "INTERRUPTED"
        assert service.status(name="w1")["session_id"] == "session-abc"

        assert service.message(name="w1", message="carry on")["code"] == "STARTED"
        wait_running(service, "w1")
        assert wait_until(lambda: harness_count(harness_log) == 2)
        wait_for_session(service, "w1")
        assert read_invocations(harness_log)[1]["session_id"] == "session-abc"
        service.interrupt(name="w1")

    def test_partial_file_modifications_survive_interrupt(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """Plan test 8: interrupting must never revert the worker's edits."""

        fake_harness(
            SLEEP=120,
            SESSION_ID="session-abc",
            WRITE_FILE="src/feature.py",
            WRITE_TEXT="PARTIAL = True\n",
        )
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_running(service, "w1")

        written = worktree / "src" / "feature.py"
        assert wait_until(written.exists, timeout=15)
        assert written.read_text(encoding="utf-8") == "PARTIAL = True\n"

        assert service.interrupt(name="w1")["code"] == "INTERRUPTED"

        assert written.exists(), "interrupt reverted a file the worker had written"
        assert written.read_text(encoding="utf-8") == "PARTIAL = True\n"

    def test_interrupt_when_already_idle_is_not_an_error(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=0.1, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_idle(service, "w1")

        result = service.interrupt(name="w1")
        assert result["ok"] is True
        assert result["code"] == "ALREADY_FINISHED"
        # Still usable afterwards.
        assert service.message(name="w1", message="next")["code"] == "STARTED"
        wait_idle(service, "w1")

    def test_interrupt_of_missing_worker(self, service: WorkerService) -> None:
        assert service.interrupt(name="ghost")["code"] == "NOT_FOUND"

    def test_interrupt_before_backend_spawn_is_honoured(
        self, service: WorkerService, tmp_path: Path, fake_harness, harness_log: Path
    ) -> None:
        """A marker written before the Harness starts must abort the turn."""

        # PRE_SLEEP makes aborting before spawn overwhelmingly likely.
        fake_harness(SLEEP=120, PRE_SLEEP=5, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_running(service, "w1")

        result = service.interrupt(name="w1")
        assert result["ok"] is True
        assert not service.registry.is_busy("w1")

        run = service.registry.get_run("w1", result["run_id"])
        assert run.status == "interrupted"
        assert run.backend_pid is None
        assert not harness_log.exists(), "the Harness must not have run"
