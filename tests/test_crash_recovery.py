"""Crash safety and stale-state recovery.

Covers plan test 9 plus the "dead runner must not leave the worker BUSY" and
"never signal a stale PID" guarantees.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

from conftest import (
    pid_alive,
    wait_for_run_field,
    wait_for_session,
    wait_running,
    wait_until,
    write_plan,
)
from worker_mcp.models import RUN_RUNNING
from worker_mcp.registry import Registry
from worker_mcp.service import WorkerService


class TestStaleStateRecovery:
    def test_status_repairs_a_stale_running_run(
        self, service: WorkerService, registry: Registry, tmp_path: Path
    ) -> None:
        """Plan test 9: run.json says running, active_run_id set, lock free."""

        worktree = tmp_path / "wt"
        worktree.mkdir()
        worker = registry.create_worker(name="w1", cwd=str(worktree), backend="deepseek-harness")
        run = registry.start_run(worker=worker, kind="start", prompt="p")
        registry.record_session_id(run, "session-keep")
        run.status = RUN_RUNNING
        registry.layout.write_run(run)

        assert registry.is_busy("w1") is False

        status = service.status(name="w1")
        assert status["ok"] is True
        assert status["state"] == "failed"
        assert status["active_run_id"] is None
        assert status["reconciled"]["run_id"] == run.run_id
        # The discovered session survives the repair, so the conversation is
        # still resumable after a crash.
        assert status["session_id"] == "session-keep"

        repaired = registry.get_run("w1", run.run_id)
        assert repaired.status == "failed"
        assert repaired.error == "runner_exited_without_final_state"
        assert repaired.finished_at is not None

        # Reconciliation is idempotent: a second status call changes nothing.
        assert "reconciled" not in service.status(name="w1")

    def test_killed_runner_does_not_leave_the_worker_busy(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """SIGKILL the turn-runner: the OS drops the lock and status recovers."""

        fake_harness(SLEEP=120, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_running(service, "w1")
        wait_for_session(service, "w1")

        runner_pid = wait_for_run_field(service.registry, "w1", "runner_pid").runner_pid
        assert runner_pid is not None and pid_alive(runner_pid)

        os.kill(runner_pid, signal.SIGKILL)
        assert wait_until(lambda: not service.registry.is_busy("w1"), timeout=15)

        status = service.status(name="w1")
        assert status["state"] != "running"
        assert status["active_run_id"] is None

        # And the worker can accept new work again.
        assert service.message(name="w1", message="after crash")["code"] == "STARTED"
        service.interrupt(name="w1")

    def test_stale_pids_are_not_signalled(
        self, service: WorkerService, registry: Registry, tmp_path: Path
    ) -> None:
        """A recorded PID that outlived its run must never be killed.

        The lock is free, so the run is stale; the recorded PID belongs to an
        unrelated live process we must leave alone.
        """

        worktree = tmp_path / "wt"
        worktree.mkdir()
        worker = registry.create_worker(name="w1", cwd=str(worktree), backend="deepseek-harness")
        run = registry.start_run(worker=worker, kind="start", prompt="p")

        # A live bystander process that must survive reconciliation.
        import subprocess
        import sys

        bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            run.status = RUN_RUNNING
            run.backend_pgid = bystander.pid
            run.backend_pid = bystander.pid
            run.runner_pid = bystander.pid
            registry.layout.write_run(run)

            assert service.interrupt(name="w1")["code"] == "ALREADY_FINISHED"
            assert service.status(name="w1")["state"] == "failed"
            time.sleep(0.3)
            assert bystander.poll() is None, "a stale recorded PID was signalled"
        finally:
            bystander.kill()
            bystander.wait(timeout=10)

    def test_missing_run_record_is_repaired(
        self, service: WorkerService, registry: Registry, tmp_path: Path
    ) -> None:
        """active_run_id pointing at a directory that is gone."""

        worktree = tmp_path / "wt"
        worktree.mkdir()
        worker = registry.create_worker(name="w1", cwd=str(worktree), backend="deepseek-harness")
        worker.active_run_id = "1700000000000-deadbeef"
        registry.layout.write_worker(worker)

        status = service.status(name="w1")
        assert status["ok"] is True
        assert status["active_run_id"] is None
        assert service.registry.get_worker("w1").last_run_id == "1700000000000-deadbeef"

    def test_worker_state_survives_a_fresh_process_view(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """Cross-client: a brand-new Registry sees exactly the same worker."""

        fake_harness(SLEEP=0.1, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_until(lambda: service.status(name="w1")["state"] != "running")

        from worker_mcp.config import load_config
        from worker_mcp.state import StateLayout

        fresh_registry = Registry(StateLayout(load_config().state_dir))
        fresh_service = WorkerService(load_config(), fresh_registry)

        assert fresh_service.status(name="w1")["state"] == "idle"
        listing = fresh_service.list_workers()
        assert [entry["name"] for entry in listing["workers"]] == ["w1"]
        assert fresh_service.result(name="w1")["session_id"] == "session-abc"


class TestStateDurability:
    def test_events_and_logs_are_preserved_on_failure(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=0.1, EXIT_CODE=3, TOOL_EVENTS=True)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        result = service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_until(lambda: service.status(name="w1")["state"] != "running")

        run_dir = tmp_path / "state" / "workers" / "w1" / "runs" / result["run_id"]
        events = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kinds = [event["type"] for event in events]
        assert "session" in kinds
        assert "tool_call" in kinds
        assert "tool_result" in kinds
        assert (run_dir / "runner.log").exists()
        assert (run_dir / "stderr.log").exists()

    def test_run_json_is_valid_json_at_all_times(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """Readers must never observe a half-written document."""

        import threading

        fake_harness(SLEEP=2.0, SESSION_ID="session-abc", INTERMEDIATE_AFTER=0.5)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        result = service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        run_file = tmp_path / "state" / "workers" / "w1" / "runs" / result["run_id"] / "run.json"

        stop = threading.Event()
        errors: list = []

        def reader() -> None:
            while not stop.is_set():
                raw = run_file.read_bytes()
                try:
                    json.loads(raw.decode("utf-8"))
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append((raw, exc))
                time.sleep(0.001)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            wait_until(lambda: service.status(name="w1")["state"] != "running", timeout=30)
        finally:
            stop.set()
            thread.join(timeout=5)
        assert not errors, f"observed a partially written run.json: {errors[:1]}"
