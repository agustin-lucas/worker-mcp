"""BUSY semantics: never queue a message, never start a second turn.

Covers plan test 3.
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from conftest import harness_count, wait_running, wait_until, write_plan
from worker_mcp.service import WorkerService


def _start_in_process(state_dir: str, worktree: str, plan: str, queue) -> None:
    """Start a worker from a *separate* process, for the cross-process test."""

    import os

    os.environ["WORKER_MCP_STATE_DIR"] = state_dir
    from worker_mcp.config import ensure_state_dirs, load_config
    from worker_mcp.service import WorkerService as Service

    config = load_config()
    ensure_state_dirs(config)
    queue.put(Service(config).start(name="w1", cwd=worktree, plan_path=plan))


class TestBusy:
    def test_message_while_running_returns_busy(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=10, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        wait_running(service, "w1")

        result = service.message(name="w1", message="do something else")
        assert result["ok"] is False
        assert result["code"] == "BUSY"
        assert result["worker"] == "w1"
        assert result["active_run_id"]
        assert "active turn" in result["message"].lower()

        service.interrupt(name="w1")

    def test_busy_does_not_start_a_second_harness(
        self, service: WorkerService, tmp_path: Path, fake_harness, harness_log: Path
    ) -> None:
        """A rejected message must not spawn anything."""

        fake_harness(SLEEP=10, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        assert wait_until(lambda: harness_count(harness_log) == 1)

        for _ in range(5):
            assert service.message(name="w1", message="nope")["code"] == "BUSY"

        # Give any (incorrectly) spawned process time to register itself.
        time.sleep(1.0)
        assert harness_count(harness_log) == 1
        service.interrupt(name="w1")

    def test_busy_is_detected_across_processes(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """Another MCP process must also see BUSY, via the on-disk lock."""

        fake_harness(SLEEP=3.0, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)

        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        child = context.Process(
            target=_start_in_process,
            args=(
                str(tmp_path / "state"),
                str(worktree),
                str(plan),
                queue,
            ),
        )
        child.start()
        try:
            started = queue.get(timeout=30)
            assert started["code"] == "STARTED"
            # This process never started the turn, yet must see the same worker.
            assert service.registry.try_get_worker("w1") is not None
            status = service.status(name="w1")
            assert status["state"] == "running"
            assert service.message(name="w1", message="from another process")["code"] == "BUSY"
        finally:
            child.join(timeout=30)
            if child.is_alive():  # pragma: no cover - safety net
                child.terminate()
            service.interrupt(name="w1")

    def test_message_succeeds_once_the_turn_finishes(
        self, service: WorkerService, tmp_path: Path, fake_harness, harness_log: Path
    ) -> None:
        fake_harness(SLEEP=0.5, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))

        assert wait_until(lambda: service.status(name="w1")["state"] != "running", timeout=30)
        assert service.message(name="w1", message="next")["code"] == "STARTED"
        assert wait_until(lambda: harness_count(harness_log) == 2, timeout=30)

    def test_one_active_turn_at_a_time_under_contention(
        self, service: WorkerService, tmp_path: Path, fake_harness, harness_log: Path
    ) -> None:
        """Hammer worker_message from threads: exactly one turn may start."""

        import threading

        fake_harness(SLEEP=2.0, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        assert wait_until(lambda: service.status(name="w1")["state"] != "running", timeout=30)

        results = []
        lock = threading.Lock()

        def send() -> None:
            outcome = service.message(name="w1", message="concurrent")
            with lock:
                results.append(outcome["code"])

        threads = [threading.Thread(target=send) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        started = [code for code in results if code == "STARTED"]
        busy = [code for code in results if code == "BUSY"]
        assert len(started) == 1, f"expected exactly one turn to start, got {results}"
        assert len(busy) == len(results) - 1
        service.interrupt(name="w1")
