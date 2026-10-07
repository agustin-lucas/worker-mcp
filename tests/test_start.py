"""worker_start behaviour: async start, session capture, duplicate guards.

Covers plan tests 1, 2, 10 and 11.
"""

from __future__ import annotations

import time
from pathlib import Path

from conftest import (
    harness_count,
    read_invocations,
    wait_for_session,
    wait_idle,
    wait_until,
    write_plan,
)
from worker_mcp.registry import Registry
from worker_mcp.service import WorkerService

idle = wait_idle


class TestStartCreatesAsyncWorker:
    def test_start_returns_before_harness_finishes(
        self,
        service: WorkerService,
        registry: Registry,
        tmp_path: Path,
        fake_harness,
    ) -> None:
        fake_harness(SLEEP=8, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path, "# Plan\n\nImplement auth.\n")

        started = time.monotonic()
        result = service.start(name="auth", cwd=str(worktree), plan_path=str(plan))
        elapsed = time.monotonic() - started

        assert result["ok"] is True
        assert result["code"] == "STARTED"
        # The whole point: the tool must not wait for DeepSeek.
        assert elapsed < 3.0, f"worker_start blocked for {elapsed:.2f}s"

        # The worker is running and its lock is held by the detached runner.
        assert registry.is_busy("auth")
        status = service.status(name="auth")
        assert status["state"] == "running"

        assert registry.get_worker("auth").active_run_id is not None
        run = service.registry.latest_run("auth")
        assert run is not None
        assert run.status in {"starting", "running"}

        idle(service, "auth")

    def test_run_directory_is_fully_populated(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=0.2)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path, "# Plan\n")
        result = service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        idle(service, "w1")

        run_dir = tmp_path / "state" / "workers" / "w1" / "runs" / result["run_id"]
        assert (run_dir / "run.json").is_file()
        prompt = (run_dir / "prompt.txt").read_text(encoding="utf-8")
        assert "BEGIN IMPLEMENTATION PLAN" in prompt
        assert "# Plan" in prompt
        events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
        assert '"type": "session"' in events
        assert '"type": "final"' in events

    def test_plan_contents_are_embedded_not_path_referenced(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """The plan may live outside the worktree."""

        fake_harness(SLEEP=0.1)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path, "# Plan\n\nUNIQUE_MARKER_12345\n")
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))

        invocation = None
        for _ in range(200):
            invocations = read_invocations(tmp_path / "harness-invocations.jsonl")
            if invocations:
                invocation = invocations[0]
                break
            time.sleep(0.05)
        assert invocation is not None
        assert "UNIQUE_MARKER_12345" in invocation["task"]

    def test_additional_instructions_are_appended(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=0.1)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(
            name="w1",
            cwd=str(worktree),
            plan_path=str(plan),
            additional_instructions="Prefer dataclasses.",
        )
        wait_until(lambda: harness_count(tmp_path / "harness-invocations.jsonl") >= 1)
        invocation = read_invocations(tmp_path / "harness-invocations.jsonl")[0]
        assert "Prefer dataclasses." in invocation["task"]
        assert "ADDITIONAL INSTRUCTIONS" in invocation["task"]


class TestSessionCapture:
    def test_session_id_captured_during_active_turn(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """Plan test 2: the session id lands before the turn finishes."""

        fake_harness(SLEEP=10, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))

        assert wait_for_session(service, "w1", timeout=15) == "session-abc"

        status = service.status(name="w1")
        assert status["state"] == "running"
        assert status["session_id"] == "session-abc"
        service.interrupt(name="w1")

    def test_no_session_event_leaves_session_unavailable(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=0.2, SKIP_SESSION_EVENT=True)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        result = service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        idle(service, "w1")

        status = service.status(name="w1")
        assert status["session_id_available"] is False
        assert service.result(name="w1", run_id=result["run_id"])["session_id_available"] is False
        # And a follow-up turn is impossible without a session.
        follow_up = service.message(name="w1", message="more")
        assert follow_up["code"] == "NO_SESSION"


class TestDuplicateGuards:
    def test_duplicate_name_rejected(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """Plan test 10."""

        fake_harness(SLEEP=0.1)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        first = service.start(name="dup", cwd=str(worktree), plan_path=str(plan))
        assert first["code"] == "STARTED"
        idle(service, "dup")

        other = tmp_path / "other-wt"
        other.mkdir()
        second = service.start(name="dup", cwd=str(other), plan_path=str(plan))
        assert second["ok"] is False
        assert second["code"] == "ALREADY_EXISTS"

    def test_duplicate_cwd_rejected(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """Plan test 11."""

        fake_harness(SLEEP=0.1)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        assert service.start(name="w1", cwd=str(worktree), plan_path=str(plan))["code"] == "STARTED"

        # A path that canonicalises to the same directory must also be refused.
        alias = worktree / "." / ".." / "wt"
        second = service.start(name="w2", cwd=str(alias), plan_path=str(plan))
        assert second["ok"] is False
        assert second["code"] == "CWD_IN_USE"
        assert second["existing_worker"] == "w1"

    def test_concurrent_start_cannot_share_one_cwd(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        """Two racing worker_start calls must not both claim the same worktree."""

        import threading

        fake_harness(SLEEP=0.1)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)

        results: list = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def attempt(name: str) -> None:
            barrier.wait(timeout=30)
            outcome = service.start(name=name, cwd=str(worktree), plan_path=str(plan))
            with lock:
                results.append((name, outcome["code"]))

        threads = [threading.Thread(target=attempt, args=(f"w{i}",)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert len(results) == 4, results
        codes = [code for _, code in results]
        assert codes.count("STARTED") == 1, f"worktree claimed more than once: {results}"
        assert codes.count("CWD_IN_USE") == 3, results
        assert service.list_workers()["count"] == 1

    def test_failed_start_does_not_reserve_name_or_cwd(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=0.1)
        plan = write_plan(tmp_path)
        missing = tmp_path / "does-not-exist"

        bad = service.start(name="w1", cwd=str(missing), plan_path=str(plan))
        assert bad["code"] == "INVALID_ARGUMENT"
        assert service.list_workers()["count"] == 0

        worktree = tmp_path / "wt"
        worktree.mkdir()
        good = service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        assert good["code"] == "STARTED"


class TestArgumentValidation:
    def test_invalid_worker_name(self, service: WorkerService, tmp_path: Path) -> None:
        plan = write_plan(tmp_path)
        result = service.start(name="../evil", cwd=str(tmp_path), plan_path=str(plan))
        assert result["code"] == "INVALID_ARGUMENT"

    def test_missing_plan_file(self, service: WorkerService, tmp_path: Path) -> None:
        result = service.start(name="w1", cwd=str(tmp_path), plan_path=str(tmp_path / "nope.md"))
        assert result["code"] == "INVALID_ARGUMENT"
        assert "plan" in result["message"].lower()

    def test_oversized_plan_rejected(self, tmp_path: Path, monkeypatch, state_dir) -> None:
        monkeypatch.setenv("WORKER_MCP_MAX_PLAN_BYTES", "64")
        (tmp_path / "big.md").write_text("x" * 200, encoding="utf-8")
        # Configuration is snapshotted at startup, so build a fresh service.
        from worker_mcp.config import ensure_state_dirs, load_config
        from worker_mcp.registry import Registry
        from worker_mcp.state import StateLayout

        config = load_config()
        ensure_state_dirs(config)
        fresh = WorkerService(config, Registry(StateLayout(config.state_dir)))
        result = fresh.start(name="w1", cwd=str(tmp_path), plan_path=str(tmp_path / "big.md"))
        assert result["code"] == "INVALID_ARGUMENT"
        assert "limit" in result["message"]

    def test_message_to_missing_worker(self, service: WorkerService) -> None:
        result = service.message(name="ghost", message="hi")
        assert result["code"] == "NOT_FOUND"

    def test_status_of_missing_worker(self, service: WorkerService) -> None:
        assert service.status(name="ghost")["code"] == "NOT_FOUND"

    def test_empty_message_rejected(
        self, service: WorkerService, tmp_path: Path, fake_harness
    ) -> None:
        fake_harness(SLEEP=0.1)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        service.start(name="w1", cwd=str(worktree), plan_path=str(plan))
        idle(service, "w1")
        assert service.message(name="w1", message="   ")["code"] == "INVALID_ARGUMENT"
