"""State, atomic IO and locking unit tests (plan phase 2)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from worker_mcp.locking import WorkerLock, is_worker_locked
from worker_mcp.models import RUN_RUNNING
from worker_mcp.registry import Registry, WorkerExists, new_run_id
from worker_mcp.state import (
    StateLayout,
    atomic_write_json,
    read_json,
    validate_worker_name,
)


class TestWorkerNames:
    @pytest.mark.parametrize("name", ["auth-refactor", "a", "A.b_c-d", "x" * 100, "0", "..hidden"])
    def test_accepts_legal_names(self, name: str) -> None:
        assert validate_worker_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "x" * 101,
            "../escape",
            "a/b",
            "a\\b",
            ".",
            "..",
            "with space",
            "semi;colon",
            "dollar$",
            "new\nline",
        ],
    )
    def test_rejects_illegal_names(self, name: str) -> None:
        with pytest.raises(ValueError):
            validate_worker_name(name)

    def test_layout_refuses_traversal(self, tmp_path: Path) -> None:
        layout = StateLayout(tmp_path)
        with pytest.raises(ValueError):
            layout.worker_dir("../../etc")


class TestAtomicWrites:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "worker.json"
        atomic_write_json(path, {"a": 1, "b": [2, 3]})
        assert read_json(path) == {"a": 1, "b": [2, 3]}

    def test_missing_file_is_none(self, tmp_path: Path) -> None:
        assert read_json(tmp_path / "nope.json") is None

    def test_no_temp_files_left_behind(self, tmp_path: Path) -> None:
        path = tmp_path / "run.json"
        for index in range(5):
            atomic_write_json(path, {"n": index})
        assert [p.name for p in tmp_path.iterdir()] == ["run.json"]

    def test_corrupt_state_raises(self, tmp_path: Path) -> None:
        from worker_mcp.state import StateError

        path = tmp_path / "worker.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(StateError):
            read_json(path)

    def test_state_files_are_private(self, tmp_path: Path) -> None:
        path = tmp_path / "worker.json"
        atomic_write_json(path, {"a": 1})
        assert (path.stat().st_mode & 0o077) == 0


class TestWorkerLock:
    def test_second_lock_in_same_process_conflicts(self, tmp_path: Path) -> None:
        """flock conflicts between open file descriptions, not just processes."""

        path = tmp_path / "lock"
        first = WorkerLock(path)
        assert first.try_acquire() is True
        second = WorkerLock(path)
        assert second.try_acquire() is False
        first.release()
        assert second.try_acquire() is True
        second.release()

    def test_lock_released_on_process_exit(self, tmp_path: Path) -> None:
        path = tmp_path / "lock"
        script = (
            "import fcntl,os,sys\n"
            "fd=os.open(sys.argv[1], os.O_RDWR|os.O_CREAT)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "print('locked', flush=True)\n"
            "import time; time.sleep(30)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(path)],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "locked"
            assert is_worker_locked(path) is True
        finally:
            process.kill()
            process.wait(timeout=10)
        # The kernel drops the lock when the owner dies; no stale BUSY.
        assert is_worker_locked(path) is False

    def test_close_without_unlock_keeps_child_lock(self, tmp_path: Path) -> None:
        """The atomic handoff: closing our descriptor does not release the lock."""

        path = tmp_path / "lock"
        lock = WorkerLock(path)
        assert lock.try_acquire()
        script = "import time; time.sleep(30)"
        child = subprocess.Popen(
            [sys.executable, "-c", script],
            pass_fds=(lock.fd,),
            close_fds=True,
        )
        try:
            lock.close_without_unlock()
            assert is_worker_locked(path) is True, "child must still hold the lock"
        finally:
            child.kill()
            child.wait(timeout=10)
        assert is_worker_locked(path) is False


class TestRegistry:
    def test_create_and_read(self, registry: Registry, tmp_path: Path) -> None:
        worker = registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        assert worker.session_id is None
        read = registry.get_worker("w1")
        assert read.name == "w1"
        assert read.cwd == str(tmp_path)
        assert registry.layout.list_worker_names() == ["w1"]

    def test_duplicate_name_rejected(self, registry: Registry, tmp_path: Path) -> None:
        registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        with pytest.raises(WorkerExists):
            registry.create_worker(
                name="w1", cwd=str(tmp_path / "other"), backend="deepseek-harness"
            )

    def test_run_ids_sort_chronologically(self) -> None:
        """Sorting run ids must match chronological order, even in one tick."""

        ids = [new_run_id() for _ in range(200)]
        assert len(set(ids)) == len(ids), "run ids must be unique"
        assert ids == sorted(ids)

    def test_record_session_id_updates_worker_and_run(
        self, registry: Registry, tmp_path: Path
    ) -> None:
        worker = registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        run = registry.start_run(worker=worker, kind="start", prompt="p")
        registry.record_session_id(run, "session-xyz")
        assert registry.get_worker("w1").session_id == "session-xyz"
        assert registry.get_run("w1", run.run_id).session_id_after == "session-xyz"

    def test_finalize_clears_active_run(self, registry: Registry, tmp_path: Path) -> None:
        worker = registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        run = registry.start_run(worker=worker, kind="start", prompt="p")
        assert registry.get_worker("w1").active_run_id == run.run_id
        registry.finalize_run(run, status="completed", exit_code=0)
        worker = registry.get_worker("w1")
        assert worker.active_run_id is None
        assert worker.last_run_id == run.run_id

    def test_worker_state_is_independent_per_worker(
        self, registry: Registry, tmp_path: Path
    ) -> None:
        registry.create_worker(name="a", cwd=str(tmp_path / "a"), backend="deepseek-harness")
        registry.create_worker(name="b", cwd=str(tmp_path / "b"), backend="deepseek-harness")
        assert sorted(registry.layout.list_worker_names()) == ["a", "b"]
        assert {w.name for w in registry.list_workers()} == {"a", "b"}

    def test_layout_files_use_expected_names(self, registry: Registry, tmp_path: Path) -> None:
        worker = registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        run = registry.start_run(worker=worker, kind="start", prompt="hello")
        layout = registry.layout
        run_dir = layout.run_dir("w1", run.run_id)
        assert (run_dir / "run.json").is_file()
        assert (run_dir / "prompt.txt").read_text(encoding="utf-8") == "hello"
        assert (run_dir / "events.jsonl").is_file()
        assert (run_dir / "stderr.log").is_file()
        assert layout.lock_file("w1").is_file()

    def test_latest_run_tracks_the_newest_run_not_the_last_finished_one(
        self, registry: Registry, tmp_path: Path
    ) -> None:
        """While a turn is active, "most recent run" is that turn."""

        worker = registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        first = registry.start_run(worker=worker, kind="start", prompt="p")
        registry.finalize_run(first, status="completed", exit_code=0)
        worker = registry.get_worker("w1")
        second = registry.start_run(worker=worker, kind="message", prompt="q")

        assert registry.latest_run("w1").run_id == second.run_id
        assert registry.last_finished_run("w1").run_id == first.run_id

        registry.finalize_run(second, status="completed", exit_code=0)
        assert registry.latest_run("w1").run_id == second.run_id
        assert registry.last_finished_run("w1").run_id == second.run_id

    def test_find_worker_for_cwd_uses_realpath(self, registry: Registry, tmp_path: Path) -> None:
        target = tmp_path / "wt"
        target.mkdir()
        registry.create_worker(name="w1", cwd=str(target), backend="deepseek-harness")
        alias = tmp_path / "alias"
        try:
            alias.symlink_to(target, target_is_directory=True)
        except OSError:  # pragma: no cover - platform without symlinks
            pytest.skip("symlinks unavailable")
        assert registry.find_worker_for_cwd(str(alias)) is not None


class TestStaleReconciliation:
    def test_stale_running_run_becomes_failed(self, registry: Registry, tmp_path: Path) -> None:
        worker = registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        run = registry.start_run(worker=worker, kind="start", prompt="p")
        # Simulate a runner that died without finalizing: run.json says running,
        # worker.json still points at it, and nobody holds the lock.
        run.status = RUN_RUNNING
        registry.layout.write_run(run)
        assert not registry.is_busy("w1")

        result = registry.reconcile_stale("w1")
        assert result.changed
        assert result.reconciled_status == "failed"
        repaired = registry.get_run("w1", run.run_id)
        assert repaired.status == "failed"
        assert repaired.error == "runner_exited_without_final_state"
        assert registry.get_worker("w1").active_run_id is None

    def test_stale_run_with_interrupt_marker_becomes_interrupted(
        self, registry: Registry, tmp_path: Path
    ) -> None:
        worker = registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        run = registry.start_run(worker=worker, kind="start", prompt="p")
        run.status = RUN_RUNNING
        registry.layout.write_run(run)
        registry.request_interrupt("w1", run.run_id)

        result = registry.reconcile_stale("w1")
        assert result.reconciled_status == "interrupted"
        assert registry.get_run("w1", run.run_id).status == "interrupted"

    def test_reconcile_preserves_session_id(self, registry: Registry, tmp_path: Path) -> None:
        worker = registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        run = registry.start_run(worker=worker, kind="start", prompt="p")
        registry.record_session_id(run, "session-keep")
        run.status = RUN_RUNNING
        registry.layout.write_run(run)

        registry.reconcile_stale("w1")
        assert registry.get_worker("w1").session_id == "session-keep"

    def test_reconcile_is_idempotent(self, registry: Registry, tmp_path: Path) -> None:
        worker = registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        registry.start_run(worker=worker, kind="start", prompt="p")
        registry.reconcile_stale("w1")
        second = registry.reconcile_stale("w1")
        assert not second.changed

    def test_interrupted_run_keeps_changes_and_session(
        self, registry: Registry, tmp_path: Path
    ) -> None:
        worker = registry.create_worker(name="w1", cwd=str(tmp_path), backend="deepseek-harness")
        run = registry.start_run(worker=worker, kind="start", prompt="p")
        registry.record_session_id(run, "session-1")
        run.status = RUN_RUNNING
        registry.layout.write_run(run)

        registry.request_interrupt("w1", run.run_id)
        registry.reconcile_stale("w1")

        final = registry.get_run("w1", run.run_id)
        assert final.status == "interrupted"
        assert final.session_id_after == "session-1"
        state, _ = registry.logical_state("w1")
        assert state == "interrupted"
