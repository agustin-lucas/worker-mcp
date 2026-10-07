"""Disk-backed worker registry.

Every operation that could observe or change "is this worker busy?" takes the
worker lock first. The registry never trusts a PID for liveness: the lock is the
single source of truth, because the kernel releases it whenever its owner dies.
"""

from __future__ import annotations

import itertools
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .locking import WorkerLock, is_worker_locked
from .models import (
    ERROR_RUNNER_DIED,
    RUN_ACTIVE_STATUSES,
    RUN_INTERRUPTED,
    RUN_RUNNING,
    RUN_STARTING,
    STATE_FAILED,
    STATE_IDLE,
    STATE_INTERRUPTED,
    STATE_RUNNING,
    RunRecord,
    WorkerRecord,
    now_iso,
)
from .state import (
    EVENTS_FILE,
    INTERRUPT_FILE,
    LOCK_FILE,
    PROMPT_FILE,
    RUN_FILE,
    RUNNER_LOG_FILE,
    STDERR_FILE,
    StateLayout,
    atomic_write_text,
    read_json,
)

logger = logging.getLogger(__name__)


class WorkerNotFound(KeyError):
    """Raised when a logical worker does not exist."""


class WorkerBusy(RuntimeError):
    """Raised when a turn is already active for this worker."""

    def __init__(self, name: str, active_run_id: str | None) -> None:
        super().__init__(f"worker {name!r} currently has an active turn")
        self.name = name
        self.active_run_id = active_run_id


class WorkerExists(RuntimeError):
    """Raised when a logical worker name is already taken."""


class CwdInUse(RuntimeError):
    """Raised when another logical worker already owns the canonical cwd."""

    def __init__(self, cwd: str, existing_worker: str) -> None:
        super().__init__(f"cwd {cwd!r} is already owned by worker {existing_worker!r}")
        self.cwd = cwd
        self.existing_worker = existing_worker


_RUN_ID_COUNTER = itertools.count()

#: Nanosecond timestamps are fixed-width for the next few centuries, so string
#: ordering matches chronological ordering -- which is what "latest run" means.
_RUN_ID_WIDTH = 19


def new_run_id() -> str:
    """Return a chronologically sortable, collision-resistant run id.

    The nanosecond timestamp orders runs across processes; the in-process
    counter keeps ordering deterministic when two runs start inside the same
    clock tick, which is easy to hit on filesystems with coarse timers.
    """

    return f"{time.time_ns():0{_RUN_ID_WIDTH}d}-{next(_RUN_ID_COUNTER):06d}-{uuid.uuid4().hex[:8]}"


@dataclass
class ReconcileResult:
    """Outcome of a stale-state sweep performed while holding the worker lock."""

    reconciled_run_id: str | None = None
    reconciled_status: str | None = None

    @property
    def changed(self) -> bool:
        return self.reconciled_run_id is not None


class Registry:
    """All persistent worker/run state, rooted at one state directory."""

    def __init__(self, layout: StateLayout) -> None:
        self.layout = layout

    # ------------------------------------------------------------------ #
    # Locking
    # ------------------------------------------------------------------ #

    def lock_for(self, name: str) -> WorkerLock:
        return WorkerLock(self.layout.lock_file(name))

    def creation_lock(self) -> WorkerLock:
        """Serialise worker creation across processes.

        Without it, two MCP processes could each pass the "is this cwd free?"
        check and then both create a worker pointing at the same worktree.
        """

        return WorkerLock(self.layout.registry_lock_file())

    def is_busy(self, name: str) -> bool:
        """Cheap probe: is some process holding this worker's lock?"""

        return is_worker_locked(self.layout.lock_file(name))

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def get_worker(self, name: str) -> WorkerRecord:
        worker = self.layout.read_worker(name)
        if worker is None:
            raise WorkerNotFound(name)
        return worker

    def try_get_worker(self, name: str) -> WorkerRecord | None:
        return self.layout.read_worker(name)

    def list_workers(self) -> list[WorkerRecord]:
        found: list[WorkerRecord] = []
        for name in self.layout.list_worker_names():
            worker = self.layout.read_worker(name)
            if worker is not None:
                found.append(worker)
        return found

    def get_active_run(self, worker: WorkerRecord) -> RunRecord | None:
        if not worker.active_run_id:
            return None
        return self.layout.read_run(worker.name, worker.active_run_id)

    def get_run(self, name: str, run_id: str) -> RunRecord | None:
        return self.layout.read_run(name, run_id)

    def latest_run(self, name: str) -> RunRecord | None:
        """Return the run created most recently, whether or not it has finished.

        "Most recent" is decided purely by run id ordering, so it stays correct
        while a turn is still active -- a worker's ``last_run_id`` only advances
        when a run finalizes.
        """

        run_ids = self.layout.list_run_ids(name)
        if not run_ids:
            return None
        return self.layout.read_run(name, run_ids[-1])

    def last_finished_run(self, name: str) -> RunRecord | None:
        """Return the most recent run that reached a terminal status."""

        worker = self.try_get_worker(name)
        if worker is not None and worker.last_run_id:
            run = self.layout.read_run(name, worker.last_run_id)
            if run is not None and run.is_terminal:
                return run
        for run_id in reversed(self.layout.list_run_ids(name)):
            run = self.layout.read_run(name, run_id)
            if run is not None and run.is_terminal:
                return run
        return None

    def find_worker_for_cwd(self, cwd: str) -> WorkerRecord | None:
        """Return the worker that owns ``cwd``, comparing canonical paths."""

        wanted = os.path.realpath(str(cwd))
        for worker in self.list_workers():
            if os.path.realpath(worker.cwd) == wanted:
                return worker
        return None

    # ------------------------------------------------------------------ #
    # Worker creation
    # ------------------------------------------------------------------ #

    def create_worker(
        self,
        *,
        name: str,
        cwd: str,
        backend: str,
        session_id: str | None = None,
    ) -> WorkerRecord:
        """Create a brand-new worker directory, or raise if the name is taken."""

        worker_dir = self.layout.worker_dir(name)
        if self.layout.worker_file(name).exists() or worker_dir.exists():
            raise WorkerExists(name)
        worker = WorkerRecord(name=name, cwd=cwd, backend=backend, session_id=session_id)
        worker_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(worker_dir, 0o700)
        self.layout.runs_dir(name).mkdir(parents=True, exist_ok=True)
        # Create the lock file up front so a probe never has to create it.
        (worker_dir / LOCK_FILE).touch(mode=0o600, exist_ok=True)
        self.layout.write_worker(worker)
        return worker

    # ------------------------------------------------------------------ #
    # Run lifecycle
    # ------------------------------------------------------------------ #

    def start_run(
        self,
        *,
        worker: WorkerRecord,
        kind: str,
        prompt: str,
        run_id: str | None = None,
    ) -> RunRecord:
        """Create run state and point the worker at it.

        The caller must already hold the worker lock; this is the second half of
        the atomic "decide idle, start turn" step.
        """

        run_id = run_id or new_run_id()
        run_dir = self.layout.run_dir(worker.name, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(run_dir, 0o700)
        atomic_write_text(run_dir / PROMPT_FILE, prompt)
        # Touch the diagnostic files so a reader always finds them.
        for filename in (EVENTS_FILE, STDERR_FILE, RUNNER_LOG_FILE):
            (run_dir / filename).touch(mode=0o600, exist_ok=True)
        run = RunRecord(
            run_id=run_id,
            worker=worker.name,
            kind=kind,
            status=RUN_STARTING,
            cwd=worker.cwd,
            session_id_before=worker.session_id,
            session_id_after=worker.session_id,
        )
        self.layout.write_run(run)
        worker.active_run_id = run_id
        worker.updated_at = now_iso()
        self.layout.write_worker(worker)
        return run

    def mark_run_running(self, run: RunRecord, *, runner_pid: int) -> RunRecord:
        run.status = RUN_RUNNING
        run.runner_pid = runner_pid
        self.layout.write_run(run)
        return run

    def record_backend_process(self, run: RunRecord, *, pid: int, pgid: int) -> RunRecord:
        run.backend_pid = pid
        run.backend_pgid = pgid
        self.layout.write_run(run)
        return run

    def record_session_id(self, run: RunRecord, session_id: str) -> bool:
        """Persist a discovered backend session id on both worker and run.

        Returns ``True`` when anything changed. Called as soon as the opening
        session event arrives, so an interruption later in the same turn still
        leaves the conversation resumable.
        """

        changed = False
        worker = self.layout.read_worker(run.worker)
        if worker is not None and worker.session_id != session_id:
            worker.session_id = session_id
            worker.updated_at = now_iso()
            self.layout.write_worker(worker)
            changed = True
        if run.session_id_after != session_id:
            run.session_id_after = session_id
            self.layout.write_run(run)
            changed = True
        return changed

    def finalize_run(
        self,
        run: RunRecord,
        *,
        status: str,
        exit_code: int | None = None,
        error: str | None = None,
        final_text: str | None = None,
        final_text_truncated: bool = False,
        stop_reason=None,
    ) -> RunRecord:
        """Write terminal run state and clear the worker's active run."""

        run.status = status
        run.exit_code = exit_code
        run.error = error
        run.finished_at = now_iso()
        if final_text is not None:
            run.final_text = final_text
            run.final_text_truncated = final_text_truncated
        if stop_reason is not None:
            run.stop_reason = stop_reason
        self.layout.write_run(run)

        worker = self.layout.read_worker(run.worker)
        if worker is not None:
            worker.active_run_id = None
            worker.last_run_id = run.run_id
            worker.updated_at = now_iso()
            self.layout.write_worker(worker)
        return run

    def request_interrupt(self, name: str, run_id: str) -> Path:
        """Drop the ``interrupt.request`` marker that tells the runner to stop."""

        path = self.layout.interrupt_file(name, run_id)
        atomic_write_text(path, now_iso())
        return path

    def interrupt_requested(self, name: str, run_id: str) -> bool:
        return self.layout.interrupt_file(name, run_id).exists()

    # ------------------------------------------------------------------ #
    # Reconciliation
    # ------------------------------------------------------------------ #

    def reconcile_stale(
        self,
        name: str,
        *,
        assume_locked: bool = False,
    ) -> ReconcileResult:
        """Repair state left behind by a turn-runner that died without finalizing.

        Must be called while this process holds the worker lock, which proves no
        live runner owns the worker. Never signals recorded PIDs: once the lock
        is free the PIDs may already belong to unrelated processes.
        """

        worker = self.try_get_worker(name)
        if worker is None:
            return ReconcileResult()

        result = ReconcileResult()
        target_run_id = worker.active_run_id
        if target_run_id is None and worker.last_run_id:
            # Defensive: a runner could have exited between writing terminal run
            # state and clearing active_run_id (it writes the run first).
            candidate = self.layout.read_run(name, worker.last_run_id)
            if candidate is not None and candidate.status in RUN_ACTIVE_STATUSES:
                target_run_id = candidate.run_id

        if target_run_id is None:
            return result

        run = self.layout.read_run(name, target_run_id)
        if run is None:
            worker.active_run_id = None
            worker.last_run_id = target_run_id
            worker.updated_at = now_iso()
            self.layout.write_worker(worker)
            result.reconciled_run_id = target_run_id
            result.reconciled_status = RUN_INTERRUPTED
            logger.warning(
                "worker %s referenced missing run %s; cleared active run",
                name,
                target_run_id,
            )
            return result

        if run.status not in RUN_ACTIVE_STATUSES:
            # Run already terminal; only the worker pointer is stale.
            if worker.active_run_id == run.run_id:
                worker.active_run_id = None
                worker.last_run_id = run.run_id
                worker.updated_at = now_iso()
                self.layout.write_worker(worker)
            return result

        interrupted = self.interrupt_requested(name, run.run_id)
        status = RUN_INTERRUPTED if interrupted else "failed"
        error = None if interrupted else ERROR_RUNNER_DIED
        logger.warning(
            "reconciling stale run %s for worker %s as %s (worker lock was free)",
            run.run_id,
            name,
            status,
        )
        self.finalize_run(run, status=status, error=error)
        result.reconciled_run_id = run.run_id
        result.reconciled_status = status
        return result

    # ------------------------------------------------------------------ #
    # Derived state
    # ------------------------------------------------------------------ #

    def logical_state(self, name: str, *, busy: bool | None = None) -> tuple[str, RunRecord | None]:
        """Return ``(state, last_run)`` for a worker without mutating state."""

        if busy is None:
            busy = self.is_busy(name)
        if busy:
            return STATE_RUNNING, self.latest_run(name)
        run = self.latest_run(name)
        if run is None:
            return STATE_IDLE, None
        if run.status == RUN_INTERRUPTED:
            return STATE_INTERRUPTED, run
        if run.status == "failed":
            return STATE_FAILED, run
        return STATE_IDLE, run


def read_run_tail(path: Path, limit: int = 4096) -> str:
    """Read at most ``limit`` bytes from the end of a log file."""

    try:
        size = path.stat().st_size
    except OSError:
        return ""
    try:
        with path.open("rb") as handle:
            if size > limit:
                handle.seek(size - limit)
            data = handle.read(limit)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


__all__ = [
    "EVENTS_FILE",
    "INTERRUPT_FILE",
    "PROMPT_FILE",
    "RUNNER_LOG_FILE",
    "RUN_FILE",
    "RUN_RUNNING",
    "RUN_STARTING",
    "STDERR_FILE",
    "CwdInUse",
    "ReconcileResult",
    "Registry",
    "WorkerBusy",
    "WorkerExists",
    "WorkerNotFound",
    "new_run_id",
    "read_json",
    "read_run_tail",
]
