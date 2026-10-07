"""Orchestration behind the MCP tools.

Everything here is synchronous and blocking only where the plan requires it:
``worker_start``/``worker_message`` return immediately after the detached runner
owns the lock, and ``worker_interrupt`` blocks until the worker is genuinely
available again.
"""

from __future__ import annotations

import logging
import shutil
import signal
import time
from pathlib import Path
from typing import Any

from .backends import build_backend
from .config import Config, ensure_state_dirs
from .launcher import launch_runner
from .models import (
    CODE_ALREADY_EXISTS,
    CODE_ALREADY_FINISHED,
    CODE_BUSY,
    CODE_CWD_IN_USE,
    CODE_FAILED,
    CODE_INTERRUPTED,
    CODE_INVALID_ARGUMENT,
    CODE_NO_SESSION,
    CODE_NOT_FOUND,
    CODE_STARTED,
    RUN_ACTIVE_STATUSES,
    RunRecord,
    WorkerRecord,
)
from .process import signal_pid, signal_process_group
from .registry import Registry, WorkerExists
from .state import StateLayout, validate_worker_name

logger = logging.getLogger(__name__)

#: How long to wait for a lock holder to publish its active run id.
ACTIVE_RUN_WAIT_SECONDS = 3.0

#: Poll interval while waiting for a worker lock to be released.
LOCK_POLL_SECONDS = 0.05

#: How long to wait for the short-lived registry lock during worker creation.
REGISTRY_LOCK_TIMEOUT_SECONDS = 10.0

INITIAL_PROMPT_TEMPLATE = """You are the implementation worker for this task.

The architecture and implementation decisions in the supplied plan have
already been made by the lead engineer. Implement that plan faithfully.

Inspect the repository as needed. Make the required code changes and tests.
Run the relevant validation before finishing.

Do not make unrelated architectural changes. If something in the plan is
impossible or materially incorrect, report it clearly instead of silently
redesigning the system.

--- BEGIN IMPLEMENTATION PLAN ---

{plan}

--- END IMPLEMENTATION PLAN ---
{additional}"""

ADDITIONAL_TEMPLATE = """
--- ADDITIONAL INSTRUCTIONS ---

{instructions}

--- END ADDITIONAL INSTRUCTIONS ---
"""


# --------------------------------------------------------------------------- #
# Structured result helpers
# --------------------------------------------------------------------------- #


def ok(code: str, **fields: Any) -> dict[str, Any]:
    """Build a successful structured result."""

    return {"ok": True, "code": code, **fields}


def err(code: str, message: str, **fields: Any) -> dict[str, Any]:
    """Build a structured failure result.

    These are *expected operational outcomes*, not tool crashes: they travel as
    normal tool results carrying ``ok: false`` and a stable ``code`` so the
    controller never has to parse English to tell BUSY from a broken server.
    """

    return {"ok": False, "code": code, "message": message, **fields}


def build_initial_prompt(plan_text: str, additional_instructions: str | None) -> str:
    additional = ""
    if additional_instructions and additional_instructions.strip():
        additional = ADDITIONAL_TEMPLATE.format(instructions=additional_instructions.strip())
    return INITIAL_PROMPT_TEMPLATE.format(plan=plan_text.rstrip("\n"), additional=additional)


def truncate_text(text: str | None, limit: int) -> tuple:
    """Return ``(text, truncated)`` bounded to ``limit`` UTF-8 bytes."""

    if text is None:
        return None, False
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


class WorkerService:
    """The full worker lifecycle, independent of the MCP surface."""

    def __init__(self, config: Config, registry: Registry | None = None) -> None:
        self.config = config
        self.layout = StateLayout(config.state_dir)
        self.registry = registry or Registry(self.layout)

    # ------------------------------------------------------------------ #
    # worker_start
    # ------------------------------------------------------------------ #

    def start(
        self,
        *,
        name: str,
        cwd: str,
        plan_path: str,
        additional_instructions: str | None = None,
    ) -> dict[str, Any]:
        try:
            validate_worker_name(name)
        except ValueError as exc:
            return err(CODE_INVALID_ARGUMENT, str(exc), worker=name)

        resolved_cwd = self._canonical_directory(cwd)
        if resolved_cwd is None:
            return err(
                CODE_INVALID_ARGUMENT,
                f"cwd must be an existing directory: {cwd}",
                worker=name,
            )

        plan, plan_error = self._read_plan(plan_path)
        if plan_error is not None:
            return err(CODE_INVALID_ARGUMENT, plan_error, worker=name)

        prompt = build_initial_prompt(plan, additional_instructions)

        # Name and cwd uniqueness are check-then-act across processes, so hold
        # the registry lock for the whole check-and-create.
        creation_lock = self.registry.creation_lock()
        if not creation_lock.acquire_with_timeout(REGISTRY_LOCK_TIMEOUT_SECONDS):
            return err(
                CODE_FAILED,
                "timed out waiting for the worker registry lock",
                worker=name,
            )
        try:
            if self.registry.try_get_worker(name) is not None:
                return err(
                    CODE_ALREADY_EXISTS,
                    f"a worker named {name!r} already exists",
                    worker=name,
                )

            owner = self.registry.find_worker_for_cwd(str(resolved_cwd))
            if owner is not None:
                return err(
                    CODE_CWD_IN_USE,
                    f"cwd {resolved_cwd} is already owned by worker {owner.name!r}",
                    worker=name,
                    existing_worker=owner.name,
                    cwd=str(resolved_cwd),
                )

            try:
                worker = self.registry.create_worker(
                    name=name, cwd=str(resolved_cwd), backend=self.config.backend
                )
            except WorkerExists:
                return err(
                    CODE_ALREADY_EXISTS, f"a worker named {name!r} already exists", worker=name
                )
            except OSError as exc:
                return err(CODE_FAILED, f"could not create worker state: {exc}", worker=name)
        finally:
            creation_lock.release()

        result = self._launch(worker=worker, kind="start", prompt=prompt)
        if not result.get("ok"):
            # A brand-new worker that never started would otherwise hold its
            # name and cwd forever: v1 has no worker_delete tool.
            self._rollback_new_worker(name)
        return result

    # ------------------------------------------------------------------ #
    # worker_message
    # ------------------------------------------------------------------ #

    def message(self, *, name: str, message: str) -> dict[str, Any]:
        worker = self.registry.try_get_worker(name)
        if worker is None:
            return err(CODE_NOT_FOUND, f"no worker named {name!r}", worker=name)

        if not isinstance(message, str) or not message.strip():
            return err(CODE_INVALID_ARGUMENT, "message must be a non-empty string", worker=name)

        if self.registry.is_busy(name):
            active = self.registry.get_active_run(worker) or self.registry.latest_run(name)
            return err(
                CODE_BUSY,
                "Worker currently has an active turn.",
                worker=name,
                active_run_id=active.run_id if active else worker.active_run_id,
            )

        if not worker.session_id:
            return err(
                CODE_NO_SESSION,
                (
                    "Worker is idle but has no DeepSeek session id yet, so there is "
                    "nothing to resume. Wait for the opening turn to report its "
                    "session event."
                ),
                worker=name,
            )

        return self._launch(worker=worker, kind="message", prompt=message)

    # ------------------------------------------------------------------ #
    # Shared turn launch (atomic lock handoff)
    # ------------------------------------------------------------------ #

    def _launch(self, *, worker: WorkerRecord, kind: str, prompt: str) -> dict[str, Any]:
        lock = self.registry.lock_for(worker.name)
        if not lock.try_acquire():
            active = self.registry.get_active_run(worker) or self.registry.latest_run(worker.name)
            return err(
                CODE_BUSY,
                "Worker currently has an active turn.",
                worker=worker.name,
                active_run_id=active.run_id if active else worker.active_run_id,
            )

        run: RunRecord | None = None
        try:
            run = self.registry.start_run(worker=worker, kind=kind, prompt=prompt)
            handle = launch_runner(
                config=self.config,
                worker=worker.name,
                run_id=run.run_id,
                lock_fd=lock.fd,
            )
        except BaseException as exc:
            logger.exception("failed to launch turn-runner for worker %s", worker.name)
            if run is not None:
                try:
                    self.registry.finalize_run(run, status="failed", error=f"launch_failed: {exc}")
                except Exception:  # pragma: no cover - best effort
                    logger.exception("rollback of run state failed")
            lock.release()
            return err(
                CODE_FAILED,
                f"could not launch the turn-runner: {exc}",
                worker=worker.name,
                run_id=run.run_id if run else None,
            )

        # Handoff complete: the runner now owns the lock through the inherited
        # descriptor. Closing ours must not unlock it.
        lock.close_without_unlock()

        return ok(
            CODE_STARTED,
            worker=worker.name,
            run_id=run.run_id,
            runner_pid=handle.pid,
            cwd=worker.cwd,
            session_id=worker.session_id,
            kind=kind,
        )

    def _rollback_new_worker(self, name: str) -> None:
        try:
            validate_worker_name(name)
            worker_dir = self.layout.worker_dir(name)
            if worker_dir.parent == self.layout.workers and worker_dir.is_dir():
                shutil.rmtree(worker_dir, ignore_errors=True)
                logger.info("rolled back partially created worker %s", name)
        except Exception:  # pragma: no cover - best effort
            logger.exception("could not roll back worker %s", name)

    # ------------------------------------------------------------------ #
    # worker_status
    # ------------------------------------------------------------------ #

    def status(self, *, name: str) -> dict[str, Any]:
        worker = self.registry.try_get_worker(name)
        if worker is None:
            return err(CODE_NOT_FOUND, f"no worker named {name!r}", worker=name)

        busy, reconciled = self._probe_and_reconcile(name)
        state, _ = self.registry.logical_state(name, busy=busy)
        worker = self.registry.try_get_worker(name) or worker
        # ``last_run_*`` describes the last run that actually finished; while a
        # turn is active that is not the newest run on disk.
        run = self.registry.last_finished_run(name) or self.registry.latest_run(name)

        payload: dict[str, Any] = {
            "worker": name,
            "state": state,
            "backend": worker.backend,
            "cwd": worker.cwd,
            "session_id_available": bool(worker.session_id),
            "session_id": worker.session_id,
            "active_run_id": worker.active_run_id,
            "last_run_id": run.run_id if run else worker.last_run_id,
            "started_at": run.started_at if run else None,
            "last_run_status": run.status if run else None,
            "updated_at": worker.updated_at,
        }
        if reconciled is not None and reconciled.changed:
            payload["reconciled"] = {
                "run_id": reconciled.reconciled_run_id,
                "status": reconciled.reconciled_status,
                "reason": "runner exited without recording terminal state",
            }
        return ok(state.upper(), **payload)

    # ------------------------------------------------------------------ #
    # worker_result
    # ------------------------------------------------------------------ #

    def result(self, *, name: str, run_id: str | None = None) -> dict[str, Any]:
        worker = self.registry.try_get_worker(name)
        if worker is None:
            return err(CODE_NOT_FOUND, f"no worker named {name!r}", worker=name)

        _, reconciled = self._probe_and_reconcile(name)
        run = self.registry.get_run(name, run_id) if run_id else self.registry.latest_run(name)
        if run is None:
            return err(
                CODE_NOT_FOUND,
                f"no run recorded for worker {name!r}" + (f" with id {run_id!r}" if run_id else ""),
                worker=name,
                run_id=run_id,
            )

        final_text, _ = truncate_text(run.final_text, self.config.max_final_text_bytes)
        payload: dict[str, Any] = {
            "worker": name,
            "run_id": run.run_id,
            "kind": run.kind,
            "status": run.status,
            "cwd": run.cwd,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "exit_code": run.exit_code,
            "interrupted": run.status == "interrupted",
            "session_id_available": bool(run.session_id_after or worker.session_id),
            "session_id": run.session_id_after or worker.session_id,
            "final_text": final_text,
            "final_text_truncated": run.final_text_truncated or (final_text != run.final_text),
            "error": run.error,
            "stop_reason": run.stop_reason,
        }
        if reconciled is not None and reconciled.changed:
            payload["reconciled"] = {
                "run_id": reconciled.reconciled_run_id,
                "status": reconciled.reconciled_status,
            }
        return ok(run.status.upper(), **payload)

    # ------------------------------------------------------------------ #
    # worker_interrupt
    # ------------------------------------------------------------------ #

    def interrupt(self, *, name: str) -> dict[str, Any]:
        """Stop the active turn and do not return until the worker is free."""

        worker = self.registry.try_get_worker(name)
        if worker is None:
            return err(CODE_NOT_FOUND, f"no worker named {name!r}", worker=name)

        lock = self.registry.lock_for(name)
        if lock.try_acquire():
            # The worker is already free. Reconcile anything a dead runner left
            # behind, then report that there was nothing to stop.
            try:
                reconciled = self.registry.reconcile_stale(name, assume_locked=True)
            finally:
                lock.release()
            return ok(
                CODE_ALREADY_FINISHED,
                worker=name,
                run_id=reconciled.reconciled_run_id or worker.last_run_id,
                status=reconciled.reconciled_status,
                message="Worker had no active turn; state reconciled.",
            )

        run = self._resolve_active_run(name)
        if run is None:
            # Lock held but no run published. Wait for the lock rather than
            # guessing at a process to kill.
            released = self._wait_for_lock_release(name, self.config.interrupt_grace_seconds)
            if not released:
                return err(
                    CODE_FAILED,
                    "a turn holds the worker lock but published no run state",
                    worker=name,
                )
            return ok(
                CODE_ALREADY_FINISHED,
                worker=name,
                message="Active turn finished before it could be interrupted.",
            )

        # 1. Marker first: the runner honours it even if we lose the race to
        #    signal, and it makes an unrecorded death reconcile as interrupted.
        self.registry.request_interrupt(name, run.run_id)

        # 2. SIGTERM the Harness process group, after re-confirming the run.
        #    ``backend_pgid`` may not have been published yet, which is exactly
        #    why the marker exists; the runner covers that case itself.
        self._signal_backend(name, run.run_id, signal.SIGTERM)

        # 3. Wait for the runner to finalize and drop the lock.
        released = self._wait_for_lock_release(name, self.config.interrupt_grace_seconds)
        if not released:
            logger.warning(
                "worker %s still busy after %.1fs; escalating",
                name,
                self.config.interrupt_grace_seconds,
            )
            self._signal_backend(name, run.run_id, signal.SIGKILL)
            released = self._wait_for_lock_release(name, 5.0)
        if not released:
            # Last resort: the runner itself holds the lock. Killing it cannot
            # lose the lock (the OS releases it) and cannot revert any file the
            # worker already wrote.
            runner_pid = self._current_runner_pid(name, run.run_id)
            if runner_pid is not None:
                logger.warning("SIGKILLing stuck turn-runner pid=%s", runner_pid)
                signal_pid(runner_pid, signal.SIGKILL)
            released = self._wait_for_lock_release(name, 5.0)
        if not released:
            return err(
                CODE_FAILED,
                "could not stop the active turn; the worker is still busy",
                worker=name,
                run_id=run.run_id,
            )

        # 4. Reconcile under the lock so the run never stays "running" forever.
        final = self._reconcile_now(name) or self.registry.get_run(name, run.run_id)
        status = final.status if final else "interrupted"
        return ok(
            CODE_INTERRUPTED,
            worker=name,
            run_id=run.run_id,
            status=status,
            session_id=worker.session_id,
            session_id_available=bool(worker.session_id),
            message=(
                "Active turn stopped. Filesystem changes already made were kept; "
                "the worker can accept another message."
            ),
        )

    # ------------------------------------------------------------------ #
    # worker_list
    # ------------------------------------------------------------------ #

    def list_workers(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for worker in self.registry.list_workers():
            busy = self.registry.is_busy(worker.name)
            state, run = self.registry.logical_state(worker.name, busy=busy)
            entries.append(
                {
                    "name": worker.name,
                    "state": state,
                    "cwd": worker.cwd,
                    "last_run_status": run.status if run else None,
                    "session_id_available": bool(worker.session_id),
                }
            )
        entries.sort(key=lambda item: item["name"])
        return ok("OK", workers=entries, count=len(entries))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _probe_and_reconcile(self, name: str):
        """Return ``(busy, reconcile_result_or_None)``.

        When the lock is free this also repairs state a dead runner left behind,
        which is what stops a crashed turn from making the worker look BUSY.
        """

        lock = self.registry.lock_for(name)
        if not lock.try_acquire():
            return True, None
        try:
            reconciled = self.registry.reconcile_stale(name, assume_locked=True)
        finally:
            lock.release()
        return False, reconciled

    def _reconcile_now(self, name: str):
        lock = self.registry.lock_for(name)
        if not lock.try_acquire():
            return None
        try:
            self.registry.reconcile_stale(name, assume_locked=True)
        finally:
            lock.release()
        return self.registry.latest_run(name)

    def _wait_for_lock_release(self, name: str, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if not self.registry.is_busy(name):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(LOCK_POLL_SECONDS)

    def _still_active(self, name: str, run_id: str) -> bool:
        """Re-check that ``run_id`` is still the worker's active run."""

        if not self.registry.is_busy(name):
            return False
        worker = self.registry.try_get_worker(name)
        return bool(worker and worker.active_run_id == run_id)

    def _signal_backend(self, name: str, run_id: str, sig: signal.Signals) -> bool:
        """Signal the backend process group for ``run_id`` if it is still live.

        The process group is re-read from disk immediately before signalling, so
        a PID recorded for a run that has since finished is never used.
        """

        if not self._still_active(name, run_id):
            return False
        run = self.registry.get_run(name, run_id)
        if run is None or not run.backend_pgid:
            return False
        return signal_process_group(run.backend_pgid, sig)

    def _current_runner_pid(self, name: str, run_id: str) -> int | None:
        if not self._still_active(name, run_id):
            return None
        run = self.registry.get_run(name, run_id)
        return run.runner_pid if run is not None else None

    def _resolve_active_run(self, name: str) -> RunRecord | None:
        """Find the run behind a held worker lock.

        ``worker.json`` is written before the runner is spawned, so this is
        normally immediate; the short poll covers the window between the lock
        being taken and the run record being published.
        """

        deadline = time.monotonic() + ACTIVE_RUN_WAIT_SECONDS
        while True:
            worker = self.registry.try_get_worker(name)
            if worker and worker.active_run_id:
                run = self.registry.get_run(name, worker.active_run_id)
                if run is not None:
                    return run
                return RunRecord(
                    run_id=worker.active_run_id,
                    worker=name,
                    kind="unknown",
                    status="running",
                    cwd=worker.cwd,
                )
            if time.monotonic() >= deadline:
                break
            time.sleep(LOCK_POLL_SECONDS)

        # Fallback: scan the run directories for a live run.
        for run_id in reversed(self.layout.list_run_ids(name)):
            run = self.registry.get_run(name, run_id)
            if run is not None and run.status in RUN_ACTIVE_STATUSES:
                return run
        return None

    def _canonical_directory(self, cwd: str) -> Path | None:
        try:
            resolved = Path(cwd).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        return resolved if resolved.is_dir() else None

    def _read_plan(self, plan_path: str) -> tuple:
        if not isinstance(plan_path, str) or not plan_path.strip():
            return "", "plan_path must be a non-empty path to a Markdown plan file"
        try:
            path = Path(plan_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return "", f"plan file does not exist: {plan_path}"
        if not path.is_file():
            return "", f"plan_path is not a regular file: {plan_path}"
        try:
            size = path.stat().st_size
        except OSError as exc:
            return "", f"could not stat the plan file: {exc}"
        if size > self.config.max_plan_bytes:
            return "", (
                f"plan file is {size} bytes, above the {self.config.max_plan_bytes} byte limit "
                f"(raise WORKER_MCP_MAX_PLAN_BYTES to allow it)"
            )
        try:
            return path.read_text(encoding="utf-8"), None
        except (OSError, UnicodeDecodeError) as exc:
            return "", f"could not read the plan file: {exc}"

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def backend_info(self) -> dict[str, Any]:
        backend = build_backend(self.config.backend, self.config)
        info = backend.describe()
        info["executable_found"] = not backend.missing_executable()
        return info

    def ensure_dirs(self) -> None:
        ensure_state_dirs(self.config)
