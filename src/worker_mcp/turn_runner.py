"""Detached turn-runner.

One process per turn. It owns:

* the worker lock it inherited from the MCP server (atomically handed over);
* the DeepSeek Harness subprocess, in its own session/process group;
* the run's diagnostic files and terminal state.

It is independent of the MCP server that launched it: the server exits, the
runner keeps going. When the runner exits -- normally, by crash, or by
``SIGKILL`` -- the kernel releases the worker lock, so a dead runner can never
leave a worker permanently BUSY.
"""

from __future__ import annotations

import argparse
import errno
import json
import logging
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .backends import BackendEvent, build_backend
from .backends.base import (
    EVENT_ERROR,
    EVENT_FINAL,
    EVENT_SESSION,
    EVENT_STATUS,
    EVENT_TEXT,
    EVENT_THINKING,
)
from .config import Config, load_config
from .locking import WorkerLock, adopt_inherited_lock
from .models import (
    ERROR_BACKEND_MISSING,
    ERROR_INTERRUPTED_BEFORE_SPAWN,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INTERRUPTED,
    RunRecord,
)
from .process import process_group_exists, signal_process_group
from .registry import Registry
from .state import (
    EVENTS_FILE,
    PROMPT_FILE,
    STDERR_FILE,
    StateLayout,
)

logger = logging.getLogger("worker_mcp.turn_runner")

#: How long to keep draining Harness stdout after the process has exited. A
#: surviving grandchild can hold the write end of the pipe open forever.
DRAIN_SECONDS = 2.0

#: Granularity of the event loop, so an interrupt marker is noticed promptly.
POLL_SECONDS = 0.25

#: Maximum bytes of run.json's ``final_text``. The lossless answer always stays
#: in ``events.jsonl``.
DEFAULT_FINAL_TEXT_LIMIT = 256 * 1024


class TurnContext:
    """Mutable state accumulated while one turn runs."""

    def __init__(self, run: RunRecord) -> None:
        self.run = run
        self.session_id: str | None = None
        self.final_text: str | None = None
        self.stop_reason: dict[str, Any] | None = None
        self.backend_error: str | None = None
        self.exit_code: int | None = None
        self.interrupt_seen = False
        self.interrupt_deadline: float | None = None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="implementation-worker-runner",
        description="Run exactly one worker turn and record its terminal state.",
    )
    parser.add_argument("--state-dir", required=True, help="absolute state directory")
    parser.add_argument("--worker", required=True, help="logical worker name")
    parser.add_argument("--run-id", required=True, help="run id to execute")
    parser.add_argument(
        "--lock-fd",
        required=True,
        type=int,
        help="inherited descriptor of the worker lock held by this process",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=os.environ.get("WORKER_MCP_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    state_dir = Path(args.state_dir).resolve()
    os.environ["WORKER_MCP_STATE_DIR"] = str(state_dir)
    config = load_config()
    layout = StateLayout(state_dir)
    registry = Registry(layout)
    lock = adopt_inherited_lock(args.lock_fd, layout.lock_file(args.worker))

    try:
        return run_turn(
            config=config,
            registry=registry,
            layout=layout,
            lock=lock,
            worker=args.worker,
            run_id=args.run_id,
        )
    except Exception:  # pragma: no cover - last-resort guard
        logger.exception("turn-runner crashed for worker=%s run=%s", args.worker, args.run_id)
        _emergency_finalize(layout, registry, args.worker, args.run_id)
        return 1


def _emergency_finalize(layout: StateLayout, registry: Registry, worker: str, run_id: str) -> None:
    """Best-effort terminal write so a crash does not leave an active run."""

    try:
        run = layout.read_run(worker, run_id)
        if run is None or run.is_terminal:
            return
        registry.finalize_run(
            run,
            status=RUN_FAILED,
            error="runner_crashed",
        )
    except Exception:  # pragma: no cover - best effort only
        logger.exception("emergency finalize failed")


# --------------------------------------------------------------------------- #
# Turn execution
# --------------------------------------------------------------------------- #


def run_turn(
    *,
    config: Config,
    registry: Registry,
    layout: StateLayout,
    lock: WorkerLock,
    worker: str,
    run_id: str,
) -> int:
    run = layout.read_run(worker, run_id)
    if run is None:
        logger.error("no run state for worker=%s run=%s", worker, run_id)
        return 1

    record = registry.get_worker(worker)
    registry.mark_run_running(run, runner_pid=os.getpid())
    context = TurnContext(run)

    if registry.interrupt_requested(worker, run_id):
        logger.info("interrupt requested before backend spawn; aborting run %s", run_id)
        return _finalize(
            registry=registry,
            layout=layout,
            context=context,
            status=RUN_INTERRUPTED,
            error=ERROR_INTERRUPTED_BEFORE_SPAWN,
        )

    cwd = Path(record.cwd)
    if not cwd.is_dir():
        logger.error("worker cwd %s is not a directory", cwd)
        return _finalize(
            registry=registry,
            layout=layout,
            context=context,
            status=RUN_FAILED,
            error=f"worker cwd is not a directory: {cwd}",
        )

    backend = build_backend(record.backend, config)
    argv = list(backend.build_command(session_id=record.session_id))
    prompt = (layout.run_dir(worker, run_id) / PROMPT_FILE).read_text(encoding="utf-8")

    events_path = layout.run_dir(worker, run_id) / EVENTS_FILE
    stderr_path = layout.run_dir(worker, run_id) / STDERR_FILE

    logger.info(
        "starting backend %s in %s (session=%s)",
        backend.name,
        cwd,
        record.session_id or "<new>",
    )

    try:
        process = _spawn_backend(argv=argv, cwd=cwd, stderr_path=stderr_path)
    except FileNotFoundError:
        logger.error("backend executable not found: %s", argv[0])
        return _finalize(
            registry=registry,
            layout=layout,
            context=context,
            status=RUN_FAILED,
            error=f"{ERROR_BACKEND_MISSING}: {argv[0]}",
        )
    except PermissionError as exc:
        logger.error("backend executable not runnable: %s (%s)", argv[0], exc)
        return _finalize(
            registry=registry,
            layout=layout,
            context=context,
            status=RUN_FAILED,
            error=f"{ERROR_BACKEND_MISSING}: {argv[0]} is not executable",
        )
    except OSError as exc:
        logger.error("backend failed to start: %s", exc)
        return _finalize(
            registry=registry,
            layout=layout,
            context=context,
            status=RUN_FAILED,
            error=f"backend_failed_to_start: {exc}",
        )

    # The backend leads its own session, so its process group id equals its pid.
    registry.record_backend_process(run, pid=process.pid, pgid=process.pid)
    logger.info("backend pid=%s pgid=%s", process.pid, process.pid)

    _start_stdin_writer(process, prompt)

    try:
        with events_path.open("ab") as events_file:
            _pump_events(
                process=process,
                events_file=events_file,
                backend=backend,
                context=context,
                registry=registry,
                config=config,
                worker=worker,
                run_id=run_id,
            )
        context.exit_code = process.wait()
    finally:
        _close_stdin(process)

    # The interrupter may have signalled while we were in the drain loop; the
    # marker on disk is the authoritative record that this turn was interrupted.
    if registry.interrupt_requested(worker, run_id):
        context.interrupt_seen = True

    return _finalize(
        registry=registry,
        layout=layout,
        context=context,
        status=_terminal_status(context, context.exit_code),
        error=context.backend_error,
    )


def _spawn_backend(*, argv: list[str], cwd: Path, stderr_path: Path):
    stderr_handle = open(stderr_path, "ab", buffering=0)
    try:
        # No shell, ever: argv is an array and the prompt goes over stdin.
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            close_fds=True,
            start_new_session=True,
        )
    finally:
        stderr_handle.close()
    return process


def _start_stdin_writer(process, prompt: str) -> None:
    """Deliver the prompt on stdin from a thread.

    A prompt larger than the pipe buffer would otherwise block the runner's
    event loop, which must stay responsive to interrupt markers.
    """

    payload = prompt.encode("utf-8")

    def write() -> None:
        try:
            assert process.stdin is not None
            process.stdin.write(payload)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            logger.warning("backend closed stdin before the prompt was fully written")
        finally:
            _close_stdin(process)

    thread = threading.Thread(target=write, name="runner-stdin", daemon=True)
    thread.start()


def _close_stdin(process) -> None:
    try:
        if process.stdin is not None:
            process.stdin.close()
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Event pump
# --------------------------------------------------------------------------- #


def _pump_events(
    *,
    process,
    events_file,
    backend,
    context: TurnContext,
    registry: Registry,
    config: Config,
    worker: str,
    run_id: str,
) -> None:
    assert process.stdout is not None
    fd = process.stdout.fileno()
    buffer = b""

    def handle_line(raw_line: bytes) -> None:
        events_file.write(raw_line + b"\n")
        events_file.flush()
        text = raw_line.strip()
        if not text:
            return
        try:
            decoded = json.loads(text.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # DeepSeek output is untrusted data for parsing purposes: keep the
            # raw line on disk and carry on.
            logger.debug("non-JSON backend line preserved: %r", raw_line[:200])
            return
        if not isinstance(decoded, dict):
            return
        _apply_event(decoded, backend=backend, context=context, registry=registry)

    def consume(chunk: bytes) -> None:
        nonlocal buffer
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            handle_line(line)

    while True:
        readable = False
        try:
            ready, _, _ = select.select([fd], [], [], POLL_SECONDS)
            readable = bool(ready)
        except (OSError, ValueError):  # pragma: no cover - descriptor closed
            readable = False

        if readable:
            try:
                chunk = os.read(fd, 65536)
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    chunk = b""
                else:
                    chunk = b""
            if chunk:
                consume(chunk)

        _handle_interrupt(
            context=context,
            registry=registry,
            config=config,
            worker=worker,
            run_id=run_id,
        )

        if process.poll() is not None:
            break

    # Bounded drain: a surviving grandchild may hold the pipe's write end.
    deadline = time.monotonic() + DRAIN_SECONDS
    while time.monotonic() < deadline:
        try:
            ready, _, _ = select.select([fd], [], [], 0.1)
        except (OSError, ValueError):
            break
        if not ready:
            break
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        consume(chunk)

    if buffer.strip():
        handle_line(buffer)
        buffer = b""


def _apply_event(
    event: dict[str, Any],
    *,
    backend,
    context: TurnContext,
    registry: Registry,
) -> None:
    parsed: BackendEvent = backend.parse_event(event)

    if parsed.kind == EVENT_SESSION and parsed.session_id:
        context.session_id = parsed.session_id
        registry.record_session_id(context.run, parsed.session_id)
        logger.info("discovered session id %s", parsed.session_id)
        return
    if parsed.kind == EVENT_FINAL:
        context.final_text = parsed.final_text or ""
        return
    if parsed.kind == EVENT_STATUS and parsed.stop_reason is not None:
        context.stop_reason = parsed.stop_reason
        return
    if parsed.kind == EVENT_ERROR:
        context.backend_error = parsed.error or "backend reported an error"
        return
    if parsed.kind in (EVENT_TEXT, EVENT_THINKING):
        return


def _handle_interrupt(
    *,
    context: TurnContext,
    registry: Registry,
    config: Config,
    worker: str,
    run_id: str,
) -> None:
    """React to an ``interrupt.request`` marker for this run.

    The MCP-side interrupter normally signals the process group itself; this
    path covers the window where the marker exists but the interrupter had no
    process group to signal yet, and escalates if SIGTERM was ignored.
    """

    if not registry.interrupt_requested(worker, run_id):
        return

    pgid = context.run.backend_pgid
    if not context.interrupt_seen:
        context.interrupt_seen = True
        context.interrupt_deadline = time.monotonic() + config.interrupt_grace_seconds
        logger.info("interrupt requested for run %s", run_id)
        # The interrupter normally signals too, but it may have written the
        # marker in the window before this process group was recorded -- or
        # after failing to read it. Do not rely on it.
        if pgid is not None and process_group_exists(pgid):
            signal_process_group(pgid, signal.SIGTERM)
        return

    if pgid is None:
        return

    deadline = context.interrupt_deadline or time.monotonic()
    if time.monotonic() < deadline:
        return

    # Past the grace period: escalate so a SIGTERM-ignoring backend cannot keep
    # the worker BUSY forever.
    if process_group_exists(pgid):
        logger.warning("escalating to SIGKILL for process group %s", pgid)
        signal_process_group(pgid, signal.SIGKILL)
        context.interrupt_deadline = time.monotonic() + 5.0


def _terminal_status(context: TurnContext, exit_code: int | None) -> str:
    if context.interrupt_seen:
        return RUN_INTERRUPTED
    if context.backend_error:
        return RUN_FAILED
    reason = context.stop_reason or {}
    if isinstance(reason, dict) and reason.get("kind") == "error":
        return RUN_FAILED
    if exit_code == 0:
        return RUN_COMPLETED
    return RUN_FAILED


def _finalize(
    *,
    registry: Registry,
    layout: StateLayout,
    context: TurnContext,
    status: str,
    error: str | None,
) -> int:
    limit = int(os.environ.get("WORKER_MCP_MAX_FINAL_TEXT_BYTES", DEFAULT_FINAL_TEXT_LIMIT))
    final_text = context.final_text
    truncated = False
    if final_text is not None:
        encoded = final_text.encode("utf-8")
        if len(encoded) > limit:
            final_text = encoded[:limit].decode("utf-8", errors="ignore")
            truncated = True
            logger.warning("final text truncated in run.json; events.jsonl keeps it whole")

    if status == RUN_FAILED and not error:
        error = _failure_diagnostic(layout, context)

    registry.finalize_run(
        context.run,
        status=status,
        exit_code=context.exit_code,
        error=error,
        final_text=final_text,
        final_text_truncated=truncated,
        stop_reason=context.stop_reason,
    )
    logger.info(
        "run %s finished status=%s exit=%s session=%s",
        context.run.run_id,
        status,
        context.exit_code,
        context.session_id or "<none>",
    )
    # Closing the inherited lock descriptor happens on process exit; the OS
    # releases the lock, which is what makes the worker available again.
    return 0 if status in (RUN_COMPLETED, RUN_INTERRUPTED) else 1


def _failure_diagnostic(layout: StateLayout, context: TurnContext) -> str:
    exit_code = context.exit_code
    base = f"backend exited with code {exit_code}" if exit_code is not None else "backend failed"
    stderr_path = layout.run_dir(context.run.worker, context.run.run_id) / STDERR_FILE
    try:
        tail = stderr_path.read_bytes()[-1000:].decode("utf-8", errors="replace").strip()
    except OSError:
        tail = ""
    return f"{base}: {tail}" if tail else base


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
