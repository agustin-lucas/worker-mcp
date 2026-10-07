"""Launching the detached turn-runner with an atomic lock handoff.

The ordering here is the whole point of the design. The MCP server acquires the
worker lock, spawns the runner while passing that *already-held* descriptor, and
only then closes its own copy. The runner inherits the same open file
description, so the lock never lapses: there is no instant at which a second MCP
process could see the worker idle while a runner is about to claim it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .state import RUNNER_LOG_FILE

logger = logging.getLogger(__name__)

RUNNER_MODULE = "worker_mcp.turn_runner"


@dataclass
class RunnerHandle:
    """Bookkeeping for a freshly spawned turn-runner."""

    pid: int
    command: list
    log_path: Path


def runner_command(
    *,
    state_dir: Path,
    worker: str,
    run_id: str,
    lock_fd: int,
) -> list:
    """Build the runner argv (never a shell string)."""

    return [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--state-dir",
        str(state_dir),
        "--worker",
        worker,
        "--run-id",
        run_id,
        "--lock-fd",
        str(lock_fd),
    ]


def launch_runner(
    *,
    config: Config,
    worker: str,
    run_id: str,
    lock_fd: int,
) -> RunnerHandle:
    """Start the detached turn-runner and hand it the worker lock.

    ``lock_fd`` must be the descriptor of a lock this process currently holds.
    On success the returned handle describes a live, detached process that owns
    the lock; the caller must close its own descriptor (see
    :meth:`worker_mcp.locking.WorkerLock.close_without_unlock`) and must *not*
    release it.
    """

    run_dir = config.state_dir / "workers" / worker / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / RUNNER_LOG_FILE
    command = runner_command(
        state_dir=config.state_dir, worker=worker, run_id=run_id, lock_fd=lock_fd
    )

    env = os.environ.copy()
    env["WORKER_MCP_STATE_DIR"] = str(config.state_dir)
    # The runner re-derives everything else from the state directory; the
    # Harness credentials (DEEPSEEK_API_KEY, DSH_HOME, PATH, ...) are inherited
    # untouched and are never written to disk.

    # stdout/stderr must never touch the MCP protocol stream, which owns the
    # server's stdout. Both go to a per-run log file instead.
    log_handle = open(log_path, "ab", buffering=0)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(config.state_dir),
            env=env,
            close_fds=True,
            pass_fds=(lock_fd,),
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        raise
    finally:
        # The child holds its own duplicate of the file description.
        log_handle.close()

    logger.info(
        "launched turn-runner pid=%s worker=%s run=%s log=%s",
        process.pid,
        worker,
        run_id,
        log_path,
    )
    return RunnerHandle(pid=process.pid, command=command, log_path=log_path)
