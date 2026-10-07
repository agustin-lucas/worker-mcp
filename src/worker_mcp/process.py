"""Process-group helpers used to stop a DeepSeek Harness turn.

The Harness subprocess is launched with ``start_new_session=True``, so it is the
leader of a fresh process group. Killing that group reaches the Harness *and*
every tool child it spawned (``bash``, ``pytest``, git, ...) without touching
the turn-runner, which must stay alive long enough to record the interruption.
"""

from __future__ import annotations

import errno
import logging
import os
import signal
import time

logger = logging.getLogger(__name__)


def pid_exists(pid: int | None) -> bool:
    """Return ``True`` when ``pid`` names a live process we may signal."""

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, but owned by someone else.
        return True
    except OSError as exc:  # pragma: no cover - defensive
        if exc.errno == errno.EPERM:
            return True
        return False
    return True


def process_group_exists(pgid: int | None) -> bool:
    """Return ``True`` when ``pgid`` names a live process group."""

    if not _signalable_pgid(pgid):
        return False
    try:
        os.killpg(int(pgid), 0)  # type: ignore[arg-type]
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:  # pragma: no cover - defensive
        return exc.errno == errno.EPERM
    return True


def _signalable_pgid(pgid: int | None) -> bool:
    if pgid is None or pgid <= 1:
        return False
    # Never signal our own process group: a turn's backend is always a session
    # leader in its own group, so a match here means the recorded value is junk.
    try:
        if int(pgid) == os.getpgrp():
            return False
    except OSError:  # pragma: no cover - defensive
        return False
    return True


def signal_process_group(pgid: int | None, sig: signal.Signals) -> bool:
    """Send ``sig`` to ``pgid``. Returns whether the signal was delivered."""

    if not _signalable_pgid(pgid):
        logger.warning("refusing to signal process group %r", pgid)
        return False
    target = int(pgid)  # type: ignore[arg-type]
    try:
        os.killpg(target, sig)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - defensive
        logger.warning("permission denied signalling process group %s", target)
        return False
    return True


def signal_pid(pid: int | None, sig: signal.Signals) -> bool:
    """Send ``sig`` to a single process. Returns whether it was delivered."""

    if pid is None or pid <= 1 or pid == os.getpid():
        return False
    try:
        os.kill(int(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def wait_for_process_group_exit(pgid: int | None, timeout: float) -> bool:
    """Poll until ``pgid`` has no live members, or ``timeout`` elapses."""

    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        if not process_group_exists(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def terminate_process_group(
    pgid: int | None,
    *,
    grace_seconds: float,
    poll_interval: float = 0.05,
) -> bool:
    """SIGTERM then, if needed, SIGKILL a backend process group.

    Returns ``True`` when the group is gone when this function returns.
    """

    if not process_group_exists(pgid):
        return True
    signal_process_group(pgid, signal.SIGTERM)
    if wait_for_process_group_exit(pgid, grace_seconds):
        return True
    logger.warning("process group %s survived SIGTERM; sending SIGKILL", pgid)
    signal_process_group(pgid, signal.SIGKILL)
    return wait_for_process_group_exit(pgid, max(grace_seconds, 5.0))


def wait_for(
    predicate,
    timeout: float,
    interval: float = 0.05,
) -> bool:
    """Poll ``predicate`` until it is true or ``timeout`` elapses."""

    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
