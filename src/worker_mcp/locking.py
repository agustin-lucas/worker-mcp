"""Per-worker process-safe locking.

The lock is a POSIX advisory ``flock`` on ``workers/<name>/lock``. It means:

    this worker currently has an active turn, or is in the atomic process of
    starting one.

Two properties make it the right primitive here:

* It conflicts between *different open file descriptions*, even inside one
  process, so a fresh ``open`` + ``LOCK_EX | LOCK_NB`` is a true
  "is a turn active?" probe rather than a same-process formality.
* It is owned by the open file description, not the descriptor number. When the
  MCP server hands its descriptor to the detached turn-runner and then closes
  its own copy, the lock never lapses: it transfers atomically. And when the
  runner dies for any reason -- crash, ``SIGKILL``, reboot -- the kernel drops
  the lock, so a dead runner can never leave a worker permanently BUSY.
"""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path


class LockUnavailable(RuntimeError):
    """Raised when a non-blocking lock acquisition fails."""


class WorkerLock:
    """An exclusive advisory lock on one worker's lock file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def try_acquire(self) -> bool:
        """Try to take the lock without blocking. Returns ``True`` on success."""

        if self._fd is not None:
            raise RuntimeError("lock already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_CLOEXEC keeps the descriptor out of unrelated children; subprocess's
        # pass_fds re-enables exactly the descriptor we intend to hand over.
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def acquire(self) -> None:
        """Take the lock without blocking, raising when it is held elsewhere."""

        if not self.try_acquire():
            raise LockUnavailable(f"worker lock {self.path} is held by another turn")

    def acquire_with_timeout(self, timeout: float, interval: float = 0.02) -> bool:
        """Wait up to ``timeout`` seconds for the lock.

        Used for the registry lock, which is held only for the duration of a
        check-and-create rather than for a whole turn.
        """

        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if self.try_acquire():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval)

    @property
    def acquired(self) -> bool:
        return self._fd is not None

    @property
    def fd(self) -> int:
        if self._fd is None:
            raise RuntimeError("lock is not held")
        return self._fd

    def detach_fd(self) -> int:
        """Give up ownership of the descriptor without unlocking it.

        Used by the MCP server after handing the descriptor to the detached
        runner: the runner now owns the lock, and closing *our* descriptor is
        what makes the handoff complete (the object no longer tracks it).
        """

        if self._fd is None:
            raise RuntimeError("lock is not held")
        fd = self._fd
        self._fd = None
        return fd

    def close_without_unlock(self) -> None:
        """Close our descriptor, leaving the lock alive in whoever else holds it."""

        fd = self.detach_fd()
        os.close(fd)

    def release(self) -> None:
        """Unlock and close. Safe to call when the lock is not held."""

        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # -- context manager ---------------------------------------------------- #

    def __enter__(self) -> WorkerLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.release()
        except Exception:
            pass


def is_worker_locked(path: Path) -> bool:
    """Return ``True`` when some process currently holds the worker lock.

    This is a pure probe: it takes and immediately releases a fresh lock, so it
    never disturbs the ownership of an existing holder.
    """

    probe = WorkerLock(Path(path))
    if not probe.try_acquire():
        return True
    probe.release()
    return False


def adopt_inherited_lock(fd: int, path: Path) -> WorkerLock:
    """Wrap a descriptor inherited from the parent as an owned :class:`WorkerLock`.

    The child must not re-``flock`` the descriptor: it already holds the lock
    through the open file description it inherited. Marking it non-inheritable
    keeps it out of the backend subprocess the runner is about to spawn.
    """

    os.set_inheritable(fd, False)
    lock = WorkerLock(path)
    lock._fd = fd
    return lock
