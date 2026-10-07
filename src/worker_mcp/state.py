"""Atomic on-disk state IO and the state directory layout.

Every mutable JSON document is written to a temporary file in the *same*
directory, flushed, fsynced, and then ``os.replace``d over the destination. A
reader therefore never observes a half-written document, which matters because
several independent MCP processes (Codex, Claude Code) share this directory.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .models import RunRecord, WorkerRecord

#: Conservative worker-name grammar. No separators, so a name can never be used
#: to construct a path outside the state directory.
WORKER_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")

WORKER_FILE = "worker.json"
LOCKS_DIR = "locks"
REGISTRY_LOCK_FILE = "registry.lock"
LOCK_FILE = "lock"
RUNS_DIR = "runs"
RUN_FILE = "run.json"
PROMPT_FILE = "prompt.txt"
EVENTS_FILE = "events.jsonl"
STDERR_FILE = "stderr.log"
RUNNER_LOG_FILE = "runner.log"
INTERRUPT_FILE = "interrupt.request"


class StateError(RuntimeError):
    """Raised when persisted state cannot be read or is structurally invalid."""


def validate_worker_name(name: str) -> str:
    """Return ``name`` if it is a legal worker name, else raise ``ValueError``."""

    if not isinstance(name, str) or not WORKER_NAME_RE.match(name):
        raise ValueError(
            "worker name must match ^[A-Za-z0-9._-]{1,100}$ (no path separators, no '..')"
        )
    if name in {".", ".."}:
        raise ValueError("worker name must not be '.' or '..'")
    return name


# --------------------------------------------------------------------------- #
# Atomic primitives
# --------------------------------------------------------------------------- #


def atomic_write_bytes(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Write ``data`` to ``path`` atomically."""

    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _fsync_directory(directory)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Serialize ``payload`` to JSON and write it atomically."""

    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, data)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _fsync_directory(directory: Path) -> None:
    """Best-effort durability for the directory entry itself."""

    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(dir_fd)
    except OSError:  # pragma: no cover - some filesystems refuse
        pass
    finally:
        os.close(dir_fd)


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning ``None`` when the file does not exist."""

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    if not raw.strip():
        return None
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StateError(f"corrupt JSON state at {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise StateError(f"corrupt JSON state at {path}: expected an object")
    return decoded


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


class StateLayout:
    """Path helper rooted at one state directory."""

    def __init__(self, state_dir: Path) -> None:
        self.root = Path(state_dir)
        self.workers = self.root / "workers"

    @property
    def locks(self) -> Path:
        return self.root / LOCKS_DIR

    def registry_lock_file(self) -> Path:
        """Global lock guarding "check the registry, then create a worker"."""

        return self.locks / REGISTRY_LOCK_FILE

    def worker_dir(self, name: str) -> Path:
        return self.workers / validate_worker_name(name)

    def worker_file(self, name: str) -> Path:
        return self.worker_dir(name) / WORKER_FILE

    def lock_file(self, name: str) -> Path:
        return self.worker_dir(name) / LOCK_FILE

    def runs_dir(self, name: str) -> Path:
        return self.worker_dir(name) / RUNS_DIR

    def run_dir(self, name: str, run_id: str) -> Path:
        return self.runs_dir(name) / run_id

    def run_file(self, name: str, run_id: str) -> Path:
        return self.run_dir(name, run_id) / RUN_FILE

    def interrupt_file(self, name: str, run_id: str) -> Path:
        return self.run_dir(name, run_id) / INTERRUPT_FILE

    # -- typed accessors ---------------------------------------------------- #

    def read_worker(self, name: str) -> WorkerRecord | None:
        raw = read_json(self.worker_file(name))
        if raw is None:
            return None
        return WorkerRecord.from_dict(raw)

    def write_worker(self, worker: WorkerRecord) -> None:
        atomic_write_json(self.worker_file(worker.name), worker.to_dict())

    def read_run(self, name: str, run_id: str) -> RunRecord | None:
        raw = read_json(self.run_file(name, run_id))
        if raw is None:
            return None
        return RunRecord.from_dict(raw)

    def write_run(self, run: RunRecord) -> None:
        atomic_write_json(self.run_file(run.worker, run.run_id), run.to_dict())

    def list_worker_names(self) -> list[str]:
        if not self.workers.is_dir():
            return []
        names: list[str] = []
        for entry in sorted(self.workers.iterdir()):
            if not entry.is_dir():
                continue
            if WORKER_NAME_RE.match(entry.name) and (entry / WORKER_FILE).is_file():
                names.append(entry.name)
        return names

    def list_run_ids(self, name: str) -> list[str]:
        runs = self.runs_dir(name)
        if not runs.is_dir():
            return []
        return sorted(entry.name for entry in runs.iterdir() if entry.is_dir())
