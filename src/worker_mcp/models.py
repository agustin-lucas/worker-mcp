"""On-disk record shapes and stable result codes.

Every persistent structure carries a ``schema_version`` so a future version can
migrate or refuse old state instead of silently misreading it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- #
# Run statuses
# --------------------------------------------------------------------------- #

RUN_STARTING = "starting"
RUN_RUNNING = "running"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_INTERRUPTED = "interrupted"

RUN_TERMINAL_STATUSES = frozenset({RUN_COMPLETED, RUN_FAILED, RUN_INTERRUPTED})
RUN_ACTIVE_STATUSES = frozenset({RUN_STARTING, RUN_RUNNING})

# --------------------------------------------------------------------------- #
# Logical worker states reported through MCP
# --------------------------------------------------------------------------- #

STATE_RUNNING = "running"
STATE_IDLE = "idle"
STATE_FAILED = "failed"
STATE_INTERRUPTED = "interrupted"

# --------------------------------------------------------------------------- #
# Stable machine-readable result codes
# --------------------------------------------------------------------------- #

CODE_STARTED = "STARTED"
CODE_BUSY = "BUSY"
CODE_IDLE = "IDLE"
CODE_INTERRUPTED = "INTERRUPTED"
CODE_ALREADY_FINISHED = "ALREADY_FINISHED"
CODE_NOT_FOUND = "NOT_FOUND"
CODE_ALREADY_EXISTS = "ALREADY_EXISTS"
CODE_NO_SESSION = "NO_SESSION"
CODE_INVALID_ARGUMENT = "INVALID_ARGUMENT"
CODE_FAILED = "FAILED"

# Extension beyond the codes listed in the plan: two logical workers may not
# share one canonical cwd (plan section 28), and telling that apart from a
# duplicate *name* is materially more useful to a controller.
CODE_CWD_IN_USE = "CWD_IN_USE"

# Run-level failure reasons recorded in ``run.json["error"]``.
ERROR_RUNNER_DIED = "runner_exited_without_final_state"
ERROR_BACKEND_MISSING = "backend_executable_missing"
ERROR_INTERRUPTED_BEFORE_SPAWN = "interrupted_before_backend_spawn"


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class WorkerRecord:
    """One logical worker: name + fixed cwd + persistent backend session."""

    name: str
    cwd: str
    backend: str
    schema_version: int = SCHEMA_VERSION
    session_id: str | None = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    active_run_id: str | None = None
    last_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerRecord:
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            name=str(data["name"]),
            backend=str(data.get("backend", "deepseek-harness")),
            cwd=str(data["cwd"]),
            session_id=data.get("session_id"),
            created_at=str(data.get("created_at") or now_iso()),
            updated_at=str(data.get("updated_at") or now_iso()),
            active_run_id=data.get("active_run_id"),
            last_run_id=data.get("last_run_id"),
        )


@dataclass
class RunRecord:
    """One turn: everything a later MCP process needs to observe or stop it."""

    run_id: str
    worker: str
    kind: str
    status: str
    cwd: str
    schema_version: int = SCHEMA_VERSION
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    runner_pid: int | None = None
    backend_pid: int | None = None
    backend_pgid: int | None = None
    exit_code: int | None = None
    session_id_before: str | None = None
    session_id_after: str | None = None
    stop_reason: dict[str, Any] | None = None
    final_text: str | None = None
    final_text_truncated: bool = False
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in RUN_TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})
