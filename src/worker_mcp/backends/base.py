"""Backend abstraction.

A backend knows two things and nothing else:

* how to build the argv for one turn, and
* how to interpret one line of its structured output.

Everything about workers, locks, runs, and processes stays in the generic
layers, so adding a second backend does not change the public MCP tools.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# BackendEvent kinds. Deliberately small and generic.
EVENT_SESSION = "session"
EVENT_TEXT = "text"
EVENT_THINKING = "thinking"
EVENT_STATUS = "status"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_FINAL = "final"
EVENT_ERROR = "error"
EVENT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class BackendEvent:
    """One normalized backend event."""

    kind: str
    session_id: str | None = None
    text: str | None = None
    final_text: str | None = None
    stop_reason: dict[str, Any] | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class WorkerBackend(Protocol):
    """Narrow interface every worker runtime implements."""

    name: str

    def build_command(self, *, session_id: str | None) -> Sequence[str]:
        """Return the argv for one turn (never a shell string)."""

    def parse_event(self, event: dict[str, Any]) -> BackendEvent:
        """Normalize one decoded JSON event."""

    def describe(self) -> dict[str, Any]:
        """Return a log-safe description of the backend configuration."""
