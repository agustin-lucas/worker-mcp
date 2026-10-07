"""DeepSeek Harness backend (``dsh --profile headless --json``).

Verified against the installed DSH rc.2 desktop runtime
(``@deepseek-ai/dsh-headless``), which is newer than some published
documentation:

* ``--json`` **is** supported and writes newline-delimited run events to stdout.
* ``--session-id <id>`` **is** supported and adopts the exact persisted Session.
  An unknown id is a hard error; there is no get-or-create. The first turn
  therefore omits the flag and the Harness generates ``session-<uuid>`` itself.
* The task is a positional argument, or stdin when the argument is omitted or is
  a lone ``-``. We always deliver the prompt on stdin so arbitrary model text
  never becomes an argument.
* Exit code 0 means the turn completed; 1 means it aborted or errored.
* The run is refused when the process cwd differs from the cwd recorded in the
  Session, which is why a worker's cwd is immutable.

The projected stream opens with ``session`` and closes with ``final``::

    {"type":"session","sessionId":"session-...","cwd":"/path"}
    {"type":"status","phase":"turn_start","turn":1}
    {"type":"thinking","text":"..."}
    {"type":"text","text":"..."}
    {"type":"tool_call","callId":"...","tool":"...","input":{...}}
    {"type":"tool_result","callId":"...","status":"completed","result":"..."}
    {"type":"status","phase":"turn_end","turn":1,"reason":{"kind":"completed"}}
    {"type":"final","text":"<final assistant text>"}
    {"type":"error","message":"..."}     # runner-level failure, no final event

Parsing stays defensive: unknown types become ``EVENT_UNKNOWN`` and the raw
events are preserved verbatim in ``events.jsonl`` regardless.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from typing import Any

from .base import (
    EVENT_ERROR,
    EVENT_FINAL,
    EVENT_SESSION,
    EVENT_STATUS,
    EVENT_TEXT,
    EVENT_THINKING,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    EVENT_UNKNOWN,
    BackendEvent,
)

DEFAULT_PROFILE = "headless"


class DeepSeekHarnessBackend:
    """Backend that drives one ``dsh --profile headless`` process per turn."""

    name = "deepseek-harness"

    def __init__(self, command: Sequence[str], profile: str = DEFAULT_PROFILE) -> None:
        if not command:
            raise ValueError("DeepSeek Harness command must not be empty")
        self._command: tuple[str, ...] = tuple(command)
        self._profile = profile

    @property
    def command(self) -> tuple[str, ...]:
        return self._command

    def build_command(self, *, session_id: str | None) -> Sequence[str]:
        argv = list(self._command)
        argv.extend(["--profile", self._profile, "--json"])
        if session_id:
            argv.extend(["--session-id", session_id])
        return argv

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self._command),
            "profile": self._profile,
            "executable": self.executable_path() or self._command[0],
        }

    def executable_path(self) -> str | None:
        """Best-effort resolution of the executable for diagnostics."""

        return shutil.which(self._command[0])

    def missing_executable(self) -> bool:
        """Whether the configured executable looks unavailable.

        Only meaningful when the command's first element is not an absolute
        path; an absolute path is checked directly for existence.
        """

        head = self._command[0]
        if "/" in head:
            import os

            return not os.path.exists(head)
        return shutil.which(head) is None

    # ------------------------------------------------------------------ #
    # Event parsing
    # ------------------------------------------------------------------ #

    def parse_event(self, event: dict[str, Any]) -> BackendEvent:
        if not isinstance(event, dict):
            return BackendEvent(kind=EVENT_UNKNOWN, raw={"value": event})

        kind = event.get("type")
        if kind == "session":
            return BackendEvent(
                kind=EVENT_SESSION,
                session_id=_first_str(event, "sessionId", "session_id", "session"),
                raw=event,
            )
        if kind == EVENT_TEXT:
            return BackendEvent(kind=EVENT_TEXT, text=_as_text(event.get("text")), raw=event)
        if kind == EVENT_THINKING:
            return BackendEvent(kind=EVENT_THINKING, text=_as_text(event.get("text")), raw=event)
        if kind == EVENT_FINAL:
            return BackendEvent(
                kind=EVENT_FINAL, final_text=_as_text(event.get("text")) or "", raw=event
            )
        if kind == EVENT_ERROR:
            message = _as_text(event.get("message")) or "backend reported an error"
            return BackendEvent(kind=EVENT_ERROR, error=message, raw=event)
        if kind == EVENT_STATUS:
            reason = event.get("reason")
            stop_reason = reason if isinstance(reason, dict) else None
            return BackendEvent(kind=EVENT_STATUS, stop_reason=stop_reason, raw=event)
        if kind in (EVENT_TOOL_CALL, EVENT_TOOL_RESULT):
            return BackendEvent(kind=str(kind), raw=event)
        return BackendEvent(kind=EVENT_UNKNOWN, raw=event)

    # ------------------------------------------------------------------ #
    # Exit-code interpretation
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_exit(exit_code: int | None, *, turn_failed: bool) -> str:
        """Map a Harness exit status onto a terminal run disposition."""

        if turn_failed:
            return "failed"
        if exit_code == 0:
            return "completed"
        return "failed"


def _first_str(event: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _as_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None
