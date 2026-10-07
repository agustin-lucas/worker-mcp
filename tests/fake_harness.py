#!/usr/bin/env python3
"""A deterministic stand-in for ``dsh --profile headless --json``.

It speaks the *real* contract of the installed DeepSeek Harness rc.2 headless
runner -- ``--profile``, ``--json``, ``--session-id``, task on stdin, newline
delimited JSON events on stdout, exit 0 only on success -- so the worker
process-management tests exercise production code paths without credentials.

Behaviour is driven entirely by ``FAKE_HARNESS_*`` environment variables, which
is convenient because the turn-runner inherits and forwards the environment.

Recording (``FAKE_HARNESS_LOG``) appends one JSON object per invocation:

    {"pid":..., "argv":[...], "cwd":"...", "session_id":"...", "task":"..."}
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

STDOUT = sys.stdout


def env(name: str, default: str = "") -> str:
    return os.environ.get(f"FAKE_HARNESS_{name}", default)


def env_flag(name: str) -> bool:
    return env(name).strip() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float = 0.0) -> float:
    raw = env(name)
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        return default


def env_int(name: str, default: int = 0) -> int:
    raw = env(name)
    try:
        return int(raw) if raw.strip() else default
    except ValueError:
        return default


def emit(payload: dict) -> None:
    STDOUT.write(json.dumps(payload) + "\n")
    STDOUT.flush()


def parse_argv(argv: list) -> dict:
    """Parse just enough of the headless CLI to behave like the real thing."""

    parsed = {"session_id": None, "json": False, "profile": None, "task": []}
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--session-id":
            index += 1
            parsed["session_id"] = argv[index] if index < len(argv) else None
        elif item == "--profile":
            index += 1
            parsed["profile"] = argv[index] if index < len(argv) else None
        elif item == "--json":
            parsed["json"] = True
        elif item.startswith("--"):
            pass
        else:
            parsed["task"].append(item)
        index += 1
    return parsed


def read_task(parsed: dict) -> str:
    if parsed["task"] and parsed["task"] != ["-"]:
        return " ".join(parsed["task"])
    return sys.stdin.read()


def record(entry: dict) -> None:
    path = env("LOG")
    if not path:
        return
    line = json.dumps(entry, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def main() -> int:
    parsed = parse_argv(sys.argv[1:])
    task = read_task(parsed)

    if env_flag("IGNORE_SIGTERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    session_id = env("SESSION_ID", "session-abc")
    resumed = parsed["session_id"]
    effective_session = resumed or session_id

    record(
        {
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "argv": sys.argv[1:],
            "cwd": os.getcwd(),
            "session_id": resumed,
            "effective_session_id": effective_session,
            "task": task,
            "started_at": time.time(),
        }
    )

    pre_sleep = env_float("PRE_SLEEP", 0.0)
    if pre_sleep:
        time.sleep(pre_sleep)

    write_file = env("WRITE_FILE")
    if write_file:
        path = write_file if os.path.isabs(write_file) else os.path.join(os.getcwd(), write_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(env("WRITE_TEXT", "partial work"))
        if env_flag("WRITE_AND_EXIT"):
            return 0

    if env_flag("SPAWN_CHILD"):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"],
        )
        child_marker = env("CHILD_PID_FILE")
        if child_marker:
            with open(child_marker, "w", encoding="utf-8") as handle:
                handle.write(str(child.pid))

    if not env_flag("SKIP_SESSION_EVENT"):
        emit({"type": "session", "sessionId": effective_session, "cwd": os.getcwd()})

    if env_flag("EMIT_ERROR"):
        emit({"type": "error", "message": env("ERROR_MESSAGE", "fake harness error")})
        return env_int("EXIT_CODE", 1)

    if env_flag("TOOL_EVENTS"):
        emit({"type": "status", "phase": "turn_start", "turn": 1})
        emit({"type": "thinking", "text": "considering"})
        emit({"type": "tool_call", "callId": "c1", "tool": "read_file", "input": {"path": "x"}})
        emit(
            {
                "type": "tool_result",
                "callId": "c1",
                "status": "completed",
                "result": "contents",
            }
        )

    sleep_for = env_float("SLEEP", 0.0)
    if sleep_for:
        intermediate = env_float("INTERMEDIATE_AFTER", 0.0)
        if intermediate:
            time.sleep(intermediate)
            emit({"type": "text", "text": "working"})
            time.sleep(max(0.0, sleep_for - intermediate))
        else:
            time.sleep(sleep_for)

    text = env("FINAL_TEXT", "fake harness done")
    if text:
        emit({"type": "text", "text": text})
    emit({"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "completed"}})
    emit({"type": "final", "text": text})

    exit_code = env_int("EXIT_CODE", 0)
    if exit_code != 0:
        # The real runner still writes `final` for an in-turn failure.
        sys.stderr.write("dsh: fake failure\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
