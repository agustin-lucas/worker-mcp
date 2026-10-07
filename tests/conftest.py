"""Shared fixtures for the worker-MCP tests.

Almost every test drives the *real* machinery: real worker locks, a real
detached turn-runner subprocess, and a fake Harness executable that speaks the
same CLI/JSON contract as the installed DeepSeek Harness. Only the model itself
is fake.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
FAKE_HARNESS = TESTS_DIR / "fake_harness.py"

if str(SRC_DIR) not in sys.path:  # pragma: no cover - editable-install fallback
    sys.path.insert(0, str(SRC_DIR))

from worker_mcp.config import Config, ensure_state_dirs
from worker_mcp.registry import Registry
from worker_mcp.service import WorkerService
from worker_mcp.state import StateLayout


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private on-disk registry per test."""

    directory = tmp_path / "state"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORKER_MCP_STATE_DIR", str(directory))
    monkeypatch.delenv("WORKER_MCP_DSH_COMMAND_JSON", raising=False)
    return directory


@pytest.fixture
def harness_log(tmp_path: Path) -> Path:
    return tmp_path / "harness-invocations.jsonl"


@pytest.fixture
def fake_harness(
    monkeypatch: pytest.MonkeyPatch,
    harness_log: Path,
) -> Callable[..., None]:
    """Point the worker at the fake Harness and configure its behaviour.

    Returns a callable that sets ``FAKE_HARNESS_*`` variables, mirroring the
    environment the detached runner will forward to the Harness process.
    """

    monkeypatch.setenv(
        "WORKER_MCP_DSH_COMMAND_JSON",
        json.dumps([sys.executable, str(FAKE_HARNESS)]),
    )
    monkeypatch.setenv("FAKE_HARNESS_LOG", str(harness_log))
    # Keep the interrupt grace period short so the tests stay fast while still
    # exercising the real SIGTERM -> wait -> SIGKILL escalation logic.
    monkeypatch.setenv("WORKER_MCP_INTERRUPT_GRACE_SECONDS", "3")

    def configure(**options: object) -> None:
        for key, value in options.items():
            name = f"FAKE_HARNESS_{key.upper()}"
            if isinstance(value, bool):
                monkeypatch.setenv(name, "1" if value else "0")
            else:
                monkeypatch.setenv(name, str(value))

    return configure


@pytest.fixture
def config(state_dir: Path, fake_harness: Callable[..., None]) -> Config:
    """Config is snapshotted at server start-up, so it must be built *after*
    the fake-Harness environment is in place."""

    from worker_mcp.config import load_config

    return load_config()


@pytest.fixture
def layout(config: Config) -> StateLayout:
    return StateLayout(config.state_dir)


@pytest.fixture
def registry(layout: StateLayout) -> Registry:
    return Registry(layout)


@pytest.fixture
def service(config: Config, registry: Registry, state_dir: Path) -> WorkerService:
    ensure_state_dirs(config)
    return WorkerService(config, registry)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def wait_until(
    predicate: Callable[[], bool], timeout: float = 20.0, interval: float = 0.05
) -> bool:
    """Poll ``predicate`` until it is true or ``timeout`` elapses."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def write_plan(
    tmp_path: Path, text: str = "# Plan\n\nDo the thing.\n", name: str = "plan.md"
) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def read_invocations(log: Path) -> list[dict]:
    if not log.exists():
        return []
    entries: list[dict] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def harness_count(log: Path) -> int:
    return len(read_invocations(log))


def last_invocation(log: Path) -> dict | None:
    entries = read_invocations(log)
    return entries[-1] if entries else None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def start_worker(
    service: WorkerService,
    tmp_path: Path,
    *,
    name: str = "auth-refactor",
    worktree: Path | None = None,
    plan_text: str = "# Plan\n\nDo the thing.\n",
    additional_instructions: str | None = None,
) -> dict:
    """Convenience wrapper around ``worker_start`` with sane defaults."""

    directory = worktree or (tmp_path / f"worktree-{name}")
    directory.mkdir(parents=True, exist_ok=True)
    plan = write_plan(tmp_path, plan_text, name=f"{name}-plan.md")
    return service.start(
        name=name,
        cwd=str(directory),
        plan_path=str(plan),
        additional_instructions=additional_instructions,
    )


# --------------------------------------------------------------------------- #
# Readiness helpers
#
# ``worker_start`` returns as soon as the detached runner owns the lock, which
# is deliberately before the runner has published its own pid or the backend's.
# Tests that inspect those fields must wait for them explicitly.
# --------------------------------------------------------------------------- #


def wait_running(service: WorkerService, name: str, timeout: float = 30.0) -> None:
    """Wait until the worker reports ``running``."""

    assert wait_until(lambda: service.status(name=name)["state"] == "running", timeout=timeout), (
        f"worker {name} never started running"
    )


def wait_idle(service: WorkerService, name: str, timeout: float = 30.0) -> None:
    """Wait until the worker reports anything other than ``running``."""

    assert wait_until(lambda: service.status(name=name)["state"] != "running", timeout=timeout), (
        f"worker {name} never left the running state"
    )


def wait_for_session(service: WorkerService, name: str, timeout: float = 30.0) -> str:
    """Wait until the backend's session id has been persisted."""

    assert wait_until(lambda: service.status(name=name)["session_id_available"], timeout=timeout), (
        f"no session id was recorded for worker {name}"
    )
    session_id = service.status(name=name)["session_id"]
    assert session_id
    return session_id


def wait_for_run_field(registry: Registry, name: str, field: str, timeout: float = 30.0):
    """Wait until the latest run publishes ``field``, then return that run."""

    def ready() -> bool:
        run = registry.latest_run(name)
        return run is not None and getattr(run, field, None) is not None

    assert wait_until(ready, timeout=timeout), f"run field {field!r} was never recorded for {name}"
    run = registry.latest_run(name)
    assert run is not None
    return run
