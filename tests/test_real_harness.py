"""Optional real DeepSeek Harness smoke test (plan section 37).

Skipped unless a usable Harness installation and credentials are visible to the
test process. Run explicitly with::

    WORKER_MCP_REAL_HARNESS=1 uv run pytest -m real_harness -s

The task runs in a disposable git repository, never in a real worktree.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from conftest import wait_until
from worker_mcp.config import load_config
from worker_mcp.registry import Registry
from worker_mcp.service import WorkerService
from worker_mcp.state import StateLayout

pytestmark = pytest.mark.real_harness


def harness_available() -> tuple:
    """Return ``(usable, reason)`` for the real Harness."""

    if os.environ.get("WORKER_MCP_REAL_HARNESS") not in {"1", "true", "yes"}:
        return False, "set WORKER_MCP_REAL_HARNESS=1 to run the real Harness smoke test"
    config = load_config()
    executable = config.dsh_command[0]
    if shutil.which(executable) is None:
        return False, f"Harness executable {executable!r} is not on PATH"
    credentials = Path.home() / ".dsh" / ".credentials.yaml"
    if not os.environ.get("DEEPSEEK_API_KEY") and not credentials.exists():
        return False, "no DEEPSEEK_API_KEY and no ~/.dsh/.credentials.yaml"
    return True, ""


@pytest.fixture(autouse=True)
def real_harness_env(tmp_path_factory, monkeypatch: pytest.MonkeyPatch):
    """Skip unless a real Harness is usable, and isolate its DSH_HOME.

    A private DSH_HOME keeps these runs out of a user's real Harness session
    store. ``DEEPSEEK_API_KEY`` is inherited by the runner and the Harness; the
    project itself never reads or records it.
    """

    usable, reason = harness_available()
    if not usable:
        pytest.skip(reason)
    home = tmp_path_factory.mktemp("dsh-home")
    monkeypatch.setenv("DSH_HOME", str(home))


@pytest.fixture(scope="module")
def real_config():
    usable, reason = harness_available()
    if not usable:
        pytest.skip(reason)
    return load_config()


@pytest.fixture
def disposable_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "disposable-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "smoke@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "smoke"], cwd=repo, check=True)
    (repo / "README.md").write_text("disposable\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("WORKER_MCP_STATE_DIR", str(state))
    return repo


def make_service() -> WorkerService:
    config = load_config()
    from worker_mcp.config import ensure_state_dirs

    ensure_state_dirs(config)
    return WorkerService(config, Registry(StateLayout(config.state_dir)))


def wait_idle(service: WorkerService, name: str, timeout: float = 600.0) -> dict:
    assert wait_until(
        lambda: service.status(name=name)["state"] != "running",
        timeout=timeout,
        interval=1.0,
    ), f"worker {name} never went idle"
    return service.status(name=name)


class TestRealHarnessSmoke:
    def test_create_file_and_resume(self, disposable_repo: Path, tmp_path: Path) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text(
            "# Smoke test\n\n"
            'Create a file hello.txt containing exactly "hello worker".\n'
            "Then report what you changed.\n",
            encoding="utf-8",
        )
        service = make_service()

        started = time.monotonic()
        result = service.start(name="smoke", cwd=str(disposable_repo), plan_path=str(plan))
        elapsed = time.monotonic() - started
        assert result["code"] == "STARTED"
        assert elapsed < 10.0, "worker_start must return long before DeepSeek finishes"
        assert service.status(name="smoke")["state"] == "running"

        assert wait_until(
            lambda: service.status(name="smoke")["session_id_available"], timeout=300, interval=1.0
        ), "no DeepSeek session id appeared"
        session_id = service.status(name="smoke")["session_id"]
        assert session_id and session_id.startswith("session-")

        status = wait_idle(service, "smoke")
        outcome = service.result(name="smoke")
        print("status:", json.dumps(status, indent=2))
        print("result:", json.dumps(outcome, indent=2))

        assert outcome["status"] == "completed", outcome["error"]
        assert (disposable_repo / "hello.txt").read_text(encoding="utf-8").strip() == "hello worker"
        assert outcome["final_text"]

        # A second turn must resume the same DeepSeek session.
        follow_up = service.message(
            name="smoke", message="Append a second line 'second pass' to hello.txt."
        )
        assert follow_up["code"] == "STARTED"
        assert service.status(name="smoke")["session_id"] == session_id
        wait_idle(service, "smoke")
        second = service.result(name="smoke")
        assert second["status"] == "completed", second["error"]
        assert second["session_id"] == session_id
        assert "second pass" in (disposable_repo / "hello.txt").read_text(encoding="utf-8")

    def test_interrupt_real_turn_keeps_filesystem_changes(
        self, disposable_repo: Path, tmp_path: Path
    ) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text(
            "# Smoke test\n\n"
            "Create work/started.txt containing 'started'.\n"
            "Then create work/step-1.txt, work/step-2.txt and so on, one file per\n"
            "tool call, until you have created 100 of them. Do not stop early.\n",
            encoding="utf-8",
        )
        service = make_service()
        assert (
            service.start(name="smoke-int", cwd=str(disposable_repo), plan_path=str(plan))["code"]
            == "STARTED"
        )
        # Interrupt *after* the worker has demonstrably started writing, so the
        # "changes are kept" assertion is meaningful while still stopping early.
        work = disposable_repo / "work"
        wait_until(lambda: work.exists() and any(work.glob("*.txt")), timeout=240, interval=1.0)

        interrupt = service.interrupt(name="smoke-int")
        assert interrupt["ok"] is True
        assert interrupt["code"] in {"INTERRUPTED", "ALREADY_FINISHED"}
        assert not service.registry.is_busy("smoke-int")

        # Whatever the worker wrote stays on disk.
        leftovers = sorted(p.name for p in work.glob("*.txt")) if work.exists() else []
        print("files after interrupt:", leftovers)
        assert leftovers, "the interrupted worker had written no files to verify"

        # And the worker is immediately usable again.
        follow_up = service.message(
            name="smoke-int", message="Stop creating files. List what already exists."
        )
        assert follow_up["code"] == "STARTED", follow_up
        wait_idle(service, "smoke-int")
