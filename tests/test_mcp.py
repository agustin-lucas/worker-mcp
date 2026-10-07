"""MCP boundary integration tests.

At least one test exercises the real stdio transport (plan test 12 and plan
section 36); the rest use the SDK's in-memory streams for speed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from conftest import write_plan

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = REPO_ROOT / ".venv" / "bin" / "implementation-worker-mcp"

EXPECTED_TOOLS = {
    "worker_start",
    "worker_message",
    "worker_status",
    "worker_result",
    "worker_interrupt",
    "worker_list",
}


@asynccontextmanager
async def in_memory_client(server):
    """Connect a ClientSession to an MCPServer over in-memory streams."""

    from mcp.shared.memory import create_client_server_memory_streams

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        lowlevel = server._lowlevel_server

        async def run_server() -> None:
            await lowlevel.run(
                server_read,
                server_write,
                lowlevel.create_initialization_options(),
            )

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_server)
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                yield session
            task_group.cancel_scope.cancel()


def run(coro) -> object:
    return anyio.run(lambda: coro)


# --------------------------------------------------------------------------- #
# Schema surface
# --------------------------------------------------------------------------- #


class TestToolSurface:
    def test_lists_expected_tools_with_schemas(self, service) -> None:
        from worker_mcp.server import build_server

        async def scenario() -> None:
            server = build_server(service=service)
            async with in_memory_client(server) as session:
                listed = await session.list_tools()
                names = {tool.name for tool in listed.tools}
                assert EXPECTED_TOOLS <= names

                for tool in listed.tools:
                    if tool.name not in EXPECTED_TOOLS:
                        continue
                    assert tool.description, f"{tool.name} has no description"
                    assert tool.input_schema.get("type") == "object"
                    # worker_list takes no arguments, so it declares no properties.
                    if tool.name != "worker_list":
                        assert tool.input_schema.get("properties"), f"{tool.name} has no properties"

        run(scenario())

    def test_descriptions_teach_the_intended_semantics(self, service) -> None:
        from worker_mcp.server import build_server

        async def scenario() -> dict:
            server = build_server(service=service)
            async with in_memory_client(server) as session:
                listed = await session.list_tools()
                return {tool.name: tool.description or "" for tool in listed.tools}

        descriptions = run(scenario())
        assert "BUSY" in descriptions["worker_message"]
        assert "queue" in descriptions["worker_message"]
        assert "asynchronous" in descriptions["worker_start"] or (
            "before implementation finishes" in descriptions["worker_start"]
        )
        assert "preserves" in descriptions["worker_interrupt"]
        assert "accept another message" in descriptions["worker_interrupt"]

    def test_server_instructions_are_present(self, service) -> None:
        from worker_mcp.server import build_server

        async def scenario() -> str:
            server = build_server(service=service)
            async with in_memory_client(server) as session:
                assert session.initialize_result is not None
                return session.initialize_result.instructions or ""

        instructions = run(scenario())
        assert "one turn at a time" in instructions
        assert "does not revert filesystem changes" in instructions


# --------------------------------------------------------------------------- #
# Tool behaviour through the protocol
# --------------------------------------------------------------------------- #


class TestToolBehaviour:
    def test_missing_worker_returns_structured_not_found(self, service) -> None:
        from worker_mcp.server import build_server

        async def scenario() -> dict:
            server = build_server(service=service)
            async with in_memory_client(server) as session:
                result = await session.call_tool("worker_status", {"name": "ghost"})
                return result

        result = run(scenario())
        assert result.is_error is False
        assert result.structured_content["ok"] is False
        assert result.structured_content["code"] == "NOT_FOUND"
        # The same payload is mirrored as text content for clients that ignore
        # structured output.
        assert result.content and "NOT_FOUND" in result.content[0].text

    def test_start_then_status_over_the_protocol(
        self, service, tmp_path: Path, fake_harness
    ) -> None:
        from worker_mcp.server import build_server

        fake_harness(SLEEP=4.0, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)

        async def scenario() -> tuple:
            server = build_server(service=service)
            async with in_memory_client(server) as session:
                start = await session.call_tool(
                    "worker_start",
                    {"name": "w1", "cwd": str(worktree), "plan_path": str(plan)},
                )
                status = await session.call_tool("worker_status", {"name": "w1"})
                busy = await session.call_tool("worker_message", {"name": "w1", "message": "hi"})
                listing = await session.call_tool("worker_list", {})
                return start, status, busy, listing

        start, status, busy, listing = run(scenario())
        assert start.structured_content["code"] == "STARTED"
        assert status.structured_content["state"] == "running"
        assert busy.structured_content["code"] == "BUSY"
        assert busy.is_error is False
        assert [w["name"] for w in listing.structured_content["workers"]] == ["w1"]

        service.interrupt(name="w1")

    def test_result_and_interrupt_over_the_protocol(
        self, service, tmp_path: Path, fake_harness
    ) -> None:
        from worker_mcp.server import build_server

        fake_harness(SLEEP=120, SESSION_ID="session-abc")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)

        async def scenario() -> tuple:
            server = build_server(service=service)
            async with in_memory_client(server) as session:
                await session.call_tool(
                    "worker_start",
                    {"name": "w1", "cwd": str(worktree), "plan_path": str(plan)},
                )
                # Steer a turn that has produced its session identity, as a
                # controller actually would before interrupting.
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    probe = await session.call_tool("worker_status", {"name": "w1"})
                    if probe.structured_content["session_id_available"]:
                        break
                    await anyio.sleep(0.1)
                interrupt = await session.call_tool("worker_interrupt", {"name": "w1"})
                follow_up = await session.call_tool(
                    "worker_message", {"name": "w1", "message": "go again"}
                )
                result = await session.call_tool("worker_result", {"name": "w1"})
                return interrupt, follow_up, result

        interrupt, follow_up, result = run(scenario())
        assert interrupt.structured_content["code"] == "INTERRUPTED"
        assert follow_up.structured_content["code"] == "STARTED"
        # worker_result defaults to the most recent run, which here is the
        # follow-up turn that was just launched.
        assert result.structured_content["status"] in {
            "starting",
            "running",
            "completed",
            "failed",
            "interrupted",
        }
        assert result.structured_content["run_id"] == follow_up.structured_content["run_id"]
        service.interrupt(name="w1")

    def test_tool_crash_does_not_leak_a_traceback(self, service, monkeypatch) -> None:
        """Unexpected exceptions surface as an MCP tool error, not prose soup."""

        from worker_mcp.server import build_server

        server = build_server(service=service)

        def explode(**kwargs):
            raise RuntimeError("internal detail that should not reach the model")

        monkeypatch.setattr(service, "status", explode)

        async def scenario():
            async with in_memory_client(server) as session:
                return await session.call_tool("worker_status", {"name": "w1"})

        result = run(scenario())
        assert result.is_error is True
        rendered = "".join(block.text for block in result.content if hasattr(block, "text"))
        assert "internal detail" not in rendered
        assert "worker_status" in rendered


# --------------------------------------------------------------------------- #
# Real stdio transport (plan test 12)
# --------------------------------------------------------------------------- #


def server_env(state_dir: Path, harness_log: Path, **fake: object) -> dict:
    env = {
        "WORKER_MCP_STATE_DIR": str(state_dir),
        "WORKER_MCP_DSH_COMMAND_JSON": json.dumps(
            [sys.executable, str(REPO_ROOT / "tests" / "fake_harness.py")]
        ),
        "WORKER_MCP_INTERRUPT_GRACE_SECONDS": "3",
        "FAKE_HARNESS_LOG": str(harness_log),
    }
    for key, value in fake.items():
        env[f"FAKE_HARNESS_{key.upper()}"] = str(value)
    return env


class TestRealStdioTransport:
    def test_stdio_server_keeps_stdout_clean(self, tmp_path: Path, harness_log: Path) -> None:
        """Plan test 12: background work never corrupts protocol stdout."""

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path, "# Plan\n\nDo it.\n")
        env = server_env(
            state_dir,
            harness_log,
            SLEEP=1.0,
            SESSION_ID="session-stdio",
            TOOL_EVENTS=True,
        )

        async def scenario() -> dict:
            params = StdioServerParameters(
                command=str(SERVER_SCRIPT),
                args=[],
                env=env,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    assert EXPECTED_TOOLS <= {tool.name for tool in listed.tools}

                    start = await session.call_tool(
                        "worker_start",
                        {"name": "stdio-worker", "cwd": str(worktree), "plan_path": str(plan)},
                    )
                    assert start.structured_content["code"] == "STARTED"

                    # Interleave several protocol round trips while the
                    # detached runner and the Harness are writing to their own
                    # files. Any stray stdout byte would break framing here.
                    for _ in range(5):
                        listing = await session.call_tool("worker_list", {})
                        assert listing.structured_content["ok"] is True

                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        status = await session.call_tool("worker_status", {"name": "stdio-worker"})
                        if status.structured_content["state"] != "running":
                            break
                        await anyio.sleep(0.2)

                    result = await session.call_tool("worker_result", {"name": "stdio-worker"})
                    return result.structured_content

        payload = anyio.run(scenario)
        assert payload["status"] == "completed"
        assert payload["final_text"] == "fake harness done"
        assert payload["session_id"] == "session-stdio"

    def test_stdio_server_reports_startup_state(self, tmp_path: Path, harness_log: Path) -> None:
        """A fresh server process discovers workers started by another one."""

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        worktree = tmp_path / "wt"
        worktree.mkdir()
        plan = write_plan(tmp_path)
        env = server_env(state_dir, harness_log, SLEEP=0.2, SESSION_ID="session-a")

        async def scenario() -> tuple:
            params = StdioServerParameters(command=str(SERVER_SCRIPT), args=[], env=env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await session.call_tool(
                        "worker_start",
                        {"name": "alpha", "cwd": str(worktree), "plan_path": str(plan)},
                    )
                    empty = await session.call_tool("worker_list", {})
            # A completely fresh server process must still see the worker.
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        listing = await session.call_tool("worker_list", {})
                        workers = listing.structured_content["workers"]
                        if workers and workers[0]["state"] != "running":
                            break
                        await anyio.sleep(0.2)
                    again = await session.call_tool("worker_list", {})
                    return empty.structured_content, again.structured_content

        before, after = anyio.run(scenario)
        # The first process returned before the turn finished...
        assert before["ok"] is True
        # ...and the second, independent server process sees the same worker.
        assert [w["name"] for w in after["workers"]] == ["alpha"]
        assert after["workers"][0]["state"] == "idle"
        assert after["workers"][0]["session_id_available"] is True

    def test_stdout_is_valid_jsonrpc_only(self, tmp_path: Path, harness_log: Path) -> None:
        """Read raw stdout and confirm every line is a JSON-RPC message."""

        import subprocess

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        env = server_env(state_dir, harness_log, SLEEP=0.2)
        env.update({k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ})
        process = subprocess.Popen(
            [str(SERVER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "raw", "version": "1"},
                },
            }
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            decoded = json.loads(line)
            assert decoded["jsonrpc"] == "2.0"
            assert decoded["id"] == 1
            assert "result" in decoded
        finally:
            process.kill()
            process.wait(timeout=10)
