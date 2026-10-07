"""MCP surface: the stdio server Codex and Claude Code talk to.

This module owns *only* the protocol-facing layer. All orchestration lives in
:mod:`worker_mcp.service`, so the tool descriptions and the result envelope can
change without touching the worker machinery.

Nothing in this process may write to stdout except the MCP transport itself:
diagnostics go to stderr (see :func:`configure_logging`) or to per-run log files.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

from . import __version__
from .config import Config, ConfigError, ensure_state_dirs, load_config
from .service import WorkerService

logger = logging.getLogger("worker_mcp.server")

SERVER_NAME = "implementation-worker"

SERVER_INSTRUCTIONS = """\
Implementation workers execute one turn at a time. worker_start and
worker_message start work asynchronously and return before the work finishes.
If worker_message returns BUSY, either check worker_status later or call
worker_interrupt when immediate steering matters. Interrupting stops the active
turn but does not revert filesystem changes already made. A worker is a
persistent DeepSeek session bound to one working directory, and the same worker
is visible to any MCP client sharing the state directory.
"""


# --------------------------------------------------------------------------- #
# Result envelope
# --------------------------------------------------------------------------- #


def tool_result(payload: dict[str, Any]) -> CallToolResult:
    """Wrap a structured payload as a normal tool result.

    Expected operational outcomes (BUSY, NOT_FOUND, ...) are returned with
    ``is_error=False`` so the structured body always reaches the controller
    intact; a genuine crash still surfaces as an MCP tool error.
    """

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, indent=2, sort_keys=False))],
        structured_content=payload,
        is_error=False,
    )


def build_server(service: WorkerService | None = None, config: Config | None = None) -> MCPServer:
    """Construct the MCP server and register the worker tools."""

    service = service or WorkerService(config or load_config())
    server = MCPServer(
        name=SERVER_NAME,
        title="Implementation Worker MCP",
        version=__version__,
        instructions=SERVER_INSTRUCTIONS,
    )

    @server.tool(
        name="worker_start",
        description=(
            "Create a persistent implementation worker bound to one absolute working "
            "directory (a git worktree) and start its first DeepSeek turn from a "
            "Markdown implementation plan. Starts work asynchronously and returns "
            "before implementation finishes; poll worker_status and read worker_result "
            "when the worker goes idle. Returns ALREADY_EXISTS if the name is taken and "
            "CWD_IN_USE if another worker already owns that directory."
        ),
    )
    def worker_start(
        name: str,
        cwd: str,
        plan_path: str,
        additional_instructions: str | None = None,
    ) -> CallToolResult:
        """Start a new implementation worker asynchronously.

        Args:
            name: Stable logical worker name, ^[A-Za-z0-9._-]{1,100}$.
            cwd: Absolute path of the worktree the worker will edit.
            plan_path: Path to the Markdown plan; its contents are embedded in the
                first task, so the file may live outside the worktree.
            additional_instructions: Optional extra guidance appended to the plan.
        """

        return tool_result(
            service.start(
                name=name,
                cwd=cwd,
                plan_path=plan_path,
                additional_instructions=additional_instructions,
            )
        )

    @server.tool(
        name="worker_message",
        description=(
            "Send the next turn to an existing worker's persisted DeepSeek session. "
            "Starts work asynchronously and returns before the turn finishes. Returns "
            "BUSY rather than queueing if the worker currently has an active turn; in "
            "that case either wait and check worker_status, or call worker_interrupt and "
            "then send this message. Returns NO_SESSION if the worker is idle but never "
            "recorded a session id."
        ),
    )
    def worker_message(name: str, message: str) -> CallToolResult:
        """Send a follow-up turn to a worker.

        Args:
            name: Logical worker name.
            message: The message to deliver; it runs in the worker's original cwd.
        """

        return tool_result(service.message(name=name, message=message))

    @server.tool(
        name="worker_status",
        description=(
            "Cheap status check for one worker: running/idle/failed/interrupted, its cwd, "
            "whether a session id is available, and the active and last run ids. Never "
            "returns logs. Also repairs state left behind by a turn-runner that died, so "
            "a crashed turn does not leave the worker permanently BUSY."
        ),
    )
    def worker_status(name: str) -> CallToolResult:
        """Return concise worker status.

        Args:
            name: Logical worker name.
        """

        return tool_result(service.status(name=name))

    @server.tool(
        name="worker_result",
        description=(
            "Read the outcome of a worker's run: status, timestamps, exit code, final "
            "assistant text, and a concise error when it failed. Defaults to the most "
            "recent run; pass run_id for an older one. Does not dump the raw event stream."
        ),
    )
    def worker_result(name: str, run_id: str | None = None) -> CallToolResult:
        """Return the latest (or a specific) run result.

        Args:
            name: Logical worker name.
            run_id: Optional run id; defaults to the most recent run.
        """

        return tool_result(service.result(name=name, run_id=run_id))

    @server.tool(
        name="worker_interrupt",
        description=(
            "Stop the worker's active turn. Terminates the DeepSeek process tree "
            "(including tools it spawned), preserves every filesystem change already "
            "made, and does not return until the worker can accept another message -- so "
            "worker_message immediately afterwards will not return BUSY. Returns "
            "ALREADY_FINISHED when the turn had already ended."
        ),
    )
    def worker_interrupt(name: str) -> CallToolResult:
        """Interrupt a worker's active turn and wait until it is free.

        Args:
            name: Logical worker name.
        """

        return tool_result(service.interrupt(name=name))

    @server.tool(
        name="worker_list",
        description=(
            "List every known logical worker with its state, cwd, and last run status. "
            "Workers are shared through the on-disk registry, so this includes workers "
            "started by other MCP clients."
        ),
    )
    def worker_list() -> CallToolResult:
        """List all known workers concisely."""

        return tool_result(service.list_workers())

    return server


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def configure_logging() -> None:
    """Send all diagnostics to stderr; stdout belongs to the MCP transport."""

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    configure_logging()
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"implementation-worker-mcp: {exc}", file=sys.stderr)
        return 2

    ensure_state_dirs(config)
    service = WorkerService(config)

    info = service.backend_info()
    logger.info("state_dir=%s", config.state_dir)
    logger.info("backend=%s", info)
    if not info.get("executable_found"):
        logger.warning(
            "DeepSeek Harness executable %r was not found on PATH; turns will fail "
            "until WORKER_MCP_DSH_COMMAND_JSON or PATH is fixed",
            info.get("command", [None])[0],
        )

    server = build_server(service=service, config=config)
    server.run("stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
