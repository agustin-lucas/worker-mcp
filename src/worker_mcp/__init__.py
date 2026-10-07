"""Persistent implementation-worker MCP server.

A stdio MCP server that lets Codex or Claude Code drive long-running DeepSeek
Harness implementation workers asynchronously.

The package is split into layers:

* :mod:`worker_mcp.server` -- the MCP surface (tools and descriptions).
* :mod:`worker_mcp.service` -- orchestration logic behind those tools.
* :mod:`worker_mcp.registry` / :mod:`worker_mcp.state` / :mod:`worker_mcp.locking`
  -- the disk-backed worker registry and process-safe locking.
* :mod:`worker_mcp.launcher` / :mod:`worker_mcp.turn_runner` -- the detached
  turn-runner process that owns a turn.
* :mod:`worker_mcp.backends` -- worker runtimes; only DeepSeek Harness in v1.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
