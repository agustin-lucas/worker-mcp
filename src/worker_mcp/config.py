"""Runtime configuration, read from the environment.

Nothing here touches credentials. API keys and Harness home directories are
inherited by the detached runner through the normal process environment and are
never copied into worker state or logs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

STATE_DIR_ENV = "WORKER_MCP_STATE_DIR"
DSH_COMMAND_ENV = "WORKER_MCP_DSH_COMMAND_JSON"
INTERRUPT_GRACE_ENV = "WORKER_MCP_INTERRUPT_GRACE_SECONDS"
MAX_PLAN_BYTES_ENV = "WORKER_MCP_MAX_PLAN_BYTES"
MAX_FINAL_TEXT_ENV = "WORKER_MCP_MAX_FINAL_TEXT_BYTES"

BACKEND_DEEPSEEK_HARNESS = "deepseek-harness"

DEFAULT_STATE_DIR_NAME = "implementation-worker-mcp"
DEFAULT_INTERRUPT_GRACE_SECONDS = 10.0
DEFAULT_MAX_PLAN_BYTES = 256 * 1024
DEFAULT_MAX_FINAL_TEXT_BYTES = 256 * 1024

#: Default Harness command. ``dsh`` is expected on ``PATH``; override with
#: ``WORKER_MCP_DSH_COMMAND_JSON`` when the installation lives elsewhere.
DEFAULT_DSH_COMMAND: tuple[str, ...] = ("dsh",)


class ConfigError(ValueError):
    """Raised when the environment configures the server inconsistently."""


def default_state_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the XDG-compatible default state directory."""

    environment = os.environ if env is None else env
    explicit = environment.get(STATE_DIR_ENV)
    if explicit:
        return Path(explicit).expanduser()
    xdg = environment.get("XDG_STATE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
    return base / DEFAULT_STATE_DIR_NAME


def _parse_positive_float(raw: str | None, default: float, name: str) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


def _parse_positive_int(raw: str | None, default: int, name: str) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


def parse_dsh_command(raw: str | None) -> tuple[str, ...]:
    """Parse ``WORKER_MCP_DSH_COMMAND_JSON`` as a JSON argv array.

    Deliberately *not* shell syntax: the value must be a JSON array of strings,
    so there is no quoting or word-splitting layer to get wrong.
    """

    if raw is None or raw.strip() == "":
        return DEFAULT_DSH_COMMAND
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{DSH_COMMAND_ENV} must be a JSON array of strings: {exc}") from exc
    if isinstance(decoded, str):
        raise ConfigError(
            f'{DSH_COMMAND_ENV} must be a JSON array such as ["dsh"], not a bare string'
        )
    if not isinstance(decoded, list) or not decoded:
        raise ConfigError(f"{DSH_COMMAND_ENV} must be a non-empty JSON array")
    command: list[str] = []
    for item in decoded:
        if not isinstance(item, str) or item == "":
            raise ConfigError(f"{DSH_COMMAND_ENV} entries must be non-empty strings")
        command.append(item)
    return tuple(command)


@dataclass(frozen=True)
class Config:
    """Frozen snapshot of the server's configuration."""

    state_dir: Path
    dsh_command: tuple[str, ...]
    interrupt_grace_seconds: float
    max_plan_bytes: int
    max_final_text_bytes: int
    backend: str = BACKEND_DEEPSEEK_HARNESS

    @property
    def workers_dir(self) -> Path:
        return self.state_dir / "workers"

    def describe(self) -> dict:
        """Return a log-safe description (no secrets, no full environment)."""

        return {
            "state_dir": str(self.state_dir),
            "backend": self.backend,
            "dsh_command": list(self.dsh_command),
            "interrupt_grace_seconds": self.interrupt_grace_seconds,
            "max_plan_bytes": self.max_plan_bytes,
            "max_final_text_bytes": self.max_final_text_bytes,
        }


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a :class:`Config` from the process environment."""

    environment = os.environ if env is None else env
    return Config(
        state_dir=default_state_dir(environment).resolve(),
        dsh_command=parse_dsh_command(environment.get(DSH_COMMAND_ENV)),
        interrupt_grace_seconds=_parse_positive_float(
            environment.get(INTERRUPT_GRACE_ENV),
            DEFAULT_INTERRUPT_GRACE_SECONDS,
            INTERRUPT_GRACE_ENV,
        ),
        max_plan_bytes=_parse_positive_int(
            environment.get(MAX_PLAN_BYTES_ENV), DEFAULT_MAX_PLAN_BYTES, MAX_PLAN_BYTES_ENV
        ),
        max_final_text_bytes=_parse_positive_int(
            environment.get(MAX_FINAL_TEXT_ENV), DEFAULT_MAX_FINAL_TEXT_BYTES, MAX_FINAL_TEXT_ENV
        ),
    )


def ensure_state_dirs(config: Config) -> None:
    """Create the state directory tree with user-only permissions."""

    config.state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(config.state_dir, 0o700)
    config.workers_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(config.workers_dir, 0o700)


def runner_env(config: Config) -> Sequence[str]:  # pragma: no cover - documentation helper
    """Names of environment variables that must reach the runner."""

    return (DSH_COMMAND_ENV, INTERRUPT_GRACE_ENV, MAX_FINAL_TEXT_ENV, "DSH_HOME", "PATH")
