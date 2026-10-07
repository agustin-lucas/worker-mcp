"""Backend registry. v1 ships exactly one: DeepSeek Harness."""

from __future__ import annotations

from ..config import BACKEND_DEEPSEEK_HARNESS, Config
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
    WorkerBackend,
)
from .deepseek_harness import DeepSeekHarnessBackend

__all__ = [
    "EVENT_ERROR",
    "EVENT_FINAL",
    "EVENT_SESSION",
    "EVENT_STATUS",
    "EVENT_TEXT",
    "EVENT_THINKING",
    "EVENT_TOOL_CALL",
    "EVENT_TOOL_RESULT",
    "EVENT_UNKNOWN",
    "BackendEvent",
    "DeepSeekHarnessBackend",
    "WorkerBackend",
    "available_backends",
    "build_backend",
]


def available_backends() -> dict[str, str]:
    return {
        BACKEND_DEEPSEEK_HARNESS: "DeepSeek Harness headless mode (dsh --profile headless --json)"
    }


def build_backend(name: str, config: Config) -> WorkerBackend:
    """Instantiate the configured backend, or raise for an unknown name."""

    if name == BACKEND_DEEPSEEK_HARNESS:
        return DeepSeekHarnessBackend(config.dsh_command)
    raise ValueError(f"unknown worker backend: {name!r}")
