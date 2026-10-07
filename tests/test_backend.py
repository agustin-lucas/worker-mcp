"""Backend unit tests for the DeepSeek Harness JSON event contract.

The schema asserted here was read out of the *installed* DSH headless bundle
(``@deepseek-ai/dsh-headless/lib/json-stream-*.js``), not guessed from
documentation. Parsing must stay defensive: the runner preserves the raw stream
in ``events.jsonl`` regardless, so an unknown or malformed event must never
raise.
"""

from __future__ import annotations

import json

import pytest

from worker_mcp.backends import DeepSeekHarnessBackend
from worker_mcp.backends.base import (
    EVENT_ERROR,
    EVENT_FINAL,
    EVENT_SESSION,
    EVENT_STATUS,
    EVENT_TEXT,
    EVENT_THINKING,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    EVENT_UNKNOWN,
)


@pytest.fixture
def backend() -> DeepSeekHarnessBackend:
    return DeepSeekHarnessBackend(["dsh"])


class TestCommandBuilding:
    def test_first_turn_omits_session_id(self, backend: DeepSeekHarnessBackend) -> None:
        assert list(backend.build_command(session_id=None)) == [
            "dsh",
            "--profile",
            "headless",
            "--json",
        ]

    def test_resume_passes_session_id(self, backend: DeepSeekHarnessBackend) -> None:
        assert list(backend.build_command(session_id="session-abc")) == [
            "dsh",
            "--profile",
            "headless",
            "--json",
            "--session-id",
            "session-abc",
        ]

    def test_task_is_never_part_of_argv(self, backend: DeepSeekHarnessBackend) -> None:
        """Prompts go over stdin; arbitrary text must not become an argument."""

        command = backend.build_command(session_id="session-1")
        assert "--" not in command
        assert all("\n" not in part for part in command)

    def test_custom_command_prefix_is_respected(self) -> None:
        backend = DeepSeekHarnessBackend(["npx", "-y", "@deepseek-ai/dsh"])
        assert list(backend.build_command(session_id=None))[:3] == [
            "npx",
            "-y",
            "@deepseek-ai/dsh",
        ]

    def test_empty_command_rejected(self) -> None:
        with pytest.raises(ValueError):
            DeepSeekHarnessBackend([])


class TestEventParsing:
    def test_opening_session_event(self, backend: DeepSeekHarnessBackend) -> None:
        event = {
            "type": "session",
            "sessionId": "session-4f6a",
            "cwd": "/home/user/worktrees/auth",
        }
        parsed = backend.parse_event(event)
        assert parsed.kind == EVENT_SESSION
        assert parsed.session_id == "session-4f6a"

    def test_session_event_accepts_snake_case(self, backend: DeepSeekHarnessBackend) -> None:
        assert backend.parse_event({"type": "session", "session_id": "s1"}).session_id == "s1"

    def test_text_and_thinking(self, backend: DeepSeekHarnessBackend) -> None:
        assert backend.parse_event({"type": "text", "text": "hi"}).text == "hi"
        assert backend.parse_event({"type": "thinking", "text": "hmm"}).kind == EVENT_THINKING

    def test_final_event_carries_lossless_answer(self, backend: DeepSeekHarnessBackend) -> None:
        parsed = backend.parse_event({"type": "final", "text": "line one\nline two"})
        assert parsed.kind == EVENT_FINAL
        assert parsed.final_text == "line one\nline two"

    def test_empty_final_is_still_a_final(self, backend: DeepSeekHarnessBackend) -> None:
        parsed = backend.parse_event({"type": "final", "text": ""})
        assert parsed.kind == EVENT_FINAL
        assert parsed.final_text == ""

    def test_turn_end_reason(self, backend: DeepSeekHarnessBackend) -> None:
        parsed = backend.parse_event(
            {
                "type": "status",
                "phase": "turn_end",
                "turn": 1,
                "reason": {"kind": "completed"},
            }
        )
        assert parsed.kind == EVENT_STATUS
        assert parsed.stop_reason == {"kind": "completed"}

    def test_error_event(self, backend: DeepSeekHarnessBackend) -> None:
        parsed = backend.parse_event({"type": "error", "message": "boom"})
        assert parsed.kind == EVENT_ERROR
        assert parsed.error == "boom"

    def test_tool_events(self, backend: DeepSeekHarnessBackend) -> None:
        assert (
            backend.parse_event(
                {"type": "tool_call", "callId": "c1", "tool": "bash", "input": {}}
            ).kind
            == EVENT_TOOL_CALL
        )
        assert (
            backend.parse_event(
                {"type": "tool_result", "callId": "c1", "status": "completed", "result": "ok"}
            ).kind
            == EVENT_TOOL_RESULT
        )

    def test_truncated_event_marker_is_tolerated(self, backend: DeepSeekHarnessBackend) -> None:
        parsed = backend.parse_event({"type": "text", "text": "abc", "truncated": True})
        assert parsed.kind == EVENT_TEXT
        assert parsed.text == "abc"

    @pytest.mark.parametrize(
        "event",
        [
            {},
            {"type": "unknown-future-event", "payload": 1},
            {"no": "type"},
            {"type": None},
        ],
    )
    def test_unknown_events_never_raise(self, backend: DeepSeekHarnessBackend, event: dict) -> None:
        assert backend.parse_event(event).kind == EVENT_UNKNOWN

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "session", "sessionId": 12345},
            {"type": "text", "text": {"nested": True}},
            {"type": "final", "text": None},
            {"type": "error", "message": ["list"]},
            {"type": "status", "reason": "not-an-object"},
        ],
    )
    def test_malformed_payloads_are_defensive(
        self, backend: DeepSeekHarnessBackend, event: dict
    ) -> None:
        parsed = backend.parse_event(event)
        assert parsed.kind in {EVENT_SESSION, EVENT_TEXT, EVENT_FINAL, EVENT_ERROR, EVENT_STATUS}

    def test_event_payload_is_preserved(self, backend: DeepSeekHarnessBackend) -> None:
        event = {"type": "text", "text": "x", "extraField": [1, 2, 3]}
        assert backend.parse_event(event).raw == event


class TestExitClassification:
    @pytest.mark.parametrize(
        "exit_code,turn_failed,expected",
        [
            (0, False, "completed"),
            (0, True, "failed"),
            (1, False, "failed"),
            (1, True, "failed"),
            (None, False, "failed"),
        ],
    )
    def test_classify_exit(self, exit_code, turn_failed, expected) -> None:
        assert DeepSeekHarnessBackend.classify_exit(exit_code, turn_failed=turn_failed) == expected


class TestRealEventStreamShape:
    def test_a_full_projected_stream_parses(self, backend: DeepSeekHarnessBackend) -> None:
        """The exact event sequence the installed DSH --json mode emits."""

        stream = [
            {"type": "session", "sessionId": "session-1", "cwd": "/w"},
            {"type": "status", "phase": "turn_start", "turn": 1},
            {"type": "status", "phase": "step_start", "turn": 1, "step": 1},
            {"type": "thinking", "text": "reasoning"},
            {"type": "tool_call", "callId": "c1", "tool": "write_file", "input": {"p": "x"}},
            {"type": "tool_result", "callId": "c1", "status": "completed", "result": "ok"},
            {"type": "text", "text": "intermediate"},
            {
                "type": "status",
                "phase": "step_end",
                "turn": 1,
                "step": 1,
                "usage": {"inputTokens": 5},
            },
            {"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "completed"}},
            {"type": "final", "text": "done"},
        ]
        kinds = [backend.parse_event(event).kind for event in stream]
        assert kinds == [
            EVENT_SESSION,
            EVENT_STATUS,
            EVENT_STATUS,
            EVENT_THINKING,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            EVENT_TEXT,
            EVENT_STATUS,
            EVENT_STATUS,
            EVENT_FINAL,
        ]

    def test_events_jsonl_lines_round_trip(self, backend: DeepSeekHarnessBackend) -> None:
        """Values including newlines and Unicode survive projection."""

        finals = 'multi\nline ✓ "quoted"'
        line = json.dumps({"type": "final", "text": finals})
        parsed = backend.parse_event(json.loads(line))
        assert parsed.final_text == finals
