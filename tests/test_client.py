import asyncio
import unittest

from litellm_acp_router.adapters.auggie import (
    AuggieTextFilter,
    auggie_tool_narrator,
)
from litellm_acp_router.client import AgentClient


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _drain(client: AgentClient):
    items = []
    while not client.queue.empty():
        items.append(client.queue.get_nowait())
    return items


class AgentClientToolNarrationTests(unittest.TestCase):
    def test_agent_message_chunk_still_streams(self) -> None:
        client = AgentClient()
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "hello"},
            },
        ))
        items = _drain(client)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "hello")

    def test_tool_call_without_narrator_is_silent(self) -> None:
        client = AgentClient()
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "Run npm install",
                "kind": "execute",
            },
        ))
        self.assertTrue(client.queue.empty())

    def test_tool_call_update_is_always_silent(self) -> None:
        client = AgentClient(tool_narrator=auggie_tool_narrator)
        for status in ("completed", "failed", "in_progress"):
            _run(client.session_update(
                session_id="s1",
                update={
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc_1",
                    "status": status,
                    "title": "Run npm install",
                },
            ))
        self.assertTrue(client.queue.empty())

    def test_tool_call_with_narrator_emits_formatted_text(self) -> None:
        client = AgentClient(tool_narrator=auggie_tool_narrator)
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "Run npm install",
                "kind": "execute",
            },
        ))
        items = _drain(client)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "\U0001F4BB terminal: Run npm install\n")

    def test_tool_call_without_title_is_silent(self) -> None:
        client = AgentClient(tool_narrator=auggie_tool_narrator)
        _run(client.session_update(
            session_id="s1",
            update={"sessionUpdate": "tool_call", "toolCallId": "tc_1", "kind": "read"},
        ))
        self.assertTrue(client.queue.empty())

    def test_suppress_stream_blocks_tool_narration(self) -> None:
        client = AgentClient(tool_narrator=auggie_tool_narrator)
        client.suppress_stream = True
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "Run npm install",
                "kind": "execute",
            },
        ))
        self.assertTrue(client.queue.empty())

    def test_emit_tool_activity_false_disables_narration(self) -> None:
        client = AgentClient(
            emit_tool_activity=False,
            tool_narrator=auggie_tool_narrator,
        )
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "Run npm install",
                "kind": "execute",
            },
        ))
        self.assertTrue(client.queue.empty())

    def test_agent_thought_chunk_emits_reasoning_event(self) -> None:
        client = AgentClient()
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "thinking about it"},
            },
        ))
        items = _drain(client)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["kind"], "reasoning")
        self.assertEqual(items[0]["text"], "thinking about it")
        # Reasoning must not pollute the final assistant transcript.
        self.assertEqual(client.get_final_text(), "")

    def test_suppress_stream_blocks_reasoning(self) -> None:
        client = AgentClient()
        client.suppress_stream = True
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "hidden"},
            },
        ))
        self.assertTrue(client.queue.empty())


class AuggieToolNarratorTests(unittest.TestCase):
    def test_known_kinds_render_expected_prefix(self) -> None:
        cases = [
            ("read", "List files", "\U0001F4D6 read: List files\n"),
            ("write", "Save README", "\u270D\uFE0F write: Save README\n"),
            ("edit", "Patch README", "\u270D\uFE0F write: Patch README\n"),
            ("execute", "npm test", "\U0001F4BB terminal: npm test\n"),
            ("fetch", "GET https://x", "\U0001F310 browser_navigate: GET https://x\n"),
        ]
        for kind, title, expected in cases:
            with self.subTest(kind=kind):
                self.assertEqual(auggie_tool_narrator(kind, title), expected)

    def test_unknown_kind_falls_back_to_thinking(self) -> None:
        self.assertEqual(
            auggie_tool_narrator("plan", "Outline approach"),
            "\U0001F9E0 thinking: Outline approach\n",
        )

    def test_empty_kind_falls_back_to_thinking(self) -> None:
        self.assertEqual(
            auggie_tool_narrator("", "Reflecting"),
            "\U0001F9E0 thinking: Reflecting\n",
        )

    def test_title_at_threshold_is_unchanged(self) -> None:
        title = "x" * 40
        self.assertEqual(
            auggie_tool_narrator("read", title),
            f"\U0001F4D6 read: {title}\n",
        )

    def test_title_longer_than_threshold_is_truncated(self) -> None:
        title = "x" * 41
        truncated = "x" * 37 + "..."
        self.assertEqual(
            auggie_tool_narrator("read", title),
            f"\U0001F4D6 read: {truncated}\n",
        )
        # The visible body (after the prefix) is exactly 40 chars.
        self.assertEqual(len(truncated), 40)

    def test_empty_title_is_silent(self) -> None:
        self.assertIsNone(auggie_tool_narrator("read", ""))

    def test_newlines_in_title_are_collapsed_to_spaces(self) -> None:
        self.assertEqual(
            auggie_tool_narrator("read", "first line\nsecond line"),
            "\U0001F4D6 read: first line second line\n",
        )
        self.assertEqual(
            auggie_tool_narrator("execute", "echo a\r\nb\rc"),
            "\U0001F4BB terminal: echo a b c\n",
        )

    def test_unbalanced_backtick_after_truncation_is_closed(self) -> None:
        # Title is well over 40 chars and contains a single opening backtick
        # that would be left dangling after truncation.
        title = "Read `/Users/miere/very/deep/path/inside/the/repo.py"
        out = auggie_tool_narrator("read", title)
        assert out is not None
        # Body strips the trailing newline for easy assertions.
        body = out.rstrip("\n").split(": ", 1)[1]
        self.assertTrue(body.endswith("...`"))
        self.assertEqual(body.count("`") % 2, 0)

    def test_balanced_backticks_are_left_alone(self) -> None:
        self.assertEqual(
            auggie_tool_narrator("read", "Read `foo.py`"),
            "\U0001F4D6 read: Read `foo.py`\n",
        )


class AuggieTextFilterTests(unittest.TestCase):
    def test_strips_complete_tag_pair_in_single_chunk(self) -> None:
        f = AuggieTextFilter()
        out = f.feed(
            'before <augment_code_snippet path="x.py" mode="EXCERPT">code'
            "</augment_code_snippet> after"
        )
        self.assertEqual(out + f.flush(), "before code after")

    def test_handles_tag_split_across_chunks(self) -> None:
        f = AuggieTextFilter()
        parts = [
            "intro ",
            "<augment_code_snip",
            'pet path="x.py" mode="EXCERPT">',
            "body ",
            "</augment_code_snippet>",
            " trail",
        ]
        emitted = "".join(f.feed(p) for p in parts) + f.flush()
        self.assertEqual(emitted, "intro body  trail")

    def test_preserves_other_angle_brackets(self) -> None:
        f = AuggieTextFilter()
        out = f.feed("if x < 3 and y > 1 then ok")
        self.assertEqual(out + f.flush(), "if x < 3 and y > 1 then ok")

    def test_partial_tag_at_end_of_stream_is_released_as_text(self) -> None:
        # If the stream truly ends mid-tag, flush releases the buffered
        # fragment as plain text rather than swallowing real content the
        # caller meant to keep.
        f = AuggieTextFilter()
        emitted = f.feed("ok <augment_code_snip") + f.flush()
        self.assertEqual(emitted, "ok <augment_code_snip")
        # A complete tag in the same chunk is stripped immediately.
        f2 = AuggieTextFilter()
        emitted2 = f2.feed("ok <augment_code_snippet>") + f2.flush()
        self.assertEqual(emitted2, "ok ")


class ToolToMessageSeparatorTests(unittest.TestCase):
    """A tool narration immediately followed by an assistant message gets a
    single leading "\\n" prepended to the queued message chunk so the two
    visually separate. The separator is display-only and must not leak into
    final_text_parts (used for stateful resume / acompletion transcripts)."""

    def _tool_call(self, client: AgentClient, title: str, kind: str = "execute"):
        return client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call",
                "toolCallId": "tc",
                "title": title,
                "kind": kind,
            },
        )

    def _message(self, client: AgentClient, text: str):
        return client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        )

    def _reasoning(self, client: AgentClient, text: str):
        return client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": text},
            },
        )

    def test_tool_then_message_prepends_newline_to_message(self) -> None:
        client = AgentClient(tool_narrator=auggie_tool_narrator)
        _run(self._tool_call(client, "Run npm install"))
        _run(self._message(client, "All done."))
        items = _drain(client)
        self.assertEqual(items[1]["text"], "\nAll done.")
        # final_text_parts stays clean — separator is display-only.
        self.assertEqual(client.get_final_text(), "All done.")

    def test_tool_tool_message_only_last_message_gets_separator(self) -> None:
        client = AgentClient(tool_narrator=auggie_tool_narrator)
        _run(self._tool_call(client, "first"))
        _run(self._tool_call(client, "second"))
        _run(self._message(client, "ok"))
        items = _drain(client)
        # Both tool narrations queued as-is, no separator between them.
        self.assertFalse(items[0]["text"].startswith("\n"))
        self.assertFalse(items[1]["text"].startswith("\n"))
        # Message gets exactly one leading "\n".
        self.assertEqual(items[2]["text"], "\nok")

    def test_tool_with_no_following_message_emits_no_spurious_newline(self) -> None:
        client = AgentClient(tool_narrator=auggie_tool_narrator)
        _run(self._tool_call(client, "lonely"))
        items = _drain(client)
        self.assertEqual(len(items), 1)
        self.assertFalse(items[0]["text"].startswith("\n"))
        self.assertEqual(client.get_final_text(), "")

    def test_reasoning_between_tool_and_message_clears_separator(self) -> None:
        client = AgentClient(tool_narrator=auggie_tool_narrator)
        _run(self._tool_call(client, "Run npm install"))
        _run(self._reasoning(client, "considering"))
        _run(self._message(client, "Here we go."))
        items = _drain(client)
        # Message has no leading separator because reasoning broke adjacency.
        message_item = items[-1]
        self.assertEqual(message_item["kind"], "assistant_text")
        self.assertEqual(message_item["text"], "Here we go.")

    def test_message_tool_message_second_message_gets_separator(self) -> None:
        client = AgentClient(tool_narrator=auggie_tool_narrator)
        _run(self._message(client, "intro"))
        _run(self._tool_call(client, "Run npm install"))
        _run(self._message(client, "outro"))
        items = _drain(client)
        self.assertEqual(items[0]["text"], "intro")
        self.assertEqual(items[2]["text"], "\noutro")

    def test_reset_turn_state_clears_pending_separator(self) -> None:
        # Simulates a stateful client reused across turns: the previous turn
        # ended on a tool narration. Without reset_turn_state, the first
        # message of the next turn would inherit a spurious leading "\n".
        client = AgentClient(tool_narrator=auggie_tool_narrator)
        _run(self._tool_call(client, "trailing tool of prev turn"))
        _drain(client)  # caller already drained the queue at turn end.
        self.assertTrue(client._last_was_tool_narration)
        client.reset_turn_state()
        self.assertFalse(client._last_was_tool_narration)
        _run(self._message(client, "fresh turn message"))
        items = _drain(client)
        self.assertEqual(items[0]["text"], "fresh turn message")


if __name__ == "__main__":
    unittest.main()
