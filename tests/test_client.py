import asyncio
import unittest

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

    def test_tool_call_start_with_title_and_kind(self) -> None:
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
        items = _drain(client)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "\n\n> [execute] Run npm install\n\n")

    def test_tool_call_start_with_title_only(self) -> None:
        client = AgentClient()
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "Read package.json",
            },
        ))
        items = _drain(client)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "\n\n> Read package.json\n\n")

    def test_tool_call_start_without_title_is_silent(self) -> None:
        client = AgentClient()
        _run(client.session_update(
            session_id="s1",
            update={"sessionUpdate": "tool_call", "toolCallId": "tc_1"},
        ))
        self.assertTrue(client.queue.empty())

    def test_tool_call_update_completed_emits_done(self) -> None:
        client = AgentClient()
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
                "title": "Run npm install",
            },
        ))
        items = _drain(client)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "\n\n> Run npm install \u2014 done\n\n")

    def test_tool_call_update_completed_without_title_is_silent(self) -> None:
        client = AgentClient()
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            },
        ))
        self.assertTrue(client.queue.empty())

    def test_tool_call_update_other_status_is_silent(self) -> None:
        client = AgentClient()
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "in_progress",
                "title": "Run npm install",
            },
        ))
        self.assertTrue(client.queue.empty())

    def test_tool_call_update_failed_with_title_emits(self) -> None:
        client = AgentClient()
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "failed",
                "title": "Run npm install",
            },
        ))
        items = _drain(client)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "\n\n> Run npm install \u2014 failed\n\n")

    def test_tool_call_update_failed_without_title_is_silent(self) -> None:
        client = AgentClient()
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "failed",
            },
        ))
        self.assertTrue(client.queue.empty())

    def test_suppress_stream_blocks_tool_narration(self) -> None:
        client = AgentClient()
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
        client = AgentClient(emit_tool_activity=False)
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "Run npm install",
                "kind": "execute",
            },
        ))
        _run(client.session_update(
            session_id="s1",
            update={
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "failed",
                "title": "Run npm install",
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


if __name__ == "__main__":
    unittest.main()
