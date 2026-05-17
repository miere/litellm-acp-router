import asyncio
import unittest

from litellm_acp_router.runtime import Runtime
from litellm_acp_router.schemas import AgentSpec
from litellm_acp_router.session_manager import SessionManager


class _FakeAcpSession:
    def __init__(self, sid: str = "acp_sid_1") -> None:
        self.session_id = sid


class _FakeConn:
    def __init__(self, client) -> None:
        self.client = client
        self.prompts: list = []
        self.assistant_replies = ["hi from agent"]
        self._reply_idx = 0

    async def initialize(self, protocol_version: int = 1) -> None:
        return None

    async def new_session(self, cwd, mcp_servers):
        return _FakeAcpSession()

    async def prompt(self, session_id, prompt):
        self.prompts.append({"session_id": session_id, "prompt": prompt})
        if self._reply_idx < len(self.assistant_replies):
            reply = self.assistant_replies[self._reply_idx]
        else:
            reply = self.assistant_replies[-1]
        self._reply_idx += 1
        await self.client.queue.put({"kind": "assistant_text", "text": reply})
        return None


class _FakeProcessCM:
    def __init__(self, client) -> None:
        self.client = client
        self.conn = _FakeConn(client)
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return self.conn, object()

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_count += 1
        return None


def _make_spawner():
    created: list = []

    def spawn(client, command, *args, **kwargs):
        cm = _FakeProcessCM(client)
        cm.spawn_kwargs = kwargs
        created.append(cm)
        return cm

    return spawn, created


def _spec() -> AgentSpec:
    return AgentSpec(
        agent_id="example",
        bin="example-bin",
        args=["--acp"],
        mode_id=None,
        bootstrap_commands=[],
    )


def _kwargs(messages, *, strategy="prompt_hashing", proxy=None):
    return {
        "optional_params": {
            "cwd": "/tmp",
            "permission_mode": "auto_allow",
            "acp_session_binding_strategy": strategy,
        },
        "messages": messages,
        "proxy_server_request": proxy,
    }


def _prompt_text(prompt_blocks) -> str:
    return " ".join(getattr(b, "text", str(b)) for b in prompt_blocks)


async def _drain(runtime, messages, *, model="acp/example", strategy="prompt_hashing", proxy=None, kwargs_override=None):
    kw = kwargs_override or _kwargs(messages, strategy=strategy, proxy=proxy)
    chunks = []
    async for c in runtime.run_stateful_stream(
        spec=_spec(),
        model=model,
        kwargs=kw,
        messages=messages,
        tools=None,
    ):
        chunks.append(c)
    return chunks


class StatefulRuntimeTests(unittest.TestCase):
    def test_first_request_creates_session_and_sends_full_prompt(self) -> None:
        spawner, created = _make_spawner()
        runtime = Runtime(session_manager=SessionManager(spawner=spawner))
        messages = [{"role": "user", "content": "explain repo"}]

        chunks = asyncio.run(_drain(runtime, messages))

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].enter_count, 1)
        self.assertEqual(len(created[0].conn.prompts), 1)
        prompt_text = _prompt_text(created[0].conn.prompts[0]["prompt"])
        self.assertIn("explain repo", prompt_text)
        self.assertEqual(chunks[-1]["finish_reason"], "stop")
        # No marker emission anymore.
        joined = "".join(c.get("text", "") for c in chunks)
        self.assertNotIn("acp-router-state", joined)

    def test_prompt_hashing_resume_sends_only_new_messages(self) -> None:
        spawner, created = _make_spawner()
        runtime = Runtime(session_manager=SessionManager(spawner=spawner))

        async def go():
            msgs1 = [{"role": "user", "content": "first turn"}]
            await _drain(runtime, msgs1)
            msgs2 = msgs1 + [
                {"role": "assistant", "content": "hi from agent"},
                {"role": "user", "content": "second turn"},
            ]
            await _drain(runtime, msgs2)

        asyncio.run(go())

        self.assertEqual(len(created), 1, "same agent process should be reused")
        self.assertEqual(len(created[0].conn.prompts), 2)
        second_prompt = _prompt_text(created[0].conn.prompts[1]["prompt"])
        self.assertIn("second turn", second_prompt)
        self.assertNotIn("first turn", second_prompt)

    def test_different_first_user_message_opens_new_session(self) -> None:
        spawner, created = _make_spawner()
        runtime = Runtime(session_manager=SessionManager(spawner=spawner))

        async def go():
            await _drain(runtime, [{"role": "user", "content": "convo A"}])
            await _drain(runtime, [{"role": "user", "content": "convo B"}])

        asyncio.run(go())
        self.assertEqual(len(created), 2)

    def test_http_header_binding_resumes_same_session(self) -> None:
        spawner, created = _make_spawner()
        runtime = Runtime(session_manager=SessionManager(spawner=spawner))
        strategy = "http_header/X-Conv-Id"
        proxy = {"headers": {"x-conv-id": "conv-42"}}

        async def go():
            msgs1 = [{"role": "user", "content": "anything"}]
            await _drain(runtime, msgs1, strategy=strategy, proxy=proxy)
            msgs2 = msgs1 + [
                {"role": "assistant", "content": "hi from agent"},
                {"role": "user", "content": "follow up"},
            ]
            await _drain(runtime, msgs2, strategy=strategy, proxy=proxy)

        asyncio.run(go())
        self.assertEqual(len(created), 1)
        self.assertEqual(len(created[0].conn.prompts), 2)
        second_prompt = _prompt_text(created[0].conn.prompts[1]["prompt"])
        self.assertIn("follow up", second_prompt)
        self.assertNotIn("anything", second_prompt)

    def test_http_header_missing_fails_fast(self) -> None:
        spawner, created = _make_spawner()
        runtime = Runtime(session_manager=SessionManager(spawner=spawner))

        async def go():
            with self.assertRaises(ValueError) as ctx:
                await _drain(
                    runtime,
                    [{"role": "user", "content": "hi"}],
                    strategy="http_header/X-Conv-Id",
                    proxy={"headers": {"X-Other": "x"}},
                )
            self.assertIn("X-Conv-Id", str(ctx.exception))

        asyncio.run(go())
        self.assertEqual(len(created), 0, "no agent should be spawned on misconfig")

    def test_empty_delta_raises_clear_error(self) -> None:
        spawner, created = _make_spawner()
        runtime = Runtime(session_manager=SessionManager(spawner=spawner))

        async def go():
            msgs1 = [{"role": "user", "content": "first"}]
            await _drain(runtime, msgs1)
            # Same messages → last_sent_message_index already covers them.
            with self.assertRaises(ValueError):
                await _drain(runtime, msgs1)

        asyncio.run(go())
        self.assertEqual(len(created[0].conn.prompts), 1)

    def test_lock_timeout_raises_clear_error(self) -> None:
        spawner, created = _make_spawner()
        sm = SessionManager(spawner=spawner)
        runtime = Runtime(session_manager=sm)

        async def go():
            msgs1 = [{"role": "user", "content": "first"}]
            await _drain(runtime, msgs1)
            live = next(iter(sm._sessions.values()))
            await live.lock.acquire()
            try:
                msgs2 = msgs1 + [
                    {"role": "assistant", "content": "hi from agent"},
                    {"role": "user", "content": "second"},
                ]
                kwargs = {
                    "optional_params": {
                        "cwd": "/tmp",
                        "acp_session_binding_strategy": "prompt_hashing",
                        "acp_session_lock_timeout_seconds": 0.05,
                    },
                    "messages": msgs2,
                }
                with self.assertRaises(TimeoutError):
                    await _drain(runtime, msgs2, kwargs_override=kwargs)
            finally:
                live.lock.release()
            self.assertEqual(len(created[0].conn.prompts), 1)

        asyncio.run(go())

    def test_stateful_spawn_uses_default_stdio_buffer_limit(self) -> None:
        from litellm_acp_router.runtime import DEFAULT_STDIO_BUFFER_BYTES

        spawner, created = _make_spawner()
        runtime = Runtime(session_manager=SessionManager(spawner=spawner))

        asyncio.run(_drain(runtime, [{"role": "user", "content": "hello"}]))

        self.assertEqual(
            created[0].spawn_kwargs.get("transport_kwargs"),
            {"limit": DEFAULT_STDIO_BUFFER_BYTES},
        )

    def test_stateful_spawn_honors_custom_stdio_buffer_limit(self) -> None:
        spawner, created = _make_spawner()
        runtime = Runtime(session_manager=SessionManager(spawner=spawner))

        messages = [{"role": "user", "content": "hello"}]
        kw = _kwargs(messages)
        kw["optional_params"]["acp_stdio_buffer_bytes"] = 4 * 1024 * 1024

        asyncio.run(_drain(runtime, messages, kwargs_override=kw))

        self.assertEqual(
            created[0].spawn_kwargs.get("transport_kwargs"),
            {"limit": 4 * 1024 * 1024},
        )


class EventToChunkTests(unittest.TestCase):
    def test_reasoning_event_maps_to_provider_specific_fields(self) -> None:
        from litellm_acp_router.runtime import _event_to_chunk

        chunk = _event_to_chunk({"kind": "reasoning", "text": "thinking"})
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["text"], "")
        self.assertEqual(
            chunk["provider_specific_fields"],
            {"reasoning_content": "thinking"},
        )
        self.assertFalse(chunk["is_finished"])
        self.assertIsNone(chunk["finish_reason"])

    def test_assistant_text_event_maps_to_text_chunk(self) -> None:
        from litellm_acp_router.runtime import _event_to_chunk

        chunk = _event_to_chunk({"kind": "assistant_text", "text": "hello"})
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["text"], "hello")
        self.assertNotIn("provider_specific_fields", chunk)

    def test_empty_text_event_is_dropped(self) -> None:
        from litellm_acp_router.runtime import _event_to_chunk

        self.assertIsNone(_event_to_chunk({"kind": "assistant_text", "text": ""}))
        self.assertIsNone(_event_to_chunk({"kind": "reasoning", "text": ""}))


if __name__ == "__main__":
    unittest.main()
