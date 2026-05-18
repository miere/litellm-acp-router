import asyncio
import time
import unittest

from litellm_acp_router.schemas import AgentSpec
from litellm_acp_router.session_manager import SessionManager


class _FakeAcpSession:
    def __init__(self, sid: str = "acp_sid_1") -> None:
        self.session_id = sid


class _FakeConn:
    def __init__(self) -> None:
        self.initialize_called = False
        self.new_session_called = False
        self.prompts: list = []

    async def initialize(self, protocol_version: int = 1) -> None:
        self.initialize_called = True

    async def new_session(self, cwd, mcp_servers):
        self.new_session_called = True
        return _FakeAcpSession()

    async def prompt(self, session_id, prompt):
        self.prompts.append((session_id, prompt))
        return None


class _FakeProcessCM:
    def __init__(self) -> None:
        self.enter_count = 0
        self.exit_count = 0
        self.conn = _FakeConn()

    async def __aenter__(self):
        self.enter_count += 1
        return self.conn, object()

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_count += 1
        return None


def _make_spawner():
    created: list = []

    def spawn(client, command, *args, **kwargs):
        cm = _FakeProcessCM()
        created.append((client, command, list(args), cm, kwargs))
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


def _run(coro):
    return asyncio.run(coro)


async def _create(mgr, key="k1"):
    return await mgr.create(
        binding_key=key,
        spec=_spec(),
        model="acp/example",
        acp_model=None,
        cwd="/tmp/example",
        options={},
    )


class SessionManagerTests(unittest.TestCase):
    def test_create_stores_session_and_get_returns_it(self) -> None:
        spawner, created = _make_spawner()
        mgr = SessionManager(spawner=spawner)

        async def go():
            session = await _create(mgr, key="key-1")
            self.assertEqual(session.acp_session_id, "acp_sid_1")
            self.assertEqual(session.binding_key, "key-1")
            self.assertEqual(session.last_sent_message_index, -1)
            fetched = await mgr.get("key-1")
            self.assertIs(fetched, session)
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0][3].enter_count, 1)

        _run(go())

    def test_close_calls_aexit_once_and_removes_session(self) -> None:
        spawner, created = _make_spawner()
        mgr = SessionManager(spawner=spawner)

        async def go():
            await _create(mgr, key="key-1")
            await mgr.close("key-1", reason="test")
            self.assertEqual(created[0][3].exit_count, 1)
            self.assertIsNone(await mgr.get("key-1"))
            await mgr.close("key-1", reason="again")
            self.assertEqual(created[0][3].exit_count, 1)

        _run(go())

    def test_ttl_eviction_closes_expired_sessions(self) -> None:
        spawner, created = _make_spawner()
        mgr = SessionManager(spawner=spawner)

        async def go():
            session = await _create(mgr, key="key-ttl")
            session.last_used_at = time.time() - 10_000
            await mgr.evict_expired(time.time(), ttl_seconds=1800)
            self.assertIsNone(await mgr.get("key-ttl"))
            self.assertEqual(created[0][3].exit_count, 1)

        _run(go())

    def test_locked_sessions_not_evicted_as_idle(self) -> None:
        spawner, _created = _make_spawner()
        mgr = SessionManager(spawner=spawner)

        async def go():
            session = await _create(mgr, key="key-busy")
            session.last_used_at = time.time() - 10_000
            await session.lock.acquire()
            try:
                await mgr.evict_expired(time.time(), ttl_seconds=1800)
                self.assertIsNotNone(await mgr.get("key-busy"))
            finally:
                session.lock.release()

        _run(go())

    def test_max_session_enforcement_evicts_lru(self) -> None:
        spawner, created = _make_spawner()
        mgr = SessionManager(spawner=spawner)

        async def go():
            sessions = []
            for i in range(3):
                sessions.append(await _create(mgr, key=f"key-{i}"))
                await asyncio.sleep(0)
            sessions[0].last_used_at = time.time() - 5_000
            sessions[1].last_used_at = time.time() - 100
            sessions[2].last_used_at = time.time()

            await mgr.enforce_max_sessions(max_sessions=2)
            self.assertIsNone(await mgr.get("key-0"))
            self.assertIsNotNone(await mgr.get("key-1"))
            self.assertIsNotNone(await mgr.get("key-2"))
            self.assertEqual(created[0][3].exit_count, 1)
            self.assertEqual(created[1][3].exit_count, 0)
            self.assertEqual(created[2][3].exit_count, 0)

        _run(go())

    def test_create_passes_default_stdio_buffer_limit(self) -> None:
        from litellm_acp_router.runtime import DEFAULT_STDIO_BUFFER_BYTES

        spawner, created = _make_spawner()
        mgr = SessionManager(spawner=spawner)

        async def go():
            await _create(mgr, key="key-buf")
            self.assertEqual(
                created[0][4].get("transport_kwargs"),
                {"limit": DEFAULT_STDIO_BUFFER_BYTES},
            )

        _run(go())

    def test_create_honors_custom_stdio_buffer_limit(self) -> None:
        spawner, created = _make_spawner()
        mgr = SessionManager(spawner=spawner)

        async def go():
            await mgr.create(
                binding_key="key-buf-custom",
                spec=_spec(),
                model="acp/example",
                acp_model=None,
                cwd="/tmp/example",
                options={"acp_stdio_buffer_bytes": 16 * 1024 * 1024},
            )
            self.assertEqual(
                created[0][4].get("transport_kwargs"),
                {"limit": 16 * 1024 * 1024},
            )

        _run(go())

    def test_create_zero_buffer_propagates_unbounded_sentinel(self) -> None:
        from litellm_acp_router.runtime import UNBOUNDED_STDIO_BUFFER_BYTES

        spawner, created = _make_spawner()
        mgr = SessionManager(spawner=spawner)

        async def go():
            await mgr.create(
                binding_key="key-buf-zero",
                spec=_spec(),
                model="acp/example",
                acp_model=None,
                cwd="/tmp/example",
                options={"acp_stdio_buffer_bytes": 0},
            )
            self.assertEqual(
                created[0][4].get("transport_kwargs"),
                {"limit": UNBOUNDED_STDIO_BUFFER_BYTES},
            )

        _run(go())


if __name__ == "__main__":
    unittest.main()
