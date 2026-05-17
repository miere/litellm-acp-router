"""Long-lived ACP session manager for stateful router mode."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from acp import spawn_agent_process, text_block

from .client import AgentClient
from .schemas import AgentSpec

LOG = logging.getLogger(__name__)


def short_key(key: str, n: int = 12) -> str:
    return (key or "")[:n]


@dataclass
class ManagedACPSession:
    binding_key: str
    acp_session_id: str
    agent_id: str
    model: str
    acp_model: Optional[str]
    cwd: str
    spec: AgentSpec
    client: AgentClient
    conn: Any
    process_cm: Any
    turn: int
    last_sent_message_index: int
    created_at: float
    last_used_at: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False


# Spawner injection makes runtime tests deterministic without touching subprocesses.
SpawnerFn = Callable[..., Any]


class SessionManager:
    def __init__(self, spawner: Optional[SpawnerFn] = None) -> None:
        self._sessions: Dict[str, ManagedACPSession] = {}
        self._sessions_lock = asyncio.Lock()
        self._spawner: SpawnerFn = spawner or spawn_agent_process

    async def get(self, binding_key: str) -> Optional[ManagedACPSession]:
        async with self._sessions_lock:
            session = self._sessions.get(binding_key)
            if session is None or session.closed:
                return None
            return session

    async def create(
        self,
        *,
        binding_key: str,
        spec: AgentSpec,
        model: str,
        acp_model: Optional[str],
        cwd: str,
        options: Dict[str, Any],
    ) -> ManagedACPSession:
        permission_mode = str(options.get("permission_mode", "auto_allow"))
        protocol_version = int(options.get("protocol_version", 1))
        mcp_servers = options.get("mcp_servers") or []

        client = AgentClient(permission_mode=permission_mode)
        process_cm = self._spawner(client, spec.bin, *spec.args)
        conn, _proc = await process_cm.__aenter__()
        try:
            await conn.initialize(protocol_version=protocol_version)
            acp_session = await conn.new_session(
                cwd=str(cwd),
                mcp_servers=mcp_servers,
            )

            if spec.mode_id:
                try:
                    set_mode = getattr(conn, "set_mode", None)
                    if callable(set_mode):
                        await set_mode(
                            session_id=acp_session.session_id,
                            mode_id=spec.mode_id,
                        )
                except Exception:
                    LOG.debug("stateful_set_mode_failed", exc_info=True)

            await self._run_bootstrap(conn, acp_session.session_id, client, spec)
        except Exception:
            try:
                await process_cm.__aexit__(None, None, None)
            except Exception:
                LOG.debug("stateful_session_create_cleanup_failed", exc_info=True)
            raise

        now = time.time()
        session = ManagedACPSession(
            binding_key=binding_key,
            acp_session_id=acp_session.session_id,
            agent_id=spec.agent_id,
            model=model,
            acp_model=acp_model,
            cwd=str(cwd),
            spec=spec,
            client=client,
            conn=conn,
            process_cm=process_cm,
            turn=0,
            last_sent_message_index=-1,
            created_at=now,
            last_used_at=now,
        )
        async with self._sessions_lock:
            self._sessions[binding_key] = session
        LOG.info(
            "stateful_session_create key=%s agent=%s model=%s",
            short_key(binding_key),
            spec.agent_id,
            model,
        )
        return session

    async def _run_bootstrap(
        self,
        conn: Any,
        acp_session_id: str,
        client: AgentClient,
        spec: AgentSpec,
    ) -> None:
        if not spec.bootstrap_commands:
            return
        client.suppress_stream = True
        try:
            for cmd in spec.bootstrap_commands:
                cmd = str(cmd).strip()
                if not cmd:
                    continue
                await conn.prompt(session_id=acp_session_id, prompt=[text_block(cmd)])
        finally:
            client.suppress_stream = False

    async def close(self, binding_key: str, reason: str = "") -> None:
        async with self._sessions_lock:
            session = self._sessions.pop(binding_key, None)
        if session is None:
            return
        if session.closed:
            return
        session.closed = True
        try:
            await session.process_cm.__aexit__(None, None, None)
        except Exception:
            LOG.debug(
                "stateful_session_close_error key=%s reason=%s",
                short_key(binding_key),
                reason,
                exc_info=True,
            )
        LOG.info(
            "stateful_session_closed key=%s reason=%s",
            short_key(binding_key),
            reason,
        )

    async def evict_expired(self, now: float, ttl_seconds: int) -> None:
        if ttl_seconds is None or ttl_seconds <= 0:
            return
        async with self._sessions_lock:
            candidates = [
                key
                for key, s in self._sessions.items()
                if not s.lock.locked() and (now - s.last_used_at) > ttl_seconds
            ]
        for key in candidates:
            LOG.info("stateful_session_expired key=%s", short_key(key))
            await self.close(key, reason="ttl")

    async def enforce_max_sessions(self, max_sessions: int) -> None:
        if max_sessions is None or max_sessions <= 0:
            return
        async with self._sessions_lock:
            if len(self._sessions) <= max_sessions:
                return
            ordered = sorted(
                self._sessions.items(),
                key=lambda kv: kv[1].last_used_at,
            )
            victims = []
            current = len(self._sessions)
            for key, s in ordered:
                if current <= max_sessions:
                    break
                if s.lock.locked():
                    continue
                victims.append(key)
                current -= 1
        for key in victims:
            LOG.info("stateful_session_evicted key=%s", short_key(key))
            await self.close(key, reason="lru")

    async def close_all(self) -> None:
        async with self._sessions_lock:
            keys = list(self._sessions.keys())
        for key in keys:
            await self.close(key, reason="shutdown")
