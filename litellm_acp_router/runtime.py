import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from acp import spawn_agent_process, text_block
from litellm.types.utils import GenericStreamingChunk

from .binding import resolve_session_key
from .client import AgentClient
from .schemas import AgentSpec
from .session_manager import ManagedACPSession, SessionManager, short_key
from .utils import (
    common_existing_parent,
    content_blocks_to_text,
    extract_existing_paths_from_text,
    messages_to_prompt,
)

LOG = logging.getLogger(__name__)

# Default StreamReader buffer for the ACP stdio transport. The upstream acp
# package falls back to asyncio's 64 KiB default, which is easily overrun by a
# single tool_call frame carrying file diffs or terminal output and crashes the
# receive loop with LimitOverrunError. 8 MiB gives ample headroom while keeping
# the per-process memory ceiling bounded.
DEFAULT_STDIO_BUFFER_BYTES = 8 * 1024 * 1024


def resolve_stdio_buffer_bytes(options: Dict[str, Any]) -> int:
    raw = options.get("acp_stdio_buffer_bytes")
    if raw is None:
        return DEFAULT_STDIO_BUFFER_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_STDIO_BUFFER_BYTES
    if value <= 0:
        return DEFAULT_STDIO_BUFFER_BYTES
    return value


def _event_to_chunk(event: Dict[str, Any]) -> Optional[GenericStreamingChunk]:
    """Translate an AgentClient queue event into a LiteLLM streaming chunk.

    Reasoning events surface via `provider_specific_fields["reasoning_content"]`
    so LiteLLM propagates them onto the OpenAI delta's `reasoning_content`
    field, leaving assistant prose in `text`.
    """
    text = event.get("text") or ""
    if not text:
        return None
    kind = event.get("kind")
    if kind == "reasoning":
        return {
            "finish_reason": None,
            "index": 0,
            "is_finished": False,
            "text": "",
            "tool_use": None,
            "usage": None,
            "provider_specific_fields": {"reasoning_content": text},
        }
    return {
        "finish_reason": None,
        "index": 0,
        "is_finished": False,
        "text": text,
        "tool_use": None,
        "usage": None,
    }


class Runtime:
    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        self.session_manager = session_manager or SessionManager()

    def resolve_cwd(
        self,
        kwargs: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> str:
        """Match reference handler: explicit cwd metadata first, then infer from paths in messages."""
        optional_params = kwargs.get("optional_params", {}) or {}
        metadata = kwargs.get("metadata") or optional_params.get("metadata") or {}

        for source in (optional_params, metadata):
            if not isinstance(source, dict):
                continue
            for key in ("cwd", "workspace_path", "project_root", "root_dir", "path"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    p = Path(value).expanduser()
                    if p.exists():
                        return str(p.resolve())

        text_blobs: List[str] = []
        for msg in messages or []:
            content = content_blocks_to_text(msg.get("content", ""))
            if content:
                text_blobs.append(content)

        found_paths: List[Path] = []
        for blob in text_blobs:
            found_paths.extend(extract_existing_paths_from_text(blob))

        inferred = common_existing_parent(found_paths)
        if inferred is not None:
            return str(inferred)

        return os.getcwd()

    async def bootstrap_agent_session(
        self,
        conn: Any,
        session_id: str,
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
                await conn.prompt(session_id=session_id, prompt=[text_block(cmd)])
        finally:
            client.suppress_stream = False

    async def run_stream(
        self,
        *,
        spec: AgentSpec,
        prompt_text: str,
        kwargs: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> AsyncIterator[GenericStreamingChunk]:
        optional_params = kwargs.get("optional_params", {}) or {}
        protocol_version = int(optional_params.get("protocol_version", 1))
        cwd = self.resolve_cwd(kwargs, messages)
        mcp_servers = optional_params.get("mcp_servers") or []
        permission_mode = str(optional_params.get("permission_mode", "auto_allow"))
        emit_tool_activity = bool(optional_params.get("acp_emit_tool_activity", True))
        stdio_buffer_bytes = resolve_stdio_buffer_bytes(optional_params)

        client = AgentClient(
            permission_mode=permission_mode,
            emit_tool_activity=emit_tool_activity,
        )

        async with spawn_agent_process(
            client,
            spec.bin,
            *spec.args,
            transport_kwargs={"limit": stdio_buffer_bytes},
        ) as (conn, _proc):
            await conn.initialize(protocol_version=protocol_version)
            session = await conn.new_session(
                cwd=str(cwd),
                mcp_servers=mcp_servers,
            )

            if spec.mode_id:
                try:
                    set_mode = getattr(conn, "set_mode", None)
                    if callable(set_mode):
                        await set_mode(session_id=session.session_id, mode_id=spec.mode_id)
                except Exception:
                    pass

            await self.bootstrap_agent_session(
                conn=conn,
                session_id=session.session_id,
                client=client,
                spec=spec,
            )

            prompt_task = asyncio.create_task(
                conn.prompt(
                    session_id=session.session_id,
                    prompt=[text_block(prompt_text)],
                )
            )

            try:
                while True:
                    if prompt_task.done() and client.queue.empty():
                        break

                    try:
                        event = await asyncio.wait_for(client.queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    chunk = _event_to_chunk(event)
                    if chunk is not None:
                        yield chunk

                await prompt_task

            finally:
                if not prompt_task.done():
                    prompt_task.cancel()

        yield {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": "",
            "tool_use": None,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


    async def run_stateful_stream(
        self,
        *,
        spec: AgentSpec,
        model: str,
        kwargs: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        optional_params = kwargs.get("optional_params", {}) or {}
        ttl_seconds = int(optional_params.get("acp_session_ttl_seconds", 1800))
        max_sessions = int(optional_params.get("acp_max_sessions", 100))
        lock_timeout = float(optional_params.get("acp_session_lock_timeout_seconds", 30))
        acp_model = optional_params.get("acp_model")
        strategy = optional_params.get("acp_session_binding_strategy")

        binding_key_raw = resolve_session_key(
            strategy,
            messages=messages,
            kwargs=kwargs,
            proxy_server_request=kwargs.get("proxy_server_request"),
        )
        cwd = self.resolve_cwd(kwargs, messages)
        # Namespace the session by adapter identity so different aliases or
        # working directories cannot collide on the same binding key.
        binding_key = (
            f"{spec.agent_id}|{model}|{acp_model or '-'}|{cwd}|{binding_key_raw}"
        )

        await self.session_manager.evict_expired(time.time(), ttl_seconds)
        await self.session_manager.enforce_max_sessions(max_sessions)

        session: Optional[ManagedACPSession] = await self.session_manager.get(binding_key)
        if session is None:
            session = await self.session_manager.create(
                binding_key=binding_key,
                spec=spec,
                model=model,
                acp_model=acp_model,
                cwd=str(cwd),
                options=optional_params,
            )
            await self.session_manager.enforce_max_sessions(max_sessions)
            send_messages = messages
            LOG.info(
                "stateful_session_open key=%s msgs=%d",
                short_key(binding_key),
                len(messages),
            )
        else:
            start = session.last_sent_message_index + 1
            send_messages = list(messages[start:])
            LOG.info(
                "stateful_session_resume key=%s delta_msgs=%d total_msgs=%d",
                short_key(binding_key),
                len(send_messages),
                len(messages),
            )
            if not send_messages:
                raise ValueError(
                    "Stateful ACP session has no new messages to send; append "
                    "a new user message and retry."
                )

        prompt_text = messages_to_prompt(send_messages, tools=tools) or "User: Hello"
        target_index = len(messages) - 1

        try:
            await asyncio.wait_for(session.lock.acquire(), timeout=lock_timeout)
        except asyncio.TimeoutError:
            LOG.info(
                "stateful_session_busy key=%s timeout=%s",
                short_key(binding_key),
                lock_timeout,
            )
            raise TimeoutError(
                "ACP stateful session is busy; another request is in flight."
            )

        try:
            async for chunk in self._stateful_turn(session, prompt_text, target_index):
                yield chunk
        finally:
            if session.lock.locked():
                session.lock.release()

    async def _stateful_turn(
        self,
        session: ManagedACPSession,
        prompt_text: str,
        target_index: int,
    ) -> AsyncIterator[GenericStreamingChunk]:
        conn = session.conn
        client = session.client
        prompt_task = asyncio.create_task(
            conn.prompt(
                session_id=session.acp_session_id,
                prompt=[text_block(prompt_text)],
            )
        )

        try:
            while True:
                if prompt_task.done() and client.queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(client.queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                chunk = _event_to_chunk(event)
                if chunk is not None:
                    yield chunk
            await prompt_task
        except Exception as exc:
            LOG.info(
                "stateful_session_error key=%s error=%s",
                short_key(session.binding_key),
                type(exc).__name__,
            )
            if not prompt_task.done():
                prompt_task.cancel()
            await self.session_manager.close(
                session.binding_key, reason="prompt_error"
            )
            raise
        finally:
            if not prompt_task.done():
                prompt_task.cancel()

        session.turn += 1
        session.last_used_at = time.time()
        session.last_sent_message_index = target_index
        yield {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": "",
            "tool_use": None,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
