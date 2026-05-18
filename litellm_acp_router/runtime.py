import asyncio
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple

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
# receive loop with LimitOverrunError. 64 MiB gives ample headroom for large
# diffs, terminal dumps, and MCP responses while keeping the per-process memory
# ceiling bounded. Operators can override via the acp_stdio_buffer_bytes
# option; setting it to 0 disables the practical limit (see
# UNBOUNDED_STDIO_BUFFER_BYTES).
DEFAULT_STDIO_BUFFER_BYTES = 64 * 1024 * 1024
# Effectively-unbounded StreamReader limit. asyncio.StreamReader requires a
# positive integer for `limit`; 2**31 - 1 (~2 GiB) is far beyond any realistic
# JSON-RPC frame and avoids LimitOverrunError for pathological agent output.
UNBOUNDED_STDIO_BUFFER_BYTES = 2**31 - 1


def resolve_stdio_buffer_bytes(options: Dict[str, Any]) -> int:
    raw = options.get("acp_stdio_buffer_bytes")
    if raw is None:
        return DEFAULT_STDIO_BUFFER_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_STDIO_BUFFER_BYTES
    if value == 0:
        return UNBOUNDED_STDIO_BUFFER_BYTES
    if value < 0:
        return DEFAULT_STDIO_BUFFER_BYTES
    return value


# Diagnostic ring buffer for receive-loop overruns. The upstream acp package's
# receive loop logs "Receive loop failed" to the root logger with exc_info when
# asyncio.StreamReader raises LimitOverrunError (the chained ValueError in
# Python 3.13's streams.readline). The receive loop then completes all pending
# requests with a generic ConnectionError("Connection closed"), which gives the
# operator no hint that the cure is to raise acp_stdio_buffer_bytes. We snoop
# the log record and remember the most recent overrun so the runtime can wrap
# the resulting ConnectionError with an actionable message.
_OVERRUN_EVENT_WINDOW_SECONDS = 30.0
_OVERRUN_EVENTS: Deque[Tuple[float, str]] = deque(maxlen=32)
_OVERRUN_MARKER = "Separator is found, but chunk is longer than limit"


def _is_limit_overrun_exc(exc: Optional[BaseException]) -> bool:
    if exc is None:
        return False
    if isinstance(exc, asyncio.LimitOverrunError):
        return True
    # Python 3.13's StreamReader.readline re-raises LimitOverrunError as
    # ValueError(e.args[0]); the original is chained via __context__.
    if isinstance(exc, ValueError) and _OVERRUN_MARKER in str(exc):
        return True
    return _is_limit_overrun_exc(exc.__context__) or _is_limit_overrun_exc(
        exc.__cause__
    )


class _ReceiveLoopOverrunFilter(logging.Filter):
    """Records LimitOverrunError occurrences from acp's receive loop.

    Always returns True so the underlying log record is still emitted; the
    filter is only used as a passive observer.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.getMessage() != "Receive loop failed":
                return True
            exc_info = record.exc_info
            if not exc_info:
                return True
            _, exc, _ = exc_info
            if _is_limit_overrun_exc(exc):
                _OVERRUN_EVENTS.append((time.monotonic(), str(exc)))
        except Exception:  # pragma: no cover - never break logging
            pass
        return True


_OVERRUN_FILTER = _ReceiveLoopOverrunFilter()
_OVERRUN_FILTER_INSTALLED = False


def _install_receive_loop_filter() -> None:
    global _OVERRUN_FILTER_INSTALLED
    if _OVERRUN_FILTER_INSTALLED:
        return
    logging.getLogger().addFilter(_OVERRUN_FILTER)
    _OVERRUN_FILTER_INSTALLED = True


_install_receive_loop_filter()


def _recent_overrun_event(
    window_seconds: float = _OVERRUN_EVENT_WINDOW_SECONDS,
) -> Optional[Tuple[float, str]]:
    if not _OVERRUN_EVENTS:
        return None
    ts, msg = _OVERRUN_EVENTS[-1]
    if (time.monotonic() - ts) > window_seconds:
        return None
    return ts, msg


def _wrap_connection_error_if_overrun(exc: BaseException) -> BaseException:
    """If exc is a ConnectionError coincident with a recent receive-loop
    overrun, return a new ConnectionError with an actionable message; else
    return exc unchanged. The original exception is preserved via __cause__.
    """
    if not isinstance(exc, ConnectionError):
        return exc
    event = _recent_overrun_event()
    if event is None:
        return exc
    _, original_msg = event
    wrapped = ConnectionError(
        "ACP receive loop failed: the agent emitted a JSON-RPC frame larger "
        "than the configured stdio buffer "
        "(acp_stdio_buffer_bytes). Increase acp_stdio_buffer_bytes, or set it "
        "to 0 for no practical limit, then retry. "
        f"Underlying error: {original_msg}"
    )
    wrapped.__cause__ = exc
    return wrapped


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
            tool_narrator=spec.tool_narrator,
            text_filter=(
                spec.text_filter_factory() if spec.text_filter_factory else None
            ),
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
                        await client.flush_text_filter()
                        if client.queue.empty():
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
                wrapped = _wrap_connection_error_if_overrun(exc)
                if wrapped is not exc:
                    raise wrapped from exc
                raise
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
        # Reset per-turn flags on the reused client so display-only state from
        # the previous turn (e.g. trailing tool-narration separator) does not
        # leak into the start of this turn.
        client.reset_turn_state()
        prompt_task = asyncio.create_task(
            conn.prompt(
                session_id=session.acp_session_id,
                prompt=[text_block(prompt_text)],
            )
        )

        try:
            while True:
                if prompt_task.done() and client.queue.empty():
                    await client.flush_text_filter()
                    if client.queue.empty():
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
            wrapped = _wrap_connection_error_if_overrun(exc)
            if wrapped is not exc:
                raise wrapped from exc
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
