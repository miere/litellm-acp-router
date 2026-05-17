import asyncio
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

from acp import spawn_agent_process, text_block
from litellm.types.utils import GenericStreamingChunk

from .client import AgentClient
from .schemas import AgentSpec
from .utils import (
    common_existing_parent,
    content_blocks_to_text,
    extract_existing_paths_from_text,
)


class Runtime:
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

        client = AgentClient(permission_mode=permission_mode)

        async with spawn_agent_process(client, spec.bin, *spec.args) as (conn, _proc):
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

                    text = event.get("text") or ""
                    if not text:
                        continue

                    yield {
                        "finish_reason": None,
                        "index": 0,
                        "is_finished": False,
                        "text": text,
                        "tool_use": None,
                        "usage": None,
                    }

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
