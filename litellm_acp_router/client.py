import asyncio
from typing import Any, Dict, List, Optional

from acp.interfaces import Client

from .schemas import TextFilter, ToolNarrator
from .utils import pick_permission_option


class AgentClient(Client):
    def __init__(
        self,
        permission_mode: str = "auto_allow",
        emit_tool_activity: bool = True,
        tool_narrator: Optional[ToolNarrator] = None,
        text_filter: Optional[TextFilter] = None,
    ) -> None:
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.final_text_parts: List[str] = []
        self.suppress_stream = False
        self.permission_mode = permission_mode.strip().lower()
        self.emit_tool_activity = bool(emit_tool_activity)
        self.tool_narrator = tool_narrator
        self.text_filter = text_filter

    async def request_permission(
        self, options, session_id=None, tool_call=None, **kwargs: Any
    ):
        if self.permission_mode in ("cancel", "deny", "reject"):
            return {"outcome": {"outcome": "cancelled"}}

        safe_options = options if isinstance(options, list) else []
        chosen_option_id = pick_permission_option(safe_options)

        if chosen_option_id:
            return {
                "outcome": {
                    "outcome": "selected",
                    "optionId": chosen_option_id,
                }
            }

        return {"outcome": {"outcome": "cancelled"}}

    async def session_update(self, session_id, update, **kwargs):
        data = update.model_dump() if hasattr(update, "model_dump") else update
        if not isinstance(data, dict):
            return

        update_kind = str(
            data.get("session_update") or data.get("sessionUpdate") or ""
        ).strip()

        if update_kind == "agent_thought_chunk":
            if self.suppress_stream:
                return
            text = self._content_block_to_text(data.get("content"))
            if text:
                # Reasoning is intentionally not appended to final_text_parts so
                # it does not pollute the assistant transcript captured for the
                # non-streaming acompletion path or for stateful history.
                await self.queue.put({"kind": "reasoning", "text": text})
            return

        if update_kind == "agent_message_chunk":
            text = self._content_block_to_text(data.get("content"))
            if text:
                if self.suppress_stream:
                    return
                if self.text_filter is not None:
                    text = self.text_filter.feed(text)
                    if not text:
                        return
                self.final_text_parts.append(text)
                await self.queue.put({"kind": "assistant_text", "text": text})
            return

        if update_kind == "tool_call":
            if self.suppress_stream or not self.emit_tool_activity:
                return
            text = self._format_tool_call_start(data)
            if text:
                await self.queue.put({"kind": "assistant_text", "text": text})
            return

        # tool_call_update events are intentionally not narrated: status flips
        # (completed/failed) would produce duplicate "done" lines and the
        # adapter-supplied narrator only formats the initial tool_call.

    def _format_tool_call_start(self, data: Dict[str, Any]) -> str:
        # Adapters opt in to tool narration by supplying a tool_narrator. When
        # none is provided (e.g. KimiAdapter) tool activity is silent.
        if self.tool_narrator is None:
            return ""
        title = self._read_string(data, "title")
        if not title:
            return ""
        kind = self._read_string(data, "kind")
        result = self.tool_narrator(kind, title)
        return result or ""

    async def flush_text_filter(self) -> None:
        # Drain any text the filter held back across chunk boundaries (e.g. a
        # partial XML tag whose closing '>' never arrived) at end-of-turn.
        if self.text_filter is None:
            return
        remaining = self.text_filter.flush()
        if remaining:
            self.final_text_parts.append(remaining)
            await self.queue.put({"kind": "assistant_text", "text": remaining})

    @staticmethod
    def _read_string(data: Dict[str, Any], key: str) -> str:
        value = data.get(key)
        if value is None:
            return ""
        return str(value).strip()

    def _content_block_to_text(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            if value.get("type") == "text":
                return str(value.get("text", ""))
            if "text" in value:
                return str(value.get("text", ""))
            if "content" in value:
                return self._content_block_to_text(value["content"])
            return ""

        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = str(item.get("text", ""))
                        if text:
                            parts.append(text)
                    elif "content" in item:
                        text = self._content_block_to_text(item["content"])
                        if text:
                            parts.append(text)
                    elif "text" in item:
                        text = str(item.get("text", ""))
                        if text:
                            parts.append(text)
                elif isinstance(item, str):
                    if item:
                        parts.append(item)
            return "".join(parts)

        return ""

    def get_final_text(self) -> str:
        return "".join(self.final_text_parts).strip()
