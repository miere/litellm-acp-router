import asyncio
from typing import Any, Dict, List

from acp.interfaces import Client

from .utils import pick_permission_option


class AgentClient(Client):
    def __init__(self, permission_mode: str = "auto_allow") -> None:
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.final_text_parts: List[str] = []
        self.suppress_stream = False
        self.permission_mode = permission_mode.strip().lower()

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
            return

        if update_kind == "agent_message_chunk":
            text = self._content_block_to_text(data.get("content"))
            if text:
                if self.suppress_stream:
                    return
                self.final_text_parts.append(text)
                await self.queue.put({"kind": "assistant_text", "text": text})

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
