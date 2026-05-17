import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional


def pick_permission_option(options: List[Dict[str, Any]]) -> Optional[str]:
    normalized: List[Dict[str, str]] = []
    for opt in options or []:
        if not isinstance(opt, dict):
            continue
        normalized.append(
            {
                "optionId": str(opt.get("optionId", "")).strip(),
                "kind": str(opt.get("kind", "")).strip().lower(),
            }
        )

    for preferred in ("allow_always", "allow_once"):
        for opt in normalized:
            if opt["kind"] == preferred and opt["optionId"]:
                return opt["optionId"]

    for opt in normalized:
        if "allow" in opt["kind"] and opt["optionId"]:
            return opt["optionId"]

    return None


def content_blocks_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                txt = item.strip()
                if txt:
                    parts.append(txt)
                continue

            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type", "")).strip().lower()

            if item_type in ("text", "input_text", "output_text"):
                txt = str(item.get("text", "")).strip()
                if txt:
                    parts.append(txt)
                continue

            if "content" in item:
                txt = content_blocks_to_text(item["content"])
                if txt:
                    parts.append(txt)
                continue

            if "text" in item:
                txt = str(item.get("text", "")).strip()
                if txt:
                    parts.append(txt)

        return "\n".join(p for p in parts if p).strip()

    if isinstance(content, dict):
        item_type = str(content.get("type", "")).strip().lower()

        if item_type in ("text", "input_text", "output_text"):
            return str(content.get("text", "")).strip()

        if "content" in content:
            return content_blocks_to_text(content["content"])

        if "text" in content:
            return str(content.get("text", "")).strip()

    return str(content).strip()


def responses_input_to_messages(
    input_value: Any,
    instructions: Optional[str] = None,
    existing_messages: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    if existing_messages:
        messages.extend(existing_messages)

    if input_value is None:
        return messages

    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    if isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, str):
                txt = item.strip()
                if txt:
                    messages.append({"role": "user", "content": txt})
                continue

            if not isinstance(item, dict):
                continue

            role = str(item.get("role", "user")).strip().lower() or "user"

            if "content" in item:
                content_text = content_blocks_to_text(item["content"])
                if content_text:
                    messages.append({"role": role, "content": content_text})
                continue

            if "text" in item:
                content_text = str(item.get("text", "")).strip()
                if content_text:
                    messages.append({"role": role, "content": content_text})

        return messages

    messages.append({"role": "user", "content": str(input_value)})
    return messages


def normalize_incoming_messages(kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = kwargs.get("messages")
    if isinstance(messages, list) and messages:
        return messages

    optional_params = kwargs.get("optional_params", {}) or {}
    input_value = kwargs.get("input", None)
    if input_value is None:
        input_value = optional_params.get("input")

    instructions = kwargs.get("instructions")
    if instructions is None:
        instructions = optional_params.get("instructions")

    return responses_input_to_messages(
        input_value=input_value,
        instructions=instructions,
        existing_messages=None,
    )


def messages_to_prompt(messages: List[Dict[str, Any]], tools: Optional[List[Any]] = None) -> str:
    system_parts: List[str] = []
    convo_parts: List[str] = []

    for msg in messages or []:
        role = str(msg.get("role", "user")).strip().lower()
        content = content_blocks_to_text(msg.get("content", ""))

        if not content:
            continue

        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            convo_parts.append(f"Assistant: {content}")
        elif role == "tool":
            name = msg.get("name") or "tool"
            convo_parts.append(f"Tool ({name}): {content}")
        else:
            convo_parts.append(f"User: {content}")

    tool_note = ""
    if tools:
        tool_names: List[str] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if str(tool.get("type", "")).strip() == "function":
                fn = tool.get("function") or {}
                name = fn.get("name")
                if name:
                    tool_names.append(str(name))

        if tool_names:
            tool_note = (
                "\n\nClient tool hints:\n"
                + ", ".join(tool_names)
                + "\nAct directly in the workspace when file or shell work is needed."
            )

    if system_parts:
        base = (
            "System instructions:\n"
            + "\n\n".join(system_parts)
            + "\n\nConversation:\n"
            + "\n\n".join(convo_parts)
        ).strip()
    else:
        base = "\n\n".join(convo_parts).strip()

    return (
        base
        + tool_note
        + "\n\nImportant:"
          "\n- Do the work directly in the workspace when the user asks to create, edit or run files."
          "\n- Prefer non-interactive commands."
          "\n- For scaffolders like Vite, always pass explicit path/name and template."
          "\n- If the latest scaffolder is incompatible with the installed Node.js, use a compatible command instead of stopping."
          "\n- Do not only describe a plan when you can execute the task."
    ).strip()


def extract_existing_paths_from_text(text: str) -> List[Path]:
    """Find filesystem paths mentioned in text that exist on disk (reference Kimi handler)."""
    candidates: List[Path] = []

    unix_paths = re.findall(r"/(?:[^\s'\":<>|]+/?)+", text)
    windows_paths = re.findall(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\s]+\\?)+', text)

    raw_paths = unix_paths + windows_paths
    for raw in raw_paths:
        cleaned = raw.rstrip('.,;:!?)"]\'')
        try:
            p = Path(cleaned).expanduser()
        except Exception:
            continue

        if p.exists():
            candidates.append(p.resolve())

    return candidates


def common_existing_parent(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None

    normalized: List[Path] = []
    for p in paths:
        normalized.append(p.parent if p.is_file() else p)

    if len(normalized) == 1:
        return normalized[0]

    try:
        common = Path(os.path.commonpath([str(p) for p in normalized]))
        if common.exists():
            return common
    except Exception:
        pass

    for p in normalized:
        if p.exists():
            return p

    return None


def coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if isinstance(value, str):
        return shlex.split(value)
    return [str(value)]
