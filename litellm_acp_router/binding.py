"""Session binding strategies for stateful ACP routing.

Two strategies are supported:

* ``prompt_hashing`` — derive a stable session key from the system prompt and
  the first user message. Reissuing the same opening turn lands on the same
  long-lived agent process, which works without any client cooperation.
* ``http_header/<NAME>`` — look up an inbound header on the LiteLLM proxy
  request. Clients control affinity by sending a stable identifier such as
  ``X-Hermes-Conversation-Id``. If the header is missing or empty, the request
  fails fast with a message that points at the easiest workaround.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Tuple

STRATEGY_PROMPT_HASHING = "prompt_hashing"
STRATEGY_HTTP_HEADER = "http_header"


def parse_binding_strategy(value: Any) -> Tuple[str, Optional[str]]:
    """Parse a configured strategy into ``(name, argument)``.

    ``prompt_hashing`` has no argument.
    ``http_header/<NAME>`` carries the header name as the argument.
    Anything else raises ``ValueError`` with a hint at valid forms.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "acp_session_binding_strategy must be a non-empty string; "
            "expected 'prompt_hashing' or 'http_header/<NAME>'."
        )
    raw = value.strip()
    if raw == STRATEGY_PROMPT_HASHING:
        return STRATEGY_PROMPT_HASHING, None
    if raw.startswith(STRATEGY_HTTP_HEADER + "/"):
        header_name = raw[len(STRATEGY_HTTP_HEADER) + 1 :].strip()
        if not header_name:
            raise ValueError(
                "http_header binding strategy requires a header name, e.g. "
                "'http_header/X-Hermes-Conversation-Id'."
            )
        return STRATEGY_HTTP_HEADER, header_name
    raise ValueError(
        f"Unknown acp_session_binding_strategy {raw!r}; expected "
        "'prompt_hashing' or 'http_header/<NAME>'."
    )


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                inner = item.get("text")
                if isinstance(inner, str):
                    parts.append(inner)
                else:
                    nested = item.get("content")
                    if nested is not None:
                        parts.append(_content_to_text(nested))
        return "\n".join(parts)
    if isinstance(content, dict):
        inner = content.get("text")
        if isinstance(inner, str):
            return inner
        nested = content.get("content")
        if nested is not None:
            return _content_to_text(nested)
    return ""


def _first_system_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages or []:
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "system":
            return _content_to_text(msg.get("content"))
    return ""


def _first_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages or []:
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
            return _content_to_text(msg.get("content"))
    return ""


def _hash_prompt(messages: List[Dict[str, Any]]) -> str:
    system = _first_system_text(messages)
    first_user = _first_user_text(messages)
    if not first_user.strip():
        raise ValueError(
            "prompt_hashing binding requires at least one user message in the "
            "request; received none."
        )
    payload = (system + "\n" + first_user).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _lookup_header(
    headers: Optional[Mapping[str, Any]],
    name: str,
) -> Optional[str]:
    if not headers:
        return None
    target = name.lower()
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == target:
            if value is None:
                return None
            return str(value)
    return None


def _extract_headers_from_kwargs(
    kwargs: Optional[Mapping[str, Any]],
) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
    """Find the inbound headers dict inside the LiteLLM CustomLLM kwargs.

    LiteLLM is inconsistent about where it places client headers depending on
    the proxy entry point and the LiteLLM version. We probe the locations we
    have actually observed in the wild, in order:

    1. ``kwargs['proxy_server_request']['headers']`` — set by the FastAPI
       proxy when running ``litellm --config ...``.
    2. ``kwargs['headers']`` — top-level passthrough used by some streaming
       paths.
    3. ``kwargs['litellm_params']['proxy_server_request']['headers']`` —
       nested copy maintained for retries/fallbacks.
    4. ``kwargs['litellm_params']['metadata']['headers']`` — set by the
       metadata-propagation layer.

    Returns ``(headers_mapping, source_label)`` where ``source_label`` is a
    human-readable hint for logging. Both are ``None`` when nothing was found.
    """
    if not isinstance(kwargs, Mapping):
        return None, None

    candidates: List[Tuple[str, Any]] = []
    psr = kwargs.get("proxy_server_request")
    if isinstance(psr, Mapping):
        candidates.append(("kwargs.proxy_server_request.headers", psr.get("headers")))
    candidates.append(("kwargs.headers", kwargs.get("headers")))
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, Mapping):
        nested_psr = litellm_params.get("proxy_server_request")
        if isinstance(nested_psr, Mapping):
            candidates.append(
                ("kwargs.litellm_params.proxy_server_request.headers", nested_psr.get("headers"))
            )
        metadata = litellm_params.get("metadata")
        if isinstance(metadata, Mapping):
            candidates.append(
                ("kwargs.litellm_params.metadata.headers", metadata.get("headers"))
            )

    for source, candidate in candidates:
        if isinstance(candidate, Mapping) and candidate:
            return candidate, source
    return None, None


def resolve_session_key(
    strategy: str,
    *,
    messages: List[Dict[str, Any]],
    kwargs: Optional[Mapping[str, Any]] = None,
    proxy_server_request: Optional[Mapping[str, Any]] = None,
) -> str:
    """Resolve the binding key for ``strategy``.

    For ``http_header`` strategies, headers are looked up first inside the full
    ``kwargs`` mapping (which is what LiteLLM hands to the custom provider) and,
    as a backwards-compatible shortcut, inside the explicit
    ``proxy_server_request`` argument.

    Raises ``ValueError`` on misconfiguration or, for ``http_header``, when the
    configured header is missing or empty from the inbound request.
    """
    name, arg = parse_binding_strategy(strategy)
    if name == STRATEGY_PROMPT_HASHING:
        return _hash_prompt(messages)
    if name == STRATEGY_HTTP_HEADER:
        headers, _source = _extract_headers_from_kwargs(kwargs)
        if headers is None and isinstance(proxy_server_request, Mapping):
            psr_headers = proxy_server_request.get("headers")
            if isinstance(psr_headers, Mapping):
                headers = psr_headers
        value = _lookup_header(headers, arg or "")
        if value is None or not value.strip():
            raise ValueError(
                f"acp_session_binding_strategy 'http_header/{arg}' requires "
                f"inbound header {arg!r} to be set; resend the request with "
                f"a non-empty {arg!r} header (e.g. a stable conversation id)."
            )
        return value.strip()
    raise ValueError(f"Unsupported binding strategy {name!r}.")
