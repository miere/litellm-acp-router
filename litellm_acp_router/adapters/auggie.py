import re
from typing import Optional, Tuple

from .static import StaticAdapter


# ACP tool `kind` -> (emoji, label) shown in the streamed narration. The
# fallback (used for any kind not listed, including empty) is the "thinking"
# entry.
_AUGGIE_TOOL_PREFIXES: dict = {
    "read": ("\U0001F4D6", "read"),            # 📖
    "write": ("\u270D\uFE0F", "write"),         # ✍️
    "edit": ("\u270D\uFE0F", "write"),          # ✍️
    "execute": ("\U0001F4BB", "terminal"),     # 💻
    "fetch": ("\U0001F310", "browser_navigate"),  # 🌐
}
_AUGGIE_TOOL_FALLBACK: Tuple[str, str] = ("\U0001F9E0", "thinking")  # 🧠

_TITLE_MAX_LEN = 40
_TITLE_TRUNCATE_AT = 37
_TITLE_NEWLINE_RE = re.compile(r"[\r\n]+")


def _sanitize_title(title: str) -> str:
    # Collapse any embedded newlines so the narration stays on a single line.
    flattened = _TITLE_NEWLINE_RE.sub(" ", title)
    if len(flattened) > _TITLE_MAX_LEN:
        flattened = flattened[:_TITLE_TRUNCATE_AT] + "..."
    # If truncation (or the original title) left an unbalanced backtick, close
    # it so downstream markdown renderers do not leak inline-code styling into
    # the rest of the stream.
    if flattened.count("`") % 2 == 1:
        flattened = flattened + "`"
    return flattened


def auggie_tool_narrator(kind: str, title: str) -> Optional[str]:
    if not title:
        return None
    emoji, label = _AUGGIE_TOOL_PREFIXES.get(kind, _AUGGIE_TOOL_FALLBACK)
    return f"{emoji} {label}: {_sanitize_title(title)}\n"


class AuggieTextFilter:
    """Strip `<augment_code_snippet ...>` / `</augment_code_snippet>` tags from
    streamed assistant text while keeping the wrapped content intact. Tags can
    arrive split across chunk boundaries, so the filter buffers any trailing
    fragment that could be the prefix of a tag until the next chunk (or flush)
    resolves it.
    """

    _TAG_RE = re.compile(r"</?augment_code_snippet\b[^>]*>")
    _PARTIAL_PREFIXES = ("<augment_code_snippet", "</augment_code_snippet")

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._pending += chunk
        # Strip any complete tags first.
        self._pending = self._TAG_RE.sub("", self._pending)
        # If the buffer ends in a fragment that could still grow into one of
        # our tags, hold it back. We only hold fragments that look like a
        # prefix of `<augment_code_snippet`/`</augment_code_snippet` or that
        # have started one of those tags but have not yet seen the closing
        # `>` — any other `<...` content is emitted immediately.
        last_lt = self._pending.rfind("<")
        if last_lt != -1:
            tail = self._pending[last_lt:]
            if ">" not in tail and self._looks_like_partial(tail):
                emit = self._pending[:last_lt]
                self._pending = tail
                return emit
        emit = self._pending
        self._pending = ""
        return emit

    def flush(self) -> str:
        # Stream is ending: drop any complete tags still in the buffer and
        # release whatever remains so we never swallow real content.
        out = self._TAG_RE.sub("", self._pending)
        self._pending = ""
        return out

    @classmethod
    def _looks_like_partial(cls, tail: str) -> bool:
        for prefix in cls._PARTIAL_PREFIXES:
            if prefix.startswith(tail) or tail.startswith(prefix):
                return True
        return False


class AuggieAdapter(StaticAdapter):
    def __init__(self) -> None:
        super().__init__(
            agent_id="auggie",
            default_bin="auggie",
            default_args=["--acp", "--allow-indexing"],
            default_mode_id=None,
            aliases=["augment", "augment-code"],
            env_var_prefix="AUGGIE",
            acp_model_arg="--model",
            acp_workspace_arg="--workspace-root",
            default_workspace_dir="/tmp/auggie-empty",
            tool_narrator=auggie_tool_narrator,
            text_filter_factory=AuggieTextFilter,
        )
