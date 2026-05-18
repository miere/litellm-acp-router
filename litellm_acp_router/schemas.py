from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, runtime_checkable


# A ToolNarrator turns an ACP tool_call (kind, title) into the text streamed to
# the client, or returns None to suppress narration for that call. Adapters that
# do not set one get no tool narration at all.
ToolNarrator = Callable[[str, str], Optional[str]]


@runtime_checkable
class TextFilter(Protocol):
    """Stateful filter applied to streamed assistant text chunks.

    `feed` receives an incoming chunk and returns the portion safe to emit
    now; the filter may hold back a trailing buffer when a chunk could be
    the prefix of something to strip. `flush` drains any remaining buffer
    at end-of-turn.
    """

    def feed(self, chunk: str) -> str: ...

    def flush(self) -> str: ...


TextFilterFactory = Callable[[], TextFilter]


@dataclass
class AgentSpec:
    agent_id: str
    bin: str
    args: List[str]
    mode_id: Optional[str] = "code"
    bootstrap_commands: List[str] = field(default_factory=list)
    tool_narrator: Optional[ToolNarrator] = None
    text_filter_factory: Optional[TextFilterFactory] = None
