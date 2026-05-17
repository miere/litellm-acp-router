from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentSpec:
    agent_id: str
    bin: str
    args: List[str]
    mode_id: Optional[str] = "code"
    bootstrap_commands: List[str] = field(default_factory=list)
