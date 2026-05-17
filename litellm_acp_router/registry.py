from typing import Any, Dict, Optional

from .adapters.base import Adapter


class Registry:
    def __init__(self, default_agent: str = "kimi") -> None:
        self._adapters: Dict[str, Adapter] = {}
        self.default_agent = default_agent.strip().lower()

    def register(self, adapter: Adapter) -> None:
        self._adapters[adapter.agent_id] = adapter

    def get(self, agent_id: str) -> Optional[Adapter]:
        return self._adapters.get(agent_id.strip().lower())

    def resolve(self, model: str, optional_params: Dict[str, Any]) -> Adapter:
        explicit_agent = str(optional_params.get("agent", "")).strip().lower()
        if explicit_agent:
            adapter = self.get(explicit_agent)
            if adapter:
                return adapter

        normalized_model = str(model or "").strip().lower()
        parts = [p.strip() for p in normalized_model.split("/") if p.strip()]

        if len(parts) >= 2 and parts[0] == "acp":
            adapter = self.get(parts[1])
            if adapter:
                return adapter

        for adapter in self._adapters.values():
            if adapter.matches(normalized_model):
                return adapter
            for alias in getattr(adapter, "aliases", []):
                if alias and alias in normalized_model:
                    return adapter

        default_adapter = self.get(self.default_agent)
        if default_adapter:
            return default_adapter

        if self._adapters:
            return next(iter(self._adapters.values()))

        raise ValueError("No adapters registered.")
