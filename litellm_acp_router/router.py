import os

from .adapters import AuggieAdapter, KimiAdapter
from .registry import Registry
from .router_handler import RouterHandler


registry = Registry(default_agent=os.getenv("ROUTER_DEFAULT_AGENT", "kimi"))

registry.register(KimiAdapter())
registry.register(AuggieAdapter())

router_handler = RouterHandler(registry)
