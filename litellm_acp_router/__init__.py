"""LiteLLM custom provider for Agent Client Protocol agents."""

import logging
import os
import sys

# LiteLLM attaches its handlers to the `litellm` logger namespace, so log
# records emitted by `litellm_acp_router.*` never reach them via propagation.
# To make operational logs (session open/resume, header lookups) visible under
# the standard LiteLLM proxy run we attach a dedicated stderr handler to our
# package logger and disable propagation so we never double-log when another
# handler is also installed on the root logger.
_level_name = os.environ.get("LITELLM_ACP_ROUTER_LOG_LEVEL", "INFO").upper()
_level = getattr(logging, _level_name, logging.INFO)
_pkg_logger = logging.getLogger(__name__)
_pkg_logger.setLevel(_level)
if not any(
    getattr(h, "_acp_router_handler", False) for h in _pkg_logger.handlers
):
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(_level)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    _handler._acp_router_handler = True  # type: ignore[attr-defined]
    _pkg_logger.addHandler(_handler)
    _pkg_logger.propagate = False

__all__: list[str] = []
