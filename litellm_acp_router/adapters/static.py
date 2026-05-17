import os
from typing import Any, Dict, List, Optional

from litellm_acp_router.schemas import AgentSpec
from litellm_acp_router.utils import coerce_list

from .base import Adapter


class StaticAdapter(Adapter):
    def __init__(
        self,
        agent_id: str,
        default_bin: str,
        default_args: List[str],
        default_mode_id: Optional[str] = "code",
        default_bootstrap_commands: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        env_var_prefix: Optional[str] = None,
        acp_model_arg: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id.strip().lower()
        self.default_bin = default_bin
        self.default_args = list(default_args)
        self.default_mode_id = default_mode_id
        self.default_bootstrap_commands = list(default_bootstrap_commands or [])
        self.aliases = [a.strip().lower() for a in (aliases or [])]
        self.env_var_prefix = (env_var_prefix or self.agent_id).upper().replace("-", "_")
        self.acp_model_arg = acp_model_arg

    def build_spec(self, optional_params: Dict[str, Any]) -> AgentSpec:
        bin_value = (
            optional_params.get(f"{self.agent_id}_bin")
            or optional_params.get("agent_bin")
            or os.getenv(f"{self.env_var_prefix}_BIN")
            or self.default_bin
        )

        args_value = (
            optional_params.get(f"{self.agent_id}_args")
            or optional_params.get("agent_args")
            or os.getenv(f"{self.env_var_prefix}_ARGS")
        )
        args = coerce_list(args_value) if args_value else list(self.default_args)

        acp_model = optional_params.get("acp_model")
        if self.acp_model_arg and acp_model:
            args.extend([self.acp_model_arg, str(acp_model)])

        mode_id = (
            optional_params.get(f"{self.agent_id}_mode_id")
            or optional_params.get("agent_mode_id")
            or os.getenv(f"{self.env_var_prefix}_MODE_ID")
            or self.default_mode_id
        )

        bootstrap_value = (
            optional_params.get(f"{self.agent_id}_bootstrap_commands")
            or optional_params.get("bootstrap_commands")
        )
        bootstrap_commands = (
            coerce_list(bootstrap_value)
            if bootstrap_value is not None
            else list(self.default_bootstrap_commands)
        )

        return AgentSpec(
            agent_id=self.agent_id,
            bin=str(bin_value),
            args=[str(x) for x in args],
            mode_id=str(mode_id) if mode_id else None,
            bootstrap_commands=[str(x) for x in bootstrap_commands],
        )
