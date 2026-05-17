from .static import StaticAdapter


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
        )
