from .static import StaticAdapter


class KimiAdapter(StaticAdapter):
    def __init__(self) -> None:
        super().__init__(
            agent_id="kimi",
            default_bin="kimi",
            default_args=["acp"],
            default_mode_id="code",
            # Match reference KimiACPHandler._bootstrap_agent_session
            default_bootstrap_commands=["/plan off", "/yolo"],
            aliases=["moonshot", "kimi-code"],
            env_var_prefix="KIMI",
        )
