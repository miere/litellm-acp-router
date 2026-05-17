# Agent Guide

This repository implements a LiteLLM custom provider that routes requests to
local Agent Client Protocol (ACP) CLI agents.

## High-level architecture

- `litellm_acp_router.router` creates the global adapter registry and exports
  `router_handler`, the object referenced by LiteLLM's `custom_provider_map`.
- `router_handler.RouterHandler` is the LiteLLM `CustomLLM` implementation. It
  normalizes incoming messages, resolves an adapter, builds a prompt, and
  delegates execution to the runtime.
- `registry.Registry` maps LiteLLM model names like `acp/kimi` or
  `acp/auggie` to adapter instances.
- `adapters/*` define how each CLI agent is launched. Most agents can extend
  `StaticAdapter` with a binary, default arguments, aliases, mode, and optional
  model flag support.
- `runtime.Runtime` owns ACP process lifecycle: spawn the agent, initialize the
  protocol session, set mode when configured, run bootstrap prompts, send the
  user prompt, and yield LiteLLM streaming chunks.
- `client.AgentClient` receives ACP session updates and converts assistant
  message chunks into streamable text events.
- `utils.py` contains request normalization, prompt formatting, path inference,
  argument coercion, and permission-option selection helpers.

## Request flow

1. LiteLLM receives a request for a configured alias, for example
   `acp-auggie`.
2. LiteLLM maps that alias to `litellm_params.model`, for example
   `acp/auggie`.
3. LiteLLM dispatches to `litellm_acp_router.router.router_handler` because the
   config maps provider `acp` to that custom handler.
4. `RouterHandler` selects the adapter from the model name and optional params.
5. The adapter returns an `AgentSpec`, such as `auggie --acp` or `kimi acp`.
6. `Runtime` starts the ACP process, sends the prompt, and yields streamed text
   back through LiteLLM.

## Adding an adapter

For a simple ACP CLI, add a file under `litellm_acp_router/adapters/` and extend
`StaticAdapter`:

```python
from .static import StaticAdapter


class ExampleAdapter(StaticAdapter):
    def __init__(self) -> None:
        super().__init__(
            agent_id="example",
            default_bin="example-agent",
            default_args=["--acp"],
            default_mode_id=None,
            aliases=["example-ai"],
            env_var_prefix="EXAMPLE",
            acp_model_arg="--model",
        )
```

If the adapter supports model selection, use the generic optional parameter
`acp_model` and set `acp_model_arg` to the CLI flag that accepts the model. If
`acp_model` is absent, do not pass a model flag; let the underlying CLI use its
own local default.

Then export and register the adapter:

- add it to `litellm_acp_router/adapters/__init__.py`
- instantiate it in `litellm_acp_router/router.py`

Users can then configure LiteLLM with `model: acp/example`.

## Configuration conventions

- Public LiteLLM aliases belong in the user's `model_list`.
- The custom provider mapping should point to
  `litellm_acp_router.router.router_handler`.
- Prefer `acp_model` for adapter model selection instead of adapter-specific
  keys.
- Adapter-specific binary, args, mode, and bootstrap overrides use existing
  keys such as `<agent>_bin`, `<agent>_args`, `<agent>_mode_id`, and
  `<agent>_bootstrap_commands`.

## Validation guidance

After code changes, run at least:

```bash
python3 -m compileall litellm_acp_router
python3 -c "from litellm_acp_router.router import router_handler; print(type(router_handler).__name__)"
```

If the relevant CLIs are installed and authenticated, smoke-test through
LiteLLM using `litellm_config.example.yaml` or a local config derived from it.