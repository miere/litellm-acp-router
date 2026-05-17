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
  user prompt, and yield LiteLLM streaming chunks. It exposes both
  `run_stream` (stateless) and `run_stateful_stream` (opt-in stateful).
- `client.AgentClient` receives ACP session updates and converts assistant
  message chunks into streamable text events. It also surfaces reasoning
  (`agent_thought_chunk`) as `{"kind": "reasoning", ...}` queue events and
  narrates tool activity on `tool_call` start plus `tool_call_update` terminal
  status (`completed` → "done", `failed` → "failed").
- `utils.py` contains request normalization, prompt formatting, path inference,
  argument coercion, and permission-option selection helpers.
- `binding.py` parses the configured `acp_session_binding_strategy` and
  resolves a session key for stateful mode. Two strategies are supported:
  `prompt_hashing` (SHA256 of `system + first user message`, truncated to 16
  hex characters) and `http_header/<NAME>` (case-insensitive lookup on the
  inbound proxy request, fail-fast when missing or empty).
- `session_manager.py` defines `ManagedACPSession` and `SessionManager`, which
  own the lifecycle of long-lived ACP processes indexed by binding key,
  including a `last_sent_message_index` for delta prompting, TTL eviction, and
  LRU max-session enforcement.

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

For stateful mode (`acp_session_binding_strategy` set), step 6 is replaced by
`Runtime.run_stateful_stream`:

1. Resolve cwd, evict expired sessions, enforce max-session cap.
2. Resolve a binding key via `binding.resolve_session_key` using the configured
   strategy. `http_header/<NAME>` reads from `kwargs["proxy_server_request"]`
   and fails fast if the header is missing or empty.
3. Look up an existing `ManagedACPSession` by binding key (namespaced by
   adapter, model, optional `acp_model`, and cwd); create a new ACP process
   via `SessionManager` if no match exists.
4. Acquire the per-session asyncio lock with a timeout.
5. Build the prompt from `messages[last_sent_message_index + 1:]` when
   resuming, or from the full history when creating a new session. Raise
   `ValueError` if a resume request has no new messages to send.
6. Stream assistant chunks, then emit the stop chunk and advance
   `last_sent_message_index` to the last message index of the request.

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
            acp_workspace_arg="--workspace-root",
            default_workspace_dir=None,
        )
```

If the adapter supports model selection, use the generic optional parameter
`acp_model` and set `acp_model_arg` to the CLI flag that accepts the model. If
`acp_model` is absent, do not pass a model flag; let the underlying CLI use its
own local default.

If the adapter exposes a workspace/project-root flag, use the generic optional
parameter `acp_workspace_dir` and set `acp_workspace_arg` to the CLI flag that
accepts it. `default_workspace_dir` provides a fallback when neither
`acp_workspace_dir` nor `<PREFIX>_WORKSPACE_DIR` is set; leave it `None` to
skip the flag entirely in that case.

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
- Prefer `acp_workspace_dir` for adapter workspace selection instead of
  adapter-specific keys. Adapters honor `<PREFIX>_WORKSPACE_DIR` as an
  environment-variable fallback.
- `acp_emit_tool_activity` (default `true`) toggles inline narration of ACP
  `tool_call` and `tool_call_update` events as assistant text chunks. Disable
  it when a deployment needs clean prose only. The flag is read in both
  `Runtime.run_stream` and `SessionManager.create` and forwarded to
  `AgentClient`. Narration covers `tool_call` start plus terminal
  `tool_call_update` status (`completed`, `failed`); intermediate progress is
  suppressed.
- Reasoning is forwarded to LiteLLM as a streaming chunk with
  `provider_specific_fields={"reasoning_content": <text>}` and an empty
  `text`. LiteLLM's streaming handler propagates that onto the OpenAI delta's
  `reasoning_content` field. Reasoning is intentionally excluded from
  `AgentClient.get_final_text()` so it does not enter the assistant transcript
  used by stateful resume or the non-streaming `acompletion` path.
- `acp_stdio_buffer_bytes` (default `8 * 1024 * 1024`, i.e. 8 MiB) sets the
  `asyncio.StreamReader` buffer used to read JSON-RPC frames from the agent's
  stdout. Threaded through both spawn sites via
  `transport_kwargs={"limit": N}` to `spawn_agent_process`. Resolved by
  `runtime.resolve_stdio_buffer_bytes`, which falls back to the default for
  missing, non-numeric, or non-positive values. Raise it when large
  `tool_call` frames (file diffs, terminal output, MCP responses) trigger
  upstream `LimitOverrunError` and surface as `ConnectionError`.
- Adapter-specific binary, args, mode, and bootstrap overrides use existing
  keys such as `<agent>_bin`, `<agent>_args`, `<agent>_mode_id`, and
  `<agent>_bootstrap_commands`.

## Validation guidance

After code changes, run at least:

```bash
python3 -m compileall litellm_acp_router tests
python3 -m unittest discover -s tests
python3 -c "from litellm_acp_router.router import router_handler; print(type(router_handler).__name__)"
```

Test expectations:

- Stateless adapter tests continue to pass.
- `binding.py` unit tests cover strategy parsing, deterministic prompt
  hashing, case-insensitive header lookup, and fail-fast behavior when a
  configured header is missing or empty.
- `SessionManager` unit tests cover create, get, close, TTL eviction,
  LRU max-session enforcement, and `acp_stdio_buffer_bytes` propagation
  (default and custom) via `transport_kwargs`, using fake process context
  managers keyed by binding key.
- Stateful runtime tests mock `spawn_agent_process` and cover first-request
  session creation, second-request delta resume, header-driven resume,
  different first messages opening new sessions, missing-header fail-fast,
  empty-delta errors, lock-timeout errors, and `acp_stdio_buffer_bytes`
  propagation (default and custom).

If the relevant CLIs are installed and authenticated, smoke-test through
LiteLLM using `litellm_config.example.yaml` or a local config derived from it.