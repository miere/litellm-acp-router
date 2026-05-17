# LiteLLM ACP Router

`litellm-acp-router` is a LiteLLM custom provider for Agent Client Protocol
(ACP) agents. It lets OpenAI-compatible clients call local ACP-capable CLI
agents through the normal LiteLLM proxy.

The package currently includes adapters for:

- Kimi via ACP (`kimi acp`)
- Auggie via ACP (`auggie --acp`)

## Installation

Install the extension into the same Python environment where LiteLLM runs:

```bash
pip install -e .
```

This installs LiteLLM proxy dependencies and the Agent Client Protocol SDK.
You must still install and authenticate the underlying CLI agents you plan to
use, such as `kimi` or `auggie`.

## LiteLLM configuration

Add the ACP provider to your LiteLLM config with `custom_provider_map`:

```yaml
litellm_settings:
  custom_provider_map:
    - provider: acp
      custom_handler: litellm_acp_router.router.router_handler
```

Then expose one or more model aliases through `model_list`:

```yaml
model_list:
  - model_name: acp-kimi
    litellm_params:
      model: acp/kimi

  - model_name: acp-auggie
    litellm_params:
      model: acp/auggie
```

See `litellm_config.example.yaml` for a complete minimal example.

## Running LiteLLM

Run LiteLLM normally with your config file:

```bash
litellm --config /path/to/litellm.yaml
```

This project no longer provides a server launcher. LiteLLM owns the proxy
process; this package only provides the custom ACP provider implementation.

## Example request

```bash
curl -X POST http://127.0.0.1:4000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "acp-auggie",
    "messages": [
      {
        "role": "user",
        "content": "Explain this repository"
      }
    ]
  }'
```

## Auggie model selection

Auggie can choose its own locally configured default model. To preserve that
behavior, omit `acp_model`:

```yaml
model_list:
  - model_name: acp-auggie
    litellm_params:
      model: acp/auggie
```

To pin an Auggie model from LiteLLM config, set the generic `acp_model`
parameter:

```yaml
model_list:
  - model_name: acp-auggie-gpt55
    litellm_params:
      model: acp/auggie
      acp_model: gpt-5.5
```

The Auggie adapter turns this into `auggie --acp --model gpt-5.5`. Use
`auggie models list` or `auggie models list --json` to confirm the exact model
IDs available to your account.

`acp_model` is intentionally generic so future adapters can reuse the same
configuration key when they support model selection.

## Adapter options

The provider accepts several optional LiteLLM parameters:

- `agent`: explicitly choose an adapter, such as `kimi` or `auggie`
- `agent_bin`: override the executable path for any adapter
- `<agent>_bin`: override one adapter executable, such as `auggie_bin`
- `agent_args`: override all launch arguments
- `<agent>_args`: override one adapter's launch arguments
- `agent_mode_id` or `<agent>_mode_id`: set an ACP mode when supported
- `bootstrap_commands` or `<agent>_bootstrap_commands`: run setup prompts
- `acp_model`: select an underlying model when the adapter supports it
- `permission_mode`: `auto_allow` by default, or `cancel`/`deny`/`reject`
- `cwd`, `workspace_path`, `project_root`, or `root_dir`: choose session cwd
- `mcp_servers`: pass MCP server definitions into the ACP session

Environment variables are also supported for executable, args, and mode
overrides using each adapter prefix, for example `KIMI_BIN`, `AUGGIE_BIN`, or
`AUGGIE_ARGS`.

## Architecture

LiteLLM imports `litellm_acp_router.router.router_handler`, which is a
`CustomLLM` instance. Requests flow through:

1. LiteLLM resolves a configured model such as `acp/auggie`.
2. `RouterHandler` normalizes the request into a text prompt.
3. `Registry` selects an adapter from the `acp/<agent>` model name.
4. The adapter builds an `AgentSpec` containing the CLI command and ACP setup.
5. `Runtime` spawns the ACP agent process and streams assistant text back.

## Project structure

```text
litellm_acp_router/
├── adapters/
│   ├── auggie.py
│   ├── base.py
│   ├── kimi.py
│   └── static.py
├── client.py
├── registry.py
├── router.py
├── router_handler.py
├── runtime.py
├── schemas.py
└── utils.py
```

## License

MIT