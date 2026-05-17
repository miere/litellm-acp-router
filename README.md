# LiteLLM ACP Router

`litellm-acp-router` is a LiteLLM custom provider for Agent Client Protocol
(ACP) agents. It lets OpenAI-compatible clients call local ACP-capable CLI
agents through the normal LiteLLM proxy.

The package currently includes adapters for:

- Kimi via ACP (`kimi acp`)
- Auggie via ACP (`auggie --acp`)

## Acknowlegment
This project is a fork from the straightforward and clean implementation made
by [nulrouter](https://github.com/nulrouter/acp-router). His work save us hours
worth of work.

## Installation

Install the extension into the same Python environment where LiteLLM runs:

```bash
# Just in case... ;)
python -m pip install --upgrade pip setuptools wheel
# assumes `python` points to your current/default python installation
python -m pip install "litellm[proxy]"
# It must be absolute path, otherwise you might run into classpath issues.
python -m pip install -e /absolute/path/to/litellm-acp-router
```

This installs LiteLLM proxy dependencies and the Agent Client Protocol SDK.
You must still install and authenticate the underlying CLI agents you plan to
use, such as `kimi` or `auggie`.

## LiteLLM configuration

Create the file `local.py` on the directory you want store your configuration file.
```python
from litellm_acp_router.router import router_handler
__all__ = ["router_handler"]
```

Add the ACP provider to your LiteLLM config with `custom_provider_map`:

```yaml
litellm_settings:
  custom_provider_map:
    - provider: acp
      custom_handler: local.router_handler
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

## Stateful ACP sessions (opt-in)

By default the router runs each Chat Completions request through a fresh ACP
agent process. The router can also keep an ACP agent process alive across
requests so agents such as Auggie can reuse cached system/context state. This
mode is opt-in and best suited to single-process LiteLLM deployments.

Stateful mode is enabled per model alias by setting
`acp_session_binding_strategy`. The strategy decides how subsequent requests
are matched to the same long-lived ACP process.

### `prompt_hashing`

Derive a stable session key from `SHA256(system_prompt + first_user_message)`,
truncated to 16 hex characters. The first user turn of a conversation acts as
its identity, so clients that always replay the full chat history (the
OpenAI-style default) will keep landing on the same agent process.

```yaml
model_list:
  - model_name: acp-auggie-stateful
    litellm_params:
      model: acp/auggie
      acp_session_binding_strategy: prompt_hashing
      acp_session_ttl_seconds: 1800
      acp_max_sessions: 100
      acp_session_lock_timeout_seconds: 30
```

### `http_header/<NAME>`

Read the session key from an inbound HTTP header on the LiteLLM proxy request.
Use this when your client can send a stable conversation identifier, such as
`X-Hermes-Conversation-Id`.

```yaml
model_list:
  - model_name: acp-auggie-stateful-header
    litellm_params:
      model: acp/auggie
      acp_session_binding_strategy: http_header/X-Hermes-Conversation-Id
```

If the configured header is missing or empty, the router fails fast with a
`ValueError` that names the header and suggests resending the request with a
non-empty value.

### How it works

- On every stateful request the router resolves a binding key from the
  configured strategy and looks up an existing ACP session for that key,
  namespaced by adapter, model, optional `acp_model`, and resolved cwd.
- If no session is found, the router spawns a fresh ACP process, runs any
  bootstrap commands, and sends the full prompt built from `messages`.
- If a session is found, the router only sends the messages that appear after
  the last index it already streamed to that process. Reissuing the same
  payload twice raises a clear error rather than re-sending stale turns.

### Limits and safety

- Sessions are held in-memory only. They do not survive a LiteLLM restart, and
  multi-worker deployments will land on a new session when a request reaches a
  worker that does not own the session.
- Idle sessions are closed after `acp_session_ttl_seconds` (default `1800`).
- Total live sessions are capped at `acp_max_sessions` (default `100`) using
  least-recently-used eviction.
- A single request per session is in flight at a time;
  `acp_session_lock_timeout_seconds` (default `30`) bounds how long concurrent
  requests wait before raising `TimeoutError`.

## License

MIT
