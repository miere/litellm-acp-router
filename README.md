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

## License

MIT
