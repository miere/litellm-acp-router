<div align="center">

# 🛰️ LiteLLM ACP Router

**Bring your local Agent Client Protocol (ACP) CLI agents to any OpenAI-compatible client.**

A LiteLLM custom provider that turns ACP-capable agents like **Kimi** and **Auggie**
into first-class chat completion models — streaming, reasoning, tool narration,
and stateful sessions included.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-custom%20provider-7c3aed.svg)](https://docs.litellm.ai/)
[![ACP](https://img.shields.io/badge/protocol-Agent%20Client%20Protocol-1f8a70.svg)](https://github.com/zed-industries/agent-client-protocol)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## ✨ Why this exists

ACP agents are powerful local CLIs, but they don't speak the OpenAI Chat
Completions dialect that the rest of the ecosystem expects. **LiteLLM ACP
Router** bridges that gap: point any OpenAI client at your LiteLLM proxy, and
get a real ACP agent on the other end — with streaming text, reasoning
channels, tool activity, and optional long-lived sessions.

### Supported agents

| Adapter  | Launch command  | Model selection | Workspace flag       |
|----------|-----------------|-----------------|----------------------|
| `kimi`   | `kimi acp`      | —               | —                    |
| `auggie` | `auggie --acp`  | `--model`       | `--workspace-root`   |

> Adding a new ACP agent is a few-line subclass of `StaticAdapter`. See
> [`AGENTS.md`](AGENTS.md) for the recipe.

---

## 🙏 Acknowledgments

This project is a fork of the clean, no-nonsense implementation by
[**nulrouter/acp-router**](https://github.com/nulrouter/acp-router). Their
groundwork saved us many hours — thank you. 💚

---

## 🚀 Quickstart

### 1. Install

Install into the same Python environment where LiteLLM runs:

```bash
# Optional but recommended
python -m pip install --upgrade pip setuptools wheel

# LiteLLM with proxy extras
python -m pip install "litellm[proxy]"

# This package — use an ABSOLUTE path to avoid classpath surprises
python -m pip install -e /absolute/path/to/litellm-acp-router
```

> 📌 You still need to install and authenticate the underlying CLI agents
> (`kimi`, `auggie`, …) on your machine.

### 2. Wire it into LiteLLM

Create a `local.py` next to your LiteLLM config:

```python
from litellm_acp_router.router import router_handler

__all__ = ["router_handler"]
```

Register the `acp` provider and expose one or more model aliases:

```yaml
litellm_settings:
  custom_provider_map:
    - provider: acp
      custom_handler: local.router_handler

model_list:
  - model_name: acp-kimi
    litellm_params:
      model: acp/kimi

  - model_name: acp-auggie
    litellm_params:
      model: acp/auggie
```

> 💡 A complete minimal config lives in
> [`litellm_config.example.yaml`](litellm_config.example.yaml).

### 3. Run the proxy

```bash
litellm --config /path/to/litellm.yaml
```

This package does **not** ship a server launcher. LiteLLM owns the proxy
process; we just plug in as a custom provider.

### 4. Send a request

```bash
curl -X POST http://127.0.0.1:4000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "acp-auggie",
    "messages": [
      { "role": "user", "content": "Explain this repository" }
    ]
  }'
```

That's it. You're now talking to a local ACP agent through the OpenAI API. 🎉

---

## 🎛️ Configuring Auggie

The Auggie adapter exposes two generic, forward-compatible knobs: `acp_model`
and `acp_workspace_dir`. Other adapters can adopt the same keys as they grow.

### Model selection — `acp_model`

By default Auggie picks its **own locally configured default model**, so most
users can simply omit `acp_model`:

```yaml
model_list:
  - model_name: acp-auggie
    litellm_params:
      model: acp/auggie
```

Pin a specific model per alias when you need it:

```yaml
model_list:
  - model_name: acp-auggie-gpt55
    litellm_params:
      model: acp/auggie
      acp_model: gpt-5.5    # → auggie --acp --model gpt-5.5
```

> 🔎 Run `auggie models list` (or `--json`) to discover the exact model IDs
> available to your account.

### Workspace directory — `acp_workspace_dir`

Auggie indexes a workspace passed via `--workspace-root`. The router defaults
to `/tmp/auggie-empty`; override it per alias:

```yaml
model_list:
  - model_name: acp-auggie-myproject
    litellm_params:
      model: acp/auggie
      acp_workspace_dir: /absolute/path/to/project
      # → auggie --acp --allow-indexing --workspace-root /absolute/path/to/project
```

| Resolution order      | Source                                              |
|-----------------------|-----------------------------------------------------|
| 1. Per-alias param    | `acp_workspace_dir` in `litellm_params`             |
| 2. Environment        | `AUGGIE_WORKSPACE_DIR`                              |
| 3. Adapter default    | `/tmp/auggie-empty`                                 |

> Both `acp_model` and `acp_workspace_dir` are deliberately generic so future
> adapters can adopt them without inventing new keys.

---

## 📡 Streaming behavior

### Tool activity narration

ACP agents like Auggie and Kimi execute tools **internally** (closed-loop) and
never surface OpenAI-style `tool_calls` for the host to run. To keep callers in
the loop, the router translates ACP `tool_call` and `tool_call_update` events
into inline assistant text chunks:

| Event                          | Emitted text             |
|--------------------------------|--------------------------|
| `tool_call` (start, with kind) | `> [kind] title`         |
| `tool_call` (start, no kind)   | `> title`                |
| `tool_call_update` → `completed` | `> title — done`       |
| `tool_call_update` → `failed`  | `> title — failed`       |
| `tool_call_update` → other     | *(silent)*               |

The narration is plain text on `delta.content` — the OpenAI `tool_calls` field
is **not** populated, so no client-side execution round-trip is triggered.

Narration is **on by default**. Disable per alias when you want clean prose:

```yaml
model_list:
  - model_name: acp-auggie
    litellm_params:
      model: acp/auggie
      acp_emit_tool_activity: false
```

### Reasoning channel

ACP agents emit reasoning as `agent_thought_chunk` frames, separate from the
final `agent_message_chunk` answer. The router forwards reasoning to LiteLLM
via `provider_specific_fields["reasoning_content"]`, which lands on the
OpenAI delta's `reasoning_content` field. Reasoning-aware clients render it as
a distinct channel; assistant prose keeps flowing through `delta.content`.

> 🧠 Reasoning is intentionally **excluded** from the assistant transcript used
> by the non-streaming `acompletion` path and by stateful session resume — it
> won't pollute downstream context.

If a client only renders `reasoning_content` on completion instead of
progressively, that's a client-side concern; the router yields each chunk as
soon as the agent emits it.

---

## 🧰 Advanced tuning

<details>
<summary><b><code>acp_stdio_buffer_bytes</code> — handling large JSON-RPC frames</b></summary>

<br>

The router reads JSON-RPC frames from the agent's stdout via
`asyncio.StreamReader`, whose default buffer is **64 KiB**. Single frames
carrying file diffs, terminal output, or large MCP responses can overrun that
ceiling and crash the receive loop with `LimitOverrunError`, surfaced to the
caller as `ConnectionError: Connection closed`.

To prevent that, the router passes `transport_kwargs={"limit": N}` to
`spawn_agent_process` for both stateless and stateful spawns. The default is
**8 MiB**. Raise it per alias when an agent legitimately emits larger frames:

```yaml
model_list:
  - model_name: acp-auggie
    litellm_params:
      model: acp/auggie
      acp_stdio_buffer_bytes: 16777216  # 16 MiB
```

Non-positive or non-numeric values fall back to the 8 MiB default.

</details>

---

## 🔁 Stateful ACP sessions *(opt-in)*

By default, every Chat Completions request spawns a **fresh** ACP agent
process. For agents like Auggie — which can reuse cached system/context state
across turns — the router can also keep a process **alive across requests**.

> ⚠️ Stateful mode is best suited to **single-process LiteLLM deployments**.
> Sessions live in memory and don't survive restarts; multi-worker setups may
> land on a fresh session when a request reaches a worker that doesn't own it.

Enable it per alias by setting `acp_session_binding_strategy`. The strategy
decides how subsequent requests are matched to the same long-lived process.

### Strategy 1 — `prompt_hashing`

Derive a stable session key from `SHA256(system_prompt + first_user_message)`,
truncated to 16 hex characters. The first user turn becomes the conversation's
identity, so clients that replay the full chat history (the OpenAI default)
keep landing on the same agent process.

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

### Strategy 2 — `http_header/<NAME>`

Read the session key from an inbound HTTP header on the LiteLLM proxy request.
Use this when your client can send a stable conversation identifier such as
`X-Hermes-Conversation-Id`.

```yaml
model_list:
  - model_name: acp-auggie-stateful-header
    litellm_params:
      model: acp/auggie
      acp_session_binding_strategy: http_header/X-Hermes-Conversation-Id
```

If the configured header is missing or empty, the router **fails fast** with a
`ValueError` naming the header and suggesting a retry with a non-empty value.

### How it works

1. The router resolves a binding key from the configured strategy and looks up
   an existing ACP session, namespaced by adapter, model, optional `acp_model`,
   and resolved cwd.
2. **Cache miss** → spawn a fresh ACP process, run any bootstrap commands, and
   send the full prompt built from `messages`.
3. **Cache hit** → send only the messages **after** the last index already
   streamed to that process. Reissuing the same payload twice raises a clear
   error rather than re-sending stale turns.

### Limits and safety

| Setting                              | Default | Purpose                                                                 |
|--------------------------------------|---------|-------------------------------------------------------------------------|
| `acp_session_ttl_seconds`            | `1800`  | Close idle sessions after this many seconds                             |
| `acp_max_sessions`                   | `100`   | Cap on live sessions; LRU eviction beyond the cap                       |
| `acp_session_lock_timeout_seconds`   | `30`    | Max wait for the per-session lock before raising `TimeoutError`         |

Only one request per session is in flight at a time — concurrent requests on
the same key queue up to the lock timeout.

---

## 🤝 Contributing

Contributions are welcome — new ACP adapters especially. The internals are
documented in [`AGENTS.md`](AGENTS.md), including:

- the request flow (stateless and stateful),
- how to add an adapter in a few lines via `StaticAdapter`,
- configuration conventions, and
- the test commands we expect to pass before merging.

Quick validation loop:

```bash
python3 -m compileall litellm_acp_router tests
python3 -m unittest discover -s tests
```

---

## 📄 License

Released under the [MIT License](LICENSE).
