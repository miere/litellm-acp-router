import uuid
from typing import AsyncIterator, List

from litellm import CustomLLM
from litellm.types.utils import (
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)

from .registry import Registry
from .runtime import Runtime
from .utils import messages_to_prompt, normalize_incoming_messages


class RouterHandler(CustomLLM):
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.runtime = Runtime()

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        model = kwargs.get("model", "acp/kimi")
        optional_params = kwargs.get("optional_params", {}) or {}
        messages = normalize_incoming_messages(kwargs)
        tools = kwargs.get("tools") or optional_params.get("tools")

        adapter = self.registry.resolve(model=model, optional_params=optional_params)
        spec = adapter.build_spec(optional_params=optional_params)

        if optional_params.get("acp_session_binding_strategy"):
            async for chunk in self.runtime.run_stateful_stream(
                spec=spec,
                model=str(model),
                kwargs=kwargs,
                messages=messages,
                tools=tools,
            ):
                yield chunk
            return

        prompt_text = messages_to_prompt(messages, tools=tools) or "User: Hello"

        async for chunk in self.runtime.run_stream(
            spec=spec,
            prompt_text=prompt_text,
            kwargs=kwargs,
            messages=messages,
        ):
            yield chunk

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        model = kwargs.get("model", "acp/kimi")

        parts: List[str] = []
        async for chunk in self.astreaming(*args, **kwargs):
            text = chunk.get("text")
            if text:
                parts.append(text)

        output_text = "".join(parts).strip()
        if not output_text:
            output_text = "No final assistant text captured from router."

        return ModelResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            model=model,
            choices=[
                Choices(
                    index=0,
                    finish_reason="stop",
                    message=Message(role="assistant", content=output_text),
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )

    def completion(self, *args, **kwargs):
        raise RuntimeError("Use the async LiteLLM proxy endpoint.")

    def streaming(self, *args, **kwargs):
        raise RuntimeError("Use the async LiteLLM proxy endpoint.")
