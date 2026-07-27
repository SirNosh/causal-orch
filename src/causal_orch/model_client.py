"""Small OpenAI-compatible client for native function calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib import error, request


class StructuredActionError(RuntimeError):
    pass


class ModelHTTPError(RuntimeError):
    def __init__(self, status: int, raw_response: Any) -> None:
        self.status = status
        self.raw_response = raw_response
        super().__init__(f"model endpoint returned HTTP {status}")


@dataclass(frozen=True)
class ModelTurn:
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str
    assistant_message: dict[str, Any]
    raw_response: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_seconds: float


class ModelClient(Protocol):
    model: str

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ModelTurn: ...


class OpenAICompatibleClient:
    """One endpoint, one model, and no routing or marketplace policy."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "temperature": 0,
        }
        encoded = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        started = time.perf_counter()
        try:
            with request.urlopen(
                request.Request(
                    f"{self.base_url}/chat/completions",
                    data=encoded,
                    headers=headers,
                    method="POST",
                ),
                timeout=self.timeout_seconds,
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                raw_error = json.loads(body)
            except json.JSONDecodeError:
                raw_error = body
            raise ModelHTTPError(exc.code, raw_error) from exc
        latency = time.perf_counter() - started
        try:
            message = raw["choices"][0]["message"]
            calls = message["tool_calls"]
            if len(calls) != 1:
                raise StructuredActionError(
                    f"expected exactly one tool call, received {len(calls)}"
                )
            call = calls[0]
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError("function arguments are not an object")
            tool_name = function["name"]
            call_id = call["id"]
            if not isinstance(tool_name, str) or not isinstance(call_id, str):
                raise TypeError("tool name and call id must be strings")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise StructuredActionError(
                "provider response did not contain one valid native function call"
            ) from exc
        usage = raw.get("usage") or {}
        return ModelTurn(
            tool_name=tool_name,
            arguments=arguments,
            tool_call_id=call_id,
            assistant_message={
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": calls,
            },
            raw_response=raw,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_seconds=latency,
        )


class ScriptedModelClient:
    """Deterministic native-call client used by harness tests."""

    model = "scripted"

    def __init__(self, calls: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
        self._calls = list(calls)
        self.call_count = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        if not self._calls:
            raise AssertionError("scripted model exhausted")
        name, arguments = self._calls.pop(0)
        call_id = f"call-{self.call_count}"
        self.call_count += 1
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(dict(arguments)),
                    },
                }
            ],
        }
        return ModelTurn(
            tool_name=name,
            arguments=dict(arguments),
            tool_call_id=call_id,
            assistant_message=assistant,
            raw_response={"choices": [{"message": assistant}], "scripted": True},
            input_tokens=0,
            output_tokens=0,
            latency_seconds=0.0,
        )
