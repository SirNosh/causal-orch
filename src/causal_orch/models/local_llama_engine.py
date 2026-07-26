"""Pinned loopback llama-server adapter for ARE's stock text interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request
from uuid import uuid4

from are.simulation.agents.llm.llm_engine import LLMEngine, LLMEngineException

from causal_orch.tracing.events import EventName, OrchestrationEvent

from .health import FailureClass, classify_http_failure
from .manifests import LocalLlamaConfig
from .openrouter_engine import (
    BackoffPolicy,
    HTTPRequest,
    HTTPResponse,
    MODEL_CALL_FAILURE_TYPES,
    Transport,
    UrllibTransport,
    _json_body,
    _normalize_trace_tags,
    _response_status,
    _safe_metadata,
)


class LocalLlamaProtocolError(LLMEngineException):
    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.protocol_violation = error_type
        self.request_id = request_id
        self.status_code = status_code


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_artifacts(config: LocalLlamaConfig, *, root: str | Path) -> None:
    model_path, server_path = config.resolved_paths(root)
    for label, path, expected in (
        ("GGUF", model_path, config.model_sha256),
        ("llama-server", server_path, config.server_binary_sha256),
    ):
        if not path.is_file():
            raise LocalLlamaProtocolError(
                f"{label} artifact is missing: {path}",
                error_type="PROVIDER_IDENTITY_UNVERIFIABLE",
            )
        if _sha256_file(path) != expected:
            raise LocalLlamaProtocolError(
                f"{label} artifact hash does not match the manifest",
                error_type="PROVIDER_IDENTITY_MISMATCH",
            )


class LocalLlamaLLMEngine(LLMEngine):
    """Call a verified local llama-server without changing ARE's interface."""

    def __init__(
        self,
        config: LocalLlamaConfig,
        trace_sink: Any | None = None,
        *,
        root: str | Path,
        transport: Transport | None = None,
        props_lookup: Callable[[], Any] | None = None,
        timeout_seconds: float = 300.0,
        backoff: BackoffPolicy | None = None,
        correlation_id_factory: Callable[[], str] | None = None,
        verify_artifacts: bool = True,
    ) -> None:
        super().__init__(config.model_slug)
        self.config = config
        self.trace_sink = trace_sink
        self.root = Path(root)
        self.transport = transport or UrllibTransport()
        self.props_lookup = props_lookup or self._default_props_lookup
        self.timeout_seconds = timeout_seconds
        self.backoff = backoff or BackoffPolicy(max_retries=1)
        self.correlation_id_factory = correlation_id_factory or (lambda: uuid4().hex)
        self._pending_request_ids: set[str] = set()
        self._server_verified = False
        self.native_exchanges: list[dict[str, Any]] = []
        if verify_artifacts:
            verify_local_artifacts(config, root=self.root)

    def _default_props_lookup(self) -> Any:
        request = urllib.request.Request(
            self.config.props_endpoint,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return HTTPResponse(response.status, response.read())
        except urllib.error.HTTPError as error:
            return HTTPResponse(error.code, error.read())

    def _verify_server(self) -> None:
        if self._server_verified:
            return
        try:
            response = self.props_lookup()
        except Exception as error:
            raise LocalLlamaProtocolError(
                "local llama-server is unavailable",
                error_type="TRANSPORT_ERROR",
            ) from error
        status = _response_status(response)
        if status != 200:
            raise LocalLlamaProtocolError(
                "local llama-server props endpoint failed",
                error_type="HTTP_OUTAGE",
                status_code=status,
            )
        try:
            body = _json_body(response)
        except Exception as error:
            raise LocalLlamaProtocolError(
                "local llama-server props response is invalid JSON",
                error_type="INVALID_JSON_BODY",
            ) from error
        if not isinstance(body, Mapping):
            raise LocalLlamaProtocolError(
                "local llama-server props response is not an object",
                error_type="INVALID_JSON_BODY",
            )
        alias = body.get("model_alias") or body.get("model")
        if alias is not None and alias != self.config.model_slug:
            raise LocalLlamaProtocolError(
                f"local llama-server model alias mismatch: {alias!r}",
                error_type="MODEL_IDENTITY_MISMATCH",
            )
        self._server_verified = True

    def _messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise LocalLlamaProtocolError(
                    "each message must be an object",
                    error_type="INVALID_TOOL_ARGUMENTS",
                )
            role = getattr(message.get("role"), "value", message.get("role"))
            content = message.get("content", "")
            normalized_role = {
                "tool-call": "assistant",
                "tool-response": "user",
            }.get(role, role)
            normalized = {"role": normalized_role, "content": content}
            for name in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
                if name in message:
                    normalized[name] = message[name]
            result.append(normalized)
        return result

    @staticmethod
    def _hash_json(value: Any) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _trace(self, event_type: EventName, **fields: Any) -> None:
        if self.trace_sink is not None:
            self.trace_sink.append(OrchestrationEvent(event_type=event_type, **fields))

    @staticmethod
    def _failure_type(error: Exception) -> str:
        value = getattr(error, "error_type", None)
        return value if value in MODEL_CALL_FAILURE_TYPES else "ENGINE_PROTOCOL_ERROR"

    @staticmethod
    def _text_response(
        message: Mapping[str, Any], request_id: str
    ) -> str:
        content = message.get("content")
        if not isinstance(content, str):
            raise LocalLlamaProtocolError(
                "completion does not contain text",
                error_type="NON_TEXT_COMPLETION",
                request_id=request_id,
            )
        if not content.strip():
            raise LocalLlamaProtocolError(
                "completion content is empty",
                error_type="EMPTY_COMPLETION",
                request_id=request_id,
            )
        return content

    def _send_completion(
        self,
        payload: Mapping[str, Any],
        correlation_id: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
        self._verify_server()
        request = HTTPRequest(
            self.config.endpoint,
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Request-ID": correlation_id,
            },
            payload,
            self.timeout_seconds,
        )
        response: Any | None = None
        status_code: int | None = None
        retry_count = 0
        for retry_index in range(self.backoff.max_retries + 1):
            retry_count = retry_index
            try:
                response = self.transport(request)
                status_code = _response_status(response)
            except Exception as error:
                if retry_index < self.backoff.max_retries:
                    self.backoff.wait(retry_index)
                    continue
                raise LocalLlamaProtocolError(
                    "local llama-server transport failed",
                    error_type="TRANSPORT_ERROR",
                ) from error
            if 200 <= status_code < 300:
                break
            classification = classify_http_failure(status_code)
            if (
                classification in {FailureClass.RATE_LIMIT, FailureClass.OUTAGE}
                and retry_index < self.backoff.max_retries
            ):
                self.backoff.wait(retry_index)
                continue
            error_type = {
                FailureClass.RATE_LIMIT: "HTTP_RATE_LIMIT",
                FailureClass.OUTAGE: "HTTP_OUTAGE",
            }.get(classification, "HTTP_NON_RETRIABLE")
            try:
                error_body = _json_body(response)
            except Exception:
                error_body = {}
            if (
                isinstance(error_body, Mapping)
                and "exceeds the available context size" in json.dumps(error_body)
            ):
                error_type = "CONTEXT_LENGTH_EXCEEDED"
            error_text = json.dumps(error_body)
            if (
                "Failed to initialize samplers" in error_text
                and "grammar" in error_text.lower()
            ):
                error_type = "LLAMA_CPP_GRAMMAR_SAMPLER_INIT_FAILURE"
            raise LocalLlamaProtocolError(
                "local llama-server request failed",
                error_type=error_type,
                status_code=status_code,
            )
        body = _json_body(response)
        if not isinstance(body, Mapping):
            raise LocalLlamaProtocolError(
                "local completion response is not a JSON object",
                error_type="INVALID_JSON_BODY",
            )
        returned_model = body.get("model")
        if returned_model != self.config.model_slug:
            raise LocalLlamaProtocolError(
                f"requested model {self.config.model_slug!r}, returned {returned_model!r}",
                error_type="MODEL_IDENTITY_MISMATCH",
            )
        choices = body.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], Mapping)
        ):
            raise LocalLlamaProtocolError(
                "local response contains no completion choice",
                error_type="NO_COMPLETION_CHOICE",
            )
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise LocalLlamaProtocolError(
                "local completion choice has no message",
                error_type="NON_TEXT_COMPLETION",
            )
        return body, message, retry_count

    def _response_metadata(
        self,
        body: Mapping[str, Any],
        *,
        started: float,
        correlation_id: str,
    ) -> dict[str, Any]:
        usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
        details = (
            usage.get("completion_tokens_details")
            if isinstance(usage.get("completion_tokens_details"), Mapping)
            else {}
        )
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "completion_duration": time.monotonic() - started,
            "request_id": str(body.get("id") or correlation_id),
            "returned_model": body.get("model"),
            "provider": self.config.provider,
        }

    def _trace_response(
        self,
        *,
        correlation_id: str,
        prompt_hash: str,
        tool_schema_hash: str | None,
        trace_tags: Mapping[str, Any],
        metadata: Mapping[str, Any],
        retry_count: int,
        response_kind: str,
        schema_strategy: str,
    ) -> None:
        self._trace(
            EventName.MODEL_RESPONSE,
            requested_model_slug=self.config.model_slug,
            returned_model_slug=metadata["returned_model"],
            provider_slug=self.config.provider,
            openrouter_request_id=correlation_id,
            prompt_hash=prompt_hash,
            tool_schema_hash=tool_schema_hash,
            prompt_tokens=metadata["prompt_tokens"],
            completion_tokens=metadata["completion_tokens"],
            reasoning_tokens=metadata["reasoning_tokens"],
            total_tokens=metadata["total_tokens"],
            wall_latency=metadata["completion_duration"],
            retry_count=retry_count,
            payload={
                "backend": "local_llama.cpp",
                "provider_identity_source": "verified_local_artifacts",
                "response_kind": response_kind,
                "request_id": metadata["request_id"],
                "schema_strategy": schema_strategy,
                "trace_tags": dict(trace_tags),
            },
        )

    def _trace_failure(self, correlation_id: str, error: Exception) -> None:
        if correlation_id not in self._pending_request_ids:
            return
        self._pending_request_ids.remove(correlation_id)
        self._trace(
            EventName.MODEL_CALL_FAILED,
            requested_model_slug=self.config.model_slug,
            provider_slug=self.config.provider,
            openrouter_request_id=correlation_id,
            error_type=self._failure_type(error),
            protocol_violation=getattr(error, "protocol_violation", None),
            payload={
                "backend": "local_llama.cpp",
                "status_code": getattr(error, "status_code", None),
                "engine_error_type": getattr(
                    error, "error_type", type(error).__name__
                ),
            },
        )

    @staticmethod
    def _normalize_native_tool_calls(
        message: Mapping[str, Any],
        *,
        request_id: str,
    ) -> list[dict[str, Any]]:
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            return []
        normalized = []
        for index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                raise LocalLlamaProtocolError(
                    "tool call must be an object",
                    error_type="INVALID_TOOL_ARGUMENTS",
                )
            function = call.get("function")
            if not isinstance(function, Mapping):
                function = call
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not name:
                raise LocalLlamaProtocolError(
                    "tool call has no function name",
                    error_type="UNKNOWN_TOOL",
                )
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise LocalLlamaProtocolError(
                        "tool call arguments are invalid JSON",
                        error_type="INVALID_TOOL_ARGUMENTS",
                    ) from error
            if not isinstance(arguments, Mapping):
                raise LocalLlamaProtocolError(
                    "tool call arguments must be an object",
                    error_type="INVALID_TOOL_ARGUMENTS",
                )
            normalized.append(
                {
                    "id": str(call.get("id") or f"call_{request_id}_{index}"),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": dict(arguments),
                    },
                }
            )
        return normalized

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        stop_sequences: list[str] | None = None,
        schema: Mapping[str, Any] | None = None,
        additional_trace_tags: Any = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        del stop_sequences, schema
        if kwargs:
            raise LocalLlamaProtocolError(
                f"local request fields cannot be overridden: {sorted(kwargs)}",
                error_type="ENGINE_PROTOCOL_ERROR",
            )
        correlation_id = self.correlation_id_factory()
        started = time.monotonic()
        trace_tags = _normalize_trace_tags(additional_trace_tags)
        safe_messages = _safe_metadata(self._messages(messages))
        prompt_hash = self._hash_json(safe_messages)
        self._trace(
            EventName.MODEL_REQUEST,
            requested_model_slug=self.config.model_slug,
            provider_slug=self.config.provider,
            openrouter_request_id=correlation_id,
            prompt_hash=prompt_hash,
            retry_count=0,
            payload={
                "backend": "local_llama.cpp",
                "schema_strategy": "stock_are_react_json",
                "trace_tags": dict(trace_tags),
            },
        )
        self._pending_request_ids.add(correlation_id)
        payload = {
            "model": self.config.model_slug,
            "messages": safe_messages,
            **self.config.sampling.to_dict(),
            "stream": False,
            "chat_template_kwargs": dict(self.config.reasoning),
        }
        try:
            body, message, retry_count = self._send_completion(
                payload, correlation_id
            )
            metadata = self._response_metadata(
                body, started=started, correlation_id=correlation_id
            )
            content = self._text_response(message, metadata["request_id"])
            self._trace_response(
                correlation_id=correlation_id,
                prompt_hash=prompt_hash,
                tool_schema_hash=None,
                trace_tags=trace_tags,
                metadata=metadata,
                retry_count=retry_count,
                response_kind="text",
                schema_strategy="stock_are_react_json",
            )
            self._pending_request_ids.remove(correlation_id)
            return content, metadata
        except Exception as error:
            self._trace_failure(correlation_id, error)
            raise

    def native_tool_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        max_tokens: int | None = None,
        additional_trace_tags: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if tool_choice == "required":
            interface_label = "NATIVE_TYPED_TOOL_INTERFACE_REQUIRED"
        elif isinstance(tool_choice, Mapping):
            function = tool_choice.get("function")
            name = function.get("name") if isinstance(function, Mapping) else None
            interface_label = (
                "NATIVE_TYPED_TOOL_INTERFACE_NAMED_FINALIZER"
                if name == "return_artifact"
                else "NATIVE_TYPED_TOOL_INTERFACE_NAMED"
            )
        else:
            interface_label = "NATIVE_TYPED_TOOL_INTERFACE_AUTO"
        correlation_id = self.correlation_id_factory()
        started = time.monotonic()
        trace_tags = _normalize_trace_tags(additional_trace_tags)
        safe_messages = _safe_metadata(self._messages(messages))
        safe_tools = _safe_metadata(tools)
        prompt_hash = self._hash_json(safe_messages)
        tool_schema_hash = self._hash_json(safe_tools)
        self._trace(
            EventName.MODEL_REQUEST,
            requested_model_slug=self.config.model_slug,
            provider_slug=self.config.provider,
            openrouter_request_id=correlation_id,
            prompt_hash=prompt_hash,
            tool_schema_hash=tool_schema_hash,
            retry_count=0,
            payload={
                "backend": "local_llama.cpp",
                "schema_strategy": interface_label,
                "trace_tags": dict(trace_tags),
            },
        )
        self._pending_request_ids.add(correlation_id)
        payload = {
            "model": self.config.model_slug,
            "messages": safe_messages,
            **self.config.sampling.to_dict(),
            "stream": False,
            "chat_template_kwargs": dict(self.config.reasoning),
            "tools": safe_tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
        }
        if max_tokens is not None:
            if type(max_tokens) is not int or max_tokens < 1:
                raise ValueError("max_tokens must be a positive integer")
            payload["max_tokens"] = min(
                max_tokens,
                self.config.sampling.max_tokens,
            )
        try:
            body, message, retry_count = self._send_completion(
                payload, correlation_id
            )
            metadata = self._response_metadata(
                body, started=started, correlation_id=correlation_id
            )
            tool_calls = self._normalize_native_tool_calls(
                message, request_id=metadata["request_id"]
            )
            normalized_message = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            }
            if "reasoning_content" in message:
                normalized_message["reasoning_content"] = message[
                    "reasoning_content"
                ]
            self._trace_response(
                correlation_id=correlation_id,
                prompt_hash=prompt_hash,
                tool_schema_hash=tool_schema_hash,
                trace_tags=trace_tags,
                metadata=metadata,
                retry_count=retry_count,
                response_kind="tool_call" if tool_calls else "text",
                schema_strategy=interface_label,
            )
            self._pending_request_ids.remove(correlation_id)
            exchange_index = len(self.native_exchanges)
            self.native_exchanges.append(
                {
                    "interface": interface_label,
                    "tool_schema": safe_tools,
                    "tool_schema_sha256": tool_schema_hash,
                    "outgoing_request": payload,
                    "raw_response": body,
                    "normalized_tool_calls": tool_calls,
                    "validator_input": None,
                    "validator_result": None,
                    "completion_tokens": metadata["completion_tokens"],
                    "reasoning_tokens": metadata["reasoning_tokens"],
                    "terminal_event": "MODEL_RESPONSE",
                }
            )
            metadata["native_exchange_index"] = exchange_index
            return normalized_message, metadata
        except Exception as error:
            self._trace_failure(correlation_id, error)
            raise

    def simple_call(self, prompt: str) -> str:
        response, _metadata = self.chat_completion(
            [{"role": "user", "content": prompt}]
        )
        return response
