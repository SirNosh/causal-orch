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
            result.append({"role": normalized_role, "content": content})
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
        tool_schema_hash = None
        prompt_hash = self._hash_json(safe_messages)
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
                "schema_strategy": "stock_are_react_json",
                "trace_tags": dict(trace_tags),
            },
        )
        self._pending_request_ids.add(correlation_id)
        try:
            self._verify_server()
            payload: dict[str, Any] = {
                "model": self.config.model_slug,
                "messages": safe_messages,
                **self.config.sampling.to_dict(),
                "stream": False,
                "chat_template_kwargs": dict(self.config.reasoning),
            }
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
                    and "exceeds the available context size"
                    in json.dumps(error_body)
                ):
                    error_type = "CONTEXT_LENGTH_EXCEEDED"
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
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
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
            request_id = str(body.get("id") or correlation_id)
            content = self._text_response(message, request_id)
            usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
            details = (
                usage.get("completion_tokens_details")
                if isinstance(usage.get("completion_tokens_details"), Mapping)
                else {}
            )
            metadata = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "reasoning_tokens": details.get("reasoning_tokens"),
                "completion_duration": time.monotonic() - started,
                "request_id": request_id,
                "returned_model": returned_model,
                "provider": self.config.provider,
            }
            self._trace(
                EventName.MODEL_RESPONSE,
                requested_model_slug=self.config.model_slug,
                returned_model_slug=returned_model,
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
                    "response_kind": "text",
                    "request_id": request_id,
                    "trace_tags": dict(trace_tags),
                },
            )
            self._pending_request_ids.remove(correlation_id)
            return content, metadata
        except Exception as error:
            if correlation_id in self._pending_request_ids:
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
            raise

    def simple_call(self, prompt: str) -> str:
        response, _metadata = self.chat_completion(
            [{"role": "user", "content": prompt}]
        )
        return response
