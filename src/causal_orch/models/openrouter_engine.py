"""A deterministic, provider-pinned OpenRouter adapter for pinned ARE."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import random
import time
from typing import Any, Callable, Mapping, Protocol
import urllib.error
import urllib.request
from uuid import uuid4

from are.simulation.agents.llm.llm_engine import LLMEngine, LLMEngineException

from .health import FailureClass, classify_http_failure
from .manifests import OpenRouterConfig


_FORBIDDEN_METADATA_KEYS = {"thought", "chain_of_thought", "hidden_reasoning", "private_reasoning"}


class OpenRouterProtocolError(LLMEngineException):
    """Deterministic protocol failure, suitable for trace classification."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        classification: str = "PROTOCOL_ERROR",
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.classification = classification
        self.protocol_violation = error_type
        self.request_id = request_id
        self.status_code = status_code

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "classification": self.classification,
            "request_id": self.request_id,
            "status_code": self.status_code,
            "protocol_violation": self.protocol_violation,
        }


@dataclass(frozen=True)
class HTTPRequest:
    url: str
    headers: Mapping[str, str]
    json_body: Mapping[str, Any]
    timeout: float


@dataclass(frozen=True)
class HTTPResponse:
    status_code: int
    body: Any
    headers: Mapping[str, str] = field(default_factory=dict)


class Transport(Protocol):
    def __call__(self, request: HTTPRequest) -> HTTPResponse:
        ...


class UrllibTransport:
    """Default transport. Constructing/importing it does not make a request."""

    def __call__(self, request: HTTPRequest) -> HTTPResponse:
        encoded = json.dumps(request.json_body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            request.url,
            data=encoded,
            headers=dict(request.headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=request.timeout) as response:
                return HTTPResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except urllib.error.HTTPError as error:
            return HTTPResponse(
                status_code=error.code,
                headers=dict(error.headers.items()) if error.headers else {},
                body=error.read(),
            )


def compute_backoff_delay(
    retry_index: int,
    *,
    base_delay_seconds: float = 1.0,
    max_delay_seconds: float = 8.0,
    jitter_ratio: float = 0.25,
    random_value: float = 0.0,
) -> float:
    """Pure bounded exponential delay with positive, deterministic injectable jitter."""

    if retry_index < 0 or base_delay_seconds < 0 or max_delay_seconds < 0:
        raise ValueError("backoff values must be non-negative")
    if base_delay_seconds > max_delay_seconds:
        raise ValueError("base delay cannot exceed max delay")
    if not 0 <= jitter_ratio <= 1 or not 0 <= random_value <= 1:
        raise ValueError("jitter values must be between zero and one")
    exponential = min(max_delay_seconds, base_delay_seconds * (2**retry_index))
    return min(max_delay_seconds, exponential * (1 + jitter_ratio * random_value))


@dataclass(frozen=True)
class BackoffPolicy:
    max_retries: int = 2
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0
    jitter_ratio: float = 0.25
    random_value: Callable[[], float] = random.random
    sleeper: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    def delay_for_retry(self, retry_index: int) -> float:
        return compute_backoff_delay(
            retry_index,
            base_delay_seconds=self.base_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            jitter_ratio=self.jitter_ratio,
            random_value=self.random_value(),
        )

    def wait(self, retry_index: int) -> float:
        delay = self.delay_for_retry(retry_index)
        self.sleeper(delay)
        return delay


def _json_body(response: Any) -> Any:
    if isinstance(response, HTTPResponse):
        body = response.body
    elif isinstance(response, Mapping):
        return response
    elif hasattr(response, "json"):
        return response.json()
    else:
        body = getattr(response, "body", getattr(response, "text", response))
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    if isinstance(body, str):
        return json.loads(body)
    return body


def _response_status(response: Any) -> int:
    if isinstance(response, HTTPResponse):
        return response.status_code
    if isinstance(response, Mapping):
        return int(response.get("status_code", 200))
    return int(getattr(response, "status_code", 200))


def _response_headers(response: Any) -> Mapping[str, str]:
    if isinstance(response, HTTPResponse):
        return response.headers
    return getattr(response, "headers", {}) or {}


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_METADATA_KEYS:
                raise OpenRouterProtocolError(
                    "provider metadata contains hidden reasoning",
                    error_type="HIDDEN_REASONING_METADATA",
                )
            result[str(key)] = _safe_metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise OpenRouterProtocolError(
        "provider metadata is not JSON-compatible",
        error_type="INVALID_PROVIDER_METADATA",
    )


class OpenRouterLLMEngine(LLMEngine):
    """ARE-compatible engine with exact model/provider identity enforcement."""

    def __init__(
        self,
        config: OpenRouterConfig,
        trace_sink: Any | None = None,
        *,
        transport: Transport | None = None,
        timeout_seconds: float = 60.0,
        backoff: BackoffPolicy | None = None,
        correlation_id_factory: Callable[[], str] | None = None,
    ) -> None:
        super().__init__(config.model_slug)
        self.config = config
        self.trace_sink = trace_sink
        self.transport = transport or UrllibTransport()
        self.timeout_seconds = timeout_seconds
        self.backoff = backoff or BackoffPolicy()
        self.correlation_id_factory = correlation_id_factory or (lambda: uuid4().hex)

    def _payload(self, messages: list[dict[str, Any]], stop_sequences: list[str] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model_slug,
            "messages": messages,
            **self.config.sampling.to_dict(),
            "provider": {
                "only": [self.config.provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        if self.config.reasoning is not None:
            payload["reasoning"] = dict(self.config.reasoning)
        if stop_sequences:
            payload["stop"] = stop_sequences
        return payload

    def _send(self, request: HTTPRequest) -> Any:
        if callable(self.transport):
            return self.transport(request)
        raise TypeError("transport must be callable with HTTPRequest")

    def _raise_provider_failure(
        self, classification: FailureClass, request_id: str, status_code: int | None, detail: Any
    ) -> None:
        raise OpenRouterProtocolError(
            f"OpenRouter {classification.value.lower()} after bounded retries",
            error_type=classification.value,
            classification=classification.value,
            request_id=request_id,
            status_code=status_code,
        ) from (detail if isinstance(detail, Exception) else None)

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        stop_sequences: list[str] | None = None,
        schema: Mapping[str, Any] | None = None,
        additional_trace_tags: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        del schema, additional_trace_tags
        forbidden_overrides = {"model", "provider", "temperature", "top_p", "max_tokens", "reasoning"}
        attempted = forbidden_overrides.intersection(kwargs)
        if attempted:
            raise OpenRouterProtocolError(
                f"fixed request fields cannot be overridden: {sorted(attempted)}",
                error_type="FIXED_CONFIGURATION_OVERRIDE",
            )
        correlation_id = self.correlation_id_factory()
        started = time.monotonic()
        payload = self._payload(messages, stop_sequences)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Request-ID": correlation_id,
        }
        if self.config.api_key is not None:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.attribution_headers)
        request = HTTPRequest(self.config.endpoint, headers, payload, self.timeout_seconds)

        response: Any | None = None
        status_code: int | None = None
        for retry_index in range(self.backoff.max_retries + 1):
            try:
                response = self._send(request)
                status_code = _response_status(response)
            except Exception as error:
                if retry_index < self.backoff.max_retries:
                    self.backoff.wait(retry_index)
                    continue
                self._raise_provider_failure(
                    FailureClass.TRANSPORT_ERROR, correlation_id, None, error
                )
            assert response is not None
            if 200 <= status_code < 300:
                break
            classification = classify_http_failure(status_code)
            if classification in {FailureClass.RATE_LIMIT, FailureClass.OUTAGE} and retry_index < self.backoff.max_retries:
                self.backoff.wait(retry_index)
                continue
            self._raise_provider_failure(classification, correlation_id, status_code, None)
        else:  # pragma: no cover - range always has a final attempt
            raise AssertionError("unreachable")

        body = _json_body(response)
        if not isinstance(body, Mapping):
            raise OpenRouterProtocolError("response body is not a JSON object", error_type="INVALID_RESPONSE")
        returned_model = body.get("model")
        if returned_model != self.config.model_slug:
            raise OpenRouterProtocolError(
                f"requested model {self.config.model_slug!r}, returned {returned_model!r}",
                error_type="MODEL_IDENTITY_MISMATCH",
                request_id=body.get("id", correlation_id),
            )
        provider = body.get("provider") or body.get("provider_name")
        if isinstance(provider, Mapping):
            provider = provider.get("name") or provider.get("slug")
        if provider is not None and provider != self.config.provider:
            raise OpenRouterProtocolError(
                f"requested provider {self.config.provider!r}, returned {provider!r}",
                error_type="PROVIDER_IDENTITY_MISMATCH",
                request_id=body.get("id", correlation_id),
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise OpenRouterProtocolError("response contains no completion choice", error_type="INVALID_RESPONSE")
        message = choices[0].get("message", {})
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, Mapping)
            )
        if not isinstance(content, str):
            raise OpenRouterProtocolError("completion content is not text", error_type="INVALID_RESPONSE")
        for stop_token in stop_sequences or ():
            content = content.split(stop_token)[0]

        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        response_headers = _response_headers(response)
        request_id = body.get("id") or response_headers.get("x-request-id") or correlation_id
        metadata = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens", usage.get("reasoning_tokens")),
            "completion_duration": time.monotonic() - started,
            "request_id": request_id,
            "returned_model": returned_model,
            "provider": provider,
            "trace_metadata": {
                "request_correlation_id": correlation_id,
                "provider_metadata": _safe_metadata(body.get("provider_metadata", body.get("metadata", {}))),
            },
        }
        return content, metadata

    def simple_call(self, prompt: str) -> str:
        response, _metadata = self.chat_completion([{"role": "user", "content": prompt}])
        return response
