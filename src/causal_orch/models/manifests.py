"""Locked model, provider, sampling, and manifest types for the protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


MODEL_CANDIDATE_ORDER = (
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
)
FIXED_TEMPERATURE = 0.2
FIXED_TOP_P = 0.9
FIXED_MAX_TOKENS = 4096
FIXED_GENERATION_SECONDS = 5.0
OPENROUTER_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
LOCAL_MODEL_SLUG = "Abiray/Nanbeige4.2-3B-GGUF:Q8_0"
LOCAL_PROVIDER = "Nanbeige/llama.cpp"
LOCAL_COMPLETIONS_URL = "http://127.0.0.1:8080/v1/chat/completions"


class ManifestError(ValueError):
    """Raised when a model/provider manifest violates the locked protocol."""


def _zero_price(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return Decimal(str(value).strip().removeprefix("$")) == 0
    except (InvalidOperation, ValueError):
        return False


def validate_model_slug(model_slug: str) -> str:
    if model_slug not in MODEL_CANDIDATE_ORDER or not model_slug.endswith(":free"):
        raise ManifestError(f"model slug is not a locked zero-price candidate: {model_slug!r}")
    return model_slug


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class SamplingConfig:
    """The sole sampling configuration permitted by the protocol."""

    temperature: float = FIXED_TEMPERATURE
    top_p: float = FIXED_TOP_P
    max_tokens: int = FIXED_MAX_TOKENS

    def __post_init__(self) -> None:
        if self.temperature != FIXED_TEMPERATURE:
            raise ManifestError("temperature is fixed at 0.2")
        if self.top_p != FIXED_TOP_P:
            raise ManifestError("top_p is fixed at 0.9")
        if self.max_tokens != FIXED_MAX_TOKENS:
            raise ManifestError("max_tokens is fixed at 4096")

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class TimeConfig:
    """Primary ARE simulated-generation-time policy."""

    mode: str = "fixed"
    orchestrator_generation_seconds: float = FIXED_GENERATION_SECONDS
    worker_generation_seconds: float = FIXED_GENERATION_SECONDS

    def __post_init__(self) -> None:
        if self.mode != "fixed":
            raise ManifestError("generation time mode is fixed")
        if self.orchestrator_generation_seconds != FIXED_GENERATION_SECONDS:
            raise ManifestError("orchestrator generation time is fixed at 5 seconds")
        if self.worker_generation_seconds != FIXED_GENERATION_SECONDS:
            raise ManifestError("worker generation time is fixed at 5 seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "orchestrator_generation_seconds": self.orchestrator_generation_seconds,
            "worker_generation_seconds": self.worker_generation_seconds,
        }


@dataclass(frozen=True)
class OpenRouterConfig:
    """Explicit engine configuration; it never reads credentials or files."""

    model_slug: str
    provider: str
    routing_provider_slug: str | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    reasoning: Mapping[str, Any] | None = None
    endpoint: str = OPENROUTER_COMPLETIONS_URL
    api_key: str | None = None
    attribution_headers: Mapping[str, str] = field(default_factory=dict)
    allow_fallbacks: bool = False
    require_parameters: bool = True
    data_collection: str = "allow"

    def __post_init__(self) -> None:
        validate_model_slug(self.model_slug)
        if not self.provider or not isinstance(self.provider, str):
            raise ManifestError("provider must be a non-empty pinned provider identity")
        if self.routing_provider_slug is not None and (
            not isinstance(self.routing_provider_slug, str) or not self.routing_provider_slug
        ):
            raise ManifestError("routing_provider_slug must be a non-empty string")
        if self.allow_fallbacks is not False:
            raise ManifestError("provider fallbacks are disabled by protocol")
        if self.require_parameters is not True:
            raise ManifestError("provider parameter requirement is fixed to true")
        if self.data_collection not in {"allow", "deny"}:
            raise ManifestError("data_collection must be 'allow' or 'deny'")
        if not self.endpoint:
            raise ManifestError("endpoint must be non-empty")
        for key, value in self.attribution_headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ManifestError("attribution headers must be string-to-string")

    @property
    def pinned_provider(self) -> str:
        return self.provider

    @property
    def provider_route(self) -> str:
        return self.routing_provider_slug or self.provider


@dataclass(frozen=True)
class LocalSamplingConfig:
    """Nanbeige's frozen tool-use sampling policy."""

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = FIXED_MAX_TOKENS

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class LocalLlamaConfig:
    """One pinned loopback llama-server and its locally verified artifacts."""

    model_slug: str = LOCAL_MODEL_SLUG
    provider: str = LOCAL_PROVIDER
    model_path: str = ""
    model_sha256: str = ""
    server_binary_path: str = ""
    server_binary_sha256: str = ""
    sampling: LocalSamplingConfig = field(default_factory=LocalSamplingConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    endpoint: str = LOCAL_COMPLETIONS_URL
    context_length: int = 32768
    reasoning: Mapping[str, Any] = field(
        default_factory=lambda: {
            "enable_thinking": True,
            "preserve_thinking": True,
            "tool_call_format": "xml",
        }
    )
    allow_fallbacks: bool = False
    require_parameters: bool = True
    data_collection: str = "deny"

    def __post_init__(self) -> None:
        if not isinstance(self.model_slug, str) or not self.model_slug:
            raise ManifestError("local model slug is required")
        if not isinstance(self.provider, str) or not self.provider:
            raise ManifestError("local provider identity is required")
        if not self.model_path or not self.server_binary_path:
            raise ManifestError("local model and server binary paths are required")
        for label, value in (
            ("model_sha256", self.model_sha256),
            ("server_binary_sha256", self.server_binary_sha256),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ManifestError(f"{label} must be a lowercase SHA-256")
        if self.context_length != 32768:
            raise ManifestError("local context length is fixed at 32768")
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ManifestError("local endpoint must be an HTTP loopback URL")
        if self.allow_fallbacks or not self.require_parameters:
            raise ManifestError("local provider fallback and parameter relaxation are disabled")
        if self.data_collection != "deny":
            raise ManifestError("local inference must not collect provider data")

    @property
    def provider_route(self) -> str:
        return self.provider

    @property
    def props_endpoint(self) -> str:
        return self.endpoint.removesuffix("/v1/chat/completions") + "/props"

    def resolved_paths(self, root: str | Path) -> tuple[Path, Path]:
        base = Path(root)
        return base / self.model_path, base / self.server_binary_path


@dataclass(frozen=True)
class ProviderManifest:
    provider_slug: str
    context_length: int
    max_output: int
    data_policy: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_slug": self.provider_slug,
            "context_length": self.context_length,
            "max_output": self.max_output,
            "data_policy": self.data_policy,
        }


@dataclass(frozen=True)
class ModelManifest:
    """Normalized, hashable snapshot of one candidate's free endpoint."""

    snapshot_timestamp_utc: str
    requested_model_slug: str
    returned_canonical_slug: str
    context_length: int
    input_price: Any
    output_price: Any
    supported_parameters: tuple[str, ...]
    available_providers: tuple[str, ...]
    selected_provider: str
    provider_context_length: int
    provider_max_output: int
    provider_data_policy: Any
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        validate_model_slug(self.requested_model_slug)
        if not isinstance(self.returned_canonical_slug, str) or not self.returned_canonical_slug:
            raise ManifestError("returned canonical model slug is required")
        if not _zero_price(self.input_price) or not _zero_price(self.output_price):
            raise ManifestError("both endpoint prices must be zero")
        if not self.snapshot_timestamp_utc:
            raise ManifestError("snapshot_timestamp_utc is required")
        if not self.supported_parameters:
            raise ManifestError("supported_parameters is required")
        if not self.available_providers or self.selected_provider not in self.available_providers:
            raise ManifestError("selected provider must be present in available_providers")
        if self.provider_data_policy is None:
            raise ManifestError("provider_data_policy is required")
        for name in self.supported_parameters + self.available_providers:
            if not isinstance(name, str) or not name:
                raise ManifestError("manifest names must be non-empty strings")
        for name, value in (
            ("context_length", self.context_length),
            ("provider_context_length", self.provider_context_length),
            ("provider_max_output", self.provider_max_output),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ManifestError(f"{name} must be a positive integer")
        computed = hashlib.sha256(self._canonical_payload().encode("utf-8")).hexdigest()
        if self.manifest_sha256 and self.manifest_sha256 != computed:
            raise ManifestError("manifest_sha256 does not match canonical manifest")
        object.__setattr__(self, "manifest_sha256", computed)

    def _canonical_payload(self) -> str:
        return _canonical_json(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "snapshot_timestamp_utc": self.snapshot_timestamp_utc,
            "requested_model_slug": self.requested_model_slug,
            "returned_canonical_slug": self.returned_canonical_slug,
            "context_length": self.context_length,
            "input_price": self.input_price,
            "output_price": self.output_price,
            "supported_parameters": list(self.supported_parameters),
            "available_providers": list(self.available_providers),
            "selected_provider": self.selected_provider,
            "provider_context_length": self.provider_context_length,
            "provider_max_output": self.provider_max_output,
            "provider_data_policy": self.provider_data_policy,
        }
        if include_hash:
            result["manifest_sha256"] = self.manifest_sha256
        return result

    def serialize(self) -> str:
        return _canonical_json(self.to_dict())

    def to_json(self) -> str:
        return self.serialize()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelManifest":
        return cls(
            snapshot_timestamp_utc=value["snapshot_timestamp_utc"],
            requested_model_slug=value["requested_model_slug"],
            returned_canonical_slug=value["returned_canonical_slug"],
            context_length=value["context_length"],
            input_price=value["input_price"],
            output_price=value["output_price"],
            supported_parameters=tuple(value.get("supported_parameters", ())),
            available_providers=tuple(value["available_providers"]),
            selected_provider=value["selected_provider"],
            provider_context_length=value["provider_context_length"],
            provider_max_output=value["provider_max_output"],
            provider_data_policy=value["provider_data_policy"],
            manifest_sha256=value.get("manifest_sha256", ""),
        )


@dataclass(frozen=True)
class LocalModelManifest:
    """Immutable identity for one exact local GGUF and llama.cpp build."""

    snapshot_timestamp_utc: str
    requested_model_slug: str
    original_model_slug: str
    gguf_repository: str
    gguf_revision: str
    gguf_filename: str
    gguf_sha256: str
    gguf_size_bytes: int
    quantization: str
    context_length: int
    selected_provider: str
    llama_cpp_repository: str
    llama_cpp_branch: str
    llama_cpp_commit: str
    server_binary_sha256: str
    cuda_version: str
    gpu_name: str
    driver_version: str
    kv_cache_type_k: str
    kv_cache_type_v: str
    jinja: bool
    tool_call_format: str
    preserve_thinking: bool
    sampling: Mapping[str, Any]
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        for label, value in (
            ("requested_model_slug", self.requested_model_slug),
            ("selected_provider", self.selected_provider),
            ("quantization", self.quantization),
            ("llama_cpp_branch", self.llama_cpp_branch),
        ):
            if not isinstance(value, str) or not value:
                raise ManifestError(f"{label} is required")
        if self.context_length != 32768:
            raise ManifestError("local qualification context is fixed to 32K")
        for label, value in (
            ("gguf_sha256", self.gguf_sha256),
            ("server_binary_sha256", self.server_binary_sha256),
            ("llama_cpp_commit", self.llama_cpp_commit),
        ):
            if len(value) not in {40, 64} or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ManifestError(f"{label} is not a lowercase hex digest")
        computed = hashlib.sha256(
            _canonical_json(self.to_dict(include_hash=False)).encode("utf-8")
        ).hexdigest()
        if self.manifest_sha256 and self.manifest_sha256 != computed:
            raise ManifestError("manifest_sha256 does not match canonical manifest")
        object.__setattr__(self, "manifest_sha256", computed)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "snapshot_timestamp_utc": self.snapshot_timestamp_utc,
            "requested_model_slug": self.requested_model_slug,
            "original_model_slug": self.original_model_slug,
            "gguf_repository": self.gguf_repository,
            "gguf_revision": self.gguf_revision,
            "gguf_filename": self.gguf_filename,
            "gguf_sha256": self.gguf_sha256,
            "gguf_size_bytes": self.gguf_size_bytes,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "selected_provider": self.selected_provider,
            "llama_cpp_repository": self.llama_cpp_repository,
            "llama_cpp_branch": self.llama_cpp_branch,
            "llama_cpp_commit": self.llama_cpp_commit,
            "server_binary_sha256": self.server_binary_sha256,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "driver_version": self.driver_version,
            "kv_cache_type_k": self.kv_cache_type_k,
            "kv_cache_type_v": self.kv_cache_type_v,
            "jinja": self.jinja,
            "tool_call_format": self.tool_call_format,
            "preserve_thinking": self.preserve_thinking,
            "sampling": dict(self.sampling),
        }
        if include_hash:
            result["manifest_sha256"] = self.manifest_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalModelManifest":
        return cls(
            **{
                key: item
                for key, item in value.items()
                if key in cls.__dataclass_fields__
            }
        )


def manifest_hash(manifest: ModelManifest) -> str:
    return manifest.manifest_sha256
