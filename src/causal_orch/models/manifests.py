"""Locked model, provider, sampling, and manifest types for the protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Any, Mapping


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
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    reasoning: Mapping[str, Any] | None = None
    endpoint: str = OPENROUTER_COMPLETIONS_URL
    api_key: str | None = None
    attribution_headers: Mapping[str, str] = field(default_factory=dict)
    allow_fallbacks: bool = False
    require_parameters: bool = True

    def __post_init__(self) -> None:
        validate_model_slug(self.model_slug)
        if not self.provider or not isinstance(self.provider, str):
            raise ManifestError("provider must be a non-empty pinned provider slug")
        if self.allow_fallbacks is not False:
            raise ManifestError("provider fallbacks are disabled by protocol")
        if self.require_parameters is not True:
            raise ManifestError("provider parameter requirement is fixed to true")
        if not self.endpoint:
            raise ManifestError("endpoint must be non-empty")
        for key, value in self.attribution_headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ManifestError("attribution headers must be string-to-string")

    @property
    def pinned_provider(self) -> str:
        return self.provider


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


def manifest_hash(manifest: ModelManifest) -> str:
    return manifest.manifest_sha256
