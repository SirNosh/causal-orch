"""Offline model/provider qualification and failure classification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Mapping

from .manifests import MODEL_CANDIDATE_ORDER, ModelManifest, validate_model_slug


class HealthError(ValueError):
    """Raised when supplied qualification evidence violates the protocol."""


class FailureClass(str, Enum):
    RATE_LIMIT = "RATE_LIMIT"
    OUTAGE = "OUTAGE"
    AUTHENTICATION = "AUTHENTICATION"
    HTTP_ERROR = "HTTP_ERROR"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"


def classify_http_failure(status_code: int) -> FailureClass:
    if status_code == 429:
        return FailureClass.RATE_LIMIT
    if status_code in {500, 502, 503, 504}:
        return FailureClass.OUTAGE
    if status_code in {401, 403}:
        return FailureClass.AUTHENTICATION
    return FailureClass.HTTP_ERROR


@dataclass(frozen=True)
class QualificationResult:
    qualified: bool
    reasons: tuple[str, ...] = ()


def validate_manifest(manifest: ModelManifest, pinned_provider: str) -> None:
    """Validate only supplied data; this helper performs no catalog/API calls."""

    validate_model_slug(manifest.requested_model_slug)
    if manifest.selected_provider != pinned_provider:
        raise HealthError("PROVIDER_PIN_MISMATCH")
    if pinned_provider not in manifest.available_providers:
        raise HealthError("PINNED_PROVIDER_NOT_AVAILABLE")
    if not manifest.supported_parameters or manifest.provider_data_policy is None:
        raise HealthError("MANIFEST_METADATA_INCOMPLETE")
    if not manifest.manifest_sha256:
        raise HealthError("MANIFEST_HASH_MISSING")


def qualify_manifest(manifest: ModelManifest, pinned_provider: str) -> QualificationResult:
    try:
        validate_manifest(manifest, pinned_provider)
    except (HealthError, ValueError) as error:
        return QualificationResult(False, (str(error),))
    return QualificationResult(True)


def select_first_qualified(
    manifests: Mapping[str, ModelManifest] | list[ModelManifest] | tuple[ModelManifest, ...],
    pinned_provider: str,
) -> ModelManifest | None:
    """Select the first passing candidate in the locked, never-effect-based order."""

    if isinstance(manifests, Mapping):
        by_slug = manifests
    else:
        by_slug = {manifest.requested_model_slug: manifest for manifest in manifests}
    for slug in MODEL_CANDIDATE_ORDER:
        manifest = by_slug.get(slug)
        if manifest is not None and qualify_manifest(manifest, pinned_provider).qualified:
            return manifest
    return None


def validate_response_metadata(
    metadata: Mapping[str, Any],
    expected_model: str,
    pinned_provider: str | None = None,
    *,
    require_provider: bool = True,
) -> None:
    """Validate identity and completeness of metadata from a supplied fake response."""

    if metadata.get("returned_model") != expected_model:
        raise HealthError("MODEL_IDENTITY_MISMATCH")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "completion_duration", "request_id"):
        if metadata.get(field) is None:
            raise HealthError(f"METADATA_INCOMPLETE:{field}")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = metadata[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise HealthError(f"METADATA_INVALID:{field}")
    if not isinstance(metadata["completion_duration"], Real) or metadata["completion_duration"] < 0:
        raise HealthError("METADATA_INVALID:completion_duration")
    if not isinstance(metadata["request_id"], str) or not metadata["request_id"]:
        raise HealthError("METADATA_INVALID:request_id")
    provider = metadata.get("provider")
    if require_provider and not provider:
        raise HealthError("PROVIDER_METADATA_MISSING")
    if pinned_provider is not None and provider is not None and provider != pinned_provider:
        raise HealthError("PROVIDER_PIN_MISMATCH")


def response_qualifies(
    metadata: Mapping[str, Any], expected_model: str, pinned_provider: str
) -> QualificationResult:
    try:
        validate_response_metadata(metadata, expected_model, pinned_provider)
    except HealthError as error:
        return QualificationResult(False, (str(error),))
    return QualificationResult(True)
