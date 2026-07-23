"""Immutable run metadata used to enrich every trace event."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


_FORBIDDEN_KEYS = {"thought", "chain_of_thought", "hidden_reasoning", "private_reasoning"}


def _freeze(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if key.lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"{path} contains a hidden chain-of-thought field")
            result[key] = _freeze(item, f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{path}[]") for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{path} is not JSON-compatible")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class RunContext:
    """Run/scenario identity that a sink applies to events automatically.

    ``temporal_batch`` and ``block_key`` identify the randomization stratum.
    ``block_metadata`` is safe, structured metadata only; it must not contain
    hidden reasoning fields.
    """

    run_id: str
    scenario_id: str | None = None
    universe_id: str | None = None
    capability: str | None = None
    model_slug: str | None = None
    provider_slug: str | None = None
    temporal_batch: str | int | None = None
    block_key: str | None = None
    block_metadata: Mapping[str, Any] = field(default_factory=dict)
    attempt_id: str | None = None
    correlation_id: str = field(default_factory=lambda: uuid4().hex)
    simulated_timestamp: int | float | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        for name in ("scenario_id", "universe_id", "capability", "model_slug", "provider_slug", "block_key", "attempt_id", "correlation_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string when supplied")
        object.__setattr__(self, "block_metadata", _freeze(self.block_metadata, "block_metadata"))

    @property
    def model(self) -> str | None:
        return self.model_slug

    @property
    def provider(self) -> str | None:
        return self.provider_slug

    @property
    def randomization_block(self) -> str | None:
        return self.block_key

    @property
    def requested_model_slug(self) -> str | None:
        return self.model_slug

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "universe_id": self.universe_id,
            "capability": self.capability,
            "model_slug": self.model_slug,
            "provider_slug": self.provider_slug,
            "temporal_batch": self.temporal_batch,
            "block_key": self.block_key,
            "block_metadata": _thaw(self.block_metadata),
            "attempt_id": self.attempt_id,
            "correlation_id": self.correlation_id,
            "simulated_timestamp": self.simulated_timestamp,
        }
