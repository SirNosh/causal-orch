"""Stable, JSON-safe public references for orchestrator-visible context."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
import dataclasses
from enum import Enum
from typing import Any

class TraceContextRegistry(Mapping[str, Any]):
    """Store exactly what a worker will resolve behind stable public handles."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def register(self, ref: str, content: Any) -> str:
        if not isinstance(ref, str) or not ref:
            raise ValueError("context reference must be a non-empty string")
        value = _json_safe(content)
        existing = self._values.get(ref)
        if existing is not None and existing != value:
            raise ValueError(f"context reference already has different content: {ref}")
        self._values[ref] = value
        return ref

    def resolve(self, ref: str) -> Any:
        return deepcopy(self._values[ref])

    def __getitem__(self, ref: str) -> Any:
        return self.resolve(ref)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    raise TypeError(f"context is not JSON-safe: {type(value).__name__}")
