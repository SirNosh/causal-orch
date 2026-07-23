"""Deterministic application-state hashing and worker write guards."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any, Mapping


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TypeError("non-finite floats are not valid application state")
        return value
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        entries = [(_canonicalize(key), _canonicalize(item)) for key, item in value.items()]
        entries.sort(key=lambda item: canonical_json(item[0]))
        return {canonical_json(key): item for key, item in entries}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=canonical_json)
    get_state = getattr(value, "get_state", None)
    if callable(get_state):
        return _canonicalize(get_state())
    if hasattr(value, "__dict__"):
        return {
            "__type__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "attributes": _canonicalize(vars(value)),
        }
    raise TypeError(f"unsupported application-state value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the stable JSON form used by all protocol hashes."""
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    """Hash deterministic state, without using object identity or logs."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def application_state_snapshot(source: Any) -> Any:
    """Read only the application snapshot from an environment-like object."""
    get_apps_state = getattr(source, "get_apps_state", None)
    if callable(get_apps_state):
        return get_apps_state()
    if callable(source):
        return source()
    return source


@dataclasses.dataclass
class StateGuard:
    """Capture application state before and after a worker attempt."""

    source: Any
    before_hash: str | None = None
    after_hash: str | None = None

    def capture_before(self) -> str:
        self.before_hash = canonical_hash(application_state_snapshot(self.source))
        return self.before_hash

    def capture_after(self) -> str:
        self.after_hash = canonical_hash(application_state_snapshot(self.source))
        return self.after_hash

    @property
    def changed(self) -> bool:
        return self.before_hash is not None and self.after_hash is not None and self.before_hash != self.after_hash

    def __enter__(self) -> "StateGuard":
        self.capture_before()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.capture_after()
        return False


def guarded_application_state(source: Any) -> StateGuard:
    """Construct a context manager for a worker's application-state guard."""
    return StateGuard(source)
