"""Deterministic application-state hashing and worker write guards."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any, Iterable, Mapping


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
    actor_id: str | None = None
    actor_role: str | None = None
    event_id: str | None = None
    event_type: str | None = None
    provenance: str | None = None
    state_records: list["StateEventRecord"] = dataclasses.field(default_factory=list)

    def capture_before(self) -> str:
        self.before_hash = canonical_hash(application_state_snapshot(self.source))
        return self.before_hash

    def capture_after(self) -> str:
        self.after_hash = canonical_hash(application_state_snapshot(self.source))
        return self.after_hash

    @property
    def changed(self) -> bool:
        return self.before_hash is not None and self.after_hash is not None and self.before_hash != self.after_hash

    @property
    def worker_write_violation(self) -> bool | None:
        """Return only an explicitly attributable worker-write observation.

        A state change by itself is not enough to attribute a write to a worker;
        tool permission/provenance is the primary guarantee and this guard is a
        secondary check for an attributable attempt.
        """

        worker = self.actor_role == "worker" or self.provenance == "worker"
        if not worker:
            return False if self.provenance == "environment" else None
        return self.changed if self.changed else None

    def record_event(
        self,
        *,
        event_id: str | None,
        event_type: str,
        actor_id: str | None = None,
        actor_role: str | None = None,
        provenance: str | None = None,
        tool_name: str | None = None,
        write_operation: bool | None = None,
        state_before_hash: str | None = None,
        state_after_hash: str | None = None,
        simulated_time_before: float | None = None,
        simulated_time_after: float | None = None,
    ) -> "StateEventRecord":
        """Record one explicitly attributed environment/state observation."""

        record = StateEventRecord(
            event_id=event_id,
            event_type=event_type,
            actor_id=actor_id,
            actor_role=actor_role,
            provenance=provenance,
            tool_name=tool_name,
            write_operation=write_operation,
            state_before_hash=state_before_hash,
            state_after_hash=state_after_hash,
            simulated_time_before=simulated_time_before,
            simulated_time_after=simulated_time_after,
        )
        self.state_records.append(record)
        return record

    def __enter__(self) -> "StateGuard":
        self.capture_before()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.capture_after()
        return False


def guarded_application_state(source: Any) -> StateGuard:
    """Construct a context manager for a worker's application-state guard."""
    return StateGuard(source)


@dataclasses.dataclass(frozen=True)
class StateEventRecord:
    """A state observation with actor/event provenance.

    ``worker_caused_write`` is deliberately tri-state.  ``False`` means the
    event is explicitly independent environment activity; ``None`` means the
    available provenance is insufficient to attribute it.
    """

    event_id: str | None
    event_type: str
    actor_id: str | None = None
    actor_role: str | None = None
    provenance: str | None = None
    tool_name: str | None = None
    write_operation: bool | None = None
    state_before_hash: str | None = None
    state_after_hash: str | None = None
    simulated_time_before: float | None = None
    simulated_time_after: float | None = None

    @property
    def state_changed(self) -> bool | None:
        if self.state_before_hash is None or self.state_after_hash is None:
            return None
        return self.state_before_hash != self.state_after_hash

    @property
    def worker_caused_write(self) -> bool | None:
        if self.actor_role == "worker" or self.provenance == "worker":
            changed = self.state_changed
            if self.write_operation is True or changed is True:
                return True
            if self.write_operation is False and changed is False:
                return False
            return None
        if self.provenance == "environment" or self.actor_role == "environment":
            return False
        return None

    @property
    def write_violation(self) -> bool | None:
        return self.worker_caused_write

    @property
    def simulated_duration(self) -> float | None:
        if self.simulated_time_before is None or self.simulated_time_after is None:
            return None
        return self.simulated_time_after - self.simulated_time_before


def worker_tool_write_violation(tool: Any) -> bool:
    """Return whether a tool is unsafe for a worker before execution.

    ARE's ``write_operation=None`` is intentionally unsafe.  This helper is a
    small provenance-layer mirror of the audited tool contract and does not
    treat a later state diff as the primary permission decision.
    """

    if isinstance(tool, Mapping):
        value = tool.get("write_operation")
    else:
        value = getattr(tool, "write_operation", None)
    return value is not False


def all_worker_tools_read_only(tools: Iterable[Any]) -> bool:
    """Return whether every supplied worker tool has explicit read-only metadata."""

    return all(not worker_tool_write_violation(tool) for tool in tools)
