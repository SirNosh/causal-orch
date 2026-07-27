"""Minimal JSONL experiment trace."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


class Trace:
    def __init__(self, run_id: str, path: str | Path | None = None) -> None:
        self.run_id = run_id
        self.path = Path(path) if path is not None else None
        self.events: list[dict[str, Any]] = []

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "sequence": len(self.events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            "payload": _json_safe(payload),
        }
        self.events.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def count(self, event: str) -> int:
        return sum(item["event"] == event for item in self.events)
