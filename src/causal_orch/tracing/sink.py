"""Append-only JSONL trace sinks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable, TextIO

from .events import OrchestrationEvent


class TraceSink:
    """Write immutable event snapshots as one deterministic JSON object per line."""

    def __init__(self, path: str | Path, *, flush: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush = flush
        self._stream: TextIO = self.path.open("a", encoding="utf-8", newline="\n")
        self._closed = False

    def append(self, event: OrchestrationEvent) -> None:
        if self._closed:
            raise ValueError("trace sink is closed")
        if not isinstance(event, OrchestrationEvent):
            raise TypeError("TraceSink accepts OrchestrationEvent values")
        self._stream.write(event.to_json() + "\n")
        if self._flush:
            self._stream.flush()

    def extend(self, events: Iterable[OrchestrationEvent]) -> None:
        for event in events:
            self.append(event)

    def close(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True

    def __enter__(self) -> "TraceSink":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class InMemoryTraceSink:
    """A tiny sink for pure tests; it stores event snapshots, not caller objects."""

    def __init__(self) -> None:
        self._events: list[OrchestrationEvent] = []

    def append(self, event: OrchestrationEvent) -> None:
        if not isinstance(event, OrchestrationEvent):
            raise TypeError("InMemoryTraceSink accepts OrchestrationEvent values")
        self._events.append(OrchestrationEvent(**deepcopy(event.__dict__)))

    def extend(self, events: Iterable[OrchestrationEvent]) -> None:
        for event in events:
            self.append(event)

    @property
    def events(self) -> tuple[OrchestrationEvent, ...]:
        return tuple(self._events)
