"""Append-only JSONL trace sinks."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Callable, Iterable, TextIO

from .context import RunContext
from .events import OrchestrationEvent


def _wall_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceSink:
    """Write immutable event snapshots as one deterministic JSON object per line."""

    def __init__(
        self,
        path: str | Path,
        *,
        flush: bool = True,
        context: RunContext | None = None,
        wall_clock: Callable[[], int | float | str] = _wall_timestamp,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush = flush
        self._stream: TextIO = self.path.open("a", encoding="utf-8", newline="\n")
        self._closed = False
        self._context = context
        self._wall_clock = wall_clock
        self._lock = RLock()
        self._sequence = 0
        self._last_event_id: str | None = None

    @property
    def context(self) -> RunContext | None:
        return self._context

    def set_context(self, context: RunContext | None) -> None:
        if context is not None and not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext or None")
        with self._lock:
            if context != self._context:
                self._sequence = 0
                self._last_event_id = None
            self._context = context

    def _enrich(self, event: OrchestrationEvent) -> OrchestrationEvent:
        context = self._context
        self._sequence += 1
        fields = {
            "event_sequence": self._sequence,
            "correlation_id": event.correlation_id or (context.correlation_id if context else None),
            "previous_event_id": event.previous_event_id or self._last_event_id,
            "wall_timestamp": event.wall_timestamp if event.wall_timestamp is not None else self._wall_clock(),
        }
        if context is not None:
            fields.update({
                "run_id": event.run_id or context.run_id,
                "scenario_id": event.scenario_id or context.scenario_id,
                "universe_id": event.universe_id or context.universe_id,
                "capability": event.capability or context.capability,
                "requested_model_slug": event.requested_model_slug or context.model_slug,
                "provider_slug": event.provider_slug or context.provider_slug,
                "attempt_id": event.attempt_id or context.attempt_id,
                "simulated_timestamp": event.simulated_timestamp if event.simulated_timestamp is not None else context.simulated_timestamp,
                "temporal_batch": event.temporal_batch if event.temporal_batch is not None else context.temporal_batch,
                "randomization_block_key": event.randomization_block_key or context.block_key,
                "block_metadata": event.block_metadata or context.to_dict()["block_metadata"],
            })
        enriched = OrchestrationEvent(**{**deepcopy(event.__dict__), **fields})
        self._last_event_id = enriched.event_id
        return enriched

    def append(self, event: OrchestrationEvent) -> None:
        if not isinstance(event, OrchestrationEvent):
            raise TypeError("TraceSink accepts OrchestrationEvent values")
        with self._lock:
            if self._closed:
                raise ValueError("trace sink is closed")
            self._stream.write(self._enrich(event).to_json() + "\n")
            if self._flush:
                self._stream.flush()

    def extend(self, events: Iterable[OrchestrationEvent]) -> None:
        for event in events:
            self.append(event)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._stream.close()
                self._closed = True

    def __enter__(self) -> "TraceSink":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class InMemoryTraceSink:
    """A tiny sink for pure tests; it stores event snapshots, not caller objects."""

    def __init__(
        self,
        *,
        context: RunContext | None = None,
        wall_clock: Callable[[], int | float | str] = _wall_timestamp,
    ) -> None:
        self._events: list[OrchestrationEvent] = []
        self._context = context
        self._wall_clock = wall_clock
        self._lock = RLock()
        self._sequence = 0
        self._last_event_id: str | None = None

    @property
    def context(self) -> RunContext | None:
        return self._context

    def set_context(self, context: RunContext | None) -> None:
        if context is not None and not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext or None")
        with self._lock:
            if context != self._context:
                self._sequence = 0
                self._last_event_id = None
            self._context = context

    def _enrich(self, event: OrchestrationEvent) -> OrchestrationEvent:
        context = self._context
        self._sequence += 1
        fields = {
            "event_sequence": self._sequence,
            "correlation_id": event.correlation_id or (context.correlation_id if context else None),
            "previous_event_id": event.previous_event_id or self._last_event_id,
            "wall_timestamp": event.wall_timestamp if event.wall_timestamp is not None else self._wall_clock(),
        }
        if context is not None:
            fields.update({
                "run_id": event.run_id or context.run_id,
                "scenario_id": event.scenario_id or context.scenario_id,
                "universe_id": event.universe_id or context.universe_id,
                "capability": event.capability or context.capability,
                "requested_model_slug": event.requested_model_slug or context.model_slug,
                "provider_slug": event.provider_slug or context.provider_slug,
                "attempt_id": event.attempt_id or context.attempt_id,
                "simulated_timestamp": event.simulated_timestamp if event.simulated_timestamp is not None else context.simulated_timestamp,
                "temporal_batch": event.temporal_batch if event.temporal_batch is not None else context.temporal_batch,
                "randomization_block_key": event.randomization_block_key or context.block_key,
                "block_metadata": event.block_metadata or context.to_dict()["block_metadata"],
            })
        return OrchestrationEvent(**{**deepcopy(event.__dict__), **fields})

    def append(self, event: OrchestrationEvent) -> None:
        if not isinstance(event, OrchestrationEvent):
            raise TypeError("InMemoryTraceSink accepts OrchestrationEvent values")
        with self._lock:
            enriched = self._enrich(event)
            self._last_event_id = enriched.event_id
            self._events.append(OrchestrationEvent(**deepcopy(enriched.__dict__)))

    def extend(self, events: Iterable[OrchestrationEvent]) -> None:
        for event in events:
            self.append(event)

    @property
    def events(self) -> tuple[OrchestrationEvent, ...]:
        return tuple(self._events)
