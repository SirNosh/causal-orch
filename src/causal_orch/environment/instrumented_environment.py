"""A non-invasive instrumentation seam around ARE's environment event path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Type

from causal_orch.runtime.state_guard import application_state_snapshot, canonical_hash

try:  # ARE is an optional runtime dependency of this package.
    from are.simulation.environment import Environment as _AREEnvironment
    from are.simulation.types import AgentValidationEvent as _AREAgentValidationEvent
except ImportError:  # pragma: no cover - exercised only without ARE installed.
    _AREEnvironment = object  # type: ignore[assignment,misc]
    _AREAgentValidationEvent = None

from causal_orch.evaluation.validator_metrics import ValidatorRecord


@dataclass(frozen=True)
class EventInstrumentationRecord:
    """Read-only metadata captured around one ``process_event`` call."""

    run_id: str | None
    event_id: str | None
    event_type: str
    state_before_hash: str | None
    state_after_hash: str | None
    validator_ids: tuple[str, ...] = ()


class InstrumentedEnvironment(_AREEnvironment):
    """Preserve ARE event processing while collecting optional observations.

    The normal construction path subclasses the pinned ARE ``Environment``.
    ``environment=`` is a small adapter path for tests or an already-created
    environment; in both paths the wrapped/native ``process_event`` result is
    returned unchanged.  Instrumentation is deliberately best-effort and never
    changes an ARE exception or event result.
    """

    def __init__(
        self,
        *args: Any,
        environment: Any | None = None,
        run_id: str | None = None,
        state_source: Any | None = None,
        validator_event_types: Iterable[Type[Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if environment is None and len(args) == 1 and hasattr(args[0], "process_event"):
            environment = args[0]
            args = ()

        self._instrumented_environment = environment
        self.run_id = run_id
        self._state_source = state_source
        default_types = () if _AREAgentValidationEvent is None else (_AREAgentValidationEvent,)
        self._validator_event_types = (
            default_types if validator_event_types is None else tuple(validator_event_types)
        )
        self._event_records: list[EventInstrumentationRecord] = []
        self._validator_records: list[ValidatorRecord] = []
        self._activation_counts: dict[str, int] = {}
        self._validator_positions: dict[str, int] = {}

        if environment is None:
            super().__init__(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        environment = self.__dict__.get("_instrumented_environment")
        if environment is not None:
            return getattr(environment, name)
        raise AttributeError(name)

    @property
    def event_records(self) -> tuple[EventInstrumentationRecord, ...]:
        return tuple(self._event_records)

    @property
    def validator_records(self) -> tuple[ValidatorRecord, ...]:
        return tuple(self._validator_records)

    def _target_environment(self) -> Any:
        return self._instrumented_environment if self._instrumented_environment is not None else self

    def _state_hash(self) -> str | None:
        source = self._state_source
        if source is None:
            source = self._target_environment()
            if not callable(getattr(source, "get_apps_state", None)):
                return None
        try:
            return canonical_hash(application_state_snapshot(source))
        except Exception:
            return None

    def _is_validator_event(self, event: Any) -> bool:
        return bool(self._validator_event_types) and isinstance(event, self._validator_event_types)

    def _validators(self) -> list[Any] | None:
        validators = getattr(self._target_environment(), "agent_action_validators", None)
        return validators if isinstance(validators, list) else None

    def _capture_validator_records(self, event: Any, before_count: int | None) -> tuple[str, ...]:
        if not self._is_validator_event(event):
            return ()
        validators = self._validators()
        if validators is None or before_count is None or len(validators) <= before_count:
            return ()

        event_id = getattr(event, "event_id", None)
        if not isinstance(event_id, str) or not event_id:
            return ()
        ids: list[str] = []
        for validator in validators[before_count:]:
            activation_index = self._activation_counts.get(event_id, 0)
            self._activation_counts[event_id] = activation_index + 1
            record = ValidatorRecord.from_validator(
                event_id=event_id,
                activation_index=activation_index,
                validator=validator,
                activated=True,
            )
            self._validator_records.append(record)
            self._validator_positions[record.validator_id] = before_count + len(ids)
            ids.append(record.validator_id)
        return tuple(ids)

    def _refresh_validator_records(self) -> None:
        validators = self._validators()
        if validators is None:
            return
        refreshed: list[ValidatorRecord] = []
        for record in self._validator_records:
            position = self._validator_positions.get(record.validator_id)
            if position is not None and position < len(validators):
                refreshed.append(record.with_validator_state(validators[position]))
            else:
                refreshed.append(record)
        self._validator_records = refreshed

    def process_event(self, event: Any) -> Any:
        """Call the native/wrapped method once and return its result unchanged."""

        state_before_hash = self._state_hash()
        validators = self._validators()
        before_count = len(validators) if validators is not None else None
        result: Any
        validator_ids: tuple[str, ...] = ()
        try:
            if self._instrumented_environment is not None:
                result = self._instrumented_environment.process_event(event)
            else:
                result = super().process_event(event)
        finally:
            validator_ids = self._capture_validator_records(event, before_count)
            self._refresh_validator_records()
            self._event_records.append(
                EventInstrumentationRecord(
                    run_id=self.run_id,
                    event_id=getattr(event, "event_id", None),
                    event_type=type(event).__name__,
                    state_before_hash=state_before_hash,
                    state_after_hash=self._state_hash(),
                    validator_ids=validator_ids,
                )
            )
        return result
