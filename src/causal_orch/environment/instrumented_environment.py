"""A non-invasive instrumentation seam around ARE's environment event path."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Type

from causal_orch.runtime.state_guard import (
    StateEventRecord,
    application_state_snapshot,
    canonical_hash,
)

try:  # ARE is an optional runtime dependency of this package.
    from are.simulation.environment import Environment as _AREEnvironment
    from are.simulation.types import (
        AgentValidationEvent as _AREAgentValidationEvent,
        ValidationException as _AREValidationException,
    )
except ImportError:  # pragma: no cover - exercised only without ARE installed.
    _AREEnvironment = object  # type: ignore[assignment,misc]
    _AREAgentValidationEvent = None
    _AREValidationException = None

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
    actor_id: str | None = None
    actor_role: str | None = "environment"
    provenance: str | None = "environment"
    simulated_time_before: float | None = None
    simulated_time_after: float | None = None
    write_operation: bool | None = None
    worker_caused_write: bool | None = False
    write_violation: bool | None = False
    validator_failures: tuple["ValidatorFailureRecord", ...] = ()

    @property
    def state_changed(self) -> bool | None:
        if self.state_before_hash is None or self.state_after_hash is None:
            return None
        return self.state_before_hash != self.state_after_hash

    @property
    def simulated_duration(self) -> float | None:
        if self.simulated_time_before is None or self.simulated_time_after is None:
            return None
        return self.simulated_time_after - self.simulated_time_before


@dataclass(frozen=True)
class ValidatorFailureRecord:
    """A validation failure observed on the native ARE event path."""

    event_id: str | None
    exception_type: str
    message: str
    triggered_minefields: int | None = None
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
        self._validator_failures: list[ValidatorFailureRecord] = []
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

    @property
    def validator_failures(self) -> tuple[ValidatorFailureRecord, ...]:
        return tuple(self._validator_failures)

    @property
    def state_records(self) -> tuple[StateEventRecord, ...]:
        """Expose event observations through the state/provenance vocabulary."""

        return tuple(
            StateEventRecord(
                event_id=record.event_id,
                event_type=record.event_type,
                actor_id=record.actor_id,
                actor_role=record.actor_role,
                provenance=record.provenance,
                write_operation=record.write_operation,
                state_before_hash=record.state_before_hash,
                state_after_hash=record.state_after_hash,
                simulated_time_before=record.simulated_time_before,
                simulated_time_after=record.simulated_time_after,
            )
            for record in self._event_records
        )

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

    def _simulated_time(self) -> float | None:
        source = self._target_environment()
        manager = getattr(source, "time_manager", None)
        time_method = getattr(manager, "time", None)
        if callable(time_method):
            try:
                value = time_method()
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None
        for name in ("current_time", "simulated_time"):
            value = getattr(source, name, None)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _event_provenance(event: Any) -> tuple[str | None, str, str]:
        actor_id = getattr(event, "actor_id", None)
        if not isinstance(actor_id, str):
            actor_id = getattr(event, "agent_id", None)
        if not isinstance(actor_id, str):
            actor_id = None
        actor_role = getattr(event, "actor_role", None)
        provenance = getattr(event, "provenance", None)
        if not isinstance(provenance, str):
            provenance = getattr(event, "source", None)
        if not isinstance(actor_role, str):
            actor_role = "worker" if provenance == "worker" else "environment"
        if not isinstance(provenance, str):
            provenance = "worker" if actor_role == "worker" else "environment"
        return actor_id, actor_role, provenance

    @staticmethod
    def _event_write_operation(event: Any) -> bool | None:
        value = getattr(event, "write_operation", None)
        return value if isinstance(value, bool) else None

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

    def _capture_validator_failure(
        self, event: Any, error: BaseException
    ) -> tuple[ValidatorFailureRecord, ...]:
        if _AREValidationException is not None and not isinstance(error, _AREValidationException):
            return ()
        message = str(error)
        match = re.search(r"triggered\s+(\d+)\s+minefields", message, re.IGNORECASE)
        triggered = int(match.group(1)) if match else None
        failure = ValidatorFailureRecord(
            event_id=getattr(event, "event_id", None),
            exception_type=type(error).__name__,
            message=message,
            triggered_minefields=triggered,
        )
        self._validator_failures.append(failure)
        return (failure,)

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
        simulated_time_before = self._simulated_time()
        actor_id, actor_role, provenance = self._event_provenance(event)
        write_operation = self._event_write_operation(event)
        validators = self._validators()
        before_count = len(validators) if validators is not None else None
        result: Any
        validator_ids: tuple[str, ...] = ()
        validator_failures: tuple[ValidatorFailureRecord, ...] = ()
        try:
            if self._instrumented_environment is not None:
                result = self._instrumented_environment.process_event(event)
            else:
                result = super().process_event(event)
        except Exception as error:
            validator_failures = self._capture_validator_failure(event, error)
            raise
        finally:
            validator_ids = self._capture_validator_records(event, before_count)
            self._refresh_validator_records()
            if validator_failures:
                validator_failures = tuple(
                    ValidatorFailureRecord(
                        event_id=failure.event_id,
                        exception_type=failure.exception_type,
                        message=failure.message,
                        triggered_minefields=failure.triggered_minefields,
                        validator_ids=validator_ids,
                    )
                    for failure in validator_failures
                )
                self._validator_failures[-1] = validator_failures[0]
            state_after_hash = self._state_hash()
            simulated_time_after = self._simulated_time()
            state_changed = (
                state_before_hash is not None
                and state_after_hash is not None
                and state_before_hash != state_after_hash
            )
            worker_caused_write = (
                True
                if (actor_role == "worker" or provenance == "worker")
                and (write_operation is True or state_changed)
                else False
                if actor_role == "environment" or provenance == "environment"
                else None
            )
            self._event_records.append(
                EventInstrumentationRecord(
                    run_id=self.run_id,
                    event_id=getattr(event, "event_id", None),
                    event_type=type(event).__name__,
                    state_before_hash=state_before_hash,
                    state_after_hash=state_after_hash,
                    validator_ids=validator_ids,
                    actor_id=actor_id,
                    actor_role=actor_role,
                    provenance=provenance,
                    simulated_time_before=simulated_time_before,
                    simulated_time_after=simulated_time_after,
                    write_operation=write_operation,
                    worker_caused_write=worker_caused_write,
                    write_violation=worker_caused_write,
                    validator_failures=validator_failures,
                )
            )
        return result
