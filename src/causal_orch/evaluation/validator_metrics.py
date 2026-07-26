"""Descriptive validator observations; these are not native Gaia2 scores."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable


_MISSING = object()


def _optional_count(value: Any) -> int | None:
    """Return a supplied count without turning an absent field into zero."""

    if value is _MISSING or value is None:
        return None
    if type(value) is int:
        if value < 0:
            raise ValueError("validator counts must be non-negative")
        return value
    try:
        count = len(value)
    except TypeError:
        return None
    if count < 0:  # pragma: no cover - Python containers cannot do this.
        raise ValueError("validator counts must be non-negative")
    return count


def _validator_count(validator: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(validator, name, _MISSING)
        if value is not _MISSING:
            return _optional_count(value)
    return None


def _validator_text(validator: Any, *names: str) -> str | None:
    for name in names:
        value = getattr(validator, name, _MISSING)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass(frozen=True)
class ValidatorRecord:
    """A supplied validator event/state observation.

    ``activated=False`` records are useful when an event was planned or
    otherwise observed but never reached ``Environment.process_event``.
    """

    event_id: str
    activation_index: int
    activated: bool
    initial_milestones: int
    achieved_milestones: int
    minefields: int | None = None
    triggered_minefields: int | None = None
    timeout: int | None = None
    timed_out: bool = False
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if self.activation_index < 0:
            raise ValueError("activation_index must be non-negative")
        for name in (
            "initial_milestones",
            "achieved_milestones",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.minefields is not None and self.minefields < 0:
            raise ValueError("minefields must be non-negative or None")
        if self.triggered_minefields is not None and self.triggered_minefields < 0:
            raise ValueError("triggered_minefields must be non-negative or None")

    @property
    def validator_id(self) -> str:
        return f"{self.event_id}:{self.activation_index}"

    @classmethod
    def from_validator(
        cls,
        *,
        event_id: str,
        activation_index: int,
        validator: Any,
        activated: bool,
    ) -> "ValidatorRecord":
        milestones = getattr(validator, "milestones", ())
        achieved = getattr(validator, "achieved_milestones", ())
        minefields = _validator_count(validator, "minefields", "minefield_count")
        timeout = getattr(validator, "timeout", None)
        internal_count = getattr(validator, "_internal_check_count", 0)
        timed_out = timeout is not None and internal_count >= timeout
        return cls(
            event_id=event_id,
            activation_index=activation_index,
            activated=activated,
            initial_milestones=len(milestones) + len(achieved),
            achieved_milestones=len(achieved),
            minefields=minefields,
            triggered_minefields=_validator_count(
                validator, "triggered_minefields", "triggered_minefield_count"
            ),
            timeout=timeout,
            timed_out=timed_out,
            failure_reason=_validator_text(validator, "failure_reason", "failure_message"),
        )

    def with_validator_state(self, validator: Any) -> "ValidatorRecord":
        achieved = getattr(validator, "achieved_milestones", ())
        minefields = _validator_count(validator, "minefields", "minefield_count")
        timeout = getattr(validator, "timeout", self.timeout)
        internal_count = getattr(validator, "_internal_check_count", 0)
        triggered = _validator_count(
            validator, "triggered_minefields", "triggered_minefield_count"
        )
        failure_reason = _validator_text(validator, "failure_reason", "failure_message")
        return replace(
            self,
            achieved_milestones=len(achieved),
            minefields=self.minefields if not hasattr(validator, "minefields") and not hasattr(validator, "minefield_count") else minefields,
            triggered_minefields=(
                self.triggered_minefields
                if not hasattr(validator, "triggered_minefields")
                and not hasattr(validator, "triggered_minefield_count")
                else triggered
            ),
            timeout=timeout,
            timed_out=timeout is not None and internal_count >= timeout,
            failure_reason=self.failure_reason if failure_reason is None else failure_reason,
        )

    def with_failure(
        self, *, triggered_minefields: int | None, failure_reason: str
    ) -> "ValidatorRecord":
        """Attach only failure information observed from ARE's validation path."""

        return replace(
            self,
            triggered_minefields=triggered_minefields,
            failure_reason=failure_reason,
        )


@dataclass(frozen=True)
class ValidatorMetric:
    """Exploratory/descriptive aggregate, never a native Gaia2 score."""

    metric_label: str = "exploratory_activated_validator_milestone_completion"
    primary_outcome: bool = False
    activated_validator_count: int = 0
    never_activated_validator_count: int = 0
    initial_milestone_count: int = 0
    achieved_milestone_count: int = 0
    completion_fraction: float | None = None
    minefield_count: int | None = None
    triggered_minefield_count: int | None = None
    timeout_count: int = 0
    validator_ids: tuple[str, ...] = ()


def aggregate_validator_milestones(records: Iterable[ValidatorRecord]) -> ValidatorMetric:
    """Aggregate only supplied records into the exploratory secondary metric."""

    records = tuple(records)
    activated = tuple(record for record in records if record.activated)
    initial = sum(record.initial_milestones for record in activated)
    achieved = sum(record.achieved_milestones for record in activated)
    known_triggered = [
        record.triggered_minefields
        for record in activated
        if record.triggered_minefields is not None
    ]
    triggered_count: int | None
    if not activated:
        triggered_count = 0
    elif len(known_triggered) != len(activated):
        triggered_count = None
    else:
        triggered_count = sum(known_triggered)
    return ValidatorMetric(
        activated_validator_count=len(activated),
        never_activated_validator_count=sum(not record.activated for record in records),
        initial_milestone_count=initial,
        achieved_milestone_count=achieved,
        completion_fraction=(achieved / initial if initial else None),
        minefield_count=(
            0
            if not activated
            else None
            if any(record.minefields is None for record in activated)
            else sum(record.minefields for record in activated)
        ),
        triggered_minefield_count=triggered_count,
        timeout_count=sum(record.timed_out for record in activated),
        validator_ids=tuple(record.validator_id for record in records),
    )
