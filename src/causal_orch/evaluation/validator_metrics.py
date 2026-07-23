"""Descriptive validator observations; these are not native Gaia2 scores."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable


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
    minefields: int
    triggered_minefields: int = 0
    timeout: int | None = None
    timed_out: bool = False

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if self.activation_index < 0:
            raise ValueError("activation_index must be non-negative")
        for name in (
            "initial_milestones",
            "achieved_milestones",
            "minefields",
            "triggered_minefields",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

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
        minefields = getattr(validator, "minefields", ())
        timeout = getattr(validator, "timeout", None)
        internal_count = getattr(validator, "_internal_check_count", 0)
        timed_out = timeout is not None and internal_count >= timeout
        return cls(
            event_id=event_id,
            activation_index=activation_index,
            activated=activated,
            initial_milestones=len(milestones) + len(achieved),
            achieved_milestones=len(achieved),
            minefields=len(minefields),
            timeout=timeout,
            timed_out=timed_out,
        )

    def with_validator_state(self, validator: Any) -> "ValidatorRecord":
        achieved = getattr(validator, "achieved_milestones", ())
        minefields = getattr(validator, "minefields", ())
        timeout = getattr(validator, "timeout", self.timeout)
        internal_count = getattr(validator, "_internal_check_count", 0)
        return replace(
            self,
            achieved_milestones=len(achieved),
            minefields=len(minefields),
            timeout=timeout,
            timed_out=timeout is not None and internal_count >= timeout,
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
    minefield_count: int = 0
    triggered_minefield_count: int = 0
    timeout_count: int = 0
    validator_ids: tuple[str, ...] = ()


def aggregate_validator_milestones(records: Iterable[ValidatorRecord]) -> ValidatorMetric:
    """Aggregate only supplied records into the exploratory secondary metric."""

    records = tuple(records)
    activated = tuple(record for record in records if record.activated)
    initial = sum(record.initial_milestones for record in activated)
    achieved = sum(record.achieved_milestones for record in activated)
    return ValidatorMetric(
        activated_validator_count=len(activated),
        never_activated_validator_count=sum(not record.activated for record in records),
        initial_milestone_count=initial,
        achieved_milestone_count=achieved,
        completion_fraction=(achieved / initial if initial else None),
        minefield_count=sum(record.minefields for record in activated),
        triggered_minefield_count=sum(record.triggered_minefields for record in activated),
        timeout_count=sum(record.timed_out for record in activated),
        validator_ids=tuple(record.validator_id for record in records),
    )
