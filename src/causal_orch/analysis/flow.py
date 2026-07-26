"""Pure run-flow and attrition accounting for the fixed experiment protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


def _arm(value: Any) -> str | None:
    if value is None:
        return None
    value = getattr(value, "label", value)
    value = getattr(value, "value", value)
    text = str(value).strip().lower()
    if text in {"treatment", "treated", "execute", "1"}:
        return "treatment"
    if text in {"control", "suppress", "suppressed", "0"}:
        return "control"
    raise ValueError(f"unknown assignment label: {value!r}")


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise TypeError("flow flags must be booleans")


@dataclass(frozen=True)
class FlowRecord:
    """One run/proposal attempt represented in a flow table."""

    run_id: str
    scheduled: bool = False
    started: bool = False
    initialized: bool = False
    proposed: bool = False
    eligible: bool = False
    assignment: str | None = None
    worker_started: bool = False
    worker_completed: bool = False
    state_violation: bool = False
    completed: bool = False
    judged: bool = False
    provider_failure: bool = False
    protocol_violation: bool = False
    success: bool | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        object.__setattr__(self, "assignment", _arm(self.assignment))
        if self.success is not None and not isinstance(self.success, bool):
            raise TypeError("success must be boolean or None")
        for name in (
            "scheduled",
            "started",
            "initialized",
            "proposed",
            "eligible",
            "worker_started",
            "worker_completed",
            "state_violation",
            "completed",
            "judged",
            "provider_failure",
            "protocol_violation",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], index: int = 0) -> "FlowRecord":
        """Convert a trace-like mapping without coupling analysis to trace classes."""

        def first(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in row:
                    return row[name]
            return default

        run_id = first("run_id", "id", "proposal_id", default=str(index))
        assignment = first("assignment", "arm", "treatment")
        if assignment is False or assignment is True:
            assignment = "treatment" if assignment else "control"
        return cls(
            run_id=str(run_id),
            scheduled=_bool(first("scheduled")),
            started=_bool(first("started")),
            initialized=_bool(first("initialized")),
            proposed=_bool(first("proposed", "has_proposal")),
            eligible=_bool(first("eligible")),
            assignment=assignment,
            worker_started=_bool(first("worker_started")),
            worker_completed=_bool(first("worker_completed")),
            state_violation=_bool(first("state_violation", "state_guard_violation")),
            completed=_bool(first("completed")),
            judged=_bool(first("judged")),
            provider_failure=_bool(first("provider_failure")),
            protocol_violation=_bool(first("protocol_violation")),
            success=first("success", "outcome", "binary_success"),
        )

    @property
    def assigned(self) -> bool:
        return self.assignment is not None

    @property
    def final_analysis_eligible(self) -> bool:
        return self.eligible and self.assigned and self.judged and isinstance(self.success, bool)


FLOW_FIELDS = (
    "runs_scheduled",
    "runs_started",
    "runs_initialized",
    "runs_with_any_delegation_proposal",
    "valid_eligible_proposals",
    "assignments_revealed",
    "treatment_assigned",
    "control_assigned",
    "workers_started",
    "workers_completed",
    "state_guard_violations",
    "runs_completed",
    "runs_judged",
    "provider_failures",
    "protocol_violations",
    "final_analysis_sample",
)


@dataclass
class FlowTable:
    """Count the protocol stages without changing the source run data."""

    records: list[FlowRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.records = [
            record if isinstance(record, FlowRecord) else FlowRecord.from_mapping(record, index=index)
            for index, record in enumerate(self.records)
        ]

    @classmethod
    def from_rows(cls, rows: Iterable[FlowRecord | Mapping[str, Any]]) -> "FlowTable":
        return cls(list(rows))

    def add(self, record: FlowRecord | Mapping[str, Any], **fields: Any) -> FlowRecord:
        if fields:
            if not isinstance(record, Mapping):
                raise TypeError("keyword fields require a mapping record")
            record = {**record, **fields}
        converted = record if isinstance(record, FlowRecord) else FlowRecord.from_mapping(record, len(self.records))
        self.records.append(converted)
        return converted

    @property
    def counts(self) -> dict[str, int]:
        rows = self.records
        return {
            "runs_scheduled": sum(row.scheduled for row in rows),
            "runs_started": sum(row.started for row in rows),
            "runs_initialized": sum(row.initialized for row in rows),
            "runs_with_any_delegation_proposal": sum(row.proposed for row in rows),
            "valid_eligible_proposals": sum(row.eligible for row in rows),
            "assignments_revealed": sum(row.assigned for row in rows),
            "treatment_assigned": sum(row.assignment == "treatment" for row in rows),
            "control_assigned": sum(row.assignment == "control" for row in rows),
            "workers_started": sum(row.worker_started for row in rows),
            "workers_completed": sum(row.worker_completed for row in rows),
            "state_guard_violations": sum(row.state_violation for row in rows),
            "runs_completed": sum(row.completed for row in rows),
            "runs_judged": sum(row.judged for row in rows),
            "provider_failures": sum(row.provider_failure for row in rows),
            "protocol_violations": sum(row.protocol_violation for row in rows),
            "final_analysis_sample": sum(row.final_analysis_eligible for row in rows),
        }

    def to_dict(self) -> dict[str, int]:
        return self.counts


def flow_counts(rows: Iterable[FlowRecord | Mapping[str, Any]]) -> dict[str, int]:
    """Return the prespecified attrition counts for an iterable of rows."""

    return FlowTable.from_rows(rows).counts

