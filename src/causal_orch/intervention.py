"""The single concealed randomization gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .trace import Trace


class Assignment(str, Enum):
    EXECUTE = "execute"
    SUPPRESS = "suppress"


class AssignmentSchedule(Protocol):
    def reveal(self, assignment_unit_id: str) -> "AssignmentRecord": ...


@dataclass(frozen=True)
class AssignmentRecord:
    assignment_id: str
    assignment: Assignment


@dataclass(frozen=True)
class FixedAssignment:
    assignment: Assignment

    def reveal(self, assignment_unit_id: str) -> AssignmentRecord:
        return AssignmentRecord(
            assignment_id=f"{assignment_unit_id}:assignment:1",
            assignment=self.assignment,
        )


@dataclass(frozen=True)
class ManifestAssignment:
    assignments: Mapping[str, Assignment]

    def reveal(self, assignment_unit_id: str) -> AssignmentRecord:
        try:
            assignment = self.assignments[assignment_unit_id]
        except KeyError as exc:
            raise KeyError(
                f"assignment unit is absent from manifest: "
                f"{assignment_unit_id}"
            ) from exc
        return AssignmentRecord(
            assignment_id=f"{assignment_unit_id}:assignment:1",
            assignment=assignment,
        )


def prepare_assignment_manifest(
    path: str | Path,
    *,
    batch_id: str,
    seed: str,
    assignment_unit_ids: list[str],
    fixed_assignment: Assignment | None = None,
) -> ManifestAssignment:
    """Create or verify one small, balanced assignment manifest."""

    if not assignment_unit_ids or len(set(assignment_unit_ids)) != len(
        assignment_unit_ids
    ):
        raise ValueError("assignment unit IDs must be nonempty and unique")
    if fixed_assignment is None:
        ranked = sorted(
            assignment_unit_ids,
            key=lambda unit: hashlib.sha256(
                f"{seed}:{unit}".encode()
            ).digest(),
        )
        execute_count = (len(ranked) + 1) // 2
        execute_units = set(ranked[:execute_count])
        assignments = {
            unit: (
                Assignment.EXECUTE
                if unit in execute_units
                else Assignment.SUPPRESS
            )
            for unit in assignment_unit_ids
        }
    else:
        assignments = {
            unit: fixed_assignment for unit in assignment_unit_ids
        }
    payload = {
        "batch_id": batch_id,
        "seed": seed,
        "assignments": {
            unit: assignments[unit].value
            for unit in sorted(assignments)
        },
    }
    manifest_path = Path(path)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"existing assignment manifest does not match this batch: "
                f"{manifest_path}"
            )
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    return ManifestAssignment(assignments)


@dataclass(frozen=True)
class InterventionResult:
    proposal_id: str
    eligible: bool
    assignment_id: str | None
    assignment: Assignment | None
    observation: str
    worker_completed: bool


class InterventionGate:
    """Reveal treatment only after the first valid delegation proposal."""

    SUPPRESSION_OBSERVATION = json.dumps(
        {
            "reason": "EXPERIMENTAL_CONTROL",
            "status": "DELEGATION_UNAVAILABLE",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    ALREADY_DECIDED_OBSERVATION = json.dumps(
        {"status": "DELEGATION_ALREADY_DECIDED"},
        sort_keys=True,
        separators=(",", ":"),
    )
    INELIGIBLE_OBSERVATION = json.dumps(
        {"status": "DELEGATION_INELIGIBLE"},
        sort_keys=True,
        separators=(",", ":"),
    )

    def __init__(
        self,
        *,
        run_id: str,
        assignment_unit_id: str | None = None,
        schedule: AssignmentSchedule,
        trace: Trace,
    ) -> None:
        self.run_id = run_id
        self.assignment_unit_id = assignment_unit_id or run_id
        self.schedule = schedule
        self.trace = trace
        self.decided = False
        self.assignment_id: str | None = None
        self.assignment: Assignment | None = None
        self.proposal_count = 0

    def intervene(
        self,
        objective: str,
        *,
        terminal: bool,
        run_worker: Callable[[str, str], str],
    ) -> InterventionResult:
        self.proposal_count += 1
        proposal_id = f"proposal-{self.proposal_count}"
        self.trace.emit(
            "delegation_proposed",
            proposal_id=proposal_id,
            objective=objective,
        )
        eligible = bool(objective.strip()) and not self.decided and not terminal
        self.trace.emit(
            "delegation_eligibility",
            proposal_id=proposal_id,
            eligible=eligible,
        )
        if not eligible:
            observation = (
                self.ALREADY_DECIDED_OBSERVATION
                if self.decided
                else self.INELIGIBLE_OBSERVATION
            )
            return InterventionResult(
                proposal_id=proposal_id,
                eligible=False,
                assignment_id=None,
                assignment=None,
                observation=observation,
                worker_completed=False,
            )

        self.decided = True
        record = self.schedule.reveal(self.assignment_unit_id)
        self.assignment_id = record.assignment_id
        self.assignment = record.assignment
        self.trace.emit(
            "assignment_revealed",
            proposal_id=proposal_id,
            assignment_unit_id=self.assignment_unit_id,
            assignment_id=self.assignment_id,
            assignment=self.assignment,
        )
        if self.assignment is Assignment.SUPPRESS:
            self.trace.emit(
                "delegation_suppressed",
                proposal_id=proposal_id,
                assignment_id=self.assignment_id,
            )
            return InterventionResult(
                proposal_id=proposal_id,
                eligible=True,
                assignment_id=self.assignment_id,
                assignment=self.assignment,
                observation=self.SUPPRESSION_OBSERVATION,
                worker_completed=False,
            )

        worker_id = f"{self.run_id}:worker:1"
        observation = run_worker(objective, worker_id)
        self.trace.emit(
            "worker_completed",
            proposal_id=proposal_id,
            assignment_id=self.assignment_id,
            worker_id=worker_id,
            worker_result=observation,
        )
        return InterventionResult(
            proposal_id=proposal_id,
            eligible=True,
            assignment_id=self.assignment_id,
            assignment=self.assignment,
            observation=observation,
            worker_completed=True,
        )
