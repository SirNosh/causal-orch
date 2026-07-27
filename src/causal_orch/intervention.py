"""The single concealed randomization gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Callable, Protocol

from .trace import Trace


class Assignment(str, Enum):
    EXECUTE = "execute"
    SUPPRESS = "suppress"


class AssignmentSchedule(Protocol):
    def reveal(self, run_id: str) -> Assignment: ...


@dataclass(frozen=True)
class FixedAssignment:
    assignment: Assignment

    def reveal(self, run_id: str) -> Assignment:
        return self.assignment


@dataclass(frozen=True)
class RandomAssignment:
    seed: str
    execute_probability: float = 0.5

    def reveal(self, run_id: str) -> Assignment:
        if not 0 <= self.execute_probability <= 1:
            raise ValueError("execute_probability must be between zero and one")
        digest = hashlib.sha256(f"{self.seed}:{run_id}".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        return (
            Assignment.EXECUTE
            if draw < self.execute_probability
            else Assignment.SUPPRESS
        )


@dataclass(frozen=True)
class InterventionResult:
    proposal_id: str
    eligible: bool
    assignment: Assignment | None
    observation: str
    worker_completed: bool


class InterventionGate:
    """Reveal treatment only after the first valid delegation proposal."""

    SUPPRESSION_OBSERVATION = (
        "Delegation was unavailable. Continue the task using your own reasoning "
        "and tools."
    )

    def __init__(
        self,
        *,
        run_id: str,
        schedule: AssignmentSchedule,
        trace: Trace,
    ) -> None:
        self.run_id = run_id
        self.schedule = schedule
        self.trace = trace
        self.decided = False
        self.assignment: Assignment | None = None
        self.proposal_count = 0

    def intervene(
        self,
        objective: str,
        *,
        terminal: bool,
        run_worker: Callable[[str], str],
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
            return InterventionResult(
                proposal_id=proposal_id,
                eligible=False,
                assignment=None,
                observation=(
                    "This delegation was not eligible. Continue without a worker."
                ),
                worker_completed=False,
            )

        self.decided = True
        self.assignment = self.schedule.reveal(self.run_id)
        self.trace.emit(
            "assignment_revealed",
            proposal_id=proposal_id,
            assignment=self.assignment,
        )
        if self.assignment is Assignment.SUPPRESS:
            self.trace.emit("delegation_suppressed", proposal_id=proposal_id)
            return InterventionResult(
                proposal_id=proposal_id,
                eligible=True,
                assignment=self.assignment,
                observation=self.SUPPRESSION_OBSERVATION,
                worker_completed=False,
            )

        observation = run_worker(objective)
        self.trace.emit("worker_completed", proposal_id=proposal_id)
        return InterventionResult(
            proposal_id=proposal_id,
            eligible=True,
            assignment=self.assignment,
            observation=observation,
            worker_completed=True,
        )
