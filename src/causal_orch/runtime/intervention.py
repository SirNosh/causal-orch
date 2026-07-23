"""Eligibility-gated delegation intervention runtime."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Callable, Iterable, Mapping

from causal_orch.agent.schemas import (
    DelegationProposal,
    EligibilityDecision,
    EligibilityStatus,
    RejectionReason,
    validate_proposal,
)
from causal_orch.runtime.randomization import AssignmentSchedule, TreatmentAssignment
from causal_orch.tracing.events import EventName, OrchestrationEvent


CONTROL_RESPONSE = {
    "status": "DELEGATION_UNAVAILABLE",
    "reason": "EXPERIMENTAL_CONTROL",
}


class DelegationInterventionGate:
    """Reveal one concealed assignment, but only after proposal eligibility."""

    def __init__(
        self,
        assignment_schedule: AssignmentSchedule,
        block_key: Any,
        trace_sink: Any,
        worker_callback: Callable[[Mapping[str, Any]], Any],
        *,
        available_context_refs: Iterable[str] = (),
        allowed_worker_tools: Iterable[str] = (),
        prior_objectives: Iterable[str] = (),
        objective_completed: bool = False,
        terminal: bool = False,
        oracle_refs: Iterable[str] = (),
        scenario_in_scope: bool = True,
    ) -> None:
        self.assignment_schedule = assignment_schedule
        self.block_key = block_key
        self.trace_sink = trace_sink
        self.worker_callback = worker_callback
        self.available_context_refs = tuple(available_context_refs)
        self.allowed_worker_tools = tuple(allowed_worker_tools)
        self.prior_objectives = tuple(prior_objectives)
        self.objective_completed = objective_completed
        self.terminal = terminal
        self.oracle_refs = tuple(oracle_refs)
        self.scenario_in_scope = scenario_in_scope
        self._assignment: TreatmentAssignment | None = None
        self._decided = False

    @property
    def assignment(self) -> TreatmentAssignment | None:
        """The assignment, which remains hidden until a decision is made."""
        return self._assignment

    @property
    def decided(self) -> bool:
        return self._decided

    def _proposal_snapshot(self, proposal: Any) -> Any:
        if isinstance(proposal, DelegationProposal):
            return proposal.to_dict()
        if isinstance(proposal, Mapping):
            snapshot = dict(proposal)
            if all(isinstance(key, str) for key in snapshot):
                try:
                    json.dumps(snapshot)
                except (TypeError, ValueError):
                    return repr(proposal)
                return deepcopy(snapshot)
        return repr(proposal)

    def _emit(self, event_type: EventName, **fields: Any) -> None:
        self.trace_sink.append(OrchestrationEvent(event_type=event_type, **fields))

    def _record_proposal(self, proposal: Any) -> None:
        self._emit(
            EventName.ORCHESTRATOR_PROPOSAL,
            proposed_action="DELEGATE",
            payload={"proposal": self._proposal_snapshot(proposal)},
        )

    def _record_validation(self, proposal: Any, decision: EligibilityDecision) -> None:
        self._emit(
            EventName.PROPOSAL_VALIDATION,
            proposed_action="DELEGATE",
            eligibility=decision.status.value,
            eligibility_reason=decision.reason,
            payload={"proposal": self._proposal_snapshot(proposal)},
        )

    def _record_eligibility(self, proposal: Any, decision: EligibilityDecision) -> None:
        self._emit(
            EventName.INTERVENTION_ELIGIBILITY,
            proposed_action="DELEGATE",
            eligibility=decision.status.value,
            eligibility_reason=decision.reason,
            payload={"proposal": self._proposal_snapshot(proposal)},
        )

    def handle_proposal(
        self,
        proposal: DelegationProposal | Mapping[str, Any] | Any,
        *,
        available_context_refs: Iterable[str] | None = None,
        allowed_worker_tools: Iterable[str] | None = None,
        prior_objectives: Iterable[str] | None = None,
        objective_completed: bool | None = None,
        terminal: bool | None = None,
        oracle_refs: Iterable[str] | None = None,
        scenario_in_scope: bool | None = None,
    ) -> dict[str, Any]:
        """Validate and, for the first eligible proposal, execute the intervention."""
        self._record_proposal(proposal)

        if self._decided:
            decision = EligibilityDecision(
                EligibilityStatus.REJECTED,
                RejectionReason.DELEGATION_ALREADY_DECIDED,
            )
            self._record_validation(proposal, decision)
            self._record_eligibility(proposal, decision)
            return {"status": RejectionReason.DELEGATION_ALREADY_DECIDED.value}

        decision = validate_proposal(
            proposal,
            available_context_refs=(
                self.available_context_refs if available_context_refs is None else available_context_refs
            ),
            allowed_worker_tools=(
                self.allowed_worker_tools if allowed_worker_tools is None else allowed_worker_tools
            ),
            prior_objectives=(
                self.prior_objectives if prior_objectives is None else prior_objectives
            ),
            objective_completed=(
                self.objective_completed if objective_completed is None else objective_completed
            ),
            terminal=self.terminal if terminal is None else terminal,
            oracle_refs=self.oracle_refs if oracle_refs is None else oracle_refs,
            scenario_in_scope=(
                self.scenario_in_scope if scenario_in_scope is None else scenario_in_scope
            ),
        )
        self._record_validation(proposal, decision)
        self._record_eligibility(proposal, decision)
        if not decision.eligible:
            return {"status": "DELEGATION_REJECTED", "reason": decision.reason}

        assignment = self.assignment_schedule.reveal(self.block_key, eligible=True)
        assert assignment is not None
        self._assignment = assignment
        self._decided = True
        self._emit(
            EventName.INTERVENTION_ASSIGNMENT,
            proposed_action="DELEGATE",
            executed_action=assignment.label,
            eligibility=decision.status.value,
            treatment_assignment=assignment.label,
            assignment_probability=0.5,
        )

        serialized_proposal = (
            proposal.to_dict() if isinstance(proposal, DelegationProposal) else deepcopy(dict(proposal))
        )
        if assignment is TreatmentAssignment.SUPPRESS:
            self._emit(
                EventName.DELEGATION_SUPPRESSED,
                proposed_action="DELEGATE",
                executed_action="SUPPRESS",
            )
            return dict(CONTROL_RESPONSE)

        artifact = self.worker_callback(serialized_proposal)
        if hasattr(artifact, "to_dict"):
            artifact = artifact.to_dict()
        self._emit(
            EventName.DELEGATION_EXECUTED,
            proposed_action="DELEGATE",
            executed_action="EXECUTE",
        )
        return {"status": "DELEGATION_EXECUTED", "artifact": artifact}
