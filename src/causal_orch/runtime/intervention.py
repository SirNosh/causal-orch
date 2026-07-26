"""Eligibility-gated delegation intervention runtime."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Callable, Iterable, Mapping
from uuid import NAMESPACE_URL, uuid5

from causal_orch.agent.schemas import (
    DelegationProposal,
    EligibilityDecision,
    EligibilityStatus,
    RejectionReason,
    validate_proposal,
)
from causal_orch.runtime.randomization import AssignmentSchedule, TreatmentAssignment
from causal_orch.tracing.events import EventName, OrchestrationEvent
from causal_orch.runtime.state_guard import canonical_json


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
        eligibility_context_provider: Callable[[], Mapping[str, Any]] | None = None,
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
        self.eligibility_context_provider = eligibility_context_provider
        self._initial_eligibility_context: dict[str, Any] = {
            "available_context_refs": self.available_context_refs,
            "allowed_worker_tools": self.allowed_worker_tools,
            "prior_objectives": self.prior_objectives,
            "prior_decision": False,
            "objective_completed": self.objective_completed,
            "terminal": self.terminal,
            "oracle_refs": self.oracle_refs,
            "scenario_in_scope": self.scenario_in_scope,
        }
        self.current_eligibility_context = dict(self._initial_eligibility_context)
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

    @staticmethod
    def _normalize_proposal(proposal: Any) -> Any:
        if not isinstance(proposal, Mapping) or "proposal_id" in proposal:
            return proposal
        candidate = dict(proposal)
        identity = canonical_json(candidate)
        candidate["proposal_id"] = str(uuid5(NAMESPACE_URL, f"causal-orch/delegate:{identity}"))
        return candidate

    def _refresh_eligibility_context(self) -> dict[str, Any]:
        current = dict(self._initial_eligibility_context)
        if self.eligibility_context_provider is not None:
            provided = self.eligibility_context_provider()
            if not isinstance(provided, Mapping):
                raise TypeError("eligibility_context_provider must return a mapping")
            current.update(provided)
            if "allowed_worker_tools" not in provided and "allowed_read_tools" in provided:
                current["allowed_worker_tools"] = provided["allowed_read_tools"]
            if "prior_decision" not in provided and "prior_delegation_state" in provided:
                current["prior_decision"] = bool(provided["prior_delegation_state"])
        self.current_eligibility_context = current
        self.available_context_refs = tuple(current.get("available_context_refs", ()))
        self.allowed_worker_tools = tuple(current.get("allowed_worker_tools", ()))
        self.prior_objectives = tuple(current.get("prior_objectives", ()))
        self.objective_completed = bool(current.get("objective_completed", False))
        self.terminal = bool(current.get("terminal", False))
        self.oracle_refs = tuple(current.get("oracle_refs", ()))
        self.scenario_in_scope = bool(current.get("scenario_in_scope", True))
        return current

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
        try:
            normalized_proposal = self._normalize_proposal(proposal)
        except Exception as exc:
            normalized_proposal = proposal
            decision = EligibilityDecision(EligibilityStatus.REJECTED, RejectionReason.INVALID_SCHEMA)
            self._record_proposal(proposal)
            self._record_validation(proposal, decision)
            self._record_eligibility(proposal, decision)
            return {
                "status": "DELEGATION_REJECTED",
                "reason": RejectionReason.INVALID_SCHEMA.value,
                "detail": f"proposal normalization failed: {type(exc).__name__}: {exc}",
            }
        self._record_proposal(normalized_proposal)

        if self._decided:
            decision = EligibilityDecision(
                EligibilityStatus.REJECTED,
                RejectionReason.DELEGATION_ALREADY_DECIDED,
            )
            self._record_validation(normalized_proposal, decision)
            self._record_eligibility(normalized_proposal, decision)
            return {"status": RejectionReason.DELEGATION_ALREADY_DECIDED.value}

        try:
            dynamic = self._refresh_eligibility_context()
        except Exception as exc:
            decision = EligibilityDecision(EligibilityStatus.REJECTED, RejectionReason.INVALID_SCHEMA)
            self._record_validation(normalized_proposal, decision)
            self._record_eligibility(normalized_proposal, decision)
            return {
                "status": "DELEGATION_REJECTED",
                "reason": RejectionReason.INVALID_SCHEMA.value,
                "detail": f"eligibility context unavailable: {type(exc).__name__}: {exc}",
            }

        decision = validate_proposal(
            normalized_proposal,
            available_context_refs=(
                dynamic["available_context_refs"] if available_context_refs is None else available_context_refs
            ),
            allowed_worker_tools=(
                dynamic.get("allowed_worker_tools", dynamic.get("allowed_read_tools", ()))
                if allowed_worker_tools is None else allowed_worker_tools
            ),
            prior_objectives=(
                dynamic["prior_objectives"] if prior_objectives is None else prior_objectives
            ),
            objective_completed=(
                dynamic["objective_completed"] if objective_completed is None else objective_completed
            ),
            terminal=dynamic["terminal"] if terminal is None else terminal,
            oracle_refs=dynamic["oracle_refs"] if oracle_refs is None else oracle_refs,
            scenario_in_scope=(
                dynamic["scenario_in_scope"] if scenario_in_scope is None else scenario_in_scope
            ),
            prior_decision=bool(dynamic.get("prior_decision", False)),
        )
        self._record_validation(normalized_proposal, decision)
        self._record_eligibility(normalized_proposal, decision)
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
            normalized_proposal.to_dict()
            if isinstance(normalized_proposal, DelegationProposal)
            else deepcopy(dict(normalized_proposal))
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
        else:
            try:
                artifact = json.loads(json.dumps(artifact, sort_keys=True, separators=(",", ":")))
            except (TypeError, ValueError):
                artifact = {
                    "status": "TREATMENT_FAILURE",
                    "reason": "RUNNER_ERROR",
                    "detail": "worker returned a non-JSON-safe result",
                }
        self._emit(
            EventName.DELEGATION_EXECUTED,
            proposed_action="DELEGATE",
            executed_action="EXECUTE",
        )
        response_key = (
            "worker_episode"
            if isinstance(artifact, Mapping) and "episode" in artifact
            else "artifact"
        )
        return {"status": "DELEGATION_EXECUTED", response_key: artifact}
