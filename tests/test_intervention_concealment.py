import json
import unittest
from uuid import uuid4

from are.simulation.agents.agent_log import ObservationLog, ToolCallLog
from are.simulation.agents.default_agent.tools.action_executor import ParsedAction

from causal_orch.agent.action_executor import InterventionActionExecutor
from causal_orch.agent.schemas import DelegationProposal
from causal_orch.runtime.intervention import CONTROL_RESPONSE, DelegationInterventionGate
from causal_orch.runtime.randomization import TreatmentAssignment, generate_balanced_schedule
from causal_orch.tracing.events import EventName
from causal_orch.tracing.sink import InMemoryTraceSink


def proposal(objective: str = "Find the relevant record.") -> DelegationProposal:
    return DelegationProposal(
        proposal_id=str(uuid4()),
        objective=objective,
        reason_code="INFORMATION_GAP",
        context_refs=("task",),
        allowed_read_tools=("read_file",),
        completion_criterion="Name the record and cite it.",
    )


def gate(seed: str = "seed", *, worker_callback=lambda value: {"value": value}):
    sink = InMemoryTraceSink()
    intervention = DelegationInterventionGate(
        generate_balanced_schedule(seed, ["block"], block_size=2),
        "block",
        sink,
        worker_callback,
        available_context_refs={"task"},
        allowed_worker_tools={"read_file"},
    )
    return intervention, sink


class InterventionTests(unittest.TestCase):
    def test_model_proposal_cannot_choose_worker_budget(self):
        raw = proposal().to_dict()
        self.assertNotIn("max_worker_steps", raw)
        self.assertNotIn("max_worker_output_tokens", raw)
        raw["max_worker_steps"] = 1
        result, _sink = gate()
        rejected = result.handle_proposal(raw)
        self.assertEqual(rejected["reason"], "INVALID_SCHEMA")

    def test_missing_proposal_id_is_deterministically_normalized_at_runtime(self):
        calls = []
        intervention, _ = gate(worker_callback=calls.append)
        raw = proposal().to_dict()
        raw.pop("proposal_id")
        first = intervention._normalize_proposal(raw)
        second = intervention._normalize_proposal(raw)
        self.assertEqual(first["proposal_id"], second["proposal_id"])
        self.assertEqual(len(first["proposal_id"]), 36)

    def test_eligibility_context_is_refreshed_at_decision_time(self):
        current = {
            "available_context_refs": set(),
            "allowed_worker_tools": set(),
            "prior_objectives": set(),
            "objective_completed": False,
            "terminal": False,
            "oracle_refs": set(),
            "scenario_in_scope": True,
        }
        sink = InMemoryTraceSink()
        intervention = DelegationInterventionGate(
            generate_balanced_schedule("dynamic", ["block"], block_size=2),
            "block",
            sink,
            lambda value: {"proposal": value},
            eligibility_context_provider=lambda: current,
        )
        self.assertEqual(
            intervention.handle_proposal(proposal()).get("reason"),
            "UNKNOWN_CONTEXT_REFERENCE",
        )
        self.assertEqual(intervention.assignment_schedule.consumed_count, 0)
        current["available_context_refs"] = {"task"}
        current["allowed_worker_tools"] = {"read_file"}
        result = intervention.handle_proposal(proposal("A new bounded objective."))
        self.assertIn(result["status"], {"DELEGATION_EXECUTED", "DELEGATION_UNAVAILABLE"})
        self.assertEqual(intervention.assignment_schedule.consumed_count, 1)

    def test_assignment_is_concealed_until_eligibility(self) -> None:
        intervention, _ = gate()
        self.assertIsNone(intervention.assignment)
        result = intervention.handle_proposal(proposal(), available_context_refs={"missing"})
        self.assertEqual(result["reason"], "UNKNOWN_CONTEXT_REFERENCE")
        self.assertIsNone(intervention.assignment)

    def test_invalid_proposal_does_not_consume_assignment(self) -> None:
        intervention, _ = gate()
        intervention.handle_proposal(proposal(), available_context_refs={"missing"})
        self.assertEqual(intervention.assignment_schedule.consumed_count, 0)

    def test_first_eligible_proposal_consumes_exactly_one_assignment(self) -> None:
        intervention, _ = gate()
        intervention.handle_proposal(proposal())
        self.assertEqual(intervention.assignment_schedule.consumed_count, 1)
        self.assertTrue(intervention.decided)

    def test_treatment_passes_exact_serialized_proposal(self) -> None:
        calls = []
        intervention, _ = gate(worker_callback=calls.append)
        raw = proposal().to_dict()
        result = intervention.handle_proposal(raw)
        if intervention.assignment is TreatmentAssignment.EXECUTE:
            self.assertEqual(calls, [raw])
            self.assertEqual(result, {"status": "DELEGATION_EXECUTED", "artifact": None})
        else:
            self.assertEqual(calls, [])
            self.assertEqual(result, CONTROL_RESPONSE)

    def test_control_has_exact_response_and_no_worker(self) -> None:
        for seed in range(20):
            calls = []
            intervention, _ = gate(str(seed), worker_callback=calls.append)
            result = intervention.handle_proposal(proposal())
            if intervention.assignment is TreatmentAssignment.SUPPRESS:
                self.assertEqual(result, CONTROL_RESPONSE)
                self.assertEqual(calls, [])

    def test_later_proposals_are_already_decided(self) -> None:
        calls = []
        intervention, sink = gate(worker_callback=calls.append)
        intervention.handle_proposal(proposal(), available_context_refs={"task"})
        consumed = intervention.assignment_schedule.consumed_count
        self.assertEqual(
            intervention.handle_proposal("malformed"),
            {"status": "DELEGATION_ALREADY_DECIDED"},
        )
        self.assertEqual(intervention.assignment_schedule.consumed_count, consumed)
        self.assertEqual(calls if intervention.assignment is TreatmentAssignment.EXECUTE else [], calls)
        self.assertEqual(
            [event.event_type for event in sink.events[-3:]],
            [
                EventName.ORCHESTRATOR_PROPOSAL,
                EventName.PROPOSAL_VALIDATION,
                EventName.INTERVENTION_ELIGIBILITY,
            ],
        )

    def test_executor_preserves_ordinary_actions_without_registering_delegate(self) -> None:
        sink = InMemoryTraceSink()
        intervention, _ = gate()
        executor = InterventionActionExecutor(intervention_gate=intervention, trace_sink=sink)
        self.assertNotIn("DELEGATE", executor.tools)
        logs = []
        executor.execute_parsed_action(
            ParsedAction(tool_name="_mock", arguments={}, rationale="ordinary"),
            logs.append,
            lambda: 1.0,
            "agent",
        )
        self.assertIsInstance(logs[1], ToolCallLog)
        self.assertIsInstance(logs[-1], ObservationLog)
        self.assertEqual(logs[-1].content, "Mocked observation")

    def test_executor_observation_is_only_serialized_intervention_response(self) -> None:
        sink = InMemoryTraceSink()
        intervention, _ = gate()
        executor = InterventionActionExecutor(intervention_gate=intervention, trace_sink=sink)
        logs = []
        result = executor.execute_parsed_action(
            ParsedAction(tool_name="DELEGATE", arguments=proposal().to_dict()),
            logs.append,
            lambda: 1.0,
            "agent",
        )
        self.assertEqual(json.loads(logs[-1].content), result)


if __name__ == "__main__":
    unittest.main()
