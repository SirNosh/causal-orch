import unittest
from uuid import uuid4

from causal_orch.agent.schemas import (
    DelegationProposal,
    EligibilityStatus,
    RejectionReason,
    validate_proposal,
)
from causal_orch.runtime.randomization import TreatmentAssignment, generate_balanced_schedule


def proposal(objective: str = "Find the relevant record.") -> DelegationProposal:
    return DelegationProposal(
        proposal_id=str(uuid4()),
        objective=objective,
        reason_code="INFORMATION_GAP",
        context_refs=("task",),
        allowed_read_tools=("read_file",),
        completion_criterion="Name the record and cite it.",
        max_worker_steps=8,
        max_worker_output_tokens=2000,
    )


class RandomizationTests(unittest.TestCase):
    def test_even_blocks_are_balanced(self) -> None:
        schedule = generate_balanced_schedule("committed-seed", ["a"], block_size=20)
        values = [schedule.reveal("a", eligible=True) for _ in range(20)]
        self.assertEqual(values.count(TreatmentAssignment.EXECUTE), 10)
        self.assertEqual(values.count(TreatmentAssignment.SUPPRESS), 10)

    def test_odd_blocks_differ_by_at_most_one(self) -> None:
        schedule = generate_balanced_schedule("committed-seed", ["a"], block_size=5)
        values = [schedule.reveal("a", eligible=True) for _ in range(5)]
        self.assertLessEqual(abs(values.count(TreatmentAssignment.EXECUTE) - values.count(TreatmentAssignment.SUPPRESS)), 1)

    def test_generation_is_deterministic(self) -> None:
        first = generate_balanced_schedule("seed", ["a", "b"], block_size=8)
        second = generate_balanced_schedule("seed", ["a", "b"], block_size=8)
        first_values = [[first.reveal(key, eligible=True) for _ in range(8)] for key in ("a", "b")]
        second_values = [[second.reveal(key, eligible=True) for _ in range(8)] for key in ("a", "b")]
        self.assertEqual(first_values, second_values)

    def test_invalid_proposal_does_not_consume(self) -> None:
        schedule = generate_balanced_schedule("seed", ["block"], block_size=2)
        decision = validate_proposal(
            proposal(), available_context_refs={"missing"}, allowed_worker_tools={"read_file"}
        )
        self.assertEqual(decision.status, EligibilityStatus.REJECTED)
        self.assertEqual(decision.rejection_reason, RejectionReason.UNKNOWN_CONTEXT_REFERENCE)
        self.assertIsNone(schedule.reveal("block", eligible=decision.eligible))
        self.assertEqual(schedule.consumed_count, 0)

    def test_one_valid_proposal_consumes_one_value(self) -> None:
        schedule = generate_balanced_schedule("seed", ["block"], block_size=4)
        decision = validate_proposal(
            proposal(), available_context_refs={"task"}, allowed_worker_tools={"read_file"}
        )
        self.assertTrue(decision.eligible)
        self.assertIsNotNone(schedule.reveal("block", eligible=decision.eligible))
        self.assertEqual(schedule.consumed_count, 1)


if __name__ == "__main__":
    unittest.main()
