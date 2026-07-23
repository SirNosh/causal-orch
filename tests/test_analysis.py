import unittest

from causal_orch.analysis.flow import FlowTable
from causal_orch.analysis.observational import adjusted_proposal_association, proposal_association
from causal_orch.analysis.randomization_inference import (
    assignment_balance,
    estimate_primary,
    randomization_inference_p_value,
)


class AnalysisTests(unittest.TestCase):
    def test_flow_counts(self) -> None:
        table = FlowTable.from_rows([
            {
                "run_id": "a", "scheduled": True, "started": True, "initialized": True,
                "proposed": True, "eligible": True, "assignment": "treatment",
                "worker_started": True, "worker_completed": True, "completed": True,
                "judged": True, "success": True,
            },
            {
                "run_id": "b", "scheduled": True, "started": True, "initialized": True,
                "proposed": True, "eligible": True, "assignment": "control",
                "completed": True, "judged": True, "success": False,
                "provider_failure": True,
            },
            {"run_id": "c", "scheduled": True, "started": True, "proposed": False},
        ])
        self.assertEqual(table.counts["runs_scheduled"], 3)
        self.assertEqual(table.counts["valid_eligible_proposals"], 2)
        self.assertEqual(table.counts["treatment_assigned"], 1)
        self.assertEqual(table.counts["control_assigned"], 1)
        self.assertEqual(table.counts["final_analysis_sample"], 2)

    def test_risk_difference_and_assignment_balance(self) -> None:
        rows = [
            {"block": "s1", "eligible": True, "assignment": "treatment", "success": True},
            {"block": "s1", "eligible": True, "assignment": "treatment", "success": False},
            {"block": "s1", "eligible": True, "assignment": "control", "success": False},
            {"block": "s1", "eligible": True, "assignment": "control", "success": False},
        ]
        estimate = estimate_primary(rows)
        self.assertEqual((estimate.treatment_count, estimate.control_count), (2, 2))
        self.assertEqual(estimate.absolute_risk_difference, 0.5)
        self.assertEqual(assignment_balance(rows).by_block["s1"], (2, 2))

    def test_exact_randomization_p_value(self) -> None:
        rows = [
            {"block": "s1", "assignment": "treatment", "success": True},
            {"block": "s1", "assignment": "treatment", "success": True},
            {"block": "s1", "assignment": "control", "success": False},
            {"block": "s1", "assignment": "control", "success": False},
        ]
        self.assertEqual(randomization_inference_p_value(rows), 1 / 3)

    def test_monte_carlo_requires_and_uses_seed(self) -> None:
        rows = [
            {"block": "s", "assignment": "treatment", "success": True},
            {"block": "s", "assignment": "treatment", "success": False},
            {"block": "s", "assignment": "control", "success": False},
            {"block": "s", "assignment": "control", "success": True},
        ] * 4
        first = estimate_primary(rows, max_exact_assignments=1, monte_carlo_samples=100, seed="fixed")
        second = estimate_primary(rows, max_exact_assignments=1, monte_carlo_samples=100, seed="fixed")
        self.assertEqual(first.p_value_method, "seeded_monte_carlo")
        self.assertEqual(first.p_value, second.p_value)
        with self.assertRaises(ValueError):
            estimate_primary(rows, max_exact_assignments=1, monte_carlo_samples=10)

    def test_observational_label_is_not_randomized_label(self) -> None:
        rows = [
            {"proposed": True, "success": False, "capability": "a"},
            {"proposed": False, "success": True, "capability": "a"},
        ]
        association = proposal_association(rows)
        self.assertFalse(association.causal)
        self.assertTrue(association.estimand.startswith("observational_"))
        with self.assertRaises(ValueError):
            adjusted_proposal_association(rows, ["worker_artifact"])


if __name__ == "__main__":
    unittest.main()

