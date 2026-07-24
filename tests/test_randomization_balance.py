import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from causal_orch.agent.schemas import (
    DelegationProposal,
    EligibilityStatus,
    RejectionReason,
    validate_proposal,
)
from causal_orch.runtime.randomization import (
    AssignmentManifest,
    AssignmentSchedule,
    BlockAssignmentManifest,
    TreatmentAssignment,
    derive_block_key,
    generate_balanced_schedule,
)


def proposal(objective: str = "Find the relevant record.") -> DelegationProposal:
    return DelegationProposal(
        proposal_id=str(uuid4()),
        objective=objective,
        reason_code="INFORMATION_GAP",
        context_refs=("task",),
        allowed_read_tools=("read_file",),
        completion_criterion="Name the record and cite it.",
    )


class RandomizationTests(unittest.TestCase):
    def test_block_manifest_preallocates_balanced_immutable_run_assignments(self):
        manifest = BlockAssignmentManifest.create(
            run_ids=("run-1", "run-2", "run-3", "run-4"),
            model_slug="model-1",
            capability="email",
            scenario_id="scenario-1",
            temporal_batch="batch-1",
            seed="seed",
        )
        assignments = [assignment for _run_id, assignment in manifest.run_assignments]
        self.assertEqual(assignments.count(TreatmentAssignment.EXECUTE), 2)
        self.assertEqual(assignments.count(TreatmentAssignment.SUPPRESS), 2)
        schedule = manifest.schedule_for("run-2")
        self.assertIsNone(schedule.reveal(manifest.block_key, eligible=False))
        self.assertEqual(
            schedule.reveal(manifest.block_key, eligible=True),
            manifest.assignment_for("run-2"),
        )
        self.assertEqual(schedule.records[0].run_id, "run-2")
        with self.assertRaises(IndexError):
            schedule.reveal(manifest.block_key, eligible=True)

    def test_manifest_round_trip_and_deterministic_block_derivation(self) -> None:
        manifest = AssignmentManifest.create(
            run_id="run-1",
            model="model-1",
            capability="delegation",
            scenario_id="scenario-1",
            temporal_batch=3,
            seed="seed",
            block_size=4,
        )
        self.assertEqual(
            manifest.block_key,
            derive_block_key("model-1", "delegation", "scenario-1", 3),
        )
        self.assertEqual(AssignmentManifest.from_json(manifest.serialize()), manifest)

    def test_manifest_schedule_consumes_once_and_records_assignment(self) -> None:
        manifest = AssignmentManifest.create(
            run_id="run-1",
            model_slug="model-1",
            capability="delegation",
            scenario_id="scenario-1",
            temporal_batch="batch-1",
            seed="seed",
            block_size=2,
        )
        schedule = generate_balanced_schedule(
            manifest.seed,
            [manifest.block_key],
            manifest.block_size,
            manifest=manifest,
        )
        self.assertIsNone(schedule.reveal(manifest.block_key, eligible=False))
        self.assertEqual(schedule.consumed_count, 0)
        self.assertIsNotNone(schedule.reveal(manifest.block_key, eligible=True))
        self.assertEqual(schedule.consumed_count, 1)
        self.assertEqual(len(schedule.records), 1)
        self.assertEqual(schedule.records[0].run_id, "run-1")
        self.assertEqual(schedule.records[0].position, "run-1:0")
        self.assertEqual(schedule.records[0].manifest_sha256, manifest.manifest_sha256)

    def test_manifest_and_consumed_records_survive_restart(self) -> None:
        manifest = AssignmentManifest.create(
            run_id="run-durable",
            model_slug="model-1",
            capability="delegation",
            scenario_id="scenario-1",
            temporal_batch="batch-1",
            seed="seed",
            block_size=4,
        )
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest_path = directory / "manifest.json"
            commitment_path = directory / "commitments.json"
            manifest.persist(manifest_path)
            self.assertEqual(
                AssignmentManifest.from_json(manifest_path.read_text(encoding="utf-8")),
                manifest,
            )

            first = generate_balanced_schedule(
                manifest.seed,
                [manifest.block_key],
                manifest.block_size,
                manifest=manifest,
                commitment_path=commitment_path,
            )
            self.assertIsNone(first.reveal(manifest.block_key, eligible=False))
            first.reveal(manifest.block_key, eligible=True)

            restarted = AssignmentSchedule.from_manifest(
                manifest, commitment_path=commitment_path
            )
            self.assertEqual(restarted.consumed_count, 1)
            self.assertEqual(len(restarted.records), 1)
            restarted.reveal(manifest.block_key, eligible=True)
            self.assertEqual(restarted.consumed_count, 2)

    def test_repeated_runs_in_one_block_have_distinct_run_positions(self) -> None:
        common = dict(
            model_slug="model-1",
            capability="delegation",
            scenario_id="scenario-1",
            temporal_batch="batch-1",
            seed="seed",
            block_size=2,
        )
        first_manifest = AssignmentManifest.create(run_id="run-1", **common)
        second_manifest = AssignmentManifest.create(run_id="run-2", **common)
        self.assertEqual(first_manifest.block_key, second_manifest.block_key)
        with TemporaryDirectory() as directory:
            first = AssignmentSchedule.from_manifest(
                first_manifest, commitment_path=Path(directory) / "first.json"
            )
            second = AssignmentSchedule.from_manifest(
                second_manifest, commitment_path=Path(directory) / "second.json"
            )
            first.reveal(first_manifest.block_key, eligible=True)
            second.reveal(second_manifest.block_key, eligible=True)

        self.assertNotEqual(first.records[0].position, second.records[0].position)
        self.assertEqual(first.records[0].run_id, "run-1")
        self.assertEqual(second.records[0].run_id, "run-2")

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
