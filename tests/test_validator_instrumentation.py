import unittest
from dataclasses import dataclass, field

from are.simulation.types import ValidationException

from causal_orch.environment.instrumented_environment import InstrumentedEnvironment
from causal_orch.evaluation.gaia2 import evaluate_gaia2_scenario
from causal_orch.evaluation.validator_metrics import (
    ValidatorRecord,
    aggregate_validator_milestones,
)


@dataclass
class FakeValidator:
    milestones: list[object] = field(default_factory=list)
    minefields: list[object] = field(default_factory=list)
    achieved_milestones: list[object] = field(default_factory=list)
    timeout: int | None = None
    _internal_check_count: int = 0


@dataclass
class ValidatorWithoutMinefieldFields:
    milestones: list[object] = field(default_factory=list)
    achieved_milestones: list[object] = field(default_factory=list)


@dataclass
class ValidatorWithExplicitFailureFields:
    milestones: list[object] = field(default_factory=list)
    achieved_milestones: list[object] = field(default_factory=list)
    minefields: list[object] = field(default_factory=list)
    triggered_minefields: list[object] = field(default_factory=list)
    failure_reason: str = ""


@dataclass
class FakeValidatorEvent:
    event_id: str


@dataclass
class WorkerWriteEvent:
    event_id: str
    actor_id: str = "worker-1"
    actor_role: str = "worker"
    provenance: str = "worker"


@dataclass
class FailingEvent:
    event_id: str


class FakeEnvironment:
    def __init__(self):
        self.state = {"value": 0}
        self.agent_action_validators = []

    def get_apps_state(self):
        return {"apps": dict(self.state)}

    def process_event(self, event):
        self.state["value"] += 1
        if isinstance(event, FakeValidatorEvent):
            self.agent_action_validators.append(
                FakeValidator(milestones=["a", "b"], minefields=["mine"], timeout=3)
            )
        if isinstance(event, FailingEvent):
            raise ValidationException("Validation event failed and triggered 2 minefields")
        return ["superclass-result", event.event_id]


class FakeScenario:
    def __init__(self, result):
        self.result = result
        self.seen_env = None

    def validate(self, env):
        self.seen_env = env
        return self.result


class ValidatorInstrumentationTests(unittest.TestCase):
    def test_validation_passthrough_calls_validate_unchanged(self):
        result = object()
        scenario = FakeScenario(result)
        env = object()

        evaluation = evaluate_gaia2_scenario(scenario, env)

        self.assertIs(evaluation.validation_result, result)
        self.assertIs(scenario.seen_env, env)
        self.assertEqual(evaluation.primary_outcome_label, "native_gaia2_binary_success")
        self.assertIsNone(evaluation.primary_success)

    def test_process_event_preserves_superclass_result_and_captures_state(self):
        base = FakeEnvironment()
        env = InstrumentedEnvironment(
            environment=base,
            run_id="run-1",
            validator_event_types=(FakeValidatorEvent,),
        )

        result = env.process_event(FakeValidatorEvent("event-1"))

        self.assertEqual(result, ["superclass-result", "event-1"])
        record = env.event_records[0]
        self.assertEqual(record.run_id, "run-1")
        self.assertIsNotNone(record.state_before_hash)
        self.assertIsNotNone(record.state_after_hash)
        self.assertNotEqual(record.state_before_hash, record.state_after_hash)
        self.assertEqual(record.actor_role, "environment")
        self.assertFalse(record.worker_caused_write)

    def test_provenance_distinguishes_worker_write_from_environment_state_change(self):
        env = InstrumentedEnvironment(environment=FakeEnvironment())

        env.process_event(WorkerWriteEvent("worker-event"))

        record = env.event_records[0]
        self.assertTrue(record.worker_caused_write)
        self.assertTrue(record.write_violation)

    def test_native_validator_failure_count_is_captured_without_inventing_unknown_values(self):
        env = InstrumentedEnvironment(environment=FakeEnvironment())

        with self.assertRaises(ValidationException):
            env.process_event(FailingEvent("failed-event"))

        failure = env.validator_failures[0]
        self.assertEqual(failure.triggered_minefields, 2)
        self.assertEqual(env.event_records[0].validator_failures[0].triggered_minefields, 2)
        self.assertIsNone(ValidatorRecord.from_validator(
            event_id="unknown", activation_index=0, validator=FakeValidator(), activated=True
        ).triggered_minefields)

    def test_missing_validator_minefield_fields_remain_unknown(self):
        record = ValidatorRecord.from_validator(
            event_id="missing",
            activation_index=0,
            validator=ValidatorWithoutMinefieldFields(milestones=["m"]),
            activated=True,
        )

        self.assertIsNone(record.minefields)
        self.assertIsNone(record.triggered_minefields)
        metric = aggregate_validator_milestones([record])
        self.assertIsNone(metric.minefield_count)
        self.assertIsNone(metric.triggered_minefield_count)

    def test_explicit_validator_failure_fields_are_preserved(self):
        record = ValidatorRecord.from_validator(
            event_id="explicit",
            activation_index=0,
            validator=ValidatorWithExplicitFailureFields(
                milestones=["m"],
                minefields=["a", "b"],
                triggered_minefields=["a"],
                failure_reason="minefield triggered",
            ),
            activated=True,
        )

        self.assertEqual(record.minefields, 2)
        self.assertEqual(record.triggered_minefields, 1)
        self.assertEqual(record.failure_reason, "minefield triggered")

    def test_activation_index_and_validator_state_are_stable(self):
        base = FakeEnvironment()
        env = InstrumentedEnvironment(
            environment=base,
            validator_event_types=(FakeValidatorEvent,),
        )

        env.process_event(FakeValidatorEvent("same-event"))
        base.agent_action_validators[0].achieved_milestones.append("a")
        env.process_event(FakeValidatorEvent("same-event"))

        self.assertEqual([r.validator_id for r in env.validator_records], ["same-event:0", "same-event:1"])
        self.assertEqual(env.validator_records[0].initial_milestones, 2)
        self.assertEqual(env.validator_records[0].achieved_milestones, 1)

    def test_metric_aggregates_activation_milestones_minefields_and_timeouts(self):
        metric = aggregate_validator_milestones(
            [
                ValidatorRecord(
                    event_id="activated",
                    activation_index=0,
                    activated=True,
                    initial_milestones=3,
                    achieved_milestones=2,
                    minefields=2,
                    triggered_minefields=1,
                    timed_out=True,
                ),
                ValidatorRecord(
                    event_id="never",
                    activation_index=0,
                    activated=False,
                    initial_milestones=4,
                    achieved_milestones=0,
                    minefields=1,
                ),
            ]
        )

        self.assertFalse(metric.primary_outcome)
        self.assertEqual(metric.activated_validator_count, 1)
        self.assertEqual(metric.never_activated_validator_count, 1)
        self.assertEqual(metric.completion_fraction, 2 / 3)
        self.assertEqual(metric.minefield_count, 2)
        self.assertEqual(metric.triggered_minefield_count, 1)
        self.assertEqual(metric.timeout_count, 1)
        self.assertEqual(metric.validator_ids, ("activated:0", "never:0"))

    def test_unknown_triggered_minefields_remain_unknown_in_aggregate(self):
        metric = aggregate_validator_milestones([
            ValidatorRecord(
                event_id="activated",
                activation_index=0,
                activated=True,
                initial_milestones=1,
                achieved_milestones=0,
                minefields=1,
            )
        ])
        self.assertIsNone(metric.triggered_minefield_count)


if __name__ == "__main__":
    unittest.main()
