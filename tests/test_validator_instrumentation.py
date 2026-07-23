import unittest
from dataclasses import dataclass, field

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
class FakeValidatorEvent:
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


if __name__ == "__main__":
    unittest.main()
