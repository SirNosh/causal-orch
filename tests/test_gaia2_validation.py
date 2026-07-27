from causal_orch.gaia2_adapter import Gaia2Adapter
from causal_orch.experiment import run_one
from causal_orch.intervention import Assignment, FixedAssignment
from causal_orch.model_client import ScriptedModelClient


class Result:
    success = True


class Scenario:
    def __init__(self):
        self.calls = 0

    def validate(self, environment):
        self.calls += 1
        return Result()


def test_gaia2_native_validation_is_called_once_and_kept_binary():
    adapter = object.__new__(Gaia2Adapter)
    adapter.scenario = Scenario()
    adapter.environment = object()

    success, result = adapter.validate()

    assert success is True
    assert isinstance(result, Result)
    assert adapter.scenario.calls == 1


class RunWorld:
    task = "Answer exactly"

    def __init__(self, _):
        self.scenario = type("ScenarioRecord", (), {"scenario_id": "gaia"})()
        self._started = False

    def start(self):
        self._started = True

    def tool_schemas(self):
        return []

    def read_only_tool_schemas(self):
        return []

    def notifications(self):
        return []

    def state_hash(self):
        return "stable"

    def finish(self, answer):
        self.answer = answer

    def validate(self):
        return True, Result()

    def close(self):
        self._started = False


def test_run_reaches_native_validation_after_final_action():
    outcome = run_one(
        scenario_path="scenario.json",
        model=ScriptedModelClient(
            [("final_answer", {"answer": "44"})]
        ),
        schedule=FixedAssignment(Assignment.SUPPRESS),
        adapter_factory=RunWorld,
    )

    assert outcome.reached_validation
    assert outcome.gaia2_success is True
    assert outcome.answer == "44"
    assert outcome.error is None
