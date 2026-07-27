import json
from pathlib import Path

from causal_orch.experiment import run_one
from causal_orch.intervention import Assignment, FixedAssignment
from causal_orch.model_client import ScriptedModelClient


class Validation:
    success = True


class FakeWorld:
    instances = []
    task = "Find the answer"

    def __init__(self, scenario_path):
        self.scenario = type(
            "Scenario", (), {"scenario_id": Path(scenario_path).stem}
        )()
        self._started = False
        self.validation_calls = 0
        self.closed = False
        self.answer = None
        self.paused = False
        self.time_offsets = []
        self.__class__.instances.append(self)

    def start(self):
        self._started = True

    def tool_schemas(self):
        return []

    def read_only_tool_schemas(self):
        return []

    def notifications(self):
        return []

    def pause_time(self):
        assert not self.paused
        self.paused = True

    def resume_time(self, fixed_offset_seconds):
        assert self.paused
        self.paused = False
        self.time_offsets.append(fixed_offset_seconds)

    def state_hash(self):
        return "stable"

    def finish(self, answer):
        self.answer = answer

    def validate(self):
        self.validation_calls += 1
        return True, Validation()

    def close(self):
        self.closed = True
        self._started = False


def test_treatment_model_failure_remains_in_assigned_outcome():
    outcome = run_one(
        scenario_path="gaia.json",
        model=ScriptedModelClient(
            [("delegate", {"objective": "research"})]
        ),
        schedule=FixedAssignment(Assignment.EXECUTE),
        adapter_factory=FakeWorld,
    )

    assert outcome.assignment == "execute"
    assert outcome.assignment_id == f"{outcome.run_id}:assignment:1"
    assert outcome.eligible_delegation
    assert not outcome.worker_completed
    assert outcome.reached_validation
    assert outcome.error is not None


def test_each_run_constructs_and_closes_a_fresh_world():
    FakeWorld.instances.clear()
    for run_id in ("run-1", "run-2"):
        outcome = run_one(
            scenario_path="gaia.json",
            model=ScriptedModelClient(
                [("final_answer", {"answer": "44"})]
            ),
            schedule=FixedAssignment(Assignment.SUPPRESS),
            run_id=run_id,
            adapter_factory=FakeWorld,
        )
        assert outcome.error is None

    assert len(FakeWorld.instances) == 2
    assert FakeWorld.instances[0] is not FakeWorld.instances[1]
    assert all(world.validation_calls == 1 for world in FakeWorld.instances)
    assert all(world.closed for world in FakeWorld.instances)


def test_forced_execute_and_suppress_both_resume_and_validate():
    execute = run_one(
        scenario_path="gaia.json",
        model=ScriptedModelClient(
            [
                (
                    "return_worker_result",
                    {"result": "finding", "evidence": []},
                ),
                ("final_answer", {"answer": "44"}),
            ]
        ),
        schedule=FixedAssignment(Assignment.EXECUTE),
        forced_delegation_objective="Find the relevant fact",
        adapter_factory=FakeWorld,
    )
    suppress = run_one(
        scenario_path="gaia.json",
        model=ScriptedModelClient(
            [("final_answer", {"answer": "44"})]
        ),
        schedule=FixedAssignment(Assignment.SUPPRESS),
        forced_delegation_objective="Find the relevant fact",
        adapter_factory=FakeWorld,
    )

    assert execute.worker_completed
    assert not suppress.worker_completed
    assert execute.answer == suppress.answer == "44"
    assert execute.reached_validation and suppress.reached_validation
    execute_world, suppress_world = FakeWorld.instances[-2:]
    assert execute_world.time_offsets == [5.0, 5.0]
    assert suppress_world.time_offsets == [5.0]


def test_jsonl_preserves_raw_canonical_executed_and_stable_ids(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    outcome = run_one(
        scenario_path="gaia.json",
        model=ScriptedModelClient(
            [("final_answer", {"answer": "44"})]
        ),
        schedule=FixedAssignment(Assignment.SUPPRESS),
        run_id="stable-run",
        attempt_id="7",
        trace_path=trace_path,
        adapter_factory=FakeWorld,
    )
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]

    raw = next(item for item in records if item["event"] == "model_response")
    canonical = next(
        item for item in records if item["event"] == "action_validation"
    )
    executed = next(
        item for item in records if item["event"] == "action_executed"
    )
    assert raw["payload"]["raw"]["scripted"] is True
    assert canonical["payload"]["canonical_action"]["answer"] == "44"
    assert executed["payload"]["action_type"] == "final"
    assert outcome.scenario_id == "gaia"
    assert all(item["scenario_id"] == "gaia" for item in records)
    assert all(item["attempt_id"] == "7" for item in records)
    assert len({item["event_id"] for item in records}) == len(records)
    assert (
        raw["payload"]["model_call_id"]
        == canonical["payload"]["model_call_id"]
        == executed["payload"]["model_call_id"]
    )
