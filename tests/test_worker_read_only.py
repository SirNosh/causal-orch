import pytest

from causal_orch.model_client import ScriptedModelClient
from causal_orch.trace import Trace
from causal_orch.worker import ReadOnlyWorker


def schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


class WorkerWorld:
    def __init__(self):
        self.calls = []
        self.paused = False
        self.time_offsets = []

    def read_only_tool_schemas(self):
        return [schema("read")]

    def execute_read_tool(self, name, arguments):
        assert self.paused
        assert name == "read"
        self.calls.append(name)
        return "evidence", "tool-result-1"

    def state_hash(self):
        assert self.paused
        return "unchanged"

    def pause_time(self):
        assert not self.paused
        self.paused = True

    def resume_time(self, fixed_offset_seconds):
        assert self.paused
        self.paused = False
        self.time_offsets.append(fixed_offset_seconds)


def test_worker_can_use_read_tool_and_return_cited_result():
    world = WorkerWorld()
    trace = Trace("run")
    model = ScriptedModelClient(
        [
            ("read", {}),
            (
                "return_worker_result",
                {"result": "The answer is 44.", "evidence": ["tool-result-1"]},
            ),
        ]
    )
    worker = ReadOnlyWorker(model=model, world=world, trace=trace)

    result = worker.run(
        original_task="answer",
        objective="research",
        scratchpad=[],
        observations=[],
    )

    assert result.result == "The answer is 44."
    assert result.evidence == ("tool-result-1",)
    assert world.calls == ["read"]
    assert world.time_offsets == [10.0]
    assert model.max_tokens_seen == [2_000, 2_000]
    responses = [
        event for event in trace.events if event["event"] == "model_response"
    ]
    validations = [
        event
        for event in trace.events
        if event["event"] == "action_validation"
    ]
    executed = [
        event for event in trace.events if event["event"] == "action_executed"
    ]
    assert len(responses) == len(validations) == len(executed) == 2
    assert [
        event["payload"]["model_call_id"] for event in responses
    ] == [event["payload"]["model_call_id"] for event in validations]
    assert [
        event["payload"]["model_call_id"] for event in responses
    ] == [event["payload"]["model_call_id"] for event in executed]


def test_worker_cannot_name_a_write_tool():
    worker = ReadOnlyWorker(
        model=ScriptedModelClient([("write", {})]),
        world=WorkerWorld(),
        trace=Trace("run"),
    )

    with pytest.raises(ValueError, match="unavailable tool"):
        worker.run(
            original_task="answer",
            objective="research",
            scratchpad=[],
            observations=[],
        )


def test_worker_state_mutation_is_a_treatment_failure():
    class MutatingWorld(WorkerWorld):
        def __init__(self):
            super().__init__()
            self.changed = False

        def execute_read_tool(self, name, arguments):
            self.changed = True
            return super().execute_read_tool(name, arguments)

        def state_hash(self):
            return "after" if self.changed else "before"

    worker = ReadOnlyWorker(
        model=ScriptedModelClient(
            [
                ("read", {}),
                (
                    "return_worker_result",
                    {"result": "finding", "evidence": ["tool-result-1"]},
                ),
            ]
        ),
        world=MutatingWorld(),
        trace=Trace("run"),
    )

    with pytest.raises(RuntimeError, match="changed Gaia2 application state"):
        worker.run(
            original_task="answer",
            objective="research",
            scratchpad=[],
            observations=[],
        )
