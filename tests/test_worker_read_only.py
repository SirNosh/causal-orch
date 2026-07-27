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

    def read_only_tool_schemas(self):
        return [schema("read")]

    def execute_read_tool(self, name, arguments):
        assert name == "read"
        self.calls.append(name)
        return "evidence", "tool-result-1"

    def state_hash(self):
        return "unchanged"


def test_worker_can_use_read_tool_and_return_cited_result():
    world = WorkerWorld()
    model = ScriptedModelClient(
        [
            ("read", {}),
            (
                "return_worker_result",
                {"result": "The answer is 44.", "evidence": ["tool-result-1"]},
            ),
        ]
    )
    worker = ReadOnlyWorker(model=model, world=world, trace=Trace("run"))

    result = worker.run(
        original_task="answer",
        objective="research",
        scratchpad=[],
        observations=[],
    )

    assert result.result == "The answer is 44."
    assert result.evidence == ("tool-result-1",)
    assert world.calls == ["read"]


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
