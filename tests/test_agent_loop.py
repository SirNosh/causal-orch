from causal_orch.agent_loop import AgentLoop
from causal_orch.intervention import (
    Assignment,
    FixedAssignment,
    InterventionGate,
)
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


class World:
    task = "Find the answer"

    def __init__(self):
        self.calls = []
        self.answer = None

    def tool_schemas(self):
        return [schema("lookup")]

    def read_only_tool_schemas(self):
        return [schema("lookup")]

    def execute_tool(self, name, arguments):
        self.calls.append(("orchestrator", name))
        return "44", "tool-result-1"

    def execute_read_tool(self, name, arguments):
        self.calls.append(("worker", name))
        return "44", "tool-result-1"

    def state_hash(self):
        return "stable"

    def notifications(self):
        return []

    def finish(self, answer):
        self.answer = answer


def test_tool_call_reaches_world_and_loop_finishes():
    model = ScriptedModelClient(
        [("lookup", {}), ("final_answer", {"answer": "44"})]
    )
    world = World()
    trace = Trace("run")
    gate = InterventionGate(
        run_id="run",
        schedule=FixedAssignment(Assignment.SUPPRESS),
        trace=trace,
    )
    loop = AgentLoop(
        model=model,
        world=world,
        worker=ReadOnlyWorker(model=model, world=world, trace=trace),
        gate=gate,
        trace=trace,
    )

    result = loop.run(world.task)

    assert result.answer == "44"
    assert world.calls == [("orchestrator", "lookup")]
    assert world.answer == "44"


def test_execute_delegation_returns_worker_result_to_same_loop():
    model = ScriptedModelClient(
        [
            ("delegate", {"objective": "Look up the number"}),
            ("lookup", {}),
            (
                "return_worker_result",
                {"result": "It is 44", "evidence": ["tool-result-1"]},
            ),
            ("final_answer", {"answer": "44"}),
        ]
    )
    world = World()
    trace = Trace("run")
    gate = InterventionGate(
        run_id="run",
        schedule=FixedAssignment(Assignment.EXECUTE),
        trace=trace,
    )
    loop = AgentLoop(
        model=model,
        world=world,
        worker=ReadOnlyWorker(model=model, world=world, trace=trace),
        gate=gate,
        trace=trace,
    )

    result = loop.run(world.task)

    assert result.intervention_occurred
    assert result.answer == "44"
    assert world.calls == [("worker", "lookup")]
    assert trace.count("worker_completed") == 1
