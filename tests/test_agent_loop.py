import pytest

from causal_orch.actions import Notification, NotificationType
from causal_orch.agent_loop import AgentLoop
from causal_orch.intervention import (
    Assignment,
    AssignmentRecord,
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
        self.paused = False
        self.time_offsets = []
        self.pending_notifications = []

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
        notices = self.pending_notifications
        self.pending_notifications = []
        return notices

    def pause_time(self):
        assert not self.paused
        self.paused = True

    def resume_time(self, fixed_offset_seconds):
        assert self.paused
        self.paused = False
        self.time_offsets.append(fixed_offset_seconds)

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
    assert world.time_offsets == [5.0, 5.0]
    response = next(
        event for event in trace.events if event["event"] == "model_response"
    )
    canonical = next(
        event
        for event in trace.events
        if event["event"] == "action_validation"
    )
    executed = next(
        event
        for event in trace.events
        if event["event"] == "action_executed"
    )
    assert response["payload"]["raw"]["scripted"] is True
    assert canonical["payload"]["status"] == "valid"
    assert canonical["payload"]["canonical_action"]["tool"] == "lookup"
    assert executed["payload"]["action_type"] == "tool"
    assert response["payload"]["model_call_id"] == canonical["payload"][
        "model_call_id"
    ]


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
    assert world.time_offsets == [5.0, 10.0, 5.0]
    assert trace.count("worker_completed") == 1


def test_invalid_action_does_not_consume_assignment():
    class SpySchedule:
        def __init__(self):
            self.reveals = 0

        def reveal(self, run_id):
            self.reveals += 1
            return AssignmentRecord(
                f"{run_id}:assignment:1", Assignment.EXECUTE
            )

    model = ScriptedModelClient([("unknown_action", {})])
    world = World()
    trace = Trace("run")
    schedule = SpySchedule()
    gate = InterventionGate(
        run_id="run", schedule=schedule, trace=trace
    )
    loop = AgentLoop(
        model=model,
        world=world,
        worker=ReadOnlyWorker(model=model, world=world, trace=trace),
        gate=gate,
        trace=trace,
    )

    with pytest.raises(ValueError, match="unknown action tool"):
        loop.run(world.task)

    assert schedule.reveals == 0
    invalid = [
        event
        for event in trace.events
        if event["event"] == "action_validation"
    ]
    assert invalid[0]["payload"]["status"] == "invalid"


def test_user_and_environment_notifications_reach_model_context():
    model = ScriptedModelClient(
        [("final_answer", {"answer": "44"})]
    )
    world = World()
    world.pending_notifications = [
        Notification(NotificationType.USER_MESSAGE, "Follow-up detail"),
        Notification(
            NotificationType.ENVIRONMENT_NOTIFICATION,
            "A new message arrived",
        ),
    ]
    trace = Trace("run")
    loop = AgentLoop(
        model=model,
        world=world,
        worker=ReadOnlyWorker(model=model, world=world, trace=trace),
        gate=InterventionGate(
            run_id="run",
            schedule=FixedAssignment(Assignment.SUPPRESS),
            trace=trace,
        ),
        trace=trace,
    )

    loop.run(world.task)

    contents = [message["content"] for message in model.messages_seen[0]]
    assert "Follow-up detail" in contents
    assert "Environment notification: A new message arrived" in contents
    assert trace.count("notification_delivered") == 2


def test_environment_stop_fails_cleanly_before_model_call():
    model = ScriptedModelClient(
        [("final_answer", {"answer": "unused"})]
    )
    world = World()
    world.pending_notifications = [
        Notification(NotificationType.ENVIRONMENT_STOP, "duration reached")
    ]
    trace = Trace("run")
    loop = AgentLoop(
        model=model,
        world=world,
        worker=ReadOnlyWorker(model=model, world=world, trace=trace),
        gate=InterventionGate(
            run_id="run",
            schedule=FixedAssignment(Assignment.SUPPRESS),
            trace=trace,
        ),
        trace=trace,
    )

    with pytest.raises(RuntimeError, match="environment stopped"):
        loop.run(world.task)

    assert model.call_count == 0
    assert trace.count("notification_delivered") == 1
