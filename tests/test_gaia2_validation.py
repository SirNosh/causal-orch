from types import SimpleNamespace

from causal_orch.actions import NotificationType
from causal_orch.gaia2_adapter import (
    Gaia2Adapter,
    worker_tool_exclusion_reason,
)
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
        self.paused = False

    def start(self):
        self._started = True

    def tool_schemas(self):
        return []

    def read_only_tool_schemas(self):
        return []

    def notifications(self):
        return []

    def pause_time(self):
        self.paused = True

    def resume_time(self, fixed_offset_seconds):
        self.paused = False

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


def test_worker_registry_rejects_write_unknown_and_control_tools():
    def tool(name, write_operation, app_name="Emails"):
        return SimpleNamespace(
            name=name,
            _public_name=name,
            app_name=app_name,
            class_name=app_name,
            function=None,
            func_name=name.rsplit("__", 1)[-1],
            write_operation=write_operation,
        )

    assert worker_tool_exclusion_reason(
        tool("Emails__list_emails", False)
    ) is None
    assert worker_tool_exclusion_reason(
        tool("Emails__delete_email", True)
    ) == "write_status_not_explicitly_read_only"
    assert worker_tool_exclusion_reason(
        tool("Mystery__lookup", None)
    ) == "write_status_not_explicitly_read_only"
    assert worker_tool_exclusion_reason(
        tool(
            "AgentUserInterface__get_messages",
            False,
            "AgentUserInterface",
        )
    ) == "agent_or_environment_control"
    assert worker_tool_exclusion_reason(
        tool("SystemApp__wait_for_notification", False, "SystemApp")
    ) == "agent_or_environment_control"


def test_gaia2_adapter_returns_all_dynamic_notification_types():
    from datetime import datetime, timezone
    from are.simulation.notification_system import Message, MessageType

    messages = [
        Message(MessageType.USER_MESSAGE, "Initial task", datetime.now(timezone.utc)),
        Message(MessageType.USER_MESSAGE, "Follow-up", datetime.now(timezone.utc)),
        Message(
            MessageType.ENVIRONMENT_NOTIFICATION,
            "Incoming event",
            datetime.now(timezone.utc),
        ),
        Message(
            MessageType.ENVIRONMENT_STOP,
            "Stopped",
            datetime.now(timezone.utc),
        ),
    ]
    queue = SimpleNamespace(get_by_timestamp=lambda _: messages)
    adapter = object.__new__(Gaia2Adapter)
    adapter.task = "Initial task"
    adapter._initial_user_message_consumed = False
    adapter.environment = SimpleNamespace(
        time_manager=SimpleNamespace(time=lambda: 0),
        notification_system=SimpleNamespace(message_queue=queue),
    )

    notices = adapter.notifications()

    assert [notice.type for notice in notices] == [
        NotificationType.USER_MESSAGE,
        NotificationType.ENVIRONMENT_NOTIFICATION,
        NotificationType.ENVIRONMENT_STOP,
    ]
    assert [notice.content for notice in notices] == [
        "Follow-up",
        "Incoming event",
        "Stopped",
    ]
