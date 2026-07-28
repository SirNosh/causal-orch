"""Thin Gaia2 environment, tool, state, and validator adapter."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .actions import Notification, NotificationType


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    get_state = getattr(value, "get_state", None)
    if callable(get_state):
        return _safe(get_state())
    return repr(value)


def _argument_schema(arg_type: Any) -> dict[str, Any]:
    text = str(arg_type).lower()
    if "bool" in text:
        return {"type": "boolean"}
    if "int" in text:
        return {"type": "integer"}
    if "float" in text or "number" in text:
        return {"type": "number"}
    if "list" in text or "tuple" in text or "sequence" in text:
        item = {"type": "string"}
        if "int" in text:
            item = {"type": "integer"}
        return {"type": "array", "items": item}
    if "dict" in text or "mapping" in text:
        return {"type": "object"}
    if "str" in text:
        return {"type": "string"}
    return {}


def app_tool_schema(tool: Any) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for arg in tool.args:
        schema = _argument_schema(arg.arg_type)
        if arg.description:
            schema["description"] = arg.description
        properties[arg.name] = schema
        if not arg.has_default:
            required.append(arg.name)
    return {
        "type": "function",
        "function": {
            "name": tool._public_name or tool.name,
            "description": tool._public_description
            or tool.function_description
            or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def worker_tool_exclusion_reason(tool: Any) -> str | None:
    """Return why a Gaia2 tool cannot cross the worker boundary."""

    if getattr(tool, "write_operation", None) is not False:
        return "write_status_not_explicitly_read_only"
    name = str(getattr(tool, "_public_name", None) or tool.name)
    app_name = str(getattr(tool, "app_name", ""))
    class_name = str(getattr(tool, "class_name", ""))
    function = getattr(tool, "function", None)
    function_name = str(
        getattr(function, "__name__", None)
        or getattr(tool, "func_name", "")
        or name.rsplit("__", 1)[-1]
    )
    identity = " ".join((name, app_name, class_name)).lower()
    if any(
        marker in identity
        for marker in (
            "agentuserinterface",
            "notification",
            "reminder",
            "systemapp",
        )
    ):
        return "agent_or_environment_control"
    lowered_function = function_name.lower()
    if any(
        marker in lowered_function
        for marker in (
            "advance_time",
            "execute",
            "pause",
            "resume",
            "run_command",
            "send_message",
            "sleep",
            "wait",
        )
    ):
        return "indirect_side_effect_or_control"
    return None


class Gaia2Adapter:
    """Own one fresh Gaia2 scenario and environment for one run."""

    def __init__(self, scenario_path: str | Path) -> None:
        from are.simulation.data_handler.importer import JsonScenarioImporter
        from are.simulation.environment import Environment
        from are.simulation.scenarios.scenario_imported_from_json.utils import (
            preprocess_scenario,
        )
        from are.simulation.validation.configs import (
            CheckerType,
            ScriptedGraphPerEventJudgeConfig,
            ToolCheckerParam,
        )

        self.scenario_path = Path(scenario_path)
        raw = self.scenario_path.read_text(encoding="utf-8")
        self.task = self._extract_task(json.loads(raw))
        self.scenario, _, _ = JsonScenarioImporter().import_from_json_to_benchmark(
            raw, load_completed_events=False
        )
        checks: dict[str, list[Any]] = {}
        for event in self.scenario.serialized_events:
            if event.class_name != "OracleEvent" or event.action is None:
                continue
            tool_name = f"{event.action.app}__{event.action.function}"
            checks[event.event_id] = [
                ToolCheckerParam(
                    arg_name=arg.name,
                    checker_type=(
                        CheckerType.eq_str_strip_checker
                        if arg.value_type == "str"
                        else CheckerType.eq_checker
                    ),
                    tool_name=tool_name,
                )
                for arg in event.action.args or ()
            ]
        preprocess_scenario(
            self.scenario,
            judge_config=ScriptedGraphPerEventJudgeConfig(
                event_id_to_checker_params=checks
            ),
            offline_validation=True,
        )
        self.environment = Environment()
        self._tools: dict[str, Any] = {}
        self._orchestrator_tools: dict[str, Any] = {}
        self._read_tools: dict[str, Any] = {}
        self._references = 0
        self._started = False
        self._initial_user_message_consumed = False

    @staticmethod
    def _extract_task(data: Mapping[str, Any]) -> str:
        for event in data.get("events", ()):
            action = event.get("action") or {}
            if (
                action.get("app") == "AgentUserInterface"
                and action.get("function") == "send_message_to_agent"
            ):
                for argument in action.get("args") or ():
                    if argument.get("name") == "content":
                        return str(argument["value"])
        raise ValueError("Gaia2 scenario has no initial user task")

    def start(self) -> None:
        if self._started:
            return
        self.environment.run(self.scenario, wait_for_end=False)
        for tool in self.scenario.get_tools():
            name = tool._public_name or tool.name
            if name in self._tools:
                raise ValueError(f"duplicate Gaia2 tool name: {name}")
            self._tools[name] = tool
            if not name.startswith("AgentUserInterface__"):
                self._orchestrator_tools[name] = tool
            if worker_tool_exclusion_reason(tool) is None:
                self._read_tools[name] = tool
            if "AgentUserInterface__send_message_to_user" == name:
                tool.class_instance.wait_for_user_response = False
        self._started = True

    def tool_schemas(self) -> Sequence[Mapping[str, Any]]:
        return [
            app_tool_schema(tool) for tool in self._orchestrator_tools.values()
        ]

    def read_only_tool_schemas(self) -> Sequence[Mapping[str, Any]]:
        return [app_tool_schema(tool) for tool in self._read_tools.values()]

    def _execute(
        self, registry: Mapping[str, Any], name: str, arguments: Mapping[str, Any]
    ) -> tuple[str, str]:
        tool = registry.get(name)
        if tool is None:
            raise ValueError(f"tool is unavailable in this registry: {name}")
        result = tool(**dict(arguments))
        self._references += 1
        reference = f"tool-result-{self._references}"
        return json.dumps(_safe(result), ensure_ascii=False), reference

    def execute_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> tuple[str, str]:
        return self._execute(self._orchestrator_tools, name, arguments)

    def execute_read_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> tuple[str, str]:
        return self._execute(self._read_tools, name, arguments)

    def pause_time(self) -> None:
        from are.simulation.types import EnvironmentState

        if self.environment.state is not EnvironmentState.RUNNING:
            raise RuntimeError(
                f"cannot pause Gaia2 environment in state "
                f"{self.environment.state}"
            )
        self.environment.pause()
        if self.environment.state is not EnvironmentState.PAUSED:
            raise RuntimeError("Gaia2 environment did not pause")

    def resume_time(self, fixed_offset_seconds: float) -> None:
        from are.simulation.types import EnvironmentState

        if fixed_offset_seconds < 0:
            raise ValueError("fixed model-time offset cannot be negative")
        if self.environment.state is not EnvironmentState.PAUSED:
            raise RuntimeError(
                f"cannot resume Gaia2 environment in state "
                f"{self.environment.state}"
            )
        self.environment.resume_with_offset(fixed_offset_seconds)

    def notifications(self) -> Sequence[Notification]:
        from datetime import datetime, timezone
        from are.simulation.notification_system import MessageType

        messages = self.environment.notification_system.message_queue.get_by_timestamp(
            datetime.fromtimestamp(
                self.environment.time_manager.time(), tz=timezone.utc
            )
        )
        type_map = {
            MessageType.USER_MESSAGE: NotificationType.USER_MESSAGE,
            MessageType.ENVIRONMENT_NOTIFICATION: (
                NotificationType.ENVIRONMENT_NOTIFICATION
            ),
            MessageType.ENVIRONMENT_STOP: NotificationType.ENVIRONMENT_STOP,
        }
        result = []
        for message in messages:
            if (
                message.message_type is MessageType.USER_MESSAGE
                and not self._initial_user_message_consumed
                and message.message == self.task
            ):
                self._initial_user_message_consumed = True
                continue
            result.append(
                Notification(
                    type=type_map[message.message_type],
                    content=message.message,
                )
            )
        return result

    def finish(self, answer: str) -> None:
        name = "AgentUserInterface__send_message_to_user"
        if name not in self._tools:
            raise RuntimeError(f"Gaia2 final-answer tool is missing: {name}")
        self._tools[name](content=answer)

    def state_hash(self) -> str:
        encoded = json.dumps(
            _safe(self.environment.get_apps_state()),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> tuple[bool, Any]:
        result = self.scenario.validate(self.environment)
        success = getattr(result, "success", None)
        if not isinstance(success, bool):
            raise TypeError("Gaia2 validation did not return a binary success value")
        return success, result

    def close(self) -> None:
        if self._started:
            self.environment.stop()
            self._started = False
