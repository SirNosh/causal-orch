"""Bounded fresh read-only worker adapters."""

from __future__ import annotations

import dataclasses
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping
from uuid import NAMESPACE_URL, uuid4, uuid5

from are.simulation.agents.default_agent.base_agent import BaseAgent, TerminationStep
from are.simulation.agents.default_agent.prompts.system_prompt import (
    DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT,
)
from are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor
from are.simulation.tools import Tool

from causal_orch.runtime.budgets import BudgetExceededError, BudgetExceededResult, WorkerBudgets
from causal_orch.runtime.read_only_tools import (
    MANUALLY_AUDITED_ALLOWLIST,
    ReadOnlyToolSelection,
    select_read_only_tools,
)
from causal_orch.runtime.state_guard import StateGuard, canonical_json
from causal_orch.tracing.events import EventName, OrchestrationEvent

from .action_executor import trace_model_output_rejection
from .schemas import (
    ArtifactStatus,
    Confidence,
    DelegationProposal,
    EvidenceFinding,
    EvidenceReport,
    ValidationError,
    diagnose_evidence_report,
)
from .schemas import MAX_WORKER_OUTPUT_TOKENS, MAX_WORKER_STEPS


STOCK_ARE_REACT_JSON = "STOCK_ARE_REACT_JSON"
NATIVE_TYPED_TOOL_INTERFACE_AUTO = "NATIVE_TYPED_TOOL_INTERFACE_AUTO"
NATIVE_TYPED_TOOL_INTERFACE_REQUIRED = "NATIVE_TYPED_TOOL_INTERFACE_REQUIRED"
NATIVE_TYPED_TOOL_INTERFACE_NAMED_FINALIZER = (
    "NATIVE_TYPED_TOOL_INTERFACE_NAMED_FINALIZER"
)


class TreatmentFailureReason(str, Enum):
    MALFORMED_ARTIFACT = "MALFORMED_ARTIFACT"
    TIMEOUT_OR_BUDGET_EXHAUSTION = "TIMEOUT_OR_BUDGET_EXHAUSTION"
    STATE_DIFF_VIOLATION = "STATE_DIFF_VIOLATION"
    RUNNER_ERROR = "RUNNER_ERROR"


@dataclasses.dataclass(frozen=True)
class TreatmentFailure:
    reason: TreatmentFailureReason
    detail: str

    classification: str = "TREATMENT_FAILURE"

    def to_dict(self) -> dict[str, str]:
        return {
            "classification": self.classification,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclasses.dataclass(frozen=True)
class WorkerResult:
    artifact: EvidenceReport | None
    failure: TreatmentFailure | None
    state_guard: StateGuard
    payload: Mapping[str, Any]

    @property
    def succeeded(self) -> bool:
        return self.artifact is not None and self.failure is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "WORKER_COMPLETED" if self.succeeded else "TREATMENT_FAILURE",
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "failure": self.failure.to_dict() if self.failure is not None else None,
            "state_guard": {
                "before_hash": self.state_guard.before_hash,
                "after_hash": self.state_guard.after_hash,
                "changed": self.state_guard.changed,
            },
            "payload": dict(self.payload),
        }


@dataclasses.dataclass(frozen=True)
class WorkerEpisode:
    """Harness-authored record of one plain-text worker episode."""

    objective: str
    final_text: str
    tool_events: tuple[Mapping[str, str], ...]
    generated_tokens: int
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "status": "COMPLETED",
            "final_text": self.final_text,
            "tool_events": [dict(event) for event in self.tool_events],
            "completion_reason": "MODEL_FINAL_TEXT",
            "token_usage": {"generated": self.generated_tokens},
            "timing": {"wall_seconds": self.wall_seconds},
        }


class ReturnArtifactTool(Tool):
    """ARE-native non-environment tool that validates the worker artifact."""

    name = "return_artifact"
    description = "Return the final structured evidence report and end the worker."
    inputs = {"artifact": {"type": "any", "description": "EvidenceReport JSON object"}}
    output_type = "any"

    def __init__(
        self,
        on_return: Callable[[EvidenceReport], None] | None = None,
        on_validation: Callable[[bool, tuple[str, ...]], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_return = on_return
        self._on_validation = on_validation

    def forward(self, artifact: Any) -> dict[str, Any]:
        try:
            if isinstance(artifact, EvidenceReport):
                validated = artifact
            elif isinstance(artifact, Mapping):
                validated = EvidenceReport.from_dict(artifact)
            else:
                raise ValidationError("artifact must be an EvidenceReport mapping")
        except (TypeError, ValueError):
            if self._on_validation is not None:
                self._on_validation(False, diagnose_evidence_report(artifact))
            raise
        if self._on_validation is not None:
            self._on_validation(True, ())
        if self._on_return is not None:
            self._on_return(validated)
        return validated.to_dict()


def native_return_artifact_tool_schema() -> dict[str, Any]:
    """Build and verify the native tool from the canonical EvidenceReport type."""

    parameters = EvidenceReport.to_json_schema()
    expected_fields = set(EvidenceReport.__dataclass_fields__)
    finding = parameters["properties"]["findings"]["items"]
    expected_finding_fields = set(EvidenceFinding.__dataclass_fields__)
    if (
        set(parameters["properties"]) != expected_fields
        or set(parameters["required"]) != expected_fields
        or parameters.get("additionalProperties") is not False
        or set(finding["properties"]) != expected_finding_fields
        or set(finding["required"]) != expected_finding_fields
        or finding.get("additionalProperties") is not False
        or parameters["properties"]["artifact_type"].get("type") != "string"
        or parameters["properties"]["artifact_type"].get("enum")
        != ["EVIDENCE_REPORT"]
        or parameters["properties"]["status"].get("enum")
        != [status.value for status in ArtifactStatus]
        or finding["properties"]["confidence"].get("enum")
        != [confidence.value for confidence in Confidence]
    ):
        raise RuntimeError("native artifact tool schema drifted from EvidenceReport")
    return {
        "type": "function",
        "function": {
            "name": "return_artifact",
            "description": ReturnArtifactTool.description,
            "parameters": parameters,
        },
    }


def _native_tool_schema(tool: Any) -> dict[str, Any]:
    if getattr(tool, "name", None) == "return_artifact":
        return native_return_artifact_tool_schema()
    inputs = _tool_inputs(tool)
    properties = {}
    required = []
    for name, value in inputs.items():
        field_type = value.get("type")
        properties[name] = {
            **(
                {"type": field_type}
                if field_type
                in {"string", "integer", "number", "boolean", "array", "object"}
                else {}
            ),
            "description": value.get("description", ""),
            **(
                {"default": value["default"]}
                if value.get("has_default")
                else {}
            ),
        }
        if value.get("required"):
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": str(tool.name),
            "description": str(getattr(tool, "description", "")),
            "parameters": {
                "type": "object",
                "required": required,
                "additionalProperties": False,
                "properties": properties,
            },
        },
    }


class _AuditableExecutableTool:
    """Expose ARE AppTool metadata without replacing its callable object."""

    def __init__(self, tool: Any) -> None:
        source = getattr(tool, "app_tool", tool)
        self._tool = tool
        self.public_name = (
            getattr(source, "_public_name", None)
            or getattr(tool, "public_name", None)
            or getattr(tool, "_public_name", None)
            or getattr(tool, "name", None)
        )
        self.name = self.public_name
        self.app_name = getattr(source, "app_name", getattr(tool, "app_name", ""))
        self.function_name = getattr(
            source,
            "func_name",
            getattr(tool, "function_name", getattr(tool, "func_name", "")),
        )
        self.write_operation = getattr(
            source, "write_operation", getattr(tool, "write_operation", None)
        )
        self.description = (
            getattr(source, "_public_description", None)
            or getattr(source, "function_description", None)
            or getattr(tool, "description", "")
        )
        self.argument_schema = getattr(tool, "argument_schema", None)
        self.args = getattr(source, "args", getattr(tool, "args", None))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tool(*args, **kwargs)


class _TracedExecutableTool(Tool):
    """ARE ``Tool`` wrapper that preserves the selected callable and emits tool events."""

    def __init__(
        self,
        tool: Any,
        public_name: str,
        trace_sink: Any,
        actor_id: str,
        evidence_refs: list[str],
    ) -> None:
        self.name = public_name
        self.description = str(getattr(tool, "description", ""))
        self.inputs = _tool_inputs(tool)
        self.output_type = "any"
        self._tool = tool
        self._trace_sink = trace_sink
        self._actor_id = actor_id
        self._evidence_refs = evidence_refs
        super().__init__()

    def _emit(self, event_type: EventName, **fields: Any) -> None:
        self._trace_sink.append(
            OrchestrationEvent(
                event_type=event_type,
                actor_id=self._actor_id,
                actor_role="worker",
                tool_refs=(self.name,),
                **fields,
            )
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        arguments = {"args": list(args), "kwargs": kwargs}
        self._emit(
            EventName.WORKER_TOOL_CALL,
            payload={"tool": self.name, "arguments": _safe_event_value(arguments)},
        )
        try:
            value = self._tool(*args, **kwargs)
        except Exception as exc:
            self._emit(
                EventName.WORKER_TOOL_RESULT,
                error_type=type(exc).__name__,
                payload={"tool": self.name, "ok": False},
            )
            raise
        result_event = OrchestrationEvent(
            event_type=EventName.WORKER_TOOL_RESULT,
            actor_id=self._actor_id,
            actor_role="worker",
            tool_refs=(self.name,),
            payload={"tool": self.name, "ok": True, "result": _safe_event_value(value)},
        )
        self._trace_sink.append(result_event)
        evidence_ref = f"worker_tool_result:{result_event.event_id}"
        self._evidence_refs.append(evidence_ref)
        return {"value": value, "evidence_ref": evidence_ref}


def _tool_inputs(tool: Any) -> dict[str, dict[str, Any]]:
    def field(
        value: Any,
        *,
        required: bool,
        has_default: bool = False,
        default: Any = None,
    ) -> dict[str, Any]:
        raw_type = (
            value.get("type", value.get("arg_type", "any"))
            if isinstance(value, Mapping)
            else "any"
        )
        type_name = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }.get(str(raw_type), str(raw_type))
        return {
            "type": type_name,
            "description": (
                str(value.get("description", ""))
                if isinstance(value, Mapping)
                else ""
            ),
            "required": required,
            "has_default": has_default,
            **({"default": _json_safe(default)} if has_default else {}),
        }

    inputs = getattr(tool, "inputs", None)
    if isinstance(inputs, Mapping):
        return {
            str(name): field(
                value,
                required=not (
                    isinstance(value, Mapping) and "default" in value
                ),
                has_default=isinstance(value, Mapping) and "default" in value,
                default=value.get("default") if isinstance(value, Mapping) else None,
            )
            for name, value in inputs.items()
        }
    schema = getattr(tool, "argument_schema", None)
    if isinstance(schema, Mapping):
        properties = schema.get("properties", {})
        required = set(schema.get("required", ()))
        return {
            str(name): field(
                value,
                required=name in required,
                has_default=isinstance(value, Mapping) and "default" in value,
                default=value.get("default") if isinstance(value, Mapping) else None,
            )
            for name, value in properties.items()
        }
    args = schema if isinstance(schema, list) else getattr(tool, "args", None)
    if not isinstance(args, (list, tuple)):
        return {}
    result = {}
    for arg in args:
        name = get_arg(arg, "name", "arg_name")
        if not isinstance(name, str) or not name:
            continue
        has_default = bool(get_arg(arg, "has_default"))
        result[name] = field(
            {
                "type": get_arg(arg, "type", "arg_type"),
                "description": get_arg(arg, "description") or "",
            },
            required=not has_default,
            has_default=has_default,
            default=get_arg(arg, "default"),
        )
    return result


def _safe_event_value(value: Any) -> Any:
    try:
        return _json_safe(value)
    except TypeError:
        return {"type": type(value).__name__}


@dataclasses.dataclass
class _WorkerBudgetState:
    budgets: WorkerBudgets
    exceeded: BudgetExceededResult | None = None
    output_tokens_used: int = 0

    def consume_output(self, count: Any) -> None:
        if type(count) is not int or count < 0:
            return
        if self.output_tokens_used + count > self.budgets.max_output_tokens:
            self.exceeded = BudgetExceededResult(
                "max_output_tokens", 0, self.output_tokens_used + count
            )
            raise BudgetExceededError(self.exceeded)
        self.output_tokens_used += count


class _BudgetedEngine:
    def __init__(self, engine: Callable[..., Any], state: _WorkerBudgetState) -> None:
        self.engine = engine
        self.state = state

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        response = self.engine(*args, **kwargs)
        metadata = response[1] if isinstance(response, tuple) and len(response) == 2 else {}
        if isinstance(metadata, Mapping):
            self.state.consume_output(metadata.get("completion_tokens"))
        return response


def _normalize_proposal(value: Mapping[str, Any]) -> DelegationProposal:
    if not isinstance(value, Mapping):
        raise ValidationError("proposal must be a mapping")
    candidate = dict(value)
    if "proposal_id" not in candidate:
        identity = json.dumps(
            {key: candidate[key] for key in sorted(candidate)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        candidate["proposal_id"] = str(uuid5(NAMESPACE_URL, f"causal-orch/delegate:{identity}"))
    return DelegationProposal.from_dict(candidate)


def _safe_tool_description(tool: Any, public_name: str) -> dict[str, Any]:
    def get(*names: str, default: Any = None) -> Any:
        if isinstance(tool, Mapping):
            for name in names:
                if name in tool:
                    return tool[name]
            return default
        for name in names:
            if hasattr(tool, name):
                return getattr(tool, name)
        return default

    description = get("description", "_public_description", "function_description", default="")
    schema = get("argument_schema", "args_schema", "schema", "inputs", default=None)
    if schema is None:
        args = get("args", default=None)
        if args is not None:
            schema = [
                {
                    "name": get_arg(arg, "name", "arg_name"),
                    "type": get_arg(arg, "type", "arg_type"),
                    "description": get_arg(arg, "description"),
                }
                for arg in args
            ]
    return {
        "public_name": public_name,
        "description": description if isinstance(description, str) else str(description),
        "argument_schema": schema if schema is not None else {},
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    raise TypeError(f"context is not JSON-safe: {type(value).__name__}")


def get_arg(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def worker_prompt(payload: Mapping[str, Any]) -> str:
    artifact_schema = {
        "artifact_type": "EVIDENCE_REPORT",
        "objective": "string",
        "status": "COMPLETE | PARTIAL | BLOCKED",
        "findings": [{"claim": "string", "evidence_refs": ["string"], "confidence": "LOW | MEDIUM | HIGH"}],
        "uncertainties": ["string"],
        "contradictions": ["string"],
        "recommended_next_action": "string | null",
    }
    worker_instructions = "\n".join(
        (
            "You are a fresh, bounded read-only evidence worker.",
            f"Objective: {payload['objective']}",
            f"Completion criterion: {payload['completion_criterion']}",
            f"Selected context: {json.dumps(payload['context'], sort_keys=True)}",
            f"Permitted tools: {json.dumps(payload['permitted_tools'], sort_keys=True)}",
            f"Initial evidence references: {json.dumps(payload['evidence_refs'])}",
            "Read-tool observations may add a worker_tool_result:<event_id> evidence reference; use that exact handle for information learned from the tool.",
            "Do not write, delete, update, send messages, contact the user, wait, control the environment, recurse, or delegate.",
            "Use ARE's required Thought / Action JSON format and call exactly one tool per action.",
            "When finished, call return_artifact with one EvidenceReport. Do not provide the report as plain text.",
            "Keep the report concise and limited to the delegated objective; do not answer or discuss the broader task.",
            "Every string field must be a single line with no newline or other control characters.",
            f"The report objective must equal this string exactly: {json.dumps(str(payload['objective']))}",
            f"The return_artifact payload must match this schema: {json.dumps(artifact_schema, sort_keys=True)}",
        )
    )
    return f"{DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT}\n\n<worker_instructions>\n{worker_instructions}\n</worker_instructions>"


def native_worker_prompt(payload: Mapping[str, Any]) -> str:
    schema_hash = hashlib.sha256(
        canonical_json(native_return_artifact_tool_schema()).encode("utf-8")
    ).hexdigest()
    return "\n".join(
        (
            "You are a fresh, bounded read-only evidence worker.",
            f"Objective: {payload['objective']}",
            f"Completion criterion: {payload['completion_criterion']}",
            f"Selected context: {json.dumps(payload['context'], sort_keys=True)}",
            f"Permitted tools: {json.dumps(payload['permitted_tools'], sort_keys=True)}",
            f"Initial evidence references: {json.dumps(payload['evidence_refs'])}",
            "Read-tool observations may add a worker_tool_result:<event_id> evidence reference; use that exact handle for information learned from the tool.",
            "Do not write, delete, update, send messages, contact the user, wait, control the environment, recurse, or delegate.",
            "Use exactly one native function tool per step.",
            "When finished, call return_artifact. Its arguments are the EvidenceReport fields directly.",
            "Keep the report concise and limited to the delegated objective; do not answer or discuss the broader task.",
            "Every string field must be a single line with no newline or other control characters.",
            f"The report objective must equal this string exactly: {json.dumps(str(payload['objective']))}",
            f"Canonical return_artifact schema SHA-256: {schema_hash}",
        )
    )


def plain_text_worker_prompt(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "You are a fresh, bounded read-only worker.",
            f"Objective: {payload['objective']}",
            f"Completion criterion: {payload['completion_criterion']}",
            f"Selected context: {json.dumps(payload['context'], sort_keys=True)}",
            f"Permitted tools: {json.dumps(payload['permitted_tools'], sort_keys=True)}",
            "Use native function tools when needed.",
            "Do not write, delete, update, send messages, contact the user, wait, control the environment, recurse, or delegate.",
            "When the bounded objective is complete, reply with concise ordinary text for the orchestrator.",
            "Do not produce an EvidenceReport or other JSON envelope.",
        )
    )


class _TracedWorkerAgent(BaseAgent):
    def __init__(self, *args: Any, trace_sink: Any | None = None, **kwargs: Any) -> None:
        self.trace_sink = trace_sink
        super().__init__(*args, **kwargs)

    def log_error(self, error: Exception) -> None:
        super().log_error(error)
        trace_model_output_rejection(
            self.trace_sink,
            error,
            actor_id=self.agent_id,
            actor_role="worker",
        )


class BaseAgentWorkerFactory:
    """Build a fresh pinned-ARE ``BaseAgent`` around actual selected tools."""

    def __init__(
        self,
        llm_engine: Callable[..., Any],
        *,
        pause_env: Callable[[], None] | None = None,
        resume_env: Callable[[float], None] | None = None,
        time_manager: Any | None = None,
        simulated_generation_time_config: Any | None = None,
        log_callback: Callable[[Any], None] | None = None,
        trace_sink: Any | None = None,
    ) -> None:
        self.llm_engine = llm_engine
        self.pause_env = pause_env
        self.resume_env = resume_env
        self.time_manager = time_manager
        self.simulated_generation_time_config = simulated_generation_time_config
        self.log_callback = log_callback
        self.trace_sink = trace_sink

    def __call__(self, payload: Mapping[str, Any], tools: tuple[Any, ...]) -> BaseAgent:
        holder: dict[str, EvidenceReport] = {}
        budgets = WorkerBudgets.from_mapping(payload["budgets"])
        budget_state = _WorkerBudgetState(budgets)

        def capture(artifact: EvidenceReport) -> None:
            holder["artifact"] = artifact

        def record_validation(accepted: bool, rejections: tuple[str, ...]) -> None:
            if self.trace_sink is not None:
                self.trace_sink.append(
                    OrchestrationEvent(
                        event_type=EventName.ARTIFACT_VALIDATION,
                        actor_id=str(payload.get("worker_id", "")) or None,
                        actor_role="worker",
                        error_type=None if accepted else rejections[0] if rejections else "INVALID_SCHEMA",
                        payload={
                            "accepted": accepted,
                            "primary_rejection": rejections[0] if rejections else None,
                            "all_rejections": list(rejections),
                        },
                    )
                )

        return_tool = ReturnArtifactTool(
            on_return=capture,
            on_validation=record_validation,
        )
        tool_map = {str(getattr(tool, "name", "")): tool for tool in tools}
        tool_map[return_tool.name] = return_tool
        executor = JsonActionExecutor(tools=tool_map)
        termination = TerminationStep(
            condition=lambda agent: (
                "artifact" in holder
                or budget_state.exceeded is not None
                or agent.iterations >= agent.max_iterations
            ),
            function=lambda _agent: holder.get("artifact"),
        )
        agent = _TracedWorkerAgent(
            llm_engine=_BudgetedEngine(self.llm_engine, budget_state),
            system_prompts={"system_prompt": str(payload["prompt"])},
            tools=tool_map,
            action_executor=executor,
            termination_step=termination,
            max_iterations=payload["budgets"]["max_steps"],
            total_iterations=payload["budgets"]["max_steps"],
            time_manager=self.time_manager,
            simulated_generation_time_config=self.simulated_generation_time_config,
            log_callback=self.log_callback,
            use_custom_logger=False,
            trace_sink=self.trace_sink,
        )
        if self.pause_env is not None:
            agent.pause_env = self.pause_env
        if self.resume_env is not None:
            agent.resume_env = self.resume_env
        if "worker_id" in payload:
            agent.agent_id = str(payload["worker_id"])
        agent.worker_artifact_holder = holder
        agent.worker_budget_state = budget_state
        return agent


class NativeTypedWorkerRunner:
    """Bounded native function-calling loop over already-audited worker tools."""

    def __init__(
        self,
        engine: Any,
        *,
        trace_sink: Any | None = None,
        pause_env: Callable[[], None] | None = None,
        resume_env: Callable[[float], None] | None = None,
        generation_seconds: float = 5.0,
    ) -> None:
        if not callable(getattr(engine, "native_tool_completion", None)):
            raise TypeError("engine must provide native_tool_completion()")
        self.engine = engine
        self.trace_sink = trace_sink
        self.pause_env = pause_env
        self.resume_env = resume_env
        self.generation_seconds = generation_seconds

    def _emit_validation(
        self,
        payload: Mapping[str, Any],
        accepted: bool,
        rejections: tuple[str, ...],
    ) -> None:
        if self.trace_sink is None:
            return
        self.trace_sink.append(
            OrchestrationEvent(
                event_type=EventName.ARTIFACT_VALIDATION,
                actor_id=str(payload.get("worker_id", "")) or None,
                actor_role="worker",
                error_type=(
                    None
                    if accepted
                    else rejections[0] if rejections else "INVALID_SCHEMA"
                ),
                payload={
                    "accepted": accepted,
                    "primary_rejection": rejections[0] if rejections else None,
                    "all_rejections": list(rejections),
                },
            )
        )

    def _exchange(self, metadata: Mapping[str, Any]) -> dict[str, Any] | None:
        index = metadata.get("native_exchange_index")
        exchanges = getattr(self.engine, "native_exchanges", None)
        if (
            type(index) is int
            and isinstance(exchanges, list)
            and 0 <= index < len(exchanges)
        ):
            return exchanges[index]
        return None

    def __call__(
        self,
        payload: Mapping[str, Any],
        tools: tuple[Any, ...],
    ) -> EvidenceReport:
        budgets = WorkerBudgets.from_mapping(payload["budgets"])
        budget_state = _WorkerBudgetState(budgets)
        tool_map = {str(tool.name): tool for tool in tools}
        native_tools = [_native_tool_schema(tool) for tool in tools]
        native_tools.append(native_return_artifact_tool_schema())
        prompt = native_worker_prompt(payload)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Complete the bounded objective using the available native function tools.",
            },
        ]
        read_tool_used = False
        named_finalizer = False
        for _step in range(budgets.max_steps):
            budget_before = max(
                0, budgets.max_output_tokens - budget_state.output_tokens_used
            )
            if budget_before < 1:
                raise BudgetExceededError(
                    BudgetExceededResult(
                        "max_output_tokens",
                        0,
                        budget_state.output_tokens_used,
                    )
                )
            if self.pause_env is not None:
                self.pause_env()
            try:
                assistant, metadata = self.engine.native_tool_completion(
                    messages,
                    tools=(
                        [native_tools[-1]]
                        if named_finalizer
                        else native_tools
                    ),
                    tool_choice=(
                        {
                            "type": "function",
                            "function": {"name": "return_artifact"},
                        }
                        if named_finalizer
                        else "auto"
                    ),
                    max_tokens=budget_before,
                    additional_trace_tags={"actor_role": "worker"},
                )
            finally:
                if self.resume_env is not None:
                    self.resume_env(self.generation_seconds)
            exchange = self._exchange(metadata)
            if exchange is not None:
                exchange["phase"] = (
                    "FORMAT_ONLY_FINALIZER"
                    if named_finalizer
                    else (
                        "POST_TOOL_ARTIFACT"
                        if read_tool_used or not tools
                        else "PRE_TOOL"
                    )
                )
                exchange["budget_before"] = budget_before
            try:
                budget_state.consume_output(metadata.get("completion_tokens"))
            finally:
                if exchange is not None:
                    exchange["budget_after"] = (
                        0
                        if budget_state.exceeded is not None
                        else max(
                            0,
                            budgets.max_output_tokens
                            - budget_state.output_tokens_used,
                        )
                    )
                    exchange["budget_exhausted"] = (
                        budget_state.exceeded is not None
                    )
            calls = assistant.get("tool_calls")
            if not isinstance(calls, list) or len(calls) != 1:
                if budget_state.output_tokens_used >= budgets.max_output_tokens:
                    if exchange is not None:
                        exchange["budget_exhausted"] = True
                    raise BudgetExceededError(
                        BudgetExceededResult(
                            "max_output_tokens",
                            0,
                            budget_state.output_tokens_used,
                        )
                    )
                no_call = not isinstance(calls, list) or not calls
                violation = (
                    "PROTOCOL_VIOLATION_PLAIN_TEXT"
                    if no_call and read_tool_used
                    else (
                        "PROTOCOL_VIOLATION_NO_TOOL_CALL"
                        if no_call
                        else "PROTOCOL_VIOLATION_MULTIPLE_TOOL_CALLS"
                    )
                )
                if exchange is not None:
                    exchange["output_rejection"] = violation
                if self.trace_sink is not None:
                    self.trace_sink.append(
                        OrchestrationEvent(
                            event_type=EventName.MODEL_OUTPUT_REJECTED,
                            actor_id=str(payload.get("worker_id", ""))
                            or None,
                            actor_role="worker",
                            error_type=violation,
                            payload={"reason": violation},
                        )
                    )
                if no_call and read_tool_used and not named_finalizer:
                    messages.append(dict(assistant))
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not a valid worker "
                                "action. Return the final EvidenceReport using "
                                "the available function."
                            ),
                        }
                    )
                    named_finalizer = True
                    continue
                error = ValidationError(
                    "native worker must emit exactly one tool call"
                )
                raise error
            call = calls[0]
            function = call["function"]
            name = function["name"]
            arguments = function["arguments"]
            messages.append(dict(assistant))
            if name == "return_artifact":
                categories = diagnose_evidence_report(
                    arguments,
                    objective=str(payload["objective"]),
                    allowed_evidence_refs=set(payload["evidence_refs"]),
                )
                if exchange is not None:
                    exchange["validator_input"] = arguments
                try:
                    artifact = FreshReadOnlyWorker._validate_artifact(
                        arguments,
                        str(payload["objective"]),
                        allowed_evidence_refs=set(payload["evidence_refs"]),
                    )
                except (ValidationError, TypeError, ValueError):
                    rejections = categories or ("INVALID_FIELD_TYPE",)
                    self._emit_validation(payload, False, rejections)
                    if exchange is not None:
                        exchange["validator_result"] = {
                            "accepted": False,
                            "rejection_categories": list(rejections),
                        }
                    raise
                self._emit_validation(payload, True, ())
                if exchange is not None:
                    exchange["validator_result"] = {
                        "accepted": True,
                        "rejection_categories": [],
                    }
                return artifact
            tool = tool_map.get(name)
            if tool is None:
                raise ValidationError(f"native worker called unknown tool: {name}")
            if named_finalizer:
                raise ValidationError(
                    "named finalizer called a non-artifact tool"
                )
            result = tool(**arguments)
            read_tool_used = True
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(
                        _safe_event_value(result),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        raise BudgetExceededError(
            BudgetExceededResult(
                "max_steps",
                budgets.max_steps,
                budget_state.output_tokens_used,
            )
        )


class PlainTextWorkerRunner:
    """Bounded read-only native-tool loop with a harness-authored episode."""

    def __init__(
        self,
        engine: Any,
        *,
        pause_env: Callable[[], None] | None = None,
        resume_env: Callable[[float], None] | None = None,
        generation_seconds: float = 5.0,
    ) -> None:
        if not callable(getattr(engine, "native_tool_completion", None)):
            raise TypeError("engine must provide native_tool_completion()")
        self.engine = engine
        self.pause_env = pause_env
        self.resume_env = resume_env
        self.generation_seconds = generation_seconds

    def __call__(
        self,
        payload: Mapping[str, Any],
        tools: tuple[Any, ...],
    ) -> WorkerEpisode:
        budgets = WorkerBudgets.from_mapping(payload["budgets"])
        budget_state = _WorkerBudgetState(budgets)
        tool_map = {str(tool.name): tool for tool in tools}
        native_tools = [_native_tool_schema(tool) for tool in tools]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": plain_text_worker_prompt(payload)},
            {
                "role": "user",
                "content": "Complete the bounded objective.",
            },
        ]
        tool_events: list[dict[str, str]] = []
        wall_seconds = 0.0
        for _step in range(budgets.max_steps):
            budget_before = max(
                0, budgets.max_output_tokens - budget_state.output_tokens_used
            )
            if budget_before < 1:
                raise BudgetExceededError(
                    BudgetExceededResult(
                        "max_output_tokens",
                        0,
                        budget_state.output_tokens_used,
                    )
                )
            if self.pause_env is not None:
                self.pause_env()
            try:
                assistant, metadata = self.engine.native_tool_completion(
                    messages,
                    tools=native_tools,
                    tool_choice="auto",
                    max_tokens=budget_before,
                    additional_trace_tags={"actor_role": "worker"},
                    interface_label="PLAIN_TEXT_WORKER_EPISODE",
                )
            finally:
                if self.resume_env is not None:
                    self.resume_env(self.generation_seconds)
            budget_state.consume_output(metadata.get("completion_tokens"))
            duration = metadata.get("completion_duration")
            if isinstance(duration, (int, float)) and duration >= 0:
                wall_seconds += float(duration)
            calls = assistant.get("tool_calls")
            if isinstance(calls, list) and calls:
                if len(calls) != 1:
                    raise ValidationError(
                        "plain-text worker must emit at most one tool call per step"
                    )
                call = calls[0]
                function = call["function"]
                name = function["name"]
                tool = tool_map.get(name)
                if tool is None:
                    raise ValidationError(
                        f"plain-text worker called unknown tool: {name}"
                    )
                result = tool(**function["arguments"])
                result_ref = (
                    str(result.get("evidence_ref"))
                    if isinstance(result, Mapping)
                    and isinstance(result.get("evidence_ref"), str)
                    else ""
                )
                tool_events.append(
                    {
                        "tool_name": name,
                        "tool_call_id": str(call["id"]),
                        "result_ref": result_ref,
                    }
                )
                messages.append(dict(assistant))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": json.dumps(
                            _safe_event_value(result),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
                continue
            content = assistant.get("content")
            if isinstance(content, str) and content.strip():
                return WorkerEpisode(
                    objective=str(payload["objective"]),
                    final_text=content.strip(),
                    tool_events=tuple(tool_events),
                    generated_tokens=budget_state.output_tokens_used,
                    wall_seconds=wall_seconds,
                )
            raise ValidationError(
                "plain-text worker returned neither a tool call nor final text"
            )
        raise BudgetExceededError(
            BudgetExceededResult(
                "max_steps",
                budgets.max_steps,
                budget_state.output_tokens_used,
            )
        )


class DelegationWorkerAdapter:
    """Callable runtime boundary for one normalized delegation proposal."""

    def __init__(
        self,
        *,
        environment: Any,
        available_tools: Iterable[Any] | Callable[[], Iterable[Any]],
        context_resolver: Callable[[str], Any] | Mapping[str, Any],
        worker_factory: Callable[[Mapping[str, Any], tuple[Any, ...]], Any] | None = None,
        worker_runner: Callable[[Mapping[str, Any], tuple[Any, ...]], Any] | None = None,
        llm_engine: Callable[..., Any] | None = None,
        trace_sink: Any | None = None,
        pause_env: Callable[[], None] | None = None,
        resume_env: Callable[[float], None] | None = None,
        time_manager: Any | None = None,
        simulated_generation_time_config: Any | None = None,
        log_callback: Callable[[Any], None] | None = None,
        audited_allowlist: frozenset[str] = MANUALLY_AUDITED_ALLOWLIST,
    ) -> None:
        if worker_factory is not None and worker_runner is not None:
            raise ValueError("provide worker_factory or worker_runner, not both")
        self.environment = environment
        self.available_tools = available_tools
        self.context_resolver = context_resolver
        self.worker_factory = worker_factory or (BaseAgentWorkerFactory(llm_engine) if llm_engine else None)
        self.worker_runner = worker_runner
        if worker_factory is None and llm_engine is not None:
            self.worker_factory = BaseAgentWorkerFactory(
                llm_engine,
                pause_env=pause_env,
                resume_env=resume_env,
                time_manager=time_manager,
                simulated_generation_time_config=simulated_generation_time_config,
                log_callback=log_callback,
                trace_sink=trace_sink,
            )
        self.trace_sink = trace_sink
        self.pause_env = pause_env
        self.resume_env = resume_env
        self.audited_allowlist = frozenset(audited_allowlist)

    def _emit(self, event_type: EventName, *, actor_id: str, **fields: Any) -> None:
        if self.trace_sink is not None:
            self.trace_sink.append(
                OrchestrationEvent(
                    event_type=event_type,
                    actor_id=actor_id,
                    actor_role="worker",
                    **fields,
                )
            )

    def _resolve_context(self, proposal: DelegationProposal) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for ref in proposal.context_refs:
            if isinstance(self.context_resolver, Mapping):
                if ref not in self.context_resolver:
                    raise KeyError(f"unknown context reference: {ref}")
                value = self.context_resolver[ref]
            else:
                value = self.context_resolver(ref)
            resolved[ref] = _json_safe(value)
        return resolved

    def _make_payload(
        self,
        proposal: DelegationProposal,
        context: Mapping[str, Any],
        selection: ReadOnlyToolSelection,
    ) -> dict[str, Any]:
        permitted = tuple(
            _safe_tool_description(tool, manifest.public_name)
            for tool, manifest in zip(selection.tools, selection.selected_manifests)
        )
        payload: dict[str, Any] = {
            "proposal_id": proposal.proposal_id,
            "objective": proposal.objective,
            "completion_criterion": proposal.completion_criterion,
            "context_refs": list(proposal.context_refs),
            "context": dict(context),
            "evidence_refs": list(proposal.context_refs),
            "permitted_tools": list(permitted),
            "tools": list(permitted),
            "budgets": {
                "max_steps": MAX_WORKER_STEPS,
                "max_output_tokens": MAX_WORKER_OUTPUT_TOKENS,
            },
        }
        payload["prompt"] = worker_prompt(payload)
        return payload

    @staticmethod
    def _invoke_session(session: Any, payload: Mapping[str, Any]) -> Any:
        if isinstance(session, BaseAgent):
            result = session.run(str(payload["prompt"]))
            state = getattr(session, "worker_budget_state", None)
            if state is not None and state.exceeded is not None:
                return state.exceeded
            if result is None and session.iterations >= session.max_iterations:
                state = getattr(session, "worker_budget_state", None)
                return BudgetExceededResult(
                    "max_steps",
                    session.iterations,
                    getattr(state, "output_tokens_used", 0),
                )
            return result
        if callable(session):
            return session(payload)
        run = getattr(session, "run", None)
        if callable(run):
            try:
                return run(payload)
            except TypeError:
                return run(str(payload["prompt"]))
        return session

    def __call__(self, proposal: Mapping[str, Any]) -> dict[str, Any]:
        guard = StateGuard(self.environment)
        payload: dict[str, Any] = {}
        worker_id = uuid4().hex
        try:
            normalized = _normalize_proposal(proposal)
            context = self._resolve_context(normalized)
            tools = self.available_tools() if callable(self.available_tools) else self.available_tools
            selection = select_read_only_tools(
                tuple(_AuditableExecutableTool(tool) for tool in tools),
                normalized.allowed_read_tools,
                audited_allowlist=self.audited_allowlist,
            )
            payload = self._make_payload(normalized, context, selection)
            payload["worker_id"] = worker_id
        except Exception as exc:
            return WorkerResult(
                None,
                TreatmentFailure(TreatmentFailureReason.RUNNER_ERROR, f"{type(exc).__name__}: {exc}"),
                guard,
                payload,
            ).to_dict()

        value: Any = None
        failure: TreatmentFailure | None = None
        evidence_refs = payload["evidence_refs"]
        state_record_count = len(getattr(self.environment, "state_records", ()))
        self._emit(
            EventName.WORKER_STARTED,
            actor_id=worker_id,
            context_refs=tuple(payload.get("context_refs", ())),
            payload={
                "proposal_id": payload.get("proposal_id"),
                "budgets": payload.get("budgets", {}),
            },
        )
        with guard:
            try:
                selected_tools = tuple(
                    _TracedExecutableTool(
                        tool,
                        manifest.public_name,
                        self.trace_sink,
                        worker_id,
                        evidence_refs,
                    )
                    if self.trace_sink is not None
                    else tool
                    for tool, manifest in zip(selection.tools, selection.selected_manifests)
                )
                if self.worker_runner is not None:
                    value = self.worker_runner(payload, selected_tools)
                elif self.worker_factory is not None:
                    value = self._invoke_session(self.worker_factory(payload, selected_tools), payload)
                else:
                    raise RuntimeError("a worker_factory, worker_runner, or llm_engine is required")
            except (TimeoutError, BudgetExceededError) as exc:
                failure = TreatmentFailure(TreatmentFailureReason.TIMEOUT_OR_BUDGET_EXHAUSTION, str(exc))
            except Exception as exc:
                failure = TreatmentFailure(TreatmentFailureReason.RUNNER_ERROR, f"{type(exc).__name__}: {exc}")

        attribution = _state_change_attribution(
            self.environment, state_record_count, worker_id, guard.changed
        )
        self._emit(
            EventName.WORKER_STATE_GUARD,
            actor_id=worker_id,
            state_before_hash=guard.before_hash,
            state_after_hash=guard.after_hash,
            payload={"changed": guard.changed, "attribution": attribution},
        )
        if guard.changed:
            self._emit(
                EventName.STATE_CHANGED_DURING_WORKER,
                actor_id=worker_id,
                state_before_hash=guard.before_hash,
                state_after_hash=guard.after_hash,
                protocol_violation=(
                    TreatmentFailureReason.STATE_DIFF_VIOLATION.value
                    if attribution == "worker"
                    else None
                ),
                payload={"attribution": attribution},
            )
        if guard.changed and attribution == "worker":
            failure = TreatmentFailure(
                TreatmentFailureReason.STATE_DIFF_VIOLATION,
                "application state changed during the worker attempt",
            )
        elif failure is None and not isinstance(value, WorkerEpisode):
            if isinstance(value, BudgetExceededResult) or (
                isinstance(value, Mapping) and value.get("status") == "BUDGET_EXCEEDED"
            ):
                failure = TreatmentFailure(
                    TreatmentFailureReason.TIMEOUT_OR_BUDGET_EXHAUSTION,
                    "worker exhausted its fixed budget",
                )
            else:
                try:
                    value = FreshReadOnlyWorker._validate_artifact(
                        value,
                        payload["objective"],
                        allowed_evidence_refs=set(evidence_refs),
                    )
                except (ValidationError, TypeError, ValueError) as exc:
                    failure = TreatmentFailure(TreatmentFailureReason.MALFORMED_ARTIFACT, str(exc))
        if isinstance(value, WorkerEpisode) and failure is None:
            result = {
                "status": "WORKER_COMPLETED",
                "episode": value.to_dict(),
                "failure": None,
                "state_guard": {
                    "before_hash": guard.before_hash,
                    "after_hash": guard.after_hash,
                    "changed": guard.changed,
                },
                "payload": dict(payload),
            }
            self._emit(
                EventName.WORKER_EPISODE,
                actor_id=worker_id,
                artifact_refs=tuple(
                    event["result_ref"]
                    for event in value.tool_events
                    if event["result_ref"]
                ),
                payload={"episode": result["episode"]},
            )
        else:
            result = WorkerResult(
                value if isinstance(value, EvidenceReport) and failure is None else None,
                failure,
                guard,
                payload,
            ).to_dict()
        if result.get("artifact") is not None:
            self._emit(
                EventName.WORKER_ARTIFACT,
                actor_id=worker_id,
                artifact_refs=(str(payload["proposal_id"]),),
                payload={"artifact": result["artifact"]},
            )
        self._emit(
            EventName.WORKER_COMPLETED,
            actor_id=worker_id,
            payload={"status": result["status"], "failure": result["failure"]},
        )
        return result


class FreshReadOnlyWorker:
    """Validate a worker request and delegate one fresh session to an injected runner."""

    def __init__(
        self,
        *,
        environment: Any,
        available_tools: tuple[Any, ...] | list[Any],
        runner: Callable[[Mapping[str, Any]], Any],
        audited_allowlist: frozenset[str] = MANUALLY_AUDITED_ALLOWLIST,
    ) -> None:
        self.environment = environment
        self.available_tools = tuple(available_tools)
        self.runner = runner
        self.audited_allowlist = frozenset(audited_allowlist)

    def _payload(
        self,
        objective: str,
        context: Any,
        selection: ReadOnlyToolSelection,
        budgets: WorkerBudgets,
    ) -> dict[str, Any]:
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be a non-empty string")
        import json

        serialized_context = json.loads(canonical_json(context))
        return {
            "objective": objective.strip(),
            "context": serialized_context,
            "tools": [manifest.to_dict() for manifest in selection.selected_manifests],
            "budgets": budgets.to_dict(),
        }

    @staticmethod
    def _validate_artifact(
        value: Any,
        objective: str,
        *,
        allowed_evidence_refs: set[str] | None = None,
    ) -> EvidenceReport:
        if isinstance(value, EvidenceReport):
            artifact = value
        elif isinstance(value, Mapping):
            artifact = EvidenceReport.from_dict(value)
        else:
            raise ValidationError("worker returned a non-mapping artifact")
        if artifact.objective != objective:
            raise ValidationError("worker artifact objective does not match the request")
        if allowed_evidence_refs is not None:
            cited = {
                ref
                for finding in artifact.findings
                for ref in finding.evidence_refs
            }
            unknown = cited - allowed_evidence_refs
            if unknown:
                raise ValidationError(
                    f"worker artifact cites unknown evidence references: {sorted(unknown)}"
                )
        return artifact

    def run(
        self,
        *,
        objective: str,
        context: Any,
        requested_tools: tuple[str, ...] | list[str],
        budgets: WorkerBudgets | Mapping[str, Any],
    ) -> WorkerResult:
        worker_budgets = WorkerBudgets.from_mapping(budgets)
        selection = select_read_only_tools(
            self.available_tools,
            requested_tools,
            audited_allowlist=self.audited_allowlist,
        )
        payload = self._payload(objective, context, selection, worker_budgets)
        guard = StateGuard(self.environment)
        value: Any = None
        failure: TreatmentFailure | None = None
        with guard:
            try:
                value = self.runner(payload)
            except (TimeoutError, BudgetExceededError) as exc:
                detail = str(exc)
                failure = TreatmentFailure(TreatmentFailureReason.TIMEOUT_OR_BUDGET_EXHAUSTION, detail)
            except Exception as exc:  # Runner failures are observable treatment outcomes.
                failure = TreatmentFailure(TreatmentFailureReason.RUNNER_ERROR, f"{type(exc).__name__}: {exc}")

        if failure is None:
            if isinstance(value, BudgetExceededResult) or (
                isinstance(value, Mapping) and value.get("status") == "BUDGET_EXCEEDED"
            ):
                failure = TreatmentFailure(
                    TreatmentFailureReason.TIMEOUT_OR_BUDGET_EXHAUSTION,
                    "worker exhausted its fixed budget",
                )
            else:
                try:
                    value = self._validate_artifact(value, payload["objective"])
                except (ValidationError, TypeError, ValueError) as exc:
                    failure = TreatmentFailure(TreatmentFailureReason.MALFORMED_ARTIFACT, str(exc))
        return WorkerResult(value if isinstance(value, EvidenceReport) and failure is None else None, failure, guard, payload)


def _state_change_attribution(
    environment: Any,
    starting_record_count: int,
    worker_id: str,
    changed: bool,
) -> str:
    """Attribute observed changes only from explicit instrumented provenance."""

    if not changed:
        return "none"
    records = tuple(getattr(environment, "state_records", ()))[starting_record_count:]
    for record in records:
        if not getattr(record, "state_changed", False):
            continue
        if (
            getattr(record, "actor_role", None) == "worker"
            or getattr(record, "provenance", None) == "worker"
            or getattr(record, "actor_id", None) == worker_id
        ):
            return "worker"
    if any(getattr(record, "state_changed", False) for record in records):
        return "environment"
    return "unknown"
