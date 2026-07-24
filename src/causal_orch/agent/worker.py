"""Bounded fresh read-only worker adapters."""

from __future__ import annotations

import dataclasses
from enum import Enum
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

from .schemas import DelegationProposal, EvidenceReport, ValidationError
from .schemas import MAX_WORKER_OUTPUT_TOKENS, MAX_WORKER_STEPS


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


class ReturnArtifactTool(Tool):
    """ARE-native non-environment tool that validates the worker artifact."""

    name = "return_artifact"
    description = "Return the final structured evidence report and end the worker."
    inputs = {"artifact": {"type": "any", "description": "EvidenceReport JSON object"}}
    output_type = "any"

    def __init__(self, on_return: Callable[[EvidenceReport], None] | None = None) -> None:
        super().__init__()
        self._on_return = on_return

    def forward(self, artifact: Any) -> dict[str, Any]:
        if isinstance(artifact, EvidenceReport):
            validated = artifact
        elif isinstance(artifact, Mapping):
            validated = EvidenceReport.from_dict(artifact)
        else:
            raise ValidationError("artifact must be an EvidenceReport mapping")
        if self._on_return is not None:
            self._on_return(validated)
        return validated.to_dict()


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
    inputs = getattr(tool, "inputs", None)
    if isinstance(inputs, Mapping):
        return {
            str(name): {
                "type": value.get("type", "any") if isinstance(value, Mapping) else "any",
                "description": str(value.get("description", "")) if isinstance(value, Mapping) else "",
            }
            for name, value in inputs.items()
        }
    schema = getattr(tool, "argument_schema", None)
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    return {
        str(name): {
            "type": value.get("type", "any") if isinstance(value, Mapping) else "any",
            "description": str(value.get("description", "")) if isinstance(value, Mapping) else "",
        }
        for name, value in properties.items()
    }


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
    ) -> None:
        self.llm_engine = llm_engine
        self.pause_env = pause_env
        self.resume_env = resume_env
        self.time_manager = time_manager
        self.simulated_generation_time_config = simulated_generation_time_config
        self.log_callback = log_callback

    def __call__(self, payload: Mapping[str, Any], tools: tuple[Any, ...]) -> BaseAgent:
        holder: dict[str, EvidenceReport] = {}
        budgets = WorkerBudgets.from_mapping(payload["budgets"])
        budget_state = _WorkerBudgetState(budgets)

        def capture(artifact: EvidenceReport) -> None:
            holder["artifact"] = artifact

        return_tool = ReturnArtifactTool(on_return=capture)
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
        agent = BaseAgent(
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
        elif failure is None:
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
        result = WorkerResult(value if isinstance(value, EvidenceReport) and failure is None else None, failure, guard, payload).to_dict()
        if result["artifact"] is not None:
            self._emit(
                EventName.WORKER_ARTIFACT,
                actor_id=worker_id,
                artifact_refs=(str(payload["proposal_id"]),),
                payload={"artifact": result["artifact"]},
            )
        self._emit(
            EventName.ORCHESTRATOR_RESUMED,
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
