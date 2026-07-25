"""Configured ARE/Gaia2 smoke contract; no live work occurs on import."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4


LOCKED_ARE_COMMIT = "7946367413129784139e785ae4c351090002a0bb"
REQUIRED_SMOKE_CHECKS = (
    "pinned_are",
    "configured_gaia2",
    "configured_provider",
    "configured_scenario",
    "direct_action",
    "forced_delegation",
    "orchestrator_continuation",
    "native_validation",
    "trace_completeness",
)


@dataclass(frozen=True)
class SmokeStep:
    name: str


@dataclass(frozen=True)
class SmokePlan:
    steps: tuple[SmokeStep, ...]
    context: Mapping[str, Any]


@dataclass(frozen=True)
class SmokeExecution:
    direct_action_result: Any
    delegation_result: Any
    validation_result: Any
    infrastructure_pass: bool
    native_validation_completed: bool
    gaia2_success: bool
    failure_stage: str | None
    trace_event_names: tuple[str, ...]
    error: str | None = None
    artifact_dir: str | None = None

    @property
    def native_success(self) -> bool:
        """Compatibility alias for the Gaia2 capability outcome."""

        return self.gaia2_success


class SmokePrerequisiteError(ValueError):
    """The configured smoke run cannot prove the locked prerequisites."""


SmokeRunner = Callable[[SmokeStep], Any]


class SmokeExecuteSchedule:
    """One-shot deterministic treatment schedule, only for integration smoke."""

    def __init__(self, block_key: Any) -> None:
        self.block_key = block_key
        self.consumed_count = 0

    def reveal(self, block_key: Any, *, eligible: bool) -> Any:
        from causal_orch.runtime.randomization import TreatmentAssignment

        if not eligible:
            return None
        if block_key != self.block_key:
            raise KeyError(f"unknown smoke block: {block_key!r}")
        if self.consumed_count:
            raise IndexError("smoke assignment already consumed")
        self.consumed_count = 1
        return TreatmentAssignment.EXECUTE


class Gaia2SmokeHarness:
    """Concrete pinned-ARE harness for one local Gaia2 scenario JSON."""

    def __init__(
        self,
        *,
        scenario_path: str | Path,
        agent_builder: Any,
        agent_config_builder: Any,
        direct_tool_name: str,
        direct_tool_arguments: Mapping[str, Any],
        delegation_proposal: Mapping[str, Any],
        task: str | None = None,
        environment_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.scenario_path = Path(scenario_path)
        self.agent_builder = agent_builder
        self.agent_config_builder = agent_config_builder
        self.direct_tool_name = direct_tool_name
        self.direct_tool_arguments = dict(direct_tool_arguments)
        self.delegation_proposal = dict(delegation_proposal)
        self.task = task
        self.environment_factory = environment_factory
        self._started = False

    def _emit(self, event_type: str, **fields: Any) -> None:
        from causal_orch.tracing.events import OrchestrationEvent

        sink = self.agent_builder.trace_sink
        sink.append(OrchestrationEvent(event_type=event_type, **fields))

    def _start(self) -> None:
        if self._started:
            return
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

        scenario_json = self.scenario_path.read_text(encoding="utf-8")
        self.scenario, _, _ = JsonScenarioImporter().import_from_json_to_benchmark(
            scenario_json,
            load_completed_events=False,
        )
        scripted_checks: dict[str, list[ToolCheckerParam]] = {}
        for event in self.scenario.serialized_events:
            if event.class_name != "OracleEvent" or event.action is None:
                continue
            tool_name = f"{event.action.app}__{event.action.function}"
            scripted_checks[event.event_id] = [
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
                event_id_to_checker_params=scripted_checks
            ),
            offline_validation=True,
        )
        self.environment = (
            self.environment_factory()
            if self.environment_factory is not None
            else Environment()
        )
        self.environment.run(self.scenario, wait_for_end=False)
        config = self.agent_config_builder.build()
        self.agent = self.agent_builder.build(config, env=self.environment)
        self.agent.prepare_are_simulation_run(
            self.scenario,
            notification_system=getattr(self.environment, "notification_system", None),
        )
        self.agent.react_agent.initialize()
        if self.task:
            from are.simulation.agents.agent_log import TaskLog

            self.agent.react_agent.append_agent_log(
                TaskLog(
                    content=self.task,
                    timestamp=self.agent.react_agent.make_timestamp(),
                    agent_id=self.agent.react_agent.agent_id,
                )
            )
            self.agent.react_agent.refresh_delegation_prompt()
        self._emit("RUN_STARTED")
        self._started = True

    def direct_action(self) -> Any:
        from are.simulation.agents.default_agent.tools.action_executor import ParsedAction

        self._start()
        result = self.agent.react_agent.action_executor.execute_parsed_action(
            ParsedAction(
                tool_name=self.direct_tool_name,
                arguments=self.direct_tool_arguments,
            ),
            self.agent.react_agent.append_agent_log,
            self.agent.react_agent.make_timestamp,
            self.agent.react_agent.agent_id,
        )
        self._emit(
            "DIRECT_ACTION",
            proposed_action=self.direct_tool_name,
            executed_action=self.direct_tool_name,
        )
        return result

    def forced_delegation(self) -> Any:
        from are.simulation.agents.default_agent.tools.action_executor import ParsedAction

        self._start()
        gate = self.agent.react_agent.action_executor.intervention_gate
        if gate.decided:
            raise SmokePrerequisiteError("SMOKE_GATE_ALREADY_DECIDED")
        gate.assignment_schedule = SmokeExecuteSchedule(gate.block_key)
        result = self.agent.react_agent.action_executor.execute_parsed_action(
            ParsedAction(tool_name="DELEGATE", arguments=self.delegation_proposal),
            self.agent.react_agent.append_agent_log,
            self.agent.react_agent.make_timestamp,
            self.agent.react_agent.agent_id,
        )
        if (
            isinstance(result, Mapping)
            and result.get("status") == "DELEGATION_EXECUTED"
        ):
            return result
        return result

    def orchestrator_continuation(self) -> Any:
        self._start()
        return self.agent.react_agent.execute_agent_loop()

    def native_validation(self) -> Any:
        self._start()
        result = self.scenario.validate(self.environment)
        self._emit(
            "RUN_COMPLETED",
            payload={
                "native_success": _native_success(result),
                "validator": type(self.scenario.judge).__name__,
            },
        )
        return result

    def trace_events(self) -> tuple[Any, ...]:
        return tuple(self.agent_builder.trace_sink.events)

    def close(self) -> None:
        if not self._started:
            return
        self.agent.stop()
        self.environment.stop()
        self._started = False


def pinned_gaia2_factory() -> Gaia2SmokeHarness:
    """Build the committed Gaia2/OpenRouter smoke harness from environment credentials."""

    from causal_orch.models.manifests import ModelManifest, OpenRouterConfig
    from causal_orch.runner.agent_builder import CausalAgentBuilder
    from causal_orch.runner.config_builder import CausalAgentConfigBuilder, ExperimentConfig
    from causal_orch.runtime.randomization import generate_balanced_schedule
    from causal_orch.tracing.context import RunContext
    from causal_orch.tracing.sink import InMemoryTraceSink

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SmokePrerequisiteError("OPENROUTER_API_KEY is not set")
    root = Path(__file__).resolve().parents[1]
    gaia = json.loads((root / "configs" / "gaia2_manifest.json").read_text(encoding="utf-8"))
    openrouter = json.loads(
        (root / "configs" / "openrouter_manifest.json").read_text(encoding="utf-8")
    )
    scenario_id = gaia["scenario_ids"][0]
    model_manifest = ModelManifest.from_dict(openrouter["model_manifest"])
    provider = openrouter["provider_manifest"]
    block_key = "smoke"
    trace_sink = InMemoryTraceSink(
        context=RunContext(
            run_id=f"smoke-{scenario_id}-{uuid4().hex[:12]}",
            scenario_id=scenario_id,
            capability=gaia["selection"]["capability"],
            model_slug=model_manifest.requested_model_slug,
            provider_slug=provider["provider_name"],
            temporal_batch="smoke",
            block_key=block_key,
            attempt_id="1",
        )
    )
    model_config = OpenRouterConfig(
        model_slug=model_manifest.requested_model_slug,
        provider=provider["provider_name"],
        routing_provider_slug=provider["provider_slug"],
        reasoning={"effort": "low"},
        api_key=api_key,
    )
    experiment = ExperimentConfig(
        model_manifest=model_manifest,
        model_config=model_config,
        intervention_schedule=generate_balanced_schedule("smoke", [block_key]),
        intervention_block=block_key,
        trace_sink=trace_sink,
        worker_callback=lambda _proposal: None,
        max_iterations=12,
    )
    return Gaia2SmokeHarness(
        scenario_path=root / gaia["scenario_files"][scenario_id],
        agent_builder=CausalAgentBuilder(experiment),
        agent_config_builder=CausalAgentConfigBuilder(experiment),
        direct_tool_name=gaia["smoke"]["direct_tool_name"],
        direct_tool_arguments=gaia["smoke"]["direct_tool_arguments"],
        delegation_proposal=gaia["smoke"]["delegation_proposal"],
        task=gaia["selection"]["task"],
    )


def build_smoke_plan(
    checks: Sequence[str] = (),
    *,
    context: Mapping[str, Any] | None = None,
) -> SmokePlan:
    """Construct labels and caller-owned context without creating ARE objects."""

    if any(not isinstance(check, str) or not check for check in checks):
        raise ValueError("smoke check names must be non-empty strings")
    if len(set(checks)) != len(checks):
        raise ValueError("smoke check names must be unique")
    return SmokePlan(tuple(SmokeStep(check) for check in checks), dict(context or {}))


def run_smoke(plan: SmokePlan, runner: SmokeRunner) -> dict[str, Any]:
    """Run only the injected checks and return their supplied results."""

    if not isinstance(plan, SmokePlan):
        raise TypeError("plan must be a SmokePlan")
    if not callable(runner):
        raise TypeError("runner must be callable")
    results = [{"check": step.name, "result": runner(step)} for step in plan.steps]
    return {"checks": results, "passed": all(bool(item["result"]) for item in results)}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def smoke_prerequisite_errors(
    *,
    experiment: Mapping[str, Any],
    gaia2_manifest: Mapping[str, Any],
    providers: Mapping[str, Any],
    scenario_factory_spec: str | None,
    are_revision: str | None,
) -> tuple[str, ...]:
    """Return all missing/placeholder prerequisites without importing a scenario."""

    errors: list[str] = []
    protocol = experiment.get("protocol", {})
    if protocol.get("are_commit") != LOCKED_ARE_COMMIT:
        errors.append("ARE_COMMIT_MISMATCH")
    if are_revision != LOCKED_ARE_COMMIT:
        errors.append("ARE_REVISION_UNVERIFIED_OR_MISMATCH")
    if not _nonempty(gaia2_manifest.get("gaia2_revision")):
        errors.append("GAIA2_REVISION_PLACEHOLDER")
    scenario_ids = gaia2_manifest.get("scenario_ids")
    if not isinstance(scenario_ids, list) or not scenario_ids or any(not _nonempty(value) for value in scenario_ids):
        errors.append("SCENARIO_IDS_PLACEHOLDER")
    for key, error in (("manifest_sha256", "GAIA2_MANIFEST_HASH_PLACEHOLDER"), ("dataset_sha256", "GAIA2_DATASET_HASH_PLACEHOLDER")):
        if not _nonempty(gaia2_manifest.get(key)):
            errors.append(error)
    if not _nonempty(providers.get("pinned_provider")):
        errors.append("PINNED_PROVIDER_PLACEHOLDER")
    if not _nonempty(providers.get("provider_manifest_sha256")):
        errors.append("PROVIDER_MANIFEST_HASH_PLACEHOLDER")
    if not _nonempty(scenario_factory_spec):
        errors.append("SCENARIO_FACTORY_REQUIRED")
    return tuple(errors)


def require_smoke_prerequisites(**kwargs: Any) -> None:
    errors = smoke_prerequisite_errors(**kwargs)
    if errors:
        raise SmokePrerequisiteError("; ".join(errors))


def load_configurations(config_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the three smoke configuration documents only when explicitly requested."""

    config_dir = Path(config_dir)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency failure is CLI-only.
        raise SmokePrerequisiteError("PyYAML is required to load smoke configuration") from exc
    experiment = yaml.safe_load((config_dir / "experiment.yaml").read_text(encoding="utf-8"))
    providers = yaml.safe_load((config_dir / "providers.yaml").read_text(encoding="utf-8"))
    gaia2_manifest = json.loads((config_dir / "gaia2_manifest.json").read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (experiment, providers, gaia2_manifest)):
        raise SmokePrerequisiteError("smoke configuration documents must contain mappings")
    return experiment, gaia2_manifest, providers


def load_symbol(spec: str) -> Any:
    """Load ``module:attribute`` on demand; importing this script remains safe."""

    if not _nonempty(spec) or ":" not in spec:
        raise SmokePrerequisiteError("scenario factory must use module:attribute syntax")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute:
        raise SmokePrerequisiteError("scenario factory must use module:attribute syntax")
    if module_name in {"__main__", "run_smoke", "scripts.run_smoke"}:
        local_value = globals().get(attribute)
        if callable(local_value):
            return local_value
    value = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(value):
        raise SmokePrerequisiteError(f"configured symbol is not callable: {spec}")
    return value


def _contract_callable(target: Any, name: str) -> Callable[[], Any]:
    value = target.get(name) if isinstance(target, Mapping) else getattr(target, name, None)
    if not callable(value):
        raise SmokePrerequisiteError(f"scenario factory result lacks callable {name}()")
    return value


def _event_name(event: Any) -> str:
    value = event.get("event_type") if isinstance(event, Mapping) else getattr(event, "event_type", event)
    return getattr(value, "value", value) if isinstance(getattr(value, "value", value), str) else str(value)


def _event_value(event: Any, name: str, default: Any = None) -> Any:
    return event.get(name, default) if isinstance(event, Mapping) else getattr(event, name, default)


def check_trace_completeness(
    events: Sequence[Any],
    *,
    expected_model: str | None = None,
    expected_provider: str | None = None,
) -> tuple[str, ...]:
    from causal_orch.agent.schemas import EvidenceReport

    names = tuple(_event_name(event) for event in events)
    if not names:
        raise SmokePrerequisiteError("TRACE_EMPTY")
    if "RUN_STARTED" not in names or "RUN_COMPLETED" not in names:
        raise SmokePrerequisiteError("TRACE_RUN_BOUNDARIES_MISSING")
    if "DIRECT_ACTION" in names:
        raise SmokePrerequisiteError("TRACE_DELEGATION_CONTAMINATED_BY_DIRECT_ACTION")
    required = (
        "INTERVENTION_ASSIGNMENT",
        "WORKER_STARTED",
        "WORKER_TOOL_CALL",
        "WORKER_TOOL_RESULT",
        "WORKER_ARTIFACT",
        "WORKER_COMPLETED",
        "DELEGATION_EXECUTED",
        "ORCHESTRATOR_RESUMED",
    )
    missing = [name for name in required if name not in names]
    if missing:
        raise SmokePrerequisiteError(f"TRACE_WORKER_PATH_MISSING:{','.join(missing)}")
    assignment_event = events[names.index("INTERVENTION_ASSIGNMENT")]
    if _event_value(assignment_event, "treatment_assignment") != "EXECUTE":
        raise SmokePrerequisiteError("SMOKE_ASSIGNMENT_NOT_EXECUTE")
    positions = [names.index(name) for name in required]
    if positions != sorted(positions):
        raise SmokePrerequisiteError("TRACE_WORKER_ORDER_INVALID")
    request_positions = [
        index for index, name in enumerate(names) if name == "MODEL_REQUEST"
    ]
    response_positions = [
        index for index, name in enumerate(names) if name == "MODEL_RESPONSE"
    ]
    requests_by_id = {
        _event_value(events[index], "openrouter_request_id"): index
        for index in request_positions
        if _event_value(events[index], "openrouter_request_id")
    }
    roundtrips = [
        (requests_by_id[request_id], index)
        for index in response_positions
        if (request_id := _event_value(events[index], "openrouter_request_id"))
        in requests_by_id
        and requests_by_id[request_id] < index
    ]
    if len(request_positions) < 3 or len(roundtrips) < 2:
        raise SmokePrerequisiteError("TRACE_WORKER_MODEL_ROUNDTRIPS_MISSING")
    worker_started = names.index("WORKER_STARTED")
    tool_call = names.index("WORKER_TOOL_CALL")
    tool_result = names.index("WORKER_TOOL_RESULT")
    artifact_position = names.index("WORKER_ARTIFACT")
    delegation_position = names.index("DELEGATION_EXECUTED")
    if not any(
        worker_started < request < response < tool_call
        for request, response in roundtrips
    ):
        raise SmokePrerequisiteError("TRACE_WORKER_READ_MODEL_ROUNDTRIP_MISSING")
    if not any(
        tool_result < request < response < artifact_position
        for request, response in roundtrips
    ):
        raise SmokePrerequisiteError("TRACE_WORKER_ARTIFACT_MODEL_ROUNDTRIP_MISSING")
    resumed_position = names.index("ORCHESTRATOR_RESUMED")
    if not any(request > resumed_position for request in request_positions):
        raise SmokePrerequisiteError("TRACE_ORCHESTRATOR_CONTINUATION_REQUEST_MISSING")
    if not any(response > resumed_position for _, response in roundtrips):
        raise SmokePrerequisiteError("TRACE_ORCHESTRATOR_CONTINUATION_RESPONSE_MISSING")
    for index in response_positions:
        event = events[index]
        requested = _event_value(event, "requested_model_slug")
        returned = _event_value(event, "returned_model_slug")
        provider = _event_value(event, "provider_slug")
        if expected_model is not None and (
            requested != expected_model or returned != expected_model
        ):
            raise SmokePrerequisiteError("TRACE_MODEL_IDENTITY_MISMATCH")
        if expected_provider is not None and provider != expected_provider:
            raise SmokePrerequisiteError("TRACE_PROVIDER_IDENTITY_MISMATCH")
    result_events = {
        f"worker_tool_result:{_event_value(event, 'event_id')}"
        for event in events
        if _event_name(event) == "WORKER_TOOL_RESULT"
        and _event_value(event, "error_type") is None
    }
    artifact_event = events[names.index("WORKER_ARTIFACT")]
    artifact_payload = _event_value(artifact_event, "payload", {})
    artifact = EvidenceReport.from_dict(artifact_payload.get("artifact"))
    citations = {
        ref for finding in artifact.findings for ref in finding.evidence_refs
    }
    if not citations.intersection(result_events):
        raise SmokePrerequisiteError("TRACE_ARTIFACT_TOOL_EVIDENCE_MISSING")
    for event in events:
        if _event_value(event, "protocol_violation") == "STATE_DIFF_VIOLATION":
            raise SmokePrerequisiteError("TRACE_WORKER_STATE_WRITE_VIOLATION")
        if _event_name(event) == "STATE_CHANGED_DURING_WORKER":
            payload = _event_value(event, "payload", {})
            if payload.get("attribution") == "worker":
                raise SmokePrerequisiteError("TRACE_WORKER_STATE_WRITE_VIOLATION")
        to_json = getattr(event, "to_json", None)
        if callable(to_json):
            json.loads(to_json())
        elif isinstance(event, Mapping):
            json.dumps(event)
        else:
            raise SmokePrerequisiteError("TRACE_EVENT_NOT_SERIALIZABLE")
    if names.index("RUN_STARTED") > names.index("RUN_COMPLETED"):
        raise SmokePrerequisiteError("TRACE_ORDER_INVALID")
    return names


def _native_success(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        value = value.get("success")
    else:
        value = getattr(value, "success", None)
    if not isinstance(value, bool):
        raise SmokePrerequisiteError("NATIVE_VALIDATION_SUCCESS_UNAVAILABLE")
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if is_dataclass(value):
        return _json_value(asdict(value))
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return _json_value(method())
    return {"type": type(value).__name__}


def _redact_error(value: Exception) -> str:
    message = f"{type(value).__name__}: {value}"
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return re.sub(r"\bsk-or-[A-Za-z0-9_-]+\b", "[REDACTED]", message)


def _redacted_trace_event(event: Any) -> dict[str, Any]:
    to_dict = getattr(event, "to_dict", None)
    value = _json_value(to_dict() if callable(to_dict) else event)
    if not isinstance(value, dict):
        raise SmokePrerequisiteError("TRACE_EVENT_NOT_SERIALIZABLE")
    if value.get("event_type") == "WORKER_TOOL_RESULT":
        payload = value.get("payload")
        if isinstance(payload, dict) and "result" in payload:
            encoded = json.dumps(
                payload["result"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            value["payload"] = {
                "ok": payload.get("ok"),
                "tool": payload.get("tool"),
                "result_sha256": hashlib.sha256(encoded).hexdigest(),
            }
    return value


def _manifest_locks(root: Path, scenario_id: str) -> dict[str, Any]:
    gaia = json.loads(
        (root / "configs" / "gaia2_manifest.json").read_text(encoding="utf-8")
    )
    openrouter = json.loads(
        (root / "configs" / "openrouter_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "are_commit": LOCKED_ARE_COMMIT,
        "gaia2_revision": gaia["gaia2_revision"],
        "gaia2_manifest_sha256": gaia["manifest_sha256"],
        "gaia2_dataset_sha256": gaia["dataset_sha256"],
        "scenario_id": scenario_id,
        "scenario_sha256": gaia["scenario_sha256"][scenario_id],
        "model_manifest_sha256": openrouter["model_manifest"]["manifest_sha256"],
        "provider_manifest_sha256": openrouter["provider_manifest_sha256"],
    }


def persist_smoke_artifacts(
    execution: SmokeExecution,
    events: Sequence[Any],
    *,
    artifact_root: str | Path,
    harness: Gaia2SmokeHarness,
) -> Path:
    """Persist a redacted causal trace and outcome without prompts or model text."""

    root = Path(__file__).resolve().parents[1]
    context = getattr(harness.agent_builder.trace_sink, "context", None)
    run_id = getattr(context, "run_id", None) or f"smoke-{uuid4().hex}"
    scenario_id = getattr(context, "scenario_id", None)
    experiment = harness.agent_builder.experiment_config
    output_dir = Path(artifact_root) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=False)

    trace_lines = []
    for event in events:
        trace_lines.append(
            json.dumps(
                _redacted_trace_event(event),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
    (output_dir / "trace.jsonl").write_text(
        "\n".join(trace_lines) + ("\n" if trace_lines else ""),
        encoding="utf-8",
    )
    summary = {
        "infrastructure_pass": execution.infrastructure_pass,
        "native_validation_completed": execution.native_validation_completed,
        "gaia2_success": execution.gaia2_success,
        "failure_stage": execution.failure_stage,
        "error": execution.error,
        "model": experiment.model_config.model_slug,
        "provider": experiment.model_config.provider,
        "scenario_id": scenario_id,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = {
        "completed": execution.native_validation_completed,
        "success": execution.gaia2_success,
        "validator": type(getattr(harness, "scenario", None).judge).__name__
        if getattr(getattr(harness, "scenario", None), "judge", None) is not None
        else None,
        "result": _json_value(execution.validation_result),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest-locks.json").write_text(
        json.dumps(_manifest_locks(root, str(scenario_id)), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def run_configured_smoke(
    scenario_factory: Callable[[], Any],
    *,
    artifact_root: str | Path | None = None,
) -> SmokeExecution:
    """Run isolated native-tool and delegation smokes on fresh environments."""

    direct_harness = scenario_factory()
    if not isinstance(direct_harness, Gaia2SmokeHarness):
        raise SmokePrerequisiteError("scenario factory must return Gaia2SmokeHarness")
    try:
        direct = direct_harness.direct_action()
    finally:
        direct_harness.close()

    harness = scenario_factory()
    if not isinstance(harness, Gaia2SmokeHarness):
        raise SmokePrerequisiteError("scenario factory must return Gaia2SmokeHarness")
    delegated: Any = None
    validation: Any = None
    infrastructure_pass = False
    native_validation_completed = False
    gaia2_success = False
    failure_stage: str | None = "DELEGATION_EXECUTION"
    error: str | None = None
    trace_names: tuple[str, ...] = ()
    events: tuple[Any, ...] = ()
    artifact_dir: str | None = None
    try:
        delegated = harness.forced_delegation()
        if not isinstance(delegated, Mapping) or delegated.get("status") != "DELEGATION_EXECUTED":
            raise SmokePrerequisiteError("SMOKE_DELEGATION_NOT_EXECUTED")
        failure_stage = "WORKER_EXECUTION"
        worker_result = delegated.get("artifact")
        if (
            not isinstance(worker_result, Mapping)
            or worker_result.get("status") != "WORKER_COMPLETED"
            or not isinstance(worker_result.get("artifact"), Mapping)
        ):
            failure = worker_result.get("failure") if isinstance(worker_result, Mapping) else None
            raise SmokePrerequisiteError(f"SMOKE_WORKER_NOT_COMPLETED:{failure}")
        failure_stage = "ORCHESTRATOR_CONTINUATION"
        harness.orchestrator_continuation()
        failure_stage = "NATIVE_VALIDATION"
        validation = harness.native_validation()
        gaia2_success = _native_success(validation)
        native_validation_completed = True
        events = harness.trace_events()
        experiment = harness.agent_builder.experiment_config
        model = experiment.model_config.model_slug
        provider = experiment.model_config.provider
        failure_stage = "TRACE_AUDIT"
        trace_names = check_trace_completeness(
            events,
            expected_model=model,
            expected_provider=provider,
        )
        infrastructure_pass = True
        failure_stage = None if gaia2_success else "ORCHESTRATOR_REASONING"
    except Exception as exc:
        error = _redact_error(exc)
        events = harness.trace_events()
    execution = SmokeExecution(
            direct,
            delegated,
            validation,
            infrastructure_pass,
            native_validation_completed,
            gaia2_success,
            failure_stage,
            trace_names,
            error,
        )
    try:
        if artifact_root is not None:
            artifact_dir = str(
                persist_smoke_artifacts(
                    execution,
                    events,
                    artifact_root=artifact_root,
                    harness=harness,
                )
            )
            execution = SmokeExecution(
                execution.direct_action_result,
                execution.delegation_result,
                execution.validation_result,
                execution.infrastructure_pass,
                execution.native_validation_completed,
                execution.gaia2_success,
                execution.failure_stage,
                execution.trace_event_names,
                execution.error,
                artifact_dir,
            )
    finally:
        harness.close()
    return execution


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config-dir", default=str(root / "configs"))
    parser.add_argument("--scenario-factory", required=True)
    parser.add_argument("--are-revision", default=os.environ.get("ARE_COMMIT"))
    parser.add_argument("--artifact-dir", default=str(root / "artifacts" / "smoke"))
    args = parser.parse_args(argv)
    try:
        experiment, gaia2_manifest, providers = load_configurations(args.config_dir)
        require_smoke_prerequisites(
            experiment=experiment,
            gaia2_manifest=gaia2_manifest,
            providers=providers,
            scenario_factory_spec=args.scenario_factory,
            are_revision=args.are_revision,
        )
        execution = run_configured_smoke(
            load_symbol(args.scenario_factory),
            artifact_root=args.artifact_dir,
        )
    except (OSError, SmokePrerequisiteError, TypeError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": _redact_error(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "passed": execution.infrastructure_pass,
                "infrastructure_pass": execution.infrastructure_pass,
                "native_validation_completed": execution.native_validation_completed,
                "gaia2_success": execution.gaia2_success,
                "failure_stage": execution.failure_stage,
                "error": execution.error,
                "artifact_dir": execution.artifact_dir,
                "trace_event_names": execution.trace_event_names,
            },
            sort_keys=True,
        )
    )
    return 0 if execution.infrastructure_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
