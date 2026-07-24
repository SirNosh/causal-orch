"""Configured ARE/Gaia2 smoke contract; no live work occurs on import."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


LOCKED_ARE_COMMIT = "7946367413129784139e785ae4c351090002a0bb"
REQUIRED_SMOKE_CHECKS = (
    "pinned_are",
    "configured_gaia2",
    "configured_provider",
    "configured_scenario",
    "direct_action",
    "forced_delegation",
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
    native_success: bool
    trace_event_names: tuple[str, ...]


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
        environment_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.scenario_path = Path(scenario_path)
        self.agent_builder = agent_builder
        self.agent_config_builder = agent_config_builder
        self.direct_tool_name = direct_tool_name
        self.direct_tool_arguments = dict(direct_tool_arguments)
        self.delegation_proposal = dict(delegation_proposal)
        self.environment_factory = environment_factory
        self._started = False

    def _emit(self, event_type: str, **fields: Any) -> None:
        from causal_orch.tracing.events import OrchestrationEvent

        sink = self.agent_builder.trace_sink
        sink.append(OrchestrationEvent(event_type=event_type, **fields))

    def _start(self) -> None:
        if self._started:
            return
        from are.simulation.environment import Environment
        from are.simulation.scenarios.utils.load_utils import load_scenario

        self.scenario = load_scenario(str(self.scenario_path))
        self.environment = (
            self.environment_factory()
            if self.environment_factory is not None
            else Environment()
        )
        self.scenario.initialize()
        self.environment.run(self.scenario, wait_for_end=False)
        config = self.agent_config_builder.build()
        self.agent = self.agent_builder.build(config, env=self.environment)
        self.agent.prepare_are_simulation_run(
            self.scenario,
            notification_system=getattr(self.environment, "notification_system", None),
        )
        self.agent.react_agent.initialize()
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
            self.agent.react_agent.step()
        return result

    def native_validation(self) -> Any:
        self._start()
        result = self.scenario.validate(self.environment)
        self._emit("RUN_COMPLETED", payload={"native_success": _native_success(result)})
        return result

    def trace_events(self) -> tuple[Any, ...]:
        return tuple(self.agent_builder.trace_sink.events)

    def close(self) -> None:
        if not self._started:
            return
        self.agent.stop()
        self.environment.stop()
        self._started = False


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
    if "DIRECT_ACTION" not in names:
        raise SmokePrerequisiteError("TRACE_DIRECT_ACTION_MISSING")
    required = (
        "INTERVENTION_ASSIGNMENT",
        "WORKER_STARTED",
        "WORKER_TOOL_CALL",
        "WORKER_TOOL_RESULT",
        "WORKER_ARTIFACT",
        "ORCHESTRATOR_RESUMED",
        "DELEGATION_EXECUTED",
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
    request_positions = [index for index, name in enumerate(names) if name == "MODEL_REQUEST"]
    response_positions = [index for index, name in enumerate(names) if name == "MODEL_RESPONSE"]
    if len(request_positions) < 3 or len(response_positions) < 3:
        raise SmokePrerequisiteError("TRACE_WORKER_MODEL_ROUNDTRIPS_MISSING")
    if any(request >= response for request, response in zip(request_positions, response_positions)):
        raise SmokePrerequisiteError("TRACE_MODEL_ORDER_INVALID")
    worker_started = names.index("WORKER_STARTED")
    tool_call = names.index("WORKER_TOOL_CALL")
    tool_result = names.index("WORKER_TOOL_RESULT")
    artifact_position = names.index("WORKER_ARTIFACT")
    delegation_position = names.index("DELEGATION_EXECUTED")
    if not any(
        worker_started < request < response < tool_call
        for request, response in zip(request_positions, response_positions)
    ):
        raise SmokePrerequisiteError("TRACE_WORKER_READ_MODEL_ROUNDTRIP_MISSING")
    if not any(
        tool_result < request < response < artifact_position
        for request, response in zip(request_positions, response_positions)
    ):
        raise SmokePrerequisiteError("TRACE_WORKER_ARTIFACT_MODEL_ROUNDTRIP_MISSING")
    if not any(
        delegation_position < request < response
        for request, response in zip(request_positions, response_positions)
    ):
        raise SmokePrerequisiteError("TRACE_ORCHESTRATOR_CONTINUATION_MISSING")
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


def run_configured_smoke(scenario_factory: Callable[[], Any]) -> SmokeExecution:
    """Execute one concrete Gaia2/ARE smoke harness."""

    harness = scenario_factory()
    if not isinstance(harness, Gaia2SmokeHarness):
        raise SmokePrerequisiteError(
            "scenario factory must return Gaia2SmokeHarness"
        )
    try:
        direct = harness.direct_action()
        delegated = harness.forced_delegation()
        if not isinstance(delegated, Mapping) or delegated.get("status") != "DELEGATION_EXECUTED":
            raise SmokePrerequisiteError("SMOKE_DELEGATION_NOT_EXECUTED")
        worker_result = delegated.get("artifact")
        if (
            not isinstance(worker_result, Mapping)
            or worker_result.get("status") != "WORKER_COMPLETED"
            or not isinstance(worker_result.get("artifact"), Mapping)
        ):
            raise SmokePrerequisiteError("SMOKE_WORKER_NOT_COMPLETED")
        validation = harness.native_validation()
        events = harness.trace_events()
        experiment = harness.agent_builder.experiment_config
        model = experiment.model_config.model_slug
        provider = experiment.model_config.provider
        return SmokeExecution(
            direct,
            delegated,
            validation,
            _native_success(validation),
            check_trace_completeness(
                events,
                expected_model=model,
                expected_provider=provider,
            ),
        )
    finally:
        harness.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config-dir", default=str(root / "configs"))
    parser.add_argument("--scenario-factory", required=True)
    parser.add_argument("--are-revision", default=os.environ.get("ARE_COMMIT"))
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
        execution = run_configured_smoke(load_symbol(args.scenario_factory))
    except (OSError, SmokePrerequisiteError, TypeError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"passed": True, "trace_event_names": execution.trace_event_names}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
