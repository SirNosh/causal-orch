"""Run five stock-ARE Gaia2 attempts with no delegation or intervention."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from are.simulation.agents.default_agent.agent_factory import (
    are_simulation_react_json_agent,
)
from are.simulation.agents.default_agent.are_simulation_main import (
    ARESimulationAgent,
)
from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactAgentConfig,
    ARESimulationReactBaseAgentConfig,
    LLMEngineConfig,
)
from are.simulation.agents.default_agent.prompts.system_prompt import (
    DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT,
)
from are.simulation.types import SimulatedGenerationTimeConfig

from causal_orch.models.local_llama_engine import LocalLlamaLLMEngine
from causal_orch.models.manifests import (
    LocalLlamaConfig,
    LocalModelManifest,
    ModelManifest,
    OpenRouterConfig,
)
from causal_orch.models.openrouter_engine import OpenRouterLLMEngine
from causal_orch.tracing.context import RunContext
from causal_orch.tracing.events import EventName
from causal_orch.tracing.sink import InMemoryTraceSink
from run_smoke import Gaia2SmokeHarness, _native_success, _redact_error


class StockAREBuilder:
    """Small adapter around Meta ARE's stock ReAct agent factory."""

    def __init__(
        self,
        model_config: LocalLlamaConfig | OpenRouterConfig,
        trace_sink: InMemoryTraceSink,
    ) -> None:
        self.model_config = model_config
        self.trace_sink = trace_sink
        self._provider_preverified = False

    def preflight(self) -> None:
        if not isinstance(self.model_config, LocalLlamaConfig):
            return
        engine = LocalLlamaLLMEngine(
            config=self.model_config,
            root=Path(__file__).resolve().parents[1],
            verify_artifacts=False,
        )
        engine._verify_server()
        self._provider_preverified = True

    def build(self, agent_config: Any, env: Any) -> ARESimulationAgent:
        if isinstance(self.model_config, LocalLlamaConfig):
            engine = LocalLlamaLLMEngine(
                config=self.model_config,
                trace_sink=self.trace_sink,
                root=Path(__file__).resolve().parents[1],
            )
            engine._server_verified = self._provider_preverified
        else:
            engine = OpenRouterLLMEngine(
                config=self.model_config,
                trace_sink=self.trace_sink,
            )
        base_config = agent_config.get_base_agent_config()
        base_agent = are_simulation_react_json_agent(engine, base_config)
        return ARESimulationAgent(
            log_callback=env.append_to_world_logs,
            pause_env=env.pause,
            resume_env=env.resume_with_offset,
            llm_engine=engine,
            base_agent=base_agent,
            time_manager=env.time_manager,
            max_iterations=base_config.max_iterations,
            max_turns=agent_config.max_turns,
            simulated_generation_time_config=base_config.simulated_generation_time_config,
        )


class StockAREConfigBuilder:
    def __init__(
        self,
        model_config: LocalLlamaConfig | OpenRouterConfig,
    ) -> None:
        self.config = ARESimulationReactAgentConfig(
            agent_name="default",
            base_agent_config=ARESimulationReactBaseAgentConfig(
                llm_engine_config=LLMEngineConfig(
                    model_name=model_config.model_slug,
                    provider=model_config.provider,
                    endpoint=model_config.endpoint,
                ),
                simulated_generation_time_config=SimulatedGenerationTimeConfig(
                    mode="fixed",
                    seconds=5.0,
                ),
                system_prompt=DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT,
                max_iterations=12,
            ),
        )

    def build(self) -> ARESimulationReactAgentConfig:
        return self.config


def _task_from_scenario(path: Path) -> str:
    scenario = json.loads(path.read_text(encoding="utf-8"))
    for event in scenario.get("events", ()):
        action = event.get("action") or {}
        if (
            event.get("class_name") == "Event"
            and action.get("app") == "AgentUserInterface"
            and action.get("function") == "send_message_to_agent"
        ):
            for argument in action.get("args") or ():
                if argument.get("name") == "content":
                    return str(argument["value"])
    raise ValueError(f"scenario has no initial user task: {path}")


def baseline_factory(
    *,
    attempt: int,
    scenario_path: Path,
    scenario_id: str,
    capability: str,
    model_manifest_path: Path,
) -> Gaia2SmokeHarness:
    local = json.loads(
        model_manifest_path.read_text(encoding="utf-8")
    )
    provider = local["provider_manifest"]
    manifest_data = local["model_manifest"]
    is_local = "gguf_filename" in manifest_data
    model_manifest = (
        LocalModelManifest.from_dict(manifest_data)
        if is_local
        else ModelManifest.from_dict(manifest_data)
    )
    trace_sink = InMemoryTraceSink(
        context=RunContext(
            run_id=f"baseline-{scenario_id}-{uuid4().hex[:12]}",
            scenario_id=scenario_id,
            capability=capability,
            model_slug=model_manifest.requested_model_slug,
            provider_slug=provider["provider_name"],
            temporal_batch="direct-baseline",
            block_key="no-intervention",
            attempt_id=str(attempt),
        )
    )
    runtime = local["runtime"]
    if is_local:
        reasoning = {
            "enable_thinking": bool(runtime.get("reasoning", True)),
            "preserve_thinking": bool(runtime.get("reasoning_preserve", True)),
        }
        if model_manifest.tool_call_format == "xml":
            reasoning["tool_call_format"] = "xml"
        model_config = LocalLlamaConfig(
            model_slug=model_manifest.requested_model_slug,
            provider=provider["provider_name"],
            model_path=runtime["model_path"],
            model_sha256=model_manifest.gguf_sha256,
            server_binary_path=runtime["server_binary_path"],
            server_binary_sha256=model_manifest.server_binary_sha256,
            endpoint=runtime["endpoint"],
            identity_endpoint=runtime.get("identity_endpoint"),
            context_length=runtime.get(
                "context_length", model_manifest.context_length
            ),
            reasoning=reasoning,
        )
    else:
        from dotenv import dotenv_values

        key = (
            dotenv_values(Path(__file__).resolve().parents[1] / ".env").get(
                runtime["api_key_env"]
            )
            or ""
        ).strip()
        if not key:
            raise RuntimeError(f"{runtime['api_key_env']} is missing from .env")
        model_config = OpenRouterConfig(
            model_slug=model_manifest.requested_model_slug,
            provider=provider["provider_name"],
            routing_provider_slug=local["routing_provider_slug"],
            endpoint=runtime["endpoint"],
            api_key=key,
            data_collection=provider["data_policy"],
            provider_stop_forwarded=runtime.get(
                "provider_stop_forwarded", True
            ),
            preserve_raw_provider_response=runtime.get(
                "preserve_raw_provider_response", False
            ),
            compatibility_condition=(
                local.get("condition", {}).get("condition_id")
            ),
        )
    return Gaia2SmokeHarness(
        scenario_path=scenario_path,
        agent_builder=StockAREBuilder(model_config, trace_sink),
        agent_config_builder=StockAREConfigBuilder(model_config),
        direct_tool_name="",
        direct_tool_arguments={},
        delegation_proposal={},
        task=_task_from_scenario(scenario_path),
    )


def pinned_baseline_factory(
    attempt: int,
    model_manifest_path: Path | None = None,
) -> Gaia2SmokeHarness:
    root = Path(__file__).resolve().parents[1]
    gaia = json.loads(
        (root / "configs" / "gaia2_manifest.json").read_text(encoding="utf-8")
    )
    scenario_id = gaia["scenario_ids"][0]
    return baseline_factory(
        attempt=attempt,
        scenario_path=root / gaia["scenario_files"][scenario_id],
        scenario_id=scenario_id,
        capability=gaia["selection"]["capability"],
        model_manifest_path=(
            model_manifest_path
            or root / "configs" / "local_model_manifest.json"
        ),
    )


def _attempt_result(
    harness: Gaia2SmokeHarness,
    *,
    attempt: int,
    scenario_id: str,
    capability: str,
) -> dict[str, Any]:
    from causal_orch.models.openrouter_engine import MODEL_CALL_FAILURE_TYPES

    evaluated = False
    success = None
    error = None
    try:
        harness.agent_builder.preflight()
        harness._start()
        if "DELEGATE" in harness.agent.react_agent.tools:
            raise RuntimeError("stock baseline unexpectedly exposed DELEGATE")
        harness.agent.react_agent.execute_agent_loop()
        success = _native_success(harness.native_validation())
        evaluated = True
    except Exception as exc:
        error = _redact_error(exc)
    events = harness.trace_events()
    responses = [
        event for event in events if event.event_type.value == "MODEL_RESPONSE"
    ]
    failures = [
        event for event in events if event.event_type.value == "MODEL_CALL_FAILED"
    ]
    model_config = harness.agent_builder.model_config
    logs = (
        harness.agent.react_agent.get_agent_logs()
        if getattr(harness, "agent", None) is not None
        else ()
    )
    tool_logs = [log for log in logs if log.get_type() == "tool_call"]
    valid_actions = sum(
        not (
            getattr(log, "tool_name", None)
            == "AgentUserInterface__send_message_to_user"
            and isinstance(getattr(log, "tool_arguments", None), dict)
            and log.tool_arguments.get("content")
            == "Max iterations (12) reached. Stopping."
        )
        for log in tool_logs
    )
    preserve_raw = bool(
        getattr(model_config, "preserve_raw_provider_response", False)
    )
    compatibility_trace = (
        [
            event.to_dict()
            for event in events
            if event.event_type
            in {
                EventName.MODEL_REQUEST,
                EventName.MODEL_RESPONSE,
                EventName.MODEL_CALL_FAILED,
                EventName.MODEL_IDENTITY_MISMATCH,
                EventName.PROVIDER_IDENTITY_MISMATCH,
                EventName.PROVIDER_IDENTITY_UNVERIFIABLE,
            }
        ]
        if preserve_raw
        else None
    )
    return {
        "attempt": attempt,
        "scenario_id": scenario_id,
        "capability": capability,
        "gaia2_evaluated": evaluated,
        "gaia2_success": success,
        "model_requests": sum(
            event.event_type.value == "MODEL_REQUEST" for event in events
        ),
        "model_responses": len(responses),
        "valid_orchestrator_actions": valid_actions,
        "fallbacks": sum(
            event.event_type == EventName.PROVIDER_IDENTITY_MISMATCH
            for event in events
        ),
        "model_provider_identity_verified": bool(responses)
        and all(
            event.requested_model_slug == model_config.model_slug
            and event.returned_model_slug == model_config.model_slug
            and event.provider_slug == model_config.provider
            for event in responses
        ),
        "unclassified_runtime_failures": sum(
            event.error_type not in MODEL_CALL_FAILURE_TYPES
            for event in failures
        ),
        "error": error,
        "compatibility_trace": compatibility_trace,
    }


def run_baseline(
    attempts: int = 5,
    model_manifest_path: Path | None = None,
) -> dict[str, Any]:
    results = []
    for attempt in range(1, attempts + 1):
        harness = pinned_baseline_factory(attempt, model_manifest_path)
        try:
            results.append(
                _attempt_result(
                    harness,
                    attempt=attempt,
                    scenario_id="scenario_universe_28_2nr5po",
                    capability="search",
                )
            )
        finally:
            harness.close()
    return {
        "condition": "STOCK_ARE_DIRECT_NO_DELEGATION",
        "attempts": attempts,
        "gaia2_evaluated": sum(
            result["gaia2_evaluated"] for result in results
        ),
        "gaia2_successes": sum(
            result["gaia2_success"] is True for result in results
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


def run_breadth_screen(
    screen_manifest_path: Path,
    model_manifest_path: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    screen = json.loads(screen_manifest_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in screen.items() if key != "manifest_sha256"}
    actual_manifest_hash = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    if actual_manifest_hash != screen["manifest_sha256"]:
        raise ValueError("breadth-screen manifest hash mismatch")
    results = []
    for scenario in screen["scenarios"]:
        scenario_path = root / scenario["scenario_file"]
        if (
            hashlib.sha256(scenario_path.read_bytes()).hexdigest()
            != scenario["scenario_sha256"]
        ):
            raise ValueError(
                f"scenario hash mismatch: {scenario['scenario_id']}"
            )
        for attempt in range(1, screen["attempts_per_scenario"] + 1):
            harness = baseline_factory(
                attempt=attempt,
                scenario_path=scenario_path,
                scenario_id=scenario["scenario_id"],
                capability=scenario["capability"],
                model_manifest_path=model_manifest_path,
            )
            try:
                results.append(
                    _attempt_result(
                        harness,
                        attempt=attempt,
                        scenario_id=scenario["scenario_id"],
                        capability=scenario["capability"],
                    )
                )
            finally:
                harness.close()
            print(
                json.dumps(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "attempt": attempt,
                        "gaia2_evaluated": results[-1]["gaia2_evaluated"],
                        "gaia2_success": results[-1]["gaia2_success"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    total_responses = sum(result["model_responses"] for result in results)
    total_valid_actions = sum(
        result["valid_orchestrator_actions"] for result in results
    )
    action_rate = (
        total_valid_actions / total_responses if total_responses else 0.0
    )
    successes = sum(result["gaia2_success"] is True for result in results)
    success_rate = successes / len(results)
    gate = screen["gate"]
    success_floor = gate.get("minimum_gaia2_successes")
    success_range = gate.get("diagnostic_success_rate_range")
    passed = (
        all(result["gaia2_evaluated"] for result in results)
        and not sum(
            result["unclassified_runtime_failures"] for result in results
        )
        and action_rate >= gate["minimum_valid_orchestrator_action_rate"]
        and (success_floor is None or successes >= success_floor)
        and (
            success_range is None
            or success_range[0] <= success_rate <= success_range[1]
        )
        and all(
            result["model_provider_identity_verified"] for result in results
        )
        and not sum(result["fallbacks"] for result in results)
    )
    model_condition = json.loads(
        model_manifest_path.read_text(encoding="utf-8")
    ).get("condition", {})
    return {
        "condition": model_condition.get(
            "condition_id", "STOCK_ARE_DIRECT_BREADTH_SCREEN"
        ),
        "qualified": passed,
        "attempts": len(results),
        "gaia2_evaluated": sum(result["gaia2_evaluated"] for result in results),
        "gaia2_successes": successes,
        "gaia2_success_rate": success_rate,
        "valid_orchestrator_action_rate": action_rate,
        "unclassified_runtime_failures": sum(
            result["unclassified_runtime_failures"] for result in results
        ),
        "identity_verified_attempts": sum(
            result["model_provider_identity_verified"] for result in results
        ),
        "fallbacks": sum(result["fallbacks"] for result in results),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--screen-manifest")
    parser.add_argument(
        "--model-manifest",
        default="configs/local_model_manifest.json",
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    report = (
        run_breadth_screen(
            Path(args.screen_manifest),
            Path(args.model_manifest),
        )
        if args.screen_manifest
        else run_baseline(args.attempts, Path(args.model_manifest))
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
