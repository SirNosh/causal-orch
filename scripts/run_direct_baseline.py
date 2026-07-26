"""Run five stock-ARE Gaia2 attempts with no delegation or intervention."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from causal_orch.models.manifests import LocalLlamaConfig, LocalModelManifest
from causal_orch.tracing.context import RunContext
from causal_orch.tracing.sink import InMemoryTraceSink
from run_smoke import Gaia2SmokeHarness, _native_success, _redact_error


class StockAREBuilder:
    """Small adapter around Meta ARE's stock ReAct agent factory."""

    def __init__(
        self,
        model_config: LocalLlamaConfig,
        trace_sink: InMemoryTraceSink,
    ) -> None:
        self.model_config = model_config
        self.trace_sink = trace_sink

    def build(self, agent_config: Any, env: Any) -> ARESimulationAgent:
        engine = LocalLlamaLLMEngine(
            config=self.model_config,
            trace_sink=self.trace_sink,
            root=Path(__file__).resolve().parents[1],
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
    def __init__(self, model_config: LocalLlamaConfig) -> None:
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


def pinned_baseline_factory(attempt: int) -> Gaia2SmokeHarness:
    root = Path(__file__).resolve().parents[1]
    gaia = json.loads(
        (root / "configs" / "gaia2_manifest.json").read_text(encoding="utf-8")
    )
    local = json.loads(
        (root / "configs" / "local_model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    scenario_id = gaia["scenario_ids"][0]
    model_manifest = LocalModelManifest.from_dict(local["model_manifest"])
    provider = local["provider_manifest"]
    trace_sink = InMemoryTraceSink(
        context=RunContext(
            run_id=f"baseline-{scenario_id}-{uuid4().hex[:12]}",
            scenario_id=scenario_id,
            capability=gaia["selection"]["capability"],
            model_slug=model_manifest.requested_model_slug,
            provider_slug=provider["provider_name"],
            temporal_batch="direct-baseline",
            block_key="no-intervention",
            attempt_id=str(attempt),
        )
    )
    runtime = local["runtime"]
    model_config = LocalLlamaConfig(
        model_slug=model_manifest.requested_model_slug,
        provider=provider["provider_name"],
        model_path=runtime["model_path"],
        model_sha256=model_manifest.gguf_sha256,
        server_binary_path=runtime["server_binary_path"],
        server_binary_sha256=model_manifest.server_binary_sha256,
        endpoint=runtime["endpoint"],
    )
    return Gaia2SmokeHarness(
        scenario_path=root / gaia["scenario_files"][scenario_id],
        agent_builder=StockAREBuilder(model_config, trace_sink),
        agent_config_builder=StockAREConfigBuilder(model_config),
        direct_tool_name=gaia["smoke"]["direct_tool_name"],
        direct_tool_arguments=gaia["smoke"]["direct_tool_arguments"],
        delegation_proposal={},
        task=gaia["selection"]["task"],
    )


def run_baseline(attempts: int = 5) -> dict[str, Any]:
    results = []
    for attempt in range(1, attempts + 1):
        harness = pinned_baseline_factory(attempt)
        try:
            harness._start()
            tools = tuple(harness.agent.react_agent.tools)
            if "DELEGATE" in tools:
                raise RuntimeError("stock baseline unexpectedly exposed DELEGATE")
            harness.agent.react_agent.execute_agent_loop()
            validation = harness.native_validation()
            results.append(
                {
                    "attempt": attempt,
                    "gaia2_evaluated": True,
                    "gaia2_success": _native_success(validation),
                    "model_requests": sum(
                        event.event_type.value == "MODEL_REQUEST"
                        for event in harness.trace_events()
                    ),
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "attempt": attempt,
                    "gaia2_evaluated": False,
                    "gaia2_success": None,
                    "model_requests": sum(
                        event.event_type.value == "MODEL_REQUEST"
                        for event in harness.trace_events()
                    ),
                    "error": _redact_error(exc),
                }
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    report = run_baseline(args.attempts)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
