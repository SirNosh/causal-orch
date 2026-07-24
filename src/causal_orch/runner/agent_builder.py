"""Pinned ARE agent-builder adapter for causal-orch."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from are.simulation.agents.agent_builder import AbstractAgentBuilder
from are.simulation.agents.default_agent.are_simulation_main import ARESimulationAgent
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.agents.are_simulation_agent_config import (
    RunnableARESimulationAgentConfig,
)

from causal_orch.agent.action_executor import InterventionActionExecutor
from causal_orch.agent.orchestrator import CausalOrchestrator
from causal_orch.models.openrouter_engine import OpenRouterLLMEngine
from causal_orch.runtime.intervention import DelegationInterventionGate
from causal_orch.runtime.read_only_tools import MANUALLY_AUDITED_ALLOWLIST, audit_tool
from causal_orch.agent.worker import DelegationWorkerAdapter

from .config_builder import (
    CAUSAL_AGENT_NAME,
    CausalAgentConfig,
    ExperimentConfig,
)


class CausalAgentBuilder(AbstractAgentBuilder):
    """Build one fresh causal agent around the pinned ARE lifecycle."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        *,
        trace_sink: Any | None = None,
        engine_factory: Callable[..., Any] | None = None,
        gate_factory: Callable[..., Any] = DelegationInterventionGate,
        worker_adapter: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        context_resolver: Callable[[str], Any] | Mapping[str, Any] | None = None,
        worker_factory: Callable[[Mapping[str, Any], tuple[Any, ...]], Any] | None = None,
        worker_runner: Callable[[Mapping[str, Any], tuple[Any, ...]], Any] | None = None,
        eligibility_context_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.experiment_config = experiment_config
        self.trace_sink = (
            experiment_config.trace_sink if trace_sink is None else trace_sink
        )
        self.engine_factory = engine_factory
        self.gate_factory = gate_factory
        self.worker_adapter = worker_adapter
        self.context_resolver = context_resolver
        self.worker_factory = worker_factory
        self.worker_runner = worker_runner
        self.eligibility_context_provider = eligibility_context_provider

    def list_agents(self) -> list[str]:
        return [CAUSAL_AGENT_NAME]

    def build(
        self,
        agent_config: RunnableARESimulationAgentConfig,
        env: Any | None = None,
        mock_responses: list[str] | None = None,
    ) -> RunnableARESimulationAgent:
        if agent_config.get_agent_name() != CAUSAL_AGENT_NAME:
            raise ValueError(f"agent {agent_config.get_agent_name()!r} not found")
        if env is None:
            raise ValueError("Environment must be provided")
        for name in ("time_manager", "append_to_world_logs", "pause", "resume_with_offset"):
            if not hasattr(env, name):
                raise ValueError(f"Environment must provide {name}")
        if mock_responses is not None and self.engine_factory is None:
            raise ValueError(
                "mock_responses requires an injected engine_factory; no provider fallback is implied"
            )

        # Construction is deliberately inside build: importing this module or
        # constructing the builder never creates an engine or reads credentials.
        if self.engine_factory is None:
            llm_engine = OpenRouterLLMEngine(
                config=self.experiment_config.model_config,
                trace_sink=self.trace_sink,
            )
        else:
            llm_engine = self.engine_factory(
                config=self.experiment_config.model_config,
                trace_sink=self.trace_sink,
                mock_responses=mock_responses,
            )

        runtime: dict[str, Any] = {}
        base_config = agent_config.get_base_agent_config()
        nested_config = getattr(agent_config, "base_agent_config", None)
        max_iterations = getattr(
            agent_config,
            "max_iterations",
            getattr(
                base_config,
                "max_iterations",
                getattr(
                    nested_config, "max_iterations", self.experiment_config.max_iterations
                ),
            ),
        )
        max_turns = getattr(agent_config, "max_turns", self.experiment_config.max_turns)
        time_config = (
            getattr(base_config, "simulated_generation_time_config", None)
            or self.experiment_config.simulated_generation_time_config
        )

        def dynamic_context() -> Mapping[str, Any]:
            if self.eligibility_context_provider is not None:
                return self.eligibility_context_provider()
            orchestrator = runtime.get("orchestrator")
            tools = tuple(getattr(orchestrator, "tools", {}).values()) if orchestrator is not None else ()
            allowed: list[str] = []
            for tool in tools:
                try:
                    audit_source = getattr(tool, "app_tool", tool)
                    manifest = audit_tool(audit_source, audited_allowlist=MANUALLY_AUDITED_ALLOWLIST)
                except Exception:
                    continue
                if manifest.allowlisted and manifest.exclusion_reason is None:
                    allowed.append(manifest.public_name)
            registry = getattr(orchestrator, "context_registry", None)
            if registry is None:
                registry = getattr(env, "context_registry", None)
            context_refs = getattr(orchestrator, "available_context_refs", None)
            if context_refs is None:
                context_refs = getattr(env, "available_context_refs", None)
            if context_refs is None:
                context_refs = getattr(env, "context_refs", None)
            if context_refs is None and isinstance(registry, Mapping):
                context_refs = tuple(registry)
            return {
                "available_context_refs": tuple(context_refs or ()),
                "allowed_worker_tools": tuple(allowed),
                "prior_objectives": tuple(getattr(orchestrator, "prior_objectives", ())),
                "objective_completed": bool(getattr(orchestrator, "objective_completed", False)),
                "terminal": bool(getattr(orchestrator, "terminal", False)),
                "oracle_refs": tuple(getattr(orchestrator, "oracle_refs", getattr(env, "oracle_refs", ()))),
                "scenario_in_scope": bool(getattr(orchestrator, "scenario_in_scope", True)),
            }

        if self.worker_adapter is not None:
            # Explicit callback injection remains available to pure/fake tests.
            worker_callback = self.worker_adapter
        else:
            def resolve_context(ref: str) -> Any:
                if self.context_resolver is not None:
                    if isinstance(self.context_resolver, Mapping):
                        return self.context_resolver[ref]
                    return self.context_resolver(ref)
                orchestrator = runtime.get("orchestrator")
                registry = getattr(orchestrator, "context_registry", {})
                if not registry:
                    registry = getattr(env, "context_registry", {})
                return registry[ref]

            worker_callback = DelegationWorkerAdapter(
                environment=env,
                available_tools=lambda: tuple(getattr(runtime.get("orchestrator"), "tools", {}).values()),
                context_resolver=resolve_context,
                worker_factory=self.worker_factory,
                worker_runner=self.worker_runner,
                llm_engine=llm_engine,
                trace_sink=self.trace_sink,
                pause_env=env.pause,
                resume_env=env.resume_with_offset,
                time_manager=env.time_manager,
                simulated_generation_time_config=time_config,
                log_callback=env.append_to_world_logs,
            )

        gate = self.gate_factory(
            assignment_schedule=self.experiment_config.intervention_schedule,
            block_key=self.experiment_config.intervention_block,
            trace_sink=self.trace_sink,
            worker_callback=worker_callback,
            eligibility_context_provider=dynamic_context,
        )
        executor = InterventionActionExecutor(
            intervention_gate=gate,
            trace_sink=self.trace_sink,
        )
        base_agent = CausalOrchestrator(
            llm_engine=llm_engine,
            action_executor=executor,
            system_prompt=str(base_config.system_prompt),
            max_iterations=max_iterations,
            time_manager=env.time_manager,
            log_callback=env.append_to_world_logs,
            simulated_generation_time_config=time_config,
            eligibility_context_provider=dynamic_context,
        )
        runtime["orchestrator"] = base_agent
        return ARESimulationAgent(
            log_callback=env.append_to_world_logs,
            pause_env=env.pause,
            resume_env=env.resume_with_offset,
            llm_engine=llm_engine,
            base_agent=base_agent,
            time_manager=env.time_manager,
            max_turns=max_turns,
            simulated_generation_time_config=time_config,
        )
