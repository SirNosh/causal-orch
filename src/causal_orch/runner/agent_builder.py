"""Pinned ARE agent-builder adapter for causal-orch."""

from __future__ import annotations

from typing import Any, Callable

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
    ) -> None:
        self.experiment_config = experiment_config
        self.trace_sink = (
            experiment_config.trace_sink if trace_sink is None else trace_sink
        )
        self.engine_factory = engine_factory
        self.gate_factory = gate_factory

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

        gate = self.gate_factory(
            assignment_schedule=self.experiment_config.intervention_schedule,
            block_key=self.experiment_config.intervention_block,
            trace_sink=self.trace_sink,
            worker_callback=self.experiment_config.worker_callback,
        )
        executor = InterventionActionExecutor(
            intervention_gate=gate,
            trace_sink=self.trace_sink,
        )
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
        base_agent = CausalOrchestrator(
            llm_engine=llm_engine,
            action_executor=executor,
            max_iterations=max_iterations,
            time_manager=env.time_manager,
            log_callback=env.append_to_world_logs,
            simulated_generation_time_config=time_config,
        )
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
