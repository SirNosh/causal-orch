"""Small validated configuration objects for the causal ARE adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping

from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactBaseAgentConfig,
    LLMEngineConfig,
    RunnableARESimulationAgentConfig,
)
from are.simulation.agents.default_agent.prompts.system_prompt import (
    DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT,
)
from are.simulation.types import SimulatedGenerationTimeConfig

from causal_orch.models.manifests import (
    FIXED_GENERATION_SECONDS,
    LocalLlamaConfig,
    LocalModelManifest,
    ModelManifest,
    OpenRouterConfig,
)


CAUSAL_AGENT_NAME = "causal_orchestrator"


@dataclass(frozen=True)
class ExperimentConfig:
    """Explicit experiment dependencies; no credential discovery is performed."""

    model_manifest: ModelManifest | LocalModelManifest
    model_config: OpenRouterConfig | LocalLlamaConfig
    intervention_schedule: Any
    intervention_block: Hashable
    trace_sink: Any
    worker_callback: Callable[[Mapping[str, Any]], Any]
    max_iterations: int = 80
    max_turns: int | None = None
    simulated_generation_time_config: SimulatedGenerationTimeConfig = field(
        default_factory=lambda: SimulatedGenerationTimeConfig(
            mode="fixed", seconds=FIXED_GENERATION_SECONDS
        )
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model_manifest, (ModelManifest, LocalModelManifest)):
            raise TypeError("model_manifest is required")
        if not isinstance(self.model_config, (OpenRouterConfig, LocalLlamaConfig)):
            raise TypeError("model_config is required")
        if self.model_manifest.requested_model_slug != self.model_config.model_slug:
            raise ValueError("model manifest and model config must use the same model")
        if self.model_manifest.selected_provider != self.model_config.provider:
            raise ValueError("model manifest and model config must use the same provider")
        if not hasattr(self.intervention_schedule, "reveal"):
            raise TypeError("intervention_schedule must provide reveal()")
        try:
            hash(self.intervention_block)
        except TypeError as error:
            raise TypeError("intervention_block must be hashable") from error
        if not hasattr(self.trace_sink, "append"):
            raise TypeError("trace_sink must provide append()")
        if not callable(self.worker_callback):
            raise TypeError("worker_callback is required")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if self.max_turns is not None and (
            type(self.max_turns) is not int or self.max_turns < 1
        ):
            raise ValueError("max_turns must be a positive integer or None")
        time_config = self.simulated_generation_time_config
        if (
            time_config.mode != "fixed"
            or time_config.seconds != FIXED_GENERATION_SECONDS
        ):
            raise ValueError("simulated generation time is fixed at 5 seconds")

    @property
    def model(self) -> OpenRouterConfig | LocalLlamaConfig:
        return self.model_config

    @property
    def block_key(self) -> Hashable:
        return self.intervention_block

    @property
    def time_config(self) -> SimulatedGenerationTimeConfig:
        return self.simulated_generation_time_config


@dataclass(frozen=True)
class CausalAgentConfig(RunnableARESimulationAgentConfig):
    """The runnable agent config accepted by pinned ``AbstractAgentBuilder``."""

    max_iterations: int = 80
    max_turns: int | None = None
    agent_name: str = CAUSAL_AGENT_NAME
    model_config: OpenRouterConfig | LocalLlamaConfig | None = None
    system_prompt: str = DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT
    simulated_generation_time_config: SimulatedGenerationTimeConfig = field(
        default_factory=lambda: SimulatedGenerationTimeConfig(
            mode="fixed", seconds=FIXED_GENERATION_SECONDS
        )
    )

    def __post_init__(self) -> None:
        if self.agent_name != CAUSAL_AGENT_NAME:
            raise ValueError(f"agent_name must be {CAUSAL_AGENT_NAME!r}")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if self.max_turns is not None and (
            type(self.max_turns) is not int or self.max_turns < 1
        ):
            raise ValueError("max_turns must be a positive integer or None")
        if (
            self.simulated_generation_time_config.mode != "fixed"
            or self.simulated_generation_time_config.seconds != FIXED_GENERATION_SECONDS
        ):
            raise ValueError("simulated generation time is fixed at 5 seconds")

    def get_agent_name(self) -> str:
        return self.agent_name

    def get_base_agent_config(self) -> ARESimulationReactBaseAgentConfig:
        llm_config = self.model_config
        return ARESimulationReactBaseAgentConfig(
            llm_engine_config=LLMEngineConfig(
                model_name=llm_config.model_slug if llm_config else "",
                provider=llm_config.provider if llm_config else None,
                endpoint=llm_config.endpoint if llm_config else None,
            ),
            simulated_generation_time_config=self.simulated_generation_time_config,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
        )

    def get_model_dump(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "max_iterations": self.max_iterations,
            "max_turns": self.max_turns,
            "model_config": (
                self.model_config.model_slug if self.model_config is not None else None
            ),
            "system_prompt": self.system_prompt,
            "simulated_generation_time_config": {
                "mode": self.simulated_generation_time_config.mode,
                "seconds": self.simulated_generation_time_config.seconds,
            },
        }

    def get_model_json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["agent_name", "max_iterations"],
            "properties": {
                "agent_name": {"const": CAUSAL_AGENT_NAME},
                "max_iterations": {"type": "integer", "minimum": 1},
                "max_turns": {"type": ["integer", "null"], "minimum": 1},
            },
        }

    def validate_model(
        self, agent_config_dict: dict[str, Any]
    ) -> "CausalAgentConfig":
        if not isinstance(agent_config_dict, dict):
            raise TypeError("agent_config_dict must be a dictionary")
        return CausalAgentConfig(
            agent_name=agent_config_dict.get("agent_name", self.agent_name),
            max_iterations=agent_config_dict.get("max_iterations", self.max_iterations),
            max_turns=agent_config_dict.get("max_turns", self.max_turns),
            model_config=self.model_config,
            system_prompt=agent_config_dict.get("system_prompt", self.system_prompt),
            simulated_generation_time_config=self.simulated_generation_time_config,
        )


class CausalAgentConfigBuilder:
    """Create one explicit runnable config from an experiment config."""

    def __init__(self, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config

    def build(
        self,
        agent_name: str = CAUSAL_AGENT_NAME,
        *,
        max_turns: int | None = None,
    ) -> CausalAgentConfig:
        if agent_name != CAUSAL_AGENT_NAME:
            raise ValueError(f"agent {agent_name!r} is not available")
        return CausalAgentConfig(
            max_iterations=self.experiment_config.max_iterations,
            max_turns=(
                self.experiment_config.max_turns if max_turns is None else max_turns
            ),
            model_config=self.experiment_config.model_config,
            simulated_generation_time_config=(
                self.experiment_config.simulated_generation_time_config
            ),
        )
