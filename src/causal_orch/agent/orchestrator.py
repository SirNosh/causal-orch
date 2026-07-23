"""The causal orchestrator's thin ARE ``BaseAgent`` specialization."""

from __future__ import annotations

from typing import Any, Callable

from are.simulation.agents.default_agent.base_agent import BaseAgent
from are.simulation.agents.default_agent.steps.are_simulation import (
    get_are_simulation_update_pre_step,
)
from are.simulation.agents.default_agent.termination_methods.are_simulation import (
    get_gaia2_termination_step,
)


DELEGATE_ACTION_SCHEMA = """
The DELEGATE pseudo-action is available to the orchestrator only. It is not an
application tool and must not be described as one. Use the normal ARE Thought /
Action format with this JSON action when a bounded read-only subtask is needed:

{
  "action": "DELEGATE",
  "action_input": {
    "proposal_id": "string",
    "objective": "string",
    "reason_code": "INFORMATION_GAP",
    "context_refs": ["string"],
    "allowed_read_tools": ["string"],
    "completion_criterion": "string",
    "max_worker_steps": 1,
    "max_worker_output_tokens": 1
  }
}

Only provide concise, task-relevant rationale. Do not provide or store hidden
chain-of-thought; the intervention response is the only delegation observation.
""".strip()


class CausalOrchestrator(BaseAgent):
    """Preserve ARE's normal history, action, and logging lifecycle."""

    def __init__(
        self,
        *,
        llm_engine: Callable,
        action_executor: Any,
        tools: dict[str, Any] | None = None,
        system_prompt: str = "<<notification_system_description>>\n<<curent_time_description>>\n<<tool_descriptions>>",
        max_iterations: int = 80,
        time_manager: Any | None = None,
        log_callback: Callable | None = None,
        simulated_generation_time_config: Any | None = None,
        use_custom_logger: bool = True,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        super().__init__(
            llm_engine=llm_engine,
            system_prompts={
                "system_prompt": f"{system_prompt}\n\n{DELEGATE_ACTION_SCHEMA}"
            },
            tools={} if tools is None else tools,
            termination_step=get_gaia2_termination_step(),
            conditional_pre_steps=[get_are_simulation_update_pre_step()],
            action_executor=action_executor,
            max_iterations=max_iterations,
            time_manager=time_manager,
            log_callback=log_callback,
            simulated_generation_time_config=simulated_generation_time_config,
            use_custom_logger=use_custom_logger,
        )

        # This is intentionally a prompt-only pseudo-action. ARE application
        # tools use the normal BaseAgent/ARESimulationAgent tool path.
        self.delegate_action_schema = DELEGATE_ACTION_SCHEMA
