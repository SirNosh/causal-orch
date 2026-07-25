"""The causal orchestrator's thin ARE ``BaseAgent`` specialization."""

from __future__ import annotations

import json
import hashlib
from typing import Any, Callable, Mapping

from are.simulation.agents.agent_log import SystemPromptLog
from are.simulation.agents.default_agent.base_agent import BaseAgent
from are.simulation.agents.default_agent.tools.action_executor import ParsedAction
from are.simulation.agents.default_agent.steps.are_simulation import (
    get_are_simulation_update_pre_step,
)
from are.simulation.agents.default_agent.termination_methods.are_simulation import (
    get_gaia2_termination_step,
)
from are.simulation.agents.default_agent.prompts.system_prompt import (
    DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT,
)

from causal_orch.runtime.context_registry import TraceContextRegistry
from causal_orch.tracing.events import EventName, OrchestrationEvent


_DELEGATE_PROMPT_START = "<!-- causal-orch delegation instructions -->"
_DELEGATE_PROMPT_END = "<!-- end causal-orch delegation instructions -->"


def _delegation_schema(
    available_context_refs: tuple[str, ...] = (),
    permitted_worker_tools: tuple[str, ...] = (),
) -> str:
    context = json.dumps(list(available_context_refs), separators=(",", ":"))
    tools = json.dumps(list(permitted_worker_tools), separators=(",", ":"))
    return f"""
The DELEGATE pseudo-action is available to the orchestrator only. It is not an
application tool and must not be described as one. Use the normal ARE Thought /
Action format with this JSON action when a bounded read-only subtask is needed:

{{
  "action": "DELEGATE",
  "action_input": {{
    "proposal_id": "optional UUID; omit to let the runtime assign one",
    "objective": "string",
    "reason_code": "INFORMATION_GAP",
    "context_refs": ["string"],
    "allowed_read_tools": ["string"],
    "completion_criterion": "string"
  }}
}}

Current eligible context references: {context}
Current permitted worker tool names: {tools}
Use only the current context references and permitted worker tool names above.
The proposal_id field may be omitted; the runtime assigns a deterministic UUID.
Worker execution always uses the fixed experimental budget of 8 steps and 2000
output tokens; these values are not proposal fields and cannot be changed.
Only provide concise, task-relevant rationale. Do not provide or store hidden
chain-of-thought; the intervention response is the only delegation observation.
""".strip()


DELEGATE_ACTION_SCHEMA = _delegation_schema()


class CausalOrchestrator(BaseAgent):
    """Preserve ARE's normal history, action, and logging lifecycle."""

    def __init__(
        self,
        *,
        llm_engine: Callable,
        action_executor: Any,
        tools: dict[str, Any] | None = None,
        system_prompt: str = DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT,
        max_iterations: int = 80,
        time_manager: Any | None = None,
        log_callback: Callable | None = None,
        simulated_generation_time_config: Any | None = None,
        use_custom_logger: bool = True,
        eligibility_context_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        self.eligibility_context_provider = eligibility_context_provider
        self._delegation_base_prompt = system_prompt
        self.context_registry = TraceContextRegistry()
        super().__init__(
            llm_engine=llm_engine,
            system_prompts={
                "system_prompt": self._prompt_with_delegation_schema(system_prompt)
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
        self.stock_system_prompt_hash = hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest()

    def _prompt_with_delegation_schema(self, prompt: str) -> str:
        return (
            f"{prompt}\n\n{_DELEGATE_PROMPT_START}\n"
            f"{self._render_delegation_schema()}\n{_DELEGATE_PROMPT_END}"
        )

    def _render_delegation_schema(self) -> str:
        context: Mapping[str, Any] = {}
        if self.eligibility_context_provider is not None:
            provided = self.eligibility_context_provider()
            if isinstance(provided, Mapping):
                context = provided
        refs = tuple(str(value) for value in context.get("available_context_refs", ()))
        tools = tuple(str(value) for value in context.get("allowed_worker_tools", ()))
        self.delegate_action_schema = _delegation_schema(refs, tools)
        return self.delegate_action_schema

    def refresh_delegation_prompt(self) -> str:
        current = self.init_system_prompts["system_prompt"]
        start = current.find(_DELEGATE_PROMPT_START)
        end = current.find(_DELEGATE_PROMPT_END)
        rendered = self._render_delegation_schema()
        if start < 0 or end < start:
            current = self._prompt_with_delegation_schema(self._delegation_base_prompt)
        else:
            current = (
                current[: start + len(_DELEGATE_PROMPT_START)]
                + "\n"
                + rendered
                + "\n"
                + current[end:]
            )
        self.init_system_prompts["system_prompt"] = current
        return current

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        self.refresh_delegation_prompt()
        super().initialize(*args, **kwargs)

    def append_agent_log(self, log: Any) -> None:
        """Index model-visible task, observation, notification, and artifact logs."""

        super().append_agent_log(log)
        log_type = log.get_type()
        content = log.get_content_for_llm()
        if content is None:
            return
        if log_type == "task":
            self.context_registry.register("task", content)
        elif log_type == "observation":
            self.context_registry.register(f"observation:{log.id}", content)
        elif log_type == "environment_notifications":
            self.context_registry.register(f"notification:{log.id}", content)
        elif log_type == "action":
            self.context_registry.register(f"tool_result:{log.id}", content)
        elif log_type == "subagent":
            self.context_registry.register(f"artifact:{log.id}", content)

    def send_message_to_user(self, content: str) -> None:
        """Preserve ARE termination for the intervention executor subclass."""

        tool_name = "AgentUserInterface__send_message_to_user"
        if tool_name not in self.tools:
            raise ValueError(f"{tool_name} is not in agent tools")
        self.action_executor.execute_parsed_action(
            ParsedAction(tool_name=tool_name, arguments={"content": content}),
            self.append_agent_log,
            self.make_timestamp,
            self.agent_id,
        )

    def step(self) -> None:
        """Refresh the bounded delegation block immediately before each LLM call."""

        prompt = self.refresh_delegation_prompt()
        self.system_prompt = prompt
        for log in self.logs:
            if isinstance(log, SystemPromptLog):
                log.content = prompt
                break
        if self.action_executor.delegation_handoff_pending:
            self.action_executor.trace_sink.append(
                OrchestrationEvent(
                    event_type=EventName.ORCHESTRATOR_RESUMED,
                    actor_id=self.agent_id,
                    actor_role="orchestrator",
                )
            )
            self.action_executor.delegation_handoff_pending = False
        super().step()
