"""ARE action executor with a non-tool delegation pseudo-action."""

from __future__ import annotations

import json
from typing import Any, Callable

from are.simulation.agents.agent_log import (
    BaseAgentLog,
    ObservationLog,
    RationaleLog,
    ToolCallLog,
)
from are.simulation.agents.default_agent.tools.action_executor import ParsedAction
from are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
    get_observation_log,
)


class InterventionActionExecutor(JsonActionExecutor):
    """Intercept ``DELEGATE`` after ARE parsing and preserve ordinary actions."""

    def __init__(self, *, intervention_gate: Any, trace_sink: Any, tools=None, **kwargs: Any):
        super().__init__(tools=tools, **kwargs)
        self.intervention_gate = intervention_gate
        self.trace_sink = trace_sink
        self.delegation_handoff_pending = False

    def execute_parsed_action(
        self,
        parsed_action: ParsedAction,
        append_agent_log: Callable[[BaseAgentLog], None],
        make_timestamp: Callable[[], float],
        agent_id: str,
    ) -> Any:
        if parsed_action.tool_name != "DELEGATE":
            return super().execute_parsed_action(
                parsed_action, append_agent_log, make_timestamp, agent_id
            )

        if parsed_action.rationale is not None:
            append_agent_log(
                RationaleLog(
                    content=parsed_action.rationale,
                    timestamp=make_timestamp(),
                    agent_id=agent_id,
                )
            )
        arguments = parsed_action.arguments if parsed_action.arguments else {}
        append_agent_log(
            ToolCallLog(
                tool_name=parsed_action.tool_name,
                tool_arguments=arguments,
                timestamp=make_timestamp(),
                agent_id=agent_id,
            )
        )
        observation = self.intervention_gate.handle_proposal(arguments)
        if (
            isinstance(observation, dict)
            and observation.get("status") == "DELEGATION_EXECUTED"
        ):
            self.delegation_handoff_pending = True
        append_agent_log(
            get_observation_log(
                make_timestamp(),
                json.dumps(observation, sort_keys=True, separators=(",", ":")),
                agent_id,
            )
        )
        return observation
