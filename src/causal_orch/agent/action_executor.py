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
from are.simulation.exceptions import (
    FormatError,
    InvalidActionAgentError,
    JsonExecutionAgentError,
    JsonParsingAgentError,
    UnavailableToolAgentError,
)

from causal_orch.tracing.events import EventName, OrchestrationEvent


def trace_model_output_rejection(
    trace_sink: Any,
    error: Exception,
    *,
    actor_id: str | None,
    actor_role: str,
) -> None:
    """Record an accepted model response that ARE could not execute."""

    if trace_sink is None:
        return
    if isinstance(error, UnavailableToolAgentError):
        reason = "UNKNOWN_TOOL"
    elif isinstance(error, JsonExecutionAgentError):
        reason = "INVALID_TOOL_ARGUMENTS"
    elif isinstance(error, JsonParsingAgentError):
        reason = "JSON_PARSE"
    elif isinstance(error, FormatError):
        reason = "ACTION_PARSE_FAILURE"
    elif isinstance(error, InvalidActionAgentError):
        reason = (
            "MISSING_ACTION_TOKEN"
            if "token" in str(error).lower()
            or "formatted correctly" in str(error).lower()
            else "ACTION_PARSE_FAILURE"
        )
    else:
        return

    events = tuple(getattr(trace_sink, "events", ()))
    response_index = next(
        (
            index
            for index in range(len(events) - 1, -1, -1)
            if events[index].event_type is EventName.MODEL_RESPONSE
        ),
        None,
    )
    request_id = (
        events[response_index].openrouter_request_id
        if response_index is not None
        else None
    )
    if response_index is not None and any(
        event.event_type is EventName.MODEL_OUTPUT_REJECTED
        and event.openrouter_request_id == request_id
        for event in events[response_index + 1 :]
    ):
        return
    trace_sink.append(
        OrchestrationEvent(
            event_type=EventName.MODEL_OUTPUT_REJECTED,
            actor_id=actor_id,
            actor_role=actor_role,
            openrouter_request_id=request_id,
            error_type=reason,
            payload={"reason": reason},
        )
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
