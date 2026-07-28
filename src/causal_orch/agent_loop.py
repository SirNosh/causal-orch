"""Minimal synchronous orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Protocol, Sequence

from .actions import (
    DELEGATE_TOOL,
    FINAL_TOOL,
    DelegateAction,
    FinalAction,
    Notification,
    NotificationType,
    ToolAction,
    decode_action,
)
from .intervention import InterventionGate
from .model_client import ModelClient
from .trace import Trace
from .worker import ReadOnlyWorker


class AgentWorld(Protocol):
    def tool_schemas(self) -> Sequence[Mapping[str, Any]]: ...

    def execute_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> tuple[str, str]: ...

    def pause_time(self) -> None: ...

    def resume_time(self, fixed_offset_seconds: float) -> None: ...

    def notifications(self) -> Sequence[Notification]: ...

    def finish(self, answer: str) -> None: ...


@dataclass(frozen=True)
class AgentResult:
    answer: str
    steps: int
    intervention_occurred: bool


class AgentLoop:
    def __init__(
        self,
        *,
        model: ModelClient,
        world: AgentWorld,
        worker: ReadOnlyWorker,
        gate: InterventionGate,
        trace: Trace,
        max_steps: int = 20,
        max_output_tokens: int = 8_192,
        model_time_seconds: float = 5.0,
        delegation_enabled: bool = True,
        forced_delegation_objective: str | None = None,
    ) -> None:
        self.model = model
        self.world = world
        self.worker = worker
        self.gate = gate
        self.trace = trace
        self.max_steps = max_steps
        self.max_output_tokens = max_output_tokens
        self.model_time_seconds = model_time_seconds
        self.delegation_enabled = delegation_enabled
        self.forced_delegation_objective = forced_delegation_objective

    def run(self, task: str) -> AgentResult:
        ordinary_tools = list(self.world.tool_schemas())
        ordinary_names = {
            item["function"]["name"]
            for item in ordinary_tools
            if item.get("type") == "function"
        }
        tools = [
            *ordinary_tools,
            *([DELEGATE_TOOL] if self.delegation_enabled else []),
            FINAL_TOOL,
        ]
        system_prompt = (
            "Solve the task using exactly one provided function per step. "
            "Use final_answer when done."
        )
        if self.delegation_enabled:
            system_prompt = (
                "Solve the task using exactly one provided function per step. "
                "Use delegate only for a focused read-only research objective. "
                "Use final_answer when done."
            )
        messages: list[Mapping[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        observations: list[str] = []
        self.trace.emit("agent_started", task=task, model=self.model.model)

        def deliver_notifications() -> None:
            for notice in self.world.notifications():
                self.trace.emit(
                    "notification_delivered",
                    notification_type=notice.type,
                    content=notice.content,
                )
                if notice.type is NotificationType.ENVIRONMENT_STOP:
                    raise RuntimeError(
                        f"Gaia2 environment stopped: {notice.content}"
                    )
                if notice.type is NotificationType.USER_MESSAGE:
                    content = notice.content
                else:
                    content = f"Environment notification: {notice.content}"
                observations.append(content)
                messages.append({"role": "user", "content": content})

        def delegate(objective: str, call_id: str):
            result = self.gate.intervene(
                objective,
                terminal=False,
                run_worker=lambda delegated_objective, worker_id: self.worker.run(
                    original_task=task,
                    objective=delegated_objective,
                    scratchpad=messages,
                    observations=observations,
                    worker_id=worker_id,
                ).observation(),
            )
            self.trace.emit(
                "action_executed",
                model_call_id=call_id,
                action_type="delegate",
                proposal_id=result.proposal_id,
                assignment_id=result.assignment_id,
            )
            return result.observation

        deliver_notifications()
        if self.forced_delegation_objective is not None:
            if not self.delegation_enabled:
                raise ValueError(
                    "forced delegation requires delegation to be enabled"
                )
            call_id = f"{self.trace.run_id}:forced-delegation"
            arguments = {"objective": self.forced_delegation_objective}
            action = decode_action("delegate", arguments, ordinary_names)
            self.trace.emit(
                "action_validation",
                model_call_id=None,
                source="forced_smoke",
                status="valid",
                canonical_action=action,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "delegate",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            )
            observation = delegate(action.objective, call_id)
            observations.append(observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": observation,
                }
            )
            deliver_notifications()

        for step in range(1, self.max_steps + 1):
            model_call_id = (
                f"{self.trace.run_id}:orchestrator:model-call:{step}"
            )
            self.trace.emit(
                "model_request",
                actor="orchestrator",
                model_call_id=model_call_id,
                step=step,
                max_tokens=self.max_output_tokens,
                fixed_model_time_seconds=self.model_time_seconds,
            )
            self.world.pause_time()
            try:
                turn = self.model.complete(
                    messages,
                    tools,
                    max_tokens=self.max_output_tokens,
                )
            except Exception as exc:
                self.trace.emit(
                    "model_call_failed",
                    actor="orchestrator",
                    model_call_id=model_call_id,
                    step=step,
                    error_type=type(exc).__name__,
                    error=f"{type(exc).__name__}: {exc}",
                    raw=getattr(exc, "raw_response", None),
                )
                raise
            finally:
                self.world.resume_time(self.model_time_seconds)
            self.trace.emit(
                "model_response",
                actor="orchestrator",
                model_call_id=model_call_id,
                step=step,
                raw=turn.raw_response,
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                latency_seconds=turn.latency_seconds,
            )
            try:
                action = decode_action(
                    turn.tool_name, turn.arguments, ordinary_names
                )
            except Exception as exc:
                self.trace.emit(
                    "action_validation",
                    actor="orchestrator",
                    model_call_id=model_call_id,
                    status="invalid",
                    error_type=type(exc).__name__,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            self.trace.emit(
                "action_validation",
                actor="orchestrator",
                model_call_id=model_call_id,
                status="valid",
                canonical_action=action,
            )
            messages.append(turn.assistant_message)

            if isinstance(action, FinalAction):
                self.world.finish(action.answer)
                self.trace.emit(
                    "action_executed",
                    model_call_id=model_call_id,
                    action_type="final",
                    answer=action.answer,
                )
                self.trace.emit("final_answer", answer=action.answer)
                return AgentResult(
                    answer=action.answer,
                    steps=step,
                    intervention_occurred=self.gate.decided,
                )

            if isinstance(action, ToolAction):
                content, reference = self.world.execute_tool(
                    action.tool, action.arguments
                )
                observation = f"{content}\nEvidence reference: {reference}"
                self.trace.emit(
                    "tool_result",
                    actor="orchestrator",
                    model_call_id=model_call_id,
                    tool=action.tool,
                    arguments=action.arguments,
                    reference=reference,
                )
                self.trace.emit(
                    "action_executed",
                    model_call_id=model_call_id,
                    action_type="tool",
                    tool=action.tool,
                    arguments=action.arguments,
                    result_reference=reference,
                )
            elif isinstance(action, DelegateAction):
                observation = delegate(action.objective, model_call_id)
            else:  # pragma: no cover - the union is exhaustive
                raise AssertionError(f"unsupported action: {action!r}")

            observations.append(observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": turn.tool_call_id,
                    "content": observation,
                }
            )
            deliver_notifications()
        raise RuntimeError("orchestrator step budget exhausted")
