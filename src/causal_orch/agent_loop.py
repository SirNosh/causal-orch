"""Minimal synchronous orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .actions import (
    DELEGATE_TOOL,
    FINAL_TOOL,
    DelegateAction,
    FinalAction,
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

    def notifications(self) -> Sequence[str]: ...

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
    ) -> None:
        self.model = model
        self.world = world
        self.worker = worker
        self.gate = gate
        self.trace = trace
        self.max_steps = max_steps

    def run(self, task: str) -> AgentResult:
        ordinary_tools = list(self.world.tool_schemas())
        ordinary_names = {
            item["function"]["name"]
            for item in ordinary_tools
            if item.get("type") == "function"
        }
        tools = [*ordinary_tools, DELEGATE_TOOL, FINAL_TOOL]
        messages: list[Mapping[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Solve the task using exactly one provided function per step. "
                    "Use delegate only for a focused read-only research objective. "
                    "Use final_answer when done."
                ),
            },
            {"role": "user", "content": task},
        ]
        observations: list[str] = []
        self.trace.emit("agent_started", task=task, model=self.model.model)
        for step in range(1, self.max_steps + 1):
            self.trace.emit("model_request", actor="orchestrator", step=step)
            try:
                turn = self.model.complete(messages, tools)
            except Exception as exc:
                self.trace.emit(
                    "model_call_failed",
                    actor="orchestrator",
                    step=step,
                    error=f"{type(exc).__name__}: {exc}",
                    raw=getattr(exc, "raw_response", None),
                )
                raise
            self.trace.emit(
                "model_response",
                actor="orchestrator",
                step=step,
                raw=turn.raw_response,
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                latency_seconds=turn.latency_seconds,
            )
            action = decode_action(
                turn.tool_name, turn.arguments, ordinary_names
            )
            self.trace.emit("action", actor="orchestrator", action=action)
            messages.append(turn.assistant_message)

            if isinstance(action, FinalAction):
                self.world.finish(action.answer)
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
                    tool=action.tool,
                    reference=reference,
                )
            elif isinstance(action, DelegateAction):
                result = self.gate.intervene(
                    action.objective,
                    terminal=False,
                    run_worker=lambda objective: self.worker.run(
                        original_task=task,
                        objective=objective,
                        scratchpad=messages,
                        observations=observations,
                    ).observation(),
                )
                observation = result.observation
            else:  # pragma: no cover - the union is exhaustive
                raise AssertionError(f"unsupported action: {action!r}")

            notices = list(self.world.notifications())
            if notices:
                observation += "\nEnvironment notifications:\n" + "\n".join(
                    notices
                )
            observations.append(observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": turn.tool_call_id,
                    "content": observation,
                }
            )
        raise RuntimeError("orchestrator step budget exhausted")
