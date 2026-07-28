"""A structurally read-only worker with one small result contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .model_client import ModelClient
from .trace import Trace


class WorkerWorld(Protocol):
    def read_only_tool_schemas(self) -> Sequence[Mapping[str, Any]]: ...

    def execute_read_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> tuple[str, str]: ...

    def state_hash(self) -> str: ...

    def pause_time(self) -> None: ...

    def resume_time(self, fixed_offset_seconds: float) -> None: ...


@dataclass(frozen=True)
class WorkerResult:
    result: str
    evidence: tuple[str, ...]

    def observation(self) -> str:
        cited = ", ".join(self.evidence) if self.evidence else "none"
        return f"Worker findings: {self.result}\nEvidence references: {cited}"


RETURN_RESULT_TOOL = {
    "type": "function",
    "function": {
        "name": "return_worker_result",
        "description": "Finish the delegated research and return concise findings.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "References returned by tools in this worker episode.",
                },
            },
            "required": ["result", "evidence"],
            "additionalProperties": False,
        },
    },
}


class ReadOnlyWorker:
    def __init__(
        self,
        *,
        model: ModelClient,
        world: WorkerWorld,
        trace: Trace,
        max_steps: int = 6,
        max_output_tokens: int = 2_000,
        model_time_seconds: float = 5.0,
    ) -> None:
        self.model = model
        self.world = world
        self.trace = trace
        self.max_steps = max_steps
        self.max_output_tokens = max_output_tokens
        self.model_time_seconds = model_time_seconds

    def run(
        self,
        *,
        original_task: str,
        objective: str,
        scratchpad: Sequence[Mapping[str, Any]],
        observations: Sequence[str],
        worker_id: str = "worker-1",
    ) -> WorkerResult:
        schemas = list(self.world.read_only_tool_schemas())
        allowed = {
            item["function"]["name"]
            for item in schemas
            if item.get("type") == "function"
        }
        messages: list[Mapping[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a read-only research worker. Use only the tools "
                    "provided. Return concise findings with evidence references "
                    "through return_worker_result."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original task:\n{original_task}\n\n"
                    f"Delegated objective:\n{objective}\n\n"
                    f"Recent observations:\n"
                    + "\n".join(observations[-6:])
                    + f"\n\nStructured scratchpad:\n{list(scratchpad)[-8:]}"
                ),
            },
        ]
        self.world.pause_time()
        before_hash = self.world.state_hash()
        evidence: set[str] = set()
        output_tokens = 0
        model_calls = 0
        self.trace.emit(
            "worker_started",
            worker_id=worker_id,
            objective=objective,
            tool_names=sorted(allowed),
            state_before_hash=before_hash,
        )
        try:
            for step in range(1, self.max_steps + 1):
                remaining_tokens = self.max_output_tokens - output_tokens
                if remaining_tokens <= 0:
                    raise RuntimeError("worker output-token budget exhausted")
                model_call_id = f"{worker_id}:model-call:{step}"
                model_calls += 1
                self.trace.emit(
                    "model_request",
                    actor="worker",
                    worker_id=worker_id,
                    model_call_id=model_call_id,
                    step=step,
                    max_tokens=remaining_tokens,
                    fixed_model_time_seconds=self.model_time_seconds,
                )
                try:
                    turn = self.model.complete(
                        messages,
                        [*schemas, RETURN_RESULT_TOOL],
                        max_tokens=remaining_tokens,
                    )
                except Exception as exc:
                    self.trace.emit(
                        "model_call_failed",
                        actor="worker",
                        worker_id=worker_id,
                        model_call_id=model_call_id,
                        step=step,
                        error_type=type(exc).__name__,
                        error=f"{type(exc).__name__}: {exc}",
                        raw=getattr(exc, "raw_response", None),
                    )
                    raise
                output_tokens += turn.output_tokens
                self.trace.emit(
                    "model_response",
                    actor="worker",
                    worker_id=worker_id,
                    model_call_id=model_call_id,
                    raw=turn.raw_response,
                    input_tokens=turn.input_tokens,
                    output_tokens=turn.output_tokens,
                    latency_seconds=turn.latency_seconds,
                )
                if output_tokens > self.max_output_tokens:
                    raise RuntimeError("worker output-token budget exhausted")
                messages.append(turn.assistant_message)
                if turn.tool_name == "return_worker_result":
                    try:
                        result = turn.arguments.get("result")
                        refs = turn.arguments.get("evidence")
                        if (
                            set(turn.arguments) != {"result", "evidence"}
                            or not isinstance(result, str)
                            or not isinstance(refs, list)
                        ):
                            raise ValueError("invalid worker result")
                        if any(not isinstance(ref, str) for ref in refs):
                            raise ValueError(
                                "worker evidence must contain strings"
                            )
                        unknown = set(refs) - evidence
                        if unknown:
                            raise ValueError(
                                "worker cited unknown evidence: "
                                f"{sorted(unknown)}"
                            )
                    except Exception as exc:
                        self.trace.emit(
                            "action_validation",
                            actor="worker",
                            worker_id=worker_id,
                            model_call_id=model_call_id,
                            status="invalid",
                            error_type=type(exc).__name__,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        raise
                    canonical = WorkerResult(
                        result=result, evidence=tuple(refs)
                    )
                    self.trace.emit(
                        "action_validation",
                        actor="worker",
                        worker_id=worker_id,
                        model_call_id=model_call_id,
                        status="valid",
                        canonical_action=canonical,
                    )
                    self.trace.emit(
                        "action_executed",
                        actor="worker",
                        worker_id=worker_id,
                        model_call_id=model_call_id,
                        action_type="worker_result",
                        worker_result=canonical,
                    )
                    return canonical
                if turn.tool_name not in allowed:
                    self.trace.emit(
                        "action_validation",
                        actor="worker",
                        worker_id=worker_id,
                        model_call_id=model_call_id,
                        status="invalid",
                        error_type="ValueError",
                        error=(
                            "ValueError: worker attempted unavailable tool: "
                            f"{turn.tool_name}"
                        ),
                    )
                    raise ValueError(
                        f"worker attempted unavailable tool: {turn.tool_name}"
                    )
                self.trace.emit(
                    "action_validation",
                    actor="worker",
                    worker_id=worker_id,
                    model_call_id=model_call_id,
                    status="valid",
                    canonical_action={
                        "type": "tool",
                        "tool": turn.tool_name,
                        "arguments": turn.arguments,
                    },
                )
                content, reference = self.world.execute_read_tool(
                    turn.tool_name, turn.arguments
                )
                evidence.add(reference)
                self.trace.emit(
                    "tool_result",
                    actor="worker",
                    worker_id=worker_id,
                    model_call_id=model_call_id,
                    tool=turn.tool_name,
                    arguments=turn.arguments,
                    reference=reference,
                )
                self.trace.emit(
                    "action_executed",
                    actor="worker",
                    worker_id=worker_id,
                    model_call_id=model_call_id,
                    action_type="tool",
                    tool=turn.tool_name,
                    arguments=turn.arguments,
                    result_reference=reference,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": turn.tool_call_id,
                        "content": content,
                    }
                )
            raise RuntimeError("worker step budget exhausted")
        finally:
            after_hash = self.world.state_hash()
            self.trace.emit(
                "worker_state_guard",
                worker_id=worker_id,
                state_before_hash=before_hash,
                state_after_hash=after_hash,
            )
            self.world.resume_time(model_calls * self.model_time_seconds)
            if after_hash != before_hash:
                raise RuntimeError("read-only worker changed Gaia2 application state")
