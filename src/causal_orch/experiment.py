"""One-run kernel plus thin smoke and pilot command-line entry points."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

from .agent_loop import AgentLoop
from .gaia2_adapter import Gaia2Adapter
from .intervention import (
    Assignment,
    AssignmentSchedule,
    FixedAssignment,
    InterventionGate,
    RandomAssignment,
)
from .model_client import ModelClient, OpenAICompatibleClient
from .trace import Trace
from .worker import ReadOnlyWorker


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    scenario_id: str
    assignment: str | None
    eligible_delegation: bool
    worker_completed: bool
    reached_validation: bool
    gaia2_success: bool | None
    answer: str | None
    model_calls: int
    input_tokens: int
    output_tokens: int
    model_latency_seconds: float
    tool_calls: int
    error: str | None


def _sum_trace(trace: Trace, field: str) -> float:
    return sum(
        float(event["payload"].get(field) or 0)
        for event in trace.events
        if event["event"] == "model_response"
    )


def run_one(
    *,
    scenario_path: str | Path,
    model: ModelClient,
    schedule: AssignmentSchedule,
    run_id: str | None = None,
    trace_path: str | Path | None = None,
    adapter_factory: Callable[[str | Path], Any] = Gaia2Adapter,
) -> RunOutcome:
    run_id = run_id or uuid4().hex
    trace = Trace(run_id, trace_path)
    world = adapter_factory(scenario_path)
    gate = InterventionGate(run_id=run_id, schedule=schedule, trace=trace)
    worker = ReadOnlyWorker(model=model, world=world, trace=trace)
    loop = AgentLoop(
        model=model, world=world, worker=worker, gate=gate, trace=trace
    )
    result = None
    validation_success = None
    reached_validation = False
    error = None
    trace.emit("run_started", scenario_path=str(scenario_path))
    try:
        world.start()
        result = loop.run(world.task)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        trace.emit("run_failed", error=error)
    try:
        if getattr(world, "_started", True):
            validation_success, validation = world.validate()
            reached_validation = True
            trace.emit(
                "gaia2_validation",
                success=validation_success,
                validator=type(validation).__name__,
            )
    except Exception as exc:
        validation_error = f"{type(exc).__name__}: {exc}"
        error = error or validation_error
        trace.emit("validation_failed", error=validation_error)
    finally:
        world.close()
    trace.emit("run_completed", success=validation_success, error=error)
    return RunOutcome(
        run_id=run_id,
        scenario_id=getattr(
            getattr(world, "scenario", None),
            "scenario_id",
            Path(scenario_path).stem,
        ),
        assignment=gate.assignment.value if gate.assignment else None,
        eligible_delegation=gate.decided,
        worker_completed=trace.count("worker_completed") == 1,
        reached_validation=reached_validation,
        gaia2_success=validation_success,
        answer=result.answer if result is not None else None,
        model_calls=trace.count("model_request"),
        input_tokens=int(_sum_trace(trace, "input_tokens")),
        output_tokens=int(_sum_trace(trace, "output_tokens")),
        model_latency_seconds=_sum_trace(trace, "latency_seconds"),
        tool_calls=trace.count("tool_result"),
        error=error,
    )


def _schedule(mode: str, seed: str) -> AssignmentSchedule:
    if mode == "smoke":
        return FixedAssignment(Assignment.EXECUTE)
    if mode == "proposal":
        return FixedAssignment(Assignment.SUPPRESS)
    return RandomAssignment(seed)


def run_batch(
    *,
    scenario_paths: Sequence[str | Path],
    model: ModelClient,
    mode: str,
    seed: str,
    trace_dir: str | Path,
) -> list[RunOutcome]:
    trace_root = Path(trace_dir)
    outcomes = []
    for index, scenario_path in enumerate(scenario_paths, 1):
        run_id = f"{mode}-{index}-{uuid4().hex[:8]}"
        outcomes.append(
            run_one(
                scenario_path=scenario_path,
                model=model,
                schedule=_schedule(mode, seed),
                run_id=run_id,
                trace_path=trace_root / f"{run_id}.jsonl",
            )
        )
    return outcomes


def main(mode: str) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", action="append", required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "CAUSAL_ORCH_BASE_URL", "https://api.openai.com/v1"
        ),
    )
    parser.add_argument("--model", default=os.environ.get("CAUSAL_ORCH_MODEL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--seed", default="paper-1-pilot")
    parser.add_argument("--trace-dir", default="artifacts/minimal")
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or CAUSAL_ORCH_MODEL is required")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.base_url.startswith(
        ("http://127.0.0.1", "http://localhost")
    ):
        parser.error(f"{args.api_key_env} is required for a remote endpoint")
    client = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
    )
    outcomes = run_batch(
        scenario_paths=args.scenario,
        model=client,
        mode=mode,
        seed=args.seed,
        trace_dir=args.trace_dir,
    )
    print(json.dumps([asdict(outcome) for outcome in outcomes], indent=2))
    return 0 if all(outcome.error is None for outcome in outcomes) else 1
