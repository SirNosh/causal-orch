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
    InterventionGate,
    prepare_assignment_manifest,
)
from .model_client import ModelClient, OpenAICompatibleClient
from .trace import Trace
from .worker import ReadOnlyWorker


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    scenario_id: str
    attempt_id: str
    assignment_unit_id: str
    assignment_id: str | None
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
    attempt_id: str = "1",
    assignment_unit_id: str | None = None,
    trace_path: str | Path | None = None,
    forced_delegation_objective: str | None = None,
    model_time_seconds: float = 5.0,
    adapter_factory: Callable[[str | Path], Any] = Gaia2Adapter,
) -> RunOutcome:
    run_id = run_id or uuid4().hex
    assignment_unit_id = assignment_unit_id or run_id
    world = adapter_factory(scenario_path)
    scenario_id = getattr(
        getattr(world, "scenario", None),
        "scenario_id",
        Path(scenario_path).stem,
    )
    trace = Trace(
        run_id,
        trace_path,
        scenario_id=scenario_id,
        attempt_id=attempt_id,
        assignment_unit_id=assignment_unit_id,
    )
    gate = InterventionGate(
        run_id=run_id,
        assignment_unit_id=assignment_unit_id,
        schedule=schedule,
        trace=trace,
    )
    worker = ReadOnlyWorker(
        model=model,
        world=world,
        trace=trace,
        model_time_seconds=model_time_seconds,
    )
    loop = AgentLoop(
        model=model,
        world=world,
        worker=worker,
        gate=gate,
        trace=trace,
        model_time_seconds=model_time_seconds,
        forced_delegation_objective=forced_delegation_objective,
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
        trace.emit(
            "run_failed", error_type=type(exc).__name__, error=error
        )
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
        trace.emit(
            "validation_failed",
            error_type=type(exc).__name__,
            error=validation_error,
        )
    finally:
        world.close()
    trace.emit("run_completed", success=validation_success, error=error)
    return RunOutcome(
        run_id=run_id,
        scenario_id=scenario_id,
        attempt_id=attempt_id,
        assignment_unit_id=assignment_unit_id,
        assignment_id=gate.assignment_id,
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


def _scenario_id_from_path(path: str | Path) -> str:
    scenario_path = Path(path)
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    return str(
        data.get("scenario_id") or data.get("id") or scenario_path.stem
    )


def _safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in value
    )


def _next_attempt_ids(
    trace_root: Path, assignment_unit_ids: list[str]
) -> dict[str, str]:
    latest = {unit: 0 for unit in assignment_unit_ids}
    for trace_path in trace_root.glob("*.jsonl"):
        try:
            with trace_path.open(encoding="utf-8") as handle:
                first_line = handle.readline()
            record = json.loads(first_line)
            unit = record.get("assignment_unit_id")
            if unit in latest:
                latest[unit] = max(
                    latest[unit], int(record.get("attempt_id") or 0)
                )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return {unit: str(value + 1) for unit, value in latest.items()}


def run_batch(
    *,
    scenario_paths: Sequence[str | Path],
    model: ModelClient,
    mode: str,
    seed: str,
    trace_dir: str | Path,
    assignment: str | None = None,
    forced_delegation_objective: str | None = None,
    batch_id: str | None = None,
    attempt_id: str | None = None,
    assignment_manifest_path: str | Path | None = None,
    model_time_seconds: float = 5.0,
) -> list[RunOutcome]:
    trace_root = Path(trace_dir)
    batch_id = batch_id or f"{mode}-{seed}"
    repetitions: dict[str, int] = {}
    units: list[tuple[str | Path, str, str]] = []
    for scenario_path in scenario_paths:
        scenario_id = _scenario_id_from_path(scenario_path)
        repetitions[scenario_id] = repetitions.get(scenario_id, 0) + 1
        assignment_unit_id = (
            f"{batch_id}:{scenario_id}:repeat-{repetitions[scenario_id]}"
        )
        units.append((scenario_path, scenario_id, assignment_unit_id))
    fixed_assignment = (
        Assignment(assignment)
        if assignment is not None
        else (
            Assignment.EXECUTE
            if mode == "smoke"
            else Assignment.SUPPRESS if mode == "proposal" else None
        )
    )
    manifest_path = Path(
        assignment_manifest_path
        or trace_root
        / f"{_safe_filename(batch_id)}-assignment-manifest.json"
    )
    schedule = prepare_assignment_manifest(
        manifest_path,
        batch_id=batch_id,
        seed=seed,
        assignment_unit_ids=[unit[2] for unit in units],
        fixed_assignment=fixed_assignment,
    )
    attempt_ids = (
        {unit[2]: attempt_id for unit in units}
        if attempt_id is not None
        else _next_attempt_ids(
            trace_root, [unit[2] for unit in units]
        )
    )
    outcomes = []
    for scenario_path, scenario_id, assignment_unit_id in units:
        unit_attempt_id = attempt_ids[assignment_unit_id]
        run_id = (
            f"{mode}-{scenario_id}-attempt-{unit_attempt_id}-"
            f"{uuid4().hex[:8]}"
        )
        outcomes.append(
            run_one(
                scenario_path=scenario_path,
                model=model,
                schedule=schedule,
                run_id=run_id,
                attempt_id=unit_attempt_id,
                assignment_unit_id=assignment_unit_id,
                trace_path=trace_root / f"{run_id}.jsonl",
                forced_delegation_objective=forced_delegation_objective,
                model_time_seconds=model_time_seconds,
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
    parser.add_argument("--batch-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--assignment-manifest")
    parser.add_argument("--model-time-seconds", type=float, default=5.0)
    parser.add_argument("--trace-dir", default="artifacts/minimal")
    parser.add_argument(
        "--assignment", choices=[item.value for item in Assignment]
    )
    parser.add_argument("--force-delegation-objective")
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
        assignment=args.assignment,
        forced_delegation_objective=args.force_delegation_objective,
        batch_id=args.batch_id,
        attempt_id=args.attempt_id,
        assignment_manifest_path=args.assignment_manifest,
        model_time_seconds=args.model_time_seconds,
    )
    print(json.dumps([asdict(outcome) for outcome in outcomes], indent=2))
    return 0 if all(outcome.error is None for outcome in outcomes) else 1
