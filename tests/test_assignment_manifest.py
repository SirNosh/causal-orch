from causal_orch.intervention import (
    Assignment,
    InterventionGate,
    prepare_assignment_manifest,
)
from causal_orch.experiment import _next_attempt_ids
from causal_orch.trace import Trace


def test_manifest_is_balanced_and_stable_when_reopened(tmp_path):
    path = tmp_path / "assignments.json"
    units = [f"batch:scenario-{index}:repeat-1" for index in range(4)]

    first = prepare_assignment_manifest(
        path,
        batch_id="batch",
        seed="seed",
        assignment_unit_ids=units,
    )
    second = prepare_assignment_manifest(
        path,
        batch_id="batch",
        seed="seed",
        assignment_unit_ids=units,
    )

    assignments = [first.reveal(unit).assignment for unit in units]
    assert assignments.count(Assignment.EXECUTE) == 2
    assert assignments.count(Assignment.SUPPRESS) == 2
    assert [
        first.reveal(unit) for unit in units
    ] == [second.reveal(unit) for unit in units]


def test_retry_gets_same_assignment_with_new_run_id(tmp_path):
    unit = "batch:scenario-a:repeat-1"
    schedule = prepare_assignment_manifest(
        tmp_path / "assignments.json",
        batch_id="batch",
        seed="seed",
        assignment_unit_ids=[unit],
    )
    results = []
    for run_id in ("run-attempt-1", "run-attempt-2"):
        gate = InterventionGate(
            run_id=run_id,
            assignment_unit_id=unit,
            schedule=schedule,
            trace=Trace(
                run_id, assignment_unit_id=unit, attempt_id=run_id[-1]
            ),
        )
        results.append(
            gate.intervene(
                "research",
                terminal=False,
                run_worker=lambda *_: "finding",
            )
        )

    assert results[0].assignment == results[1].assignment
    assert results[0].assignment_id == results[1].assignment_id
    assert results[0].assignment_id == f"{unit}:assignment:1"


def test_attempt_id_increments_from_existing_run_trace(tmp_path):
    unit = "batch:scenario-a:repeat-1"
    Trace(
        "run-1",
        tmp_path / "run-1.jsonl",
        assignment_unit_id=unit,
        attempt_id="1",
    ).emit("run_started")
    Trace(
        "run-2",
        tmp_path / "run-2.jsonl",
        assignment_unit_id=unit,
        attempt_id="2",
    ).emit("run_started")

    assert _next_attempt_ids(tmp_path, [unit]) == {unit: "3"}
