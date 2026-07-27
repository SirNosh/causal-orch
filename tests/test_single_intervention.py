import json

from causal_orch.intervention import (
    Assignment,
    AssignmentRecord,
    InterventionGate,
)
from causal_orch.trace import Trace


class CountingSchedule:
    def __init__(self, assignment):
        self.assignment = assignment
        self.reveals = 0

    def reveal(self, run_id):
        self.reveals += 1
        return AssignmentRecord(
            f"{run_id}:assignment:1", self.assignment
        )


def test_only_first_valid_delegation_is_randomized():
    schedule = CountingSchedule(Assignment.EXECUTE)
    gate = InterventionGate(
        run_id="run", schedule=schedule, trace=Trace("run")
    )
    workers = []

    first = gate.intervene(
        "first",
        terminal=False,
        run_worker=lambda objective, _: workers.append(objective) or "done",
    )
    second = gate.intervene(
        "second",
        terminal=False,
        run_worker=lambda objective, _: workers.append(objective) or "done",
    )

    assert first.eligible
    assert not second.eligible
    assert schedule.reveals == 1
    assert workers == ["first"]


def test_suppression_never_launches_worker():
    gate = InterventionGate(
        run_id="run",
        schedule=CountingSchedule(Assignment.SUPPRESS),
        trace=Trace("run"),
    )
    launched = False

    def worker(*_):
        nonlocal launched
        launched = True
        return "unexpected"

    result = gate.intervene("objective", terminal=False, run_worker=worker)

    assert result.assignment is Assignment.SUPPRESS
    assert not launched


def test_later_delegations_receive_one_fixed_already_decided_result():
    gate = InterventionGate(
        run_id="run",
        schedule=CountingSchedule(Assignment.SUPPRESS),
        trace=Trace("run"),
    )
    gate.intervene("first", terminal=False, run_worker=lambda *_: "unused")

    second = gate.intervene(
        "second", terminal=False, run_worker=lambda *_: "unused"
    )
    third = gate.intervene(
        "third", terminal=False, run_worker=lambda *_: "unused"
    )

    expected = {"status": "DELEGATION_ALREADY_DECIDED"}
    assert json.loads(second.observation) == expected
    assert second.observation == third.observation


def test_suppression_observation_is_fixed_and_strategy_neutral():
    observations = []
    for run_id in ("run-1", "run-2"):
        gate = InterventionGate(
            run_id=run_id,
            schedule=CountingSchedule(Assignment.SUPPRESS),
            trace=Trace(run_id),
        )
        observations.append(
            gate.intervene(
                "task-specific objective",
                terminal=False,
                run_worker=lambda *_: "unused",
            ).observation
        )

    assert observations[0] == observations[1]
    assert json.loads(observations[0]) == {
        "reason": "EXPERIMENTAL_CONTROL",
        "status": "DELEGATION_UNAVAILABLE",
    }
