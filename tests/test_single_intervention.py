from causal_orch.intervention import Assignment, InterventionGate
from causal_orch.trace import Trace


class CountingSchedule:
    def __init__(self, assignment):
        self.assignment = assignment
        self.reveals = 0

    def reveal(self, run_id):
        self.reveals += 1
        return self.assignment


def test_only_first_valid_delegation_is_randomized():
    schedule = CountingSchedule(Assignment.EXECUTE)
    gate = InterventionGate(
        run_id="run", schedule=schedule, trace=Trace("run")
    )
    workers = []

    first = gate.intervene(
        "first",
        terminal=False,
        run_worker=lambda objective: workers.append(objective) or "done",
    )
    second = gate.intervene(
        "second",
        terminal=False,
        run_worker=lambda objective: workers.append(objective) or "done",
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

    def worker(_):
        nonlocal launched
        launched = True
        return "unexpected"

    result = gate.intervene("objective", terminal=False, run_worker=worker)

    assert result.assignment is Assignment.SUPPRESS
    assert not launched
