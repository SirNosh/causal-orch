from causal_orch.intervention import (
    Assignment,
    AssignmentRecord,
    InterventionGate,
)
from causal_orch.trace import Trace


class SpySchedule:
    def __init__(self):
        self.reveals = 0

    def reveal(self, run_id):
        self.reveals += 1
        return AssignmentRecord(f"{run_id}:assignment:1", Assignment.EXECUTE)


def test_assignment_is_revealed_only_after_eligibility():
    trace = Trace("run")
    schedule = SpySchedule()
    gate = InterventionGate(run_id="run", schedule=schedule, trace=trace)

    rejected = gate.intervene(
        "", terminal=False, run_worker=lambda *_: "unused"
    )
    assert not rejected.eligible
    assert schedule.reveals == 0

    accepted = gate.intervene(
        "Find the policy", terminal=False, run_worker=lambda *_: "found"
    )
    assert accepted.eligible
    assert schedule.reveals == 1
    assert [event["event"] for event in trace.events][-3:] == [
        "delegation_eligibility",
        "assignment_revealed",
        "worker_completed",
    ]
