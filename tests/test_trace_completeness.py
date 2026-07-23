import json
import tempfile
import unittest
from pathlib import Path

from causal_orch.tracing.events import EventName, OrchestrationEvent
from causal_orch.tracing.sink import InMemoryTraceSink, TraceSink


CHAIN = [
    EventName.ORCHESTRATOR_PROPOSAL,
    EventName.PROPOSAL_VALIDATION,
    EventName.INTERVENTION_ELIGIBILITY,
    EventName.INTERVENTION_ASSIGNMENT,
    EventName.DELEGATION_EXECUTED,
    EventName.ORCHESTRATOR_RESUMED,
    EventName.JUDGE_RESULT,
]


class TraceTests(unittest.TestCase):
    def test_ordering_and_json_serialization(self) -> None:
        sink = InMemoryTraceSink()
        parent = "event-0"
        for index, event_name in enumerate(CHAIN):
            event = OrchestrationEvent(
                event_type=event_name,
                event_id=f"event-{index + 1}",
                causal_parent_ids=() if index == 0 else (parent,),
                payload={"index": index},
            )
            sink.append(event)
            parent = event.event_id
        self.assertEqual([event.event_type for event in sink.events], CHAIN)
        encoded = sink.events[0].to_json()
        self.assertEqual(json.loads(encoded)["event_id"], "event-1")
        self.assertEqual(encoded, sink.events[0].to_json())

    def test_append_only_and_snapshot_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "trace.jsonl"
            payload = {"value": "before"}
            first = OrchestrationEvent(event_type=EventName.RUN_STARTED, event_id="one", payload=payload)
            with TraceSink(path) as sink:
                sink.append(first)
                payload["value"] = "after"
                sink.append(OrchestrationEvent(event_type=EventName.RUN_COMPLETED, event_id="two"))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["payload"]["value"], "before")
            self.assertEqual(json.loads(lines[1])["event_id"], "two")

    def test_hidden_reasoning_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OrchestrationEvent(event_type=EventName.MODEL_RESPONSE, payload={"thought": "private"})


if __name__ == "__main__":
    unittest.main()
