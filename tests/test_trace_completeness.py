import json
import tempfile
import unittest
from pathlib import Path

from causal_orch.tracing.context import RunContext
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
    def test_sink_context_adds_identity_sequence_timestamp_and_causal_parent(self) -> None:
        context = RunContext(
            run_id="run-1",
            scenario_id="scenario-1",
            universe_id="universe-1",
            capability="delegation",
            model_slug="model-1",
            provider_slug="provider-1",
            temporal_batch="batch-1",
            block_key="block-1",
            block_metadata={"arm": "balanced"},
            simulated_timestamp=12.5,
            correlation_id="corr-1",
        )
        sink = InMemoryTraceSink(context=context, wall_clock=lambda: "2026-07-23T00:00:00+00:00")
        sink.append(OrchestrationEvent(event_type=EventName.RUN_STARTED, event_id="first"))
        sink.append(OrchestrationEvent(event_type=EventName.MODEL_REQUEST, event_id="second"))

        first, second = sink.events
        self.assertEqual(first.run_id, "run-1")
        self.assertEqual(first.provider_slug, "provider-1")
        self.assertEqual(first.event_sequence, 1)
        self.assertEqual(first.wall_timestamp, "2026-07-23T00:00:00+00:00")
        self.assertEqual(first.correlation_id, "corr-1")
        self.assertEqual(second.event_sequence, 2)
        self.assertEqual(second.causal_parent_ids, ("first",))
        self.assertEqual(second.randomization_block_key, "block-1")

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
