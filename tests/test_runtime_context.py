import unittest

from causal_orch.tracing.context import RunContext
from causal_orch.tracing.events import EventName, OrchestrationEvent
from causal_orch.tracing.sink import InMemoryTraceSink


class RuntimeContextTests(unittest.TestCase):
    def test_context_is_immutable_and_metadata_is_snapshotted(self) -> None:
        metadata = {"batch": {"index": 1}}
        context = RunContext(run_id="run", block_metadata=metadata)
        metadata["batch"]["index"] = 2

        self.assertEqual(context.block_metadata["batch"]["index"], 1)
        with self.assertRaises(TypeError):
            context.block_metadata["new"] = "value"
        with self.assertRaises(Exception):
            context.run_id = "other"

    def test_runtime_events_inherit_context_without_manual_chain_fields(self) -> None:
        sink = InMemoryTraceSink(context=RunContext(run_id="run", scenario_id="scenario"))
        sink.append(OrchestrationEvent(event_type=EventName.ORCHESTRATOR_PROPOSAL, event_id="proposal"))
        sink.append(OrchestrationEvent(event_type=EventName.PROPOSAL_VALIDATION, event_id="validation"))

        self.assertEqual(sink.events[0].scenario_id, "scenario")
        self.assertEqual(sink.events[1].causal_parent_ids, ("proposal",))
        self.assertEqual([event.event_sequence for event in sink.events], [1, 2])


if __name__ == "__main__":
    unittest.main()
