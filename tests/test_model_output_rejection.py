import unittest

from are.simulation.exceptions import (
    InvalidActionAgentError,
    JsonExecutionAgentError,
    JsonParsingAgentError,
    UnavailableToolAgentError,
)

from causal_orch.agent.action_executor import trace_model_output_rejection
from causal_orch.tracing.events import EventName, OrchestrationEvent
from causal_orch.tracing.sink import InMemoryTraceSink


class ModelOutputRejectionTests(unittest.TestCase):
    def test_are_rejections_are_safe_classified_and_deduplicated(self):
        cases = (
            (InvalidActionAgentError("missing Action: token"), "MISSING_ACTION_TOKEN"),
            (JsonParsingAgentError("bad JSON"), "JSON_PARSE"),
            (UnavailableToolAgentError("unknown"), "UNKNOWN_TOOL"),
            (JsonExecutionAgentError("bad args"), "INVALID_TOOL_ARGUMENTS"),
        )
        for index, (error, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                sink = InMemoryTraceSink()
                sink.append(
                    OrchestrationEvent(
                        event_type=EventName.MODEL_RESPONSE,
                        openrouter_request_id=f"request-{index}",
                    )
                )
                trace_model_output_rejection(
                    sink,
                    error,
                    actor_id="agent",
                    actor_role="worker",
                )
                trace_model_output_rejection(
                    sink,
                    error,
                    actor_id="agent",
                    actor_role="worker",
                )
                rejections = [
                    event
                    for event in sink.events
                    if event.event_type is EventName.MODEL_OUTPUT_REJECTED
                ]
                self.assertEqual(len(rejections), 1)
                self.assertEqual(rejections[0].error_type, expected)
                self.assertEqual(rejections[0].payload, {"reason": expected})
                self.assertEqual(
                    rejections[0].openrouter_request_id,
                    f"request-{index}",
                )


if __name__ == "__main__":
    unittest.main()
