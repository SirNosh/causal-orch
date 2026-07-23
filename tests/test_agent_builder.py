import unittest

from are.simulation.agents.default_agent.are_simulation_main import ARESimulationAgent
from are.simulation.agents.default_agent.tools.action_executor import ParsedAction
from are.simulation.tool_utils import AppTool

from causal_orch.agent.orchestrator import CausalOrchestrator
from causal_orch.models.manifests import MODEL_CANDIDATE_ORDER, ModelManifest, OpenRouterConfig
from causal_orch.runner.agent_builder import CausalAgentBuilder
from causal_orch.runner.config_builder import CausalAgentConfigBuilder, ExperimentConfig
from causal_orch.runtime.randomization import generate_balanced_schedule
from causal_orch.tracing.sink import InMemoryTraceSink


def make_experiment(**overrides):
    model = MODEL_CANDIDATE_ORDER[0]
    manifest = ModelManifest(
        snapshot_timestamp_utc="2026-07-23T00:00:00Z",
        requested_model_slug=model,
        returned_canonical_slug=model,
        context_length=32768,
        input_price="0",
        output_price="0",
        supported_parameters=("temperature",),
        available_providers=("PinnedProvider",),
        selected_provider="PinnedProvider",
        provider_context_length=32768,
        provider_max_output=4096,
        provider_data_policy="none",
    )
    values = {
        "model_manifest": manifest,
        "model_config": OpenRouterConfig(model, "PinnedProvider"),
        "intervention_schedule": generate_balanced_schedule("test", ["block"]),
        "intervention_block": "block",
        "trace_sink": InMemoryTraceSink(),
        "worker_callback": lambda proposal: {"proposal": proposal},
    }
    values.update(overrides)
    return ExperimentConfig(**values)


class FakeEnvironment:
    def __init__(self):
        self.time_manager = object()
        self.world_logs = []
        self.pause_calls = 0
        self.resume_calls = []

    def append_to_world_logs(self, value):
        self.world_logs.append(value)

    def pause(self):
        self.pause_calls += 1

    def resume_with_offset(self, offset):
        self.resume_calls.append(offset)



class FakeScenario:
    additional_system_prompt = None
    apps = []
    start_time = 0

    def get_tools(self):
        aui = type("AUI", (), {"wait_for_user_response": True})()
        return [
            AppTool(
                class_name="AgentUserInterface",
                app_name="AgentUserInterface",
                name="AgentUserInterface__get_last_message_from_user",
                function_description="Read the last user message",
                args=[],
                function=lambda: "",
                class_instance=aui,
                _public_name="AgentUserInterface__get_last_message_from_user",
                _public_description="Read the last user message",
            ),
            AppTool(
                class_name="FakeApp",
                app_name="FakeApp",
                name="read",
                function_description="Read fake data",
                args=[],
                function=lambda: "ok",
                _public_name="FakeApp__read",
                _public_description="Read fake data",
            ),
        ]


class BuilderTests(unittest.TestCase):
    def test_orchestrator_exposes_delegate_without_an_application_tool(self):
        experiment = make_experiment()
        gate = type(
            "Gate",
            (),
            {"handle_proposal": lambda self, _proposal: {"status": "ok"}},
        )()
        executor = __import__(
            "causal_orch.agent.action_executor", fromlist=["InterventionActionExecutor"]
        ).InterventionActionExecutor(intervention_gate=gate, trace_sink=experiment.trace_sink)
        orchestrator = CausalOrchestrator(
            llm_engine=lambda *_args, **_kwargs: "",
            action_executor=executor,
            tools={"FakeApp__read": object()},
        )
        self.assertIn('"action": "DELEGATE"', orchestrator.init_system_prompts["system_prompt"])
        self.assertNotIn("DELEGATE", orchestrator.tools)
        self.assertNotIn("DELEGATE", executor.tools)

        result = executor.execute_parsed_action(
            ParsedAction(tool_name="DELEGATE", arguments={"proposal_id": "p1"}),
            lambda _log: None,
            lambda: 1.0,
            "agent",
        )
        self.assertEqual(result, {"status": "ok"})

    def test_builder_wires_fake_engine_gate_and_pinned_lifecycle(self):
        experiment = make_experiment()
        env = FakeEnvironment()
        engine_calls = []
        gate_calls = []

        class FakeEngine:
            model_name = MODEL_CANDIDATE_ORDER[0]

        def engine_factory(**kwargs):
            engine_calls.append(kwargs)
            return FakeEngine()

        def gate_factory(**kwargs):
            gate_calls.append(kwargs)
            return type("Gate", (), {"handle_proposal": lambda self, _proposal: {"status": "ok"}})()

        builder = CausalAgentBuilder(
            experiment,
            engine_factory=engine_factory,
            gate_factory=gate_factory,
        )
        config = CausalAgentConfigBuilder(experiment).build()
        built = builder.build(config, env=env)

        self.assertIsInstance(built, ARESimulationAgent)
        self.assertIsInstance(built.react_agent, CausalOrchestrator)
        self.assertEqual(len(engine_calls), 1)
        self.assertIs(engine_calls[0]["config"], experiment.model_config)
        self.assertEqual(gate_calls[0]["block_key"], "block")
        self.assertIs(built.pause_env.__self__, env)
        self.assertIs(built.resume_env.__self__, env)
        self.assertEqual(built.react_agent.tools, {})

        notification_system = type(
            "NotificationSystem",
            (),
            {"config": type("Config", (), {"notified_tools": []})()},
        )()
        built.prepare_are_simulation_run(
            FakeScenario(), notification_system=notification_system
        )
        self.assertIn("FakeApp__read", built.react_agent.tools)
        self.assertNotIn("DELEGATE", built.react_agent.tools)
        built.react_agent.init_tools()
        self.assertIn("FakeApp__read", built.react_agent.action_executor.tools)
        self.assertNotIn("DELEGATE", built.react_agent.action_executor.tools)

    def test_config_rejects_manifest_provider_mismatch(self):
        with self.assertRaises(ValueError):
            make_experiment(model_config=OpenRouterConfig(MODEL_CANDIDATE_ORDER[0], "Other"))

    def test_config_rejects_non_fixed_generation_time(self):
        from are.simulation.types import SimulatedGenerationTimeConfig

        with self.assertRaises(ValueError):
            make_experiment(
                simulated_generation_time_config=SimulatedGenerationTimeConfig(
                    mode="fixed", seconds=1.0
                )
            )


if __name__ == "__main__":
    unittest.main()
