import unittest

from causal_orch.runner.scenario_runner import CausalScenarioRunner


class FakeScenario:
    def __init__(self, events):
        self.events = events

    def initialize(self):
        self.events.append("scenario.initialize")

    def validate(self, env):
        self.events.append("scenario.validate")
        return {"valid": True}


class FakeEnvironment:
    def __init__(self, events):
        self.events = events
        self.notification_system = "notifications"

    def run(self, scenario, *, wait_for_end):
        self.events.append(("env.run", scenario, wait_for_end))

    def stop(self):
        self.events.append("env.stop")


class FakeConfigBuilder:
    def __init__(self, events):
        self.events = events

    def build(self):
        self.events.append("config.build")
        return "agent-config"


class FakeAgent:
    def __init__(self, events):
        self.events = events

    def run_scenario(self, *, scenario, notification_system):
        self.events.append(("agent.run_scenario", scenario, notification_system))
        return "agent-result"

    def stop(self):
        self.events.append("agent.stop")


class FakeAgentBuilder:
    def __init__(self, events):
        self.events = events

    def build(self, *, agent_config, env):
        self.events.append(("agent.build", agent_config, env))
        return FakeAgent(self.events)


class RunnerTests(unittest.TestCase):
    def test_stock_sequence_and_fresh_objects_are_preserved_per_run(self):
        events = []
        scenarios = []
        environments = []

        def scenario_factory():
            scenario = FakeScenario(events)
            scenarios.append(scenario)
            return scenario

        def environment_factory():
            environment = FakeEnvironment(events)
            environments.append(environment)
            return environment

        def exporter(**kwargs):
            events.append(("export", kwargs["scenario"], kwargs["env"]))

        runner = CausalScenarioRunner(
            scenario_factory=scenario_factory,
            environment_factory=environment_factory,
            agent_config_builder=FakeConfigBuilder(events),
            agent_builder=FakeAgentBuilder(events),
            exporter=exporter,
            trace_sink_closer=lambda: events.append("trace.close"),
        )

        self.assertEqual(runner.run(), {"valid": True})
        self.assertEqual(runner.run(), {"valid": True})
        self.assertEqual(len({id(value) for value in scenarios}), 2)
        self.assertEqual(len({id(value) for value in environments}), 2)
        self.assertEqual(
            [value for value in events if isinstance(value, str)],
            [
                "scenario.initialize",
                "config.build",
                "scenario.validate",
                "trace.close",
                "agent.stop",
                "env.stop",
                "scenario.initialize",
                "config.build",
                "scenario.validate",
                "trace.close",
                "agent.stop",
                "env.stop",
            ],
        )
        run_events = [value for value in events if isinstance(value, tuple)]
        self.assertEqual(run_events[0][0], "env.run")
        self.assertEqual(run_events[1][0], "agent.build")
        self.assertEqual(run_events[2][0], "agent.run_scenario")


if __name__ == "__main__":
    unittest.main()
