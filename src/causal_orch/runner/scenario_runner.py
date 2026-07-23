"""A dependency-injected wrapper preserving ARE's stock run sequence."""

from __future__ import annotations

from typing import Any, Callable


class CausalScenarioRunner:
    """Run fresh scenario/environment objects without owning an event loop."""

    def __init__(
        self,
        *,
        scenario_factory: Callable[[], Any],
        environment_factory: Callable[[], Any],
        agent_config_builder: Any,
        agent_builder: Any,
        exporter: Callable[..., Any] | None = None,
        trace_sink_closer: Callable[[], Any] | None = None,
    ) -> None:
        self.scenario_factory = scenario_factory
        self.environment_factory = environment_factory
        self.agent_config_builder = agent_config_builder
        self.agent_builder = agent_builder
        self.exporter = exporter
        self.trace_sink_closer = trace_sink_closer

    @staticmethod
    def _fresh(value: Any) -> Any:
        return value() if callable(value) and not hasattr(value, "build") else value

    def run(self) -> Any:
        scenario = self.scenario_factory()
        env = self.environment_factory()
        agent = None
        builder = self._fresh(self.agent_builder)
        config_builder = self._fresh(self.agent_config_builder)
        close_trace = self.trace_sink_closer
        if close_trace is None:
            trace_sink = getattr(builder, "trace_sink", None)
            close_trace = getattr(trace_sink, "close", None)
        try:
            scenario.initialize()
            env.run(scenario, wait_for_end=False)
            agent_config = config_builder.build()
            agent = builder.build(agent_config=agent_config, env=env)
            result = agent.run_scenario(
                scenario=scenario,
                notification_system=getattr(env, "notification_system", None),
            )
            validation = scenario.validate(env)
            if self.exporter is not None:
                self.exporter(
                    env=env,
                    scenario=scenario,
                    result=result,
                    validation=validation,
                )
            return validation
        finally:
            if close_trace is not None:
                close_trace()
            if agent is not None and hasattr(agent, "stop"):
                agent.stop()
            env.stop()
