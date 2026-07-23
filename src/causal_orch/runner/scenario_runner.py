"""A dependency-injected wrapper preserving ARE's stock run sequence."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Callable
from uuid import uuid4

from causal_orch.tracing.context import RunContext
from causal_orch.tracing.sink import InMemoryTraceSink


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
        trace_sink_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.scenario_factory = scenario_factory
        self.environment_factory = environment_factory
        self.agent_config_builder = agent_config_builder
        self.agent_builder = agent_builder
        self.exporter = exporter
        self.trace_sink_closer = trace_sink_closer
        self.trace_sink_factory = trace_sink_factory

    @staticmethod
    def _fresh(value: Any, *args: Any) -> Any:
        if not callable(value) or hasattr(value, "build"):
            return value
        if args:
            try:
                signature = inspect.signature(value)
                signature.bind(*args)
            except (TypeError, ValueError):
                return value()
            return value(*args)
        return value()

    def _fresh_agent_builder(self, trace_sink: Any | None) -> Any:
        builder = self._fresh(self.agent_builder, trace_sink) if trace_sink is not None else self._fresh(self.agent_builder)
        if trace_sink is None or not hasattr(builder, "trace_sink"):
            return builder
        builder = copy.copy(builder)
        builder.trace_sink = trace_sink
        return builder

    def _run_context(self, scenario: Any) -> RunContext:
        builder = self.agent_builder
        experiment = getattr(builder, "experiment_config", None)
        manifest = getattr(experiment, "model_manifest", None)
        model_config = getattr(experiment, "model_config", None)
        scenario_id = getattr(scenario, "scenario_id", getattr(scenario, "id", None))
        return RunContext(
            run_id=uuid4().hex,
            scenario_id=str(scenario_id) if scenario_id is not None else None,
            universe_id=(
                str(getattr(scenario, "universe_id"))
                if getattr(scenario, "universe_id", None) is not None
                else None
            ),
            capability="causal-orch-delegation",
            model_slug=getattr(manifest, "requested_model_slug", getattr(model_config, "model_slug", None)),
            provider_slug=getattr(manifest, "selected_provider", getattr(model_config, "provider", None)),
            block_key=(
                str(getattr(experiment, "intervention_block"))
                if getattr(experiment, "intervention_block", None) is not None
                else None
            ),
        )

    def _fresh_trace_sink(self, context: RunContext) -> Any:
        if self.trace_sink_factory is None:
            sink = InMemoryTraceSink(context=context)
        else:
            factory = self.trace_sink_factory
            try:
                inspect.signature(factory).bind(context)
            except (TypeError, ValueError):
                sink = factory()
            else:
                sink = factory(context)
        set_context = getattr(sink, "set_context", None)
        if callable(set_context):
            set_context(context)
        return sink

    def close(self) -> None:
        """Close a shared sink once, after the caller's batch is complete."""

        if self.trace_sink_closer is not None:
            self.trace_sink_closer()

    def run(self) -> Any:
        scenario = self.scenario_factory()
        env = self.environment_factory()
        agent = None
        run_context = self._run_context(scenario)
        trace_sink = self._fresh_trace_sink(run_context)
        builder = self._fresh_agent_builder(trace_sink)
        config_builder = self._fresh(self.agent_config_builder)
        close_trace = None
        close_trace = self.trace_sink_closer or getattr(trace_sink, "close", None)
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
