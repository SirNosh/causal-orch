import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]


def load_smoke():
    path = ROOT / "scripts" / "run_smoke.py"
    spec = importlib.util.spec_from_file_location("test_run_smoke_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SmokeContractTests(unittest.TestCase):
    def test_pinned_factory_uses_local_model_without_environment_credential(self):
        smoke = load_smoke()
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            harness = smoke.pinned_gaia2_factory()
        self.assertEqual(
            harness.agent_builder.experiment_config.model_config.provider,
            "Nanbeige/llama.cpp",
        )

    def test_smoke_schedule_forces_exactly_one_execute_assignment(self):
        smoke = load_smoke()
        from causal_orch.runtime.randomization import TreatmentAssignment

        schedule = smoke.SmokeExecuteSchedule("smoke-block")
        self.assertIsNone(schedule.reveal("smoke-block", eligible=False))
        self.assertEqual(
            schedule.reveal("smoke-block", eligible=True),
            TreatmentAssignment.EXECUTE,
        )
        with self.assertRaises(IndexError):
            schedule.reveal("smoke-block", eligible=True)

    def test_configured_placeholders_are_rejected(self):
        smoke = load_smoke()
        errors = smoke.smoke_prerequisite_errors(
            experiment={"protocol": {"are_commit": smoke.LOCKED_ARE_COMMIT}},
            gaia2_manifest={"gaia2_revision": "", "scenario_ids": [], "manifest_sha256": "", "dataset_sha256": ""},
            providers={"pinned_provider": "", "provider_manifest_sha256": ""},
            scenario_factory_spec="test_module:factory",
            are_revision=smoke.LOCKED_ARE_COMMIT,
        )
        self.assertIn("GAIA2_REVISION_PLACEHOLDER", errors)
        self.assertIn("PINNED_PROVIDER_PLACEHOLDER", errors)
        self.assertIn("SCENARIO_IDS_PLACEHOLDER", errors)
        with self.assertRaises(smoke.SmokePrerequisiteError):
            smoke.require_smoke_prerequisites(
                experiment={"protocol": {"are_commit": smoke.LOCKED_ARE_COMMIT}},
                gaia2_manifest={"gaia2_revision": "", "scenario_ids": []},
                providers={},
                scenario_factory_spec="test_module:factory",
                are_revision=smoke.LOCKED_ARE_COMMIT,
            )

    def test_cli_factory_spec_resolves_without_reimporting_the_script(self):
        smoke = load_smoke()
        self.assertIs(
            smoke.load_symbol("scripts.run_smoke:pinned_gaia2_factory"),
            smoke.pinned_gaia2_factory,
        )

    def test_direct_action_returns_the_logged_tool_observation(self):
        smoke = load_smoke()
        from are.simulation.agents.agent_log import ObservationLog

        forwarded_logs = []

        class Executor:
            def execute_parsed_action(
                self, parsed_action, append_log, make_timestamp, agent_id
            ):
                append_log(
                    ObservationLog(
                        content="actual tool result",
                        timestamp=make_timestamp(),
                        agent_id=agent_id,
                    )
                )

        harness = object.__new__(smoke.Gaia2SmokeHarness)
        harness._start = lambda: None
        harness._emit = lambda *args, **kwargs: None
        harness.direct_tool_name = "Emails__list_emails"
        harness.direct_tool_arguments = {"folder_name": "INBOX", "limit": 5}
        harness.agent = SimpleNamespace(
            react_agent=SimpleNamespace(
                action_executor=Executor(),
                append_agent_log=forwarded_logs.append,
                make_timestamp=lambda: 0.0,
                agent_id="orchestrator",
            )
        )

        self.assertEqual(harness.direct_action(), "actual tool result")
        self.assertEqual(len(forwarded_logs), 1)

    def test_required_harness_paths_native_validation_and_trace_are_executed(self):
        smoke = load_smoke()
        calls = []

        class Harness(smoke.Gaia2SmokeHarness):
            validation_success = True

            def __init__(self):
                self.agent_builder = SimpleNamespace(
                    experiment_config=SimpleNamespace(
                        model_config=SimpleNamespace(
                            model_slug="model-1", provider="provider-1"
                        )
                    )
                )

            def direct_action(self):
                calls.append("direct")
                return "direct-result"

            def forced_delegation(self):
                calls.append("delegation")
                return {
                    "status": "DELEGATION_EXECUTED",
                    "artifact": {
                        "status": "WORKER_COMPLETED",
                        "artifact": {"artifact_type": "EVIDENCE_REPORT"},
                    },
                }

            def orchestrator_continuation(self):
                calls.append("continuation")

            def native_validation(self):
                calls.append("validation")
                return {"success": self.validation_success}

            def trace_events(self):
                calls.append("trace")
                return [
                    {"event_type": "RUN_STARTED"},
                    {
                        "event_type": "INTERVENTION_ASSIGNMENT",
                        "treatment_assignment": "EXECUTE",
                    },
                    {"event_type": "WORKER_STARTED"},
                    {
                        "event_type": "MODEL_REQUEST",
                        "openrouter_request_id": "failed-worker-attempt",
                    },
                    {
                        "event_type": "MODEL_CALL_FAILED",
                        "openrouter_request_id": "failed-worker-attempt",
                        "error_type": "TRANSPORT_ERROR",
                    },
                    {
                        "event_type": "MODEL_REQUEST",
                        "openrouter_request_id": "worker-read",
                    },
                    {
                        "event_type": "MODEL_RESPONSE",
                        "openrouter_request_id": "worker-read",
                        "requested_model_slug": "model-1",
                        "returned_model_slug": "model-1",
                        "provider_slug": "provider-1",
                    },
                    {"event_type": "WORKER_TOOL_CALL"},
                    {
                        "event_type": "WORKER_TOOL_RESULT",
                        "event_id": "result-1",
                        "error_type": None,
                    },
                    {
                        "event_type": "MODEL_REQUEST",
                        "openrouter_request_id": "worker-artifact",
                    },
                    {
                        "event_type": "MODEL_RESPONSE",
                        "openrouter_request_id": "worker-artifact",
                        "requested_model_slug": "model-1",
                        "returned_model_slug": "model-1",
                        "provider_slug": "provider-1",
                    },
                    {
                        "event_type": "WORKER_ARTIFACT",
                        "payload": {
                            "artifact": {
                                "artifact_type": "EVIDENCE_REPORT",
                                "objective": "Find the record.",
                                "status": "COMPLETE",
                                "findings": [
                                    {
                                        "claim": "Found it.",
                                        "evidence_refs": [
                                            "worker_tool_result:result-1"
                                        ],
                                        "confidence": "HIGH",
                                    }
                                ],
                                "uncertainties": [],
                                "contradictions": [],
                                "recommended_next_action": None,
                            }
                        },
                    },
                    {"event_type": "WORKER_COMPLETED"},
                    {"event_type": "DELEGATION_EXECUTED"},
                    {"event_type": "ORCHESTRATOR_RESUMED"},
                    {
                        "event_type": "MODEL_REQUEST",
                        "openrouter_request_id": "orchestrator",
                    },
                    {
                        "event_type": "MODEL_RESPONSE",
                        "openrouter_request_id": "orchestrator",
                        "requested_model_slug": "model-1",
                        "returned_model_slug": "model-1",
                        "provider_slug": "provider-1",
                    },
                    {"event_type": "RUN_COMPLETED"},
                ]

            def close(self):
                calls.append("close")

        execution = smoke.run_configured_smoke(Harness)

        self.assertEqual(
            calls,
            [
                "direct",
                "close",
                "delegation",
                "continuation",
                "validation",
                "trace",
                "close",
            ],
        )
        self.assertTrue(execution.infrastructure_pass)
        self.assertTrue(execution.native_validation_completed)
        self.assertTrue(execution.gaia2_success)
        self.assertTrue(execution.native_success)
        self.assertEqual(execution.trace_event_names[0], "RUN_STARTED")

        Harness.validation_success = False
        wrong_answer = smoke.run_configured_smoke(Harness)
        self.assertTrue(wrong_answer.infrastructure_pass)
        self.assertTrue(wrong_answer.native_validation_completed)
        self.assertFalse(wrong_answer.gaia2_success)
        self.assertEqual(wrong_answer.failure_stage, "ORCHESTRATOR_REASONING")

    def test_incomplete_trace_is_rejected(self):
        smoke = load_smoke()
        with self.assertRaises(smoke.SmokePrerequisiteError):
            smoke.check_trace_completeness([{"event_type": "RUN_STARTED"}, {"event_type": "RUN_COMPLETED"}])

    def test_frozen_five_attempt_qualification_gate(self):
        smoke = load_smoke()
        passing = {
            "infrastructure_pass": True,
            "model_identity_verified": True,
            "provider_identity_verified": True,
            "worker_tool_call_success": True,
            "worker_artifact_valid": True,
            "continuation_response_accepted": True,
            "gaia2_success": False,
            "terminal_coverage": True,
            "unclassified_model_call_failures": 0,
        }
        attempts = [{**passing, "gaia2_success": index == 0} for index in range(5)]
        report = smoke.qualification_gate(attempts)
        self.assertTrue(report["qualified"])
        self.assertEqual(report["counts"]["infrastructure_passes"], 5)

        one_infrastructure_failure = [
            {**attempt, "infrastructure_pass": index != 0}
            for index, attempt in enumerate(attempts)
        ]
        self.assertTrue(
            smoke.qualification_gate(one_infrastructure_failure)["qualified"]
        )
        identity_failure = [
            {**attempt, "model_identity_verified": index != 0}
            for index, attempt in enumerate(attempts)
        ]
        self.assertFalse(smoke.qualification_gate(identity_failure)["qualified"])
        with self.assertRaises(ValueError):
            smoke.qualification_gate(attempts[:4])

    def test_suppressed_smoke_assignment_is_rejected(self):
        smoke = load_smoke()
        with self.assertRaisesRegex(
            smoke.SmokePrerequisiteError, "TRACE_WORKER_PATH_MISSING"
        ):
            smoke.check_trace_completeness(
                [
                    {"event_type": "RUN_STARTED"},
                    {
                        "event_type": "INTERVENTION_ASSIGNMENT",
                        "treatment_assignment": "SUPPRESS",
                    },
                    {"event_type": "DELEGATION_SUPPRESSED"},
                    {"event_type": "RUN_COMPLETED"},
                ]
            )

    def test_persisted_smoke_report_is_redacted_and_complete(self):
        smoke = load_smoke()
        from causal_orch.tracing.context import RunContext
        from causal_orch.tracing.events import OrchestrationEvent
        from causal_orch.tracing.sink import InMemoryTraceSink
        from causal_orch.models.manifests import ModelManifest, OpenRouterConfig

        scenario_id = "scenario_universe_28_2nr5po"
        sink = InMemoryTraceSink(
            context=RunContext(
                run_id="smoke-report-test",
                scenario_id=scenario_id,
            )
        )
        sink.append(OrchestrationEvent(event_type="RUN_STARTED"))
        sink.append(
            OrchestrationEvent(
                event_type="WORKER_TOOL_RESULT",
                payload={
                    "ok": True,
                    "tool": "Emails__list_emails",
                    "result": {"content": "private synthetic email body"},
                },
            )
        )
        openrouter = json.loads(
            (ROOT / "configs" / "openrouter_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        model_manifest = ModelManifest.from_dict(openrouter["model_manifest"])
        model_config = OpenRouterConfig(
            model_manifest.requested_model_slug,
            model_manifest.selected_provider,
            routing_provider_slug=openrouter["routing_provider_slug"],
        )
        harness = SimpleNamespace(
            direct_tool_name="Emails__list_emails",
            direct_tool_arguments={"folder_name": "INBOX", "limit": 5},
            agent_builder=SimpleNamespace(
                trace_sink=sink,
                experiment_config=SimpleNamespace(
                    model_manifest=model_manifest,
                    model_config=model_config,
                    max_iterations=12,
                ),
            )
        )
        execution = smoke.SmokeExecution(
            direct_action_result="ok",
            delegation_result={},
            validation_result={"success": False},
            infrastructure_pass=True,
            native_validation_completed=True,
            gaia2_success=False,
            failure_stage="ORCHESTRATOR_REASONING",
            trace_event_names=("RUN_STARTED",),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = smoke.persist_smoke_artifacts(
                execution,
                sink.events,
                artifact_root=directory,
                harness=harness,
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "trace.jsonl",
                    "summary.json",
                    "validation.json",
                    "manifest-locks.json",
                    "direct-sanity.json",
                    "model-manifest.json",
                    "provider-manifest.json",
                    "request-policy.json",
                },
            )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["infrastructure_pass"])
            self.assertTrue(summary["legacy_infrastructure_pass"])
            self.assertTrue(summary["end_to_end_contract_pass"])
            self.assertFalse(summary["runtime_available"])
            self.assertTrue(summary["environment_initialized"])
            self.assertIsNone(summary["worker_tool_executed"])
            self.assertIsNone(summary["artifact_validator_reached"])
            self.assertIsNone(summary["artifact_schema_valid"])
            self.assertIsNone(summary["delegation_completed"])
            self.assertIsNone(summary["orchestrator_resumed"])
            self.assertIsNone(summary["gaia2_evaluated"])
            self.assertIsNone(summary["gaia2_success"])
            self.assertEqual(summary["failure_stage"], "ORCHESTRATOR_REASONING")
            persisted = "".join(
                path.read_text(encoding="utf-8") for path in output.iterdir()
            )
            self.assertNotIn("OPENROUTER_API_KEY", persisted)
            self.assertNotIn("sk-or-", persisted)
            self.assertNotIn("private synthetic email body", persisted)
            self.assertIn("result_sha256", persisted)


if __name__ == "__main__":
    unittest.main()
