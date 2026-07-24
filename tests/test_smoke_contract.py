import importlib.util
from pathlib import Path
import sys
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
    def test_pinned_factory_requires_environment_credential(self):
        smoke = load_smoke()
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(smoke.SmokePrerequisiteError) as error:
                smoke.pinned_gaia2_factory()
        self.assertEqual(str(error.exception), "OPENROUTER_API_KEY is not set")

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

    def test_required_harness_paths_native_validation_and_trace_are_executed(self):
        smoke = load_smoke()
        calls = []

        class Harness(smoke.Gaia2SmokeHarness):
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

            def native_validation(self):
                calls.append("validation")
                return {"success": True}

            def trace_events(self):
                calls.append("trace")
                return [
                    {"event_type": "RUN_STARTED"},
                    {"event_type": "DIRECT_ACTION"},
                    {
                        "event_type": "INTERVENTION_ASSIGNMENT",
                        "treatment_assignment": "EXECUTE",
                    },
                    {"event_type": "WORKER_STARTED"},
                    {"event_type": "MODEL_REQUEST"},
                    {
                        "event_type": "MODEL_RESPONSE",
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
                    {"event_type": "MODEL_REQUEST"},
                    {
                        "event_type": "MODEL_RESPONSE",
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
                    {"event_type": "ORCHESTRATOR_RESUMED"},
                    {"event_type": "DELEGATION_EXECUTED"},
                    {"event_type": "MODEL_REQUEST"},
                    {
                        "event_type": "MODEL_RESPONSE",
                        "requested_model_slug": "model-1",
                        "returned_model_slug": "model-1",
                        "provider_slug": "provider-1",
                    },
                    {"event_type": "RUN_COMPLETED"},
                ]

            def close(self):
                calls.append("close")

        execution = smoke.run_configured_smoke(Harness)

        self.assertEqual(calls, ["direct", "delegation", "validation", "trace", "close"])
        self.assertTrue(execution.native_success)
        self.assertEqual(execution.trace_event_names[0], "RUN_STARTED")

    def test_incomplete_trace_is_rejected(self):
        smoke = load_smoke()
        with self.assertRaises(smoke.SmokePrerequisiteError):
            smoke.check_trace_completeness([{"event_type": "RUN_STARTED"}, {"event_type": "RUN_COMPLETED"}])

    def test_suppressed_smoke_assignment_is_rejected(self):
        smoke = load_smoke()
        with self.assertRaisesRegex(
            smoke.SmokePrerequisiteError, "TRACE_WORKER_PATH_MISSING"
        ):
            smoke.check_trace_completeness(
                [
                    {"event_type": "RUN_STARTED"},
                    {"event_type": "DIRECT_ACTION"},
                    {
                        "event_type": "INTERVENTION_ASSIGNMENT",
                        "treatment_assignment": "SUPPRESS",
                    },
                    {"event_type": "DELEGATION_SUPPRESSED"},
                    {"event_type": "RUN_COMPLETED"},
                ]
            )


if __name__ == "__main__":
    unittest.main()
