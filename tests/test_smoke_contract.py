import importlib.util
from pathlib import Path
import sys
import unittest


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

        class Harness:
            def direct_action(self):
                calls.append("direct")
                return "direct-result"

            def forced_mock_delegation(self):
                calls.append("delegation")
                return "delegation-result"

            def native_validation(self):
                calls.append("validation")
                return {"success": True}

            def trace_events(self):
                calls.append("trace")
                return [
                    {"event_type": "RUN_STARTED"},
                    {"event_type": "DIRECT_ACTION"},
                    {"event_type": "FORCED_MOCK_DELEGATION"},
                    {"event_type": "RUN_COMPLETED"},
                ]

        execution = smoke.run_configured_smoke(Harness)

        self.assertEqual(calls, ["direct", "delegation", "validation", "trace"])
        self.assertTrue(execution.native_success)
        self.assertEqual(execution.trace_event_names[0], "RUN_STARTED")

    def test_incomplete_trace_is_rejected(self):
        smoke = load_smoke()
        with self.assertRaises(smoke.SmokePrerequisiteError):
            smoke.check_trace_completeness([{"event_type": "RUN_STARTED"}, {"event_type": "RUN_COMPLETED"}])


if __name__ == "__main__":
    unittest.main()
