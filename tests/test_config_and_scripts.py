import contextlib
import hashlib
import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import unittest

import yaml

from causal_orch.models.manifests import MODEL_CANDIDATE_ORDER, ModelManifest


ROOT = Path(__file__).parents[1]


def _script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(slug: str) -> dict:
    value = ModelManifest(
        snapshot_timestamp_utc="2026-07-23T00:00:00Z",
        requested_model_slug=slug,
        returned_canonical_slug=slug,
        context_length=32768,
        input_price="0",
        output_price="0",
        supported_parameters=("temperature", "top_p"),
        available_providers=("PinnedProvider",),
        selected_provider="PinnedProvider",
        provider_context_length=32768,
        provider_max_output=4096,
        provider_data_policy="none",
    )
    return value.to_dict()


class ConfigAndScriptTests(unittest.TestCase):
    def test_configs_lock_values_and_pin_gaia2_while_provider_remains_unresolved(self):
        experiment = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
        models = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
        providers = yaml.safe_load((ROOT / "configs/providers.yaml").read_text())
        randomization = yaml.safe_load((ROOT / "configs/randomization.yaml").read_text())
        gaia = json.loads((ROOT / "configs/gaia2_manifest.json").read_text())
        self.assertEqual(experiment["protocol"]["are_commit"], "7946367413129784139e785ae4c351090002a0bb")
        self.assertEqual(experiment["protocol"]["fixed_generation_seconds"], 5)
        self.assertEqual(models["candidate_order"], list(MODEL_CANDIDATE_ORDER))
        self.assertFalse(providers["allow_fallbacks"])
        self.assertEqual(randomization["assignment"]["ratio"], "50/50")
        self.assertTrue(randomization["seed"])
        self.assertEqual(providers["pinned_provider"], "")
        self.assertEqual(gaia["gaia2_revision"], "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99")
        self.assertEqual(gaia["scenario_ids"], ["scenario_universe_28_2nr5po"])
        self.assertEqual(gaia["row_count"], 160)
        self.assertEqual(gaia["smoke"]["direct_tool_name"], "Emails__list_emails")
        self.assertEqual(
            gaia["smoke"]["delegation_proposal"]["allowed_read_tools"],
            ["Emails__list_emails"],
        )
        payload = {key: value for key, value in gaia.items() if key != "manifest_sha256"}
        expected_manifest_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        self.assertEqual(gaia["manifest_sha256"], expected_manifest_hash)
        self.assertEqual(experiment["manifests"]["model_sha256"], {})

    def test_scripts_import_without_side_effects(self):
        for name in (
            "pin_upstream",
            "snapshot_openrouter",
            "build_tool_allowlist",
            "qualify_models",
            "run_smoke",
            "fetch_pinned_gaia2",
        ):
            with self.subTest(name=name):
                _script(name)

    def test_pin_verifies_commit_and_clean_state(self):
        module = _script("pin_upstream")
        checkout = ROOT / "src"
        calls = []

        def fake_git(command, **kwargs):
            calls.append((command, kwargs))
            output = module.LOCKED_ARE_COMMIT if command[1] == "rev-parse" else ""
            return subprocess.CompletedProcess(command, 0, output, "")

        result = module.verify_checkout(checkout, git_runner=fake_git)
        self.assertEqual(result.commit, module.LOCKED_ARE_COMMIT)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("reset", repr(calls).lower())

    def test_snapshot_uses_catalog_transport_and_does_not_print_or_persist_key(self):
        module = _script("snapshot_openrouter")
        requests = []

        def transport(request):
            requests.append(request)
            return module.CatalogResponse(
                200,
                {"data": [{"id": MODEL_CANDIDATE_ORDER[0], "pricing": {"prompt": "0"}}]},
            )

        secret = "sk-test-never-print"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            snapshot = module.snapshot_catalog(
                transport=transport,
                api_key=secret,
                now=datetime(2026, 7, 23, tzinfo=timezone.utc),
            )
        self.assertEqual(requests[0].method, "GET")
        self.assertEqual(requests[0].url, module.CATALOG_URL)
        self.assertIn(secret, requests[0].headers["Authorization"])
        self.assertNotIn(secret, json.dumps(snapshot))
        self.assertNotIn(secret, output.getvalue())

    def test_allowlist_treats_unknown_write_state_as_unsafe(self):
        module = _script("build_tool_allowlist")
        manifest = module.build_reviewed_manifest(
            [
                {"public_name": "read", "app_name": "App", "function_name": "read", "write_operation": False},
                {"public_name": "unknown", "app_name": "App", "function_name": "unknown", "write_operation": None},
            ],
            reviewed_names={"read", "unknown"},
        )
        self.assertEqual(manifest["selected_read_only_tools"], ["read"])
        unknown = next(row for row in manifest["tools"] if row["public_name"] == "unknown")
        self.assertTrue(unknown["allowlisted"])
        self.assertEqual(unknown["exclusion_reason"], "write_operation_unknown_or_unsafe")
        payload = dict(manifest)
        digest = payload.pop("manifest_sha256")
        self.assertEqual(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(), digest)

    def test_qualification_reports_invalid_provider_without_silent_substitution(self):
        module = _script("qualify_models")
        models_config = {"candidate_order": list(MODEL_CANDIDATE_ORDER)}
        providers_config = {"pinned_provider": "", "allow_fallbacks": False, "require_parameters": True}
        report = module.qualify_models([_manifest(MODEL_CANDIDATE_ORDER[0])], models_config, providers_config)
        self.assertFalse(report["valid"])
        self.assertIsNone(report["selected_model"])
        self.assertIn("PINNED_PROVIDER_PLACEHOLDER", report["config_errors"])
        self.assertNotIn("treatment_effect", json.dumps(report))

    def test_smoke_is_runner_injected(self):
        module = _script("run_smoke")
        plan = module.build_smoke_plan(["config", "identity"], context={"scenario_ids": []})
        seen = []
        report = module.run_smoke(plan, lambda step: seen.append(step.name) or True)
        self.assertEqual(seen, ["config", "identity"])
        self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
