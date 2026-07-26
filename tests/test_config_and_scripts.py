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

from causal_orch.models.manifests import (
    LOCAL_MODEL_SLUG,
    LOCAL_PROVIDER,
    LocalModelManifest,
    MODEL_CANDIDATE_ORDER,
    ModelManifest,
)


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
    def test_configs_lock_values_and_pin_gaia2_and_local_provider(self):
        experiment = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
        models = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
        providers = yaml.safe_load((ROOT / "configs/providers.yaml").read_text())
        randomization = yaml.safe_load((ROOT / "configs/randomization.yaml").read_text())
        gaia = json.loads((ROOT / "configs/gaia2_manifest.json").read_text())
        local = json.loads((ROOT / "configs/local_model_manifest.json").read_text())
        qwen = json.loads(
            (ROOT / "configs/qwen_local_model_manifest.json").read_text()
        )
        screen = json.loads(
            (ROOT / "configs/gaia2_breadth_screen.json").read_text()
        )
        self.assertEqual(experiment["protocol"]["are_commit"], "7946367413129784139e785ae4c351090002a0bb")
        self.assertEqual(experiment["protocol"]["fixed_generation_seconds"], 5)
        self.assertEqual(
            models["candidate_order"],
            ["qwen3.6-35b-a3b", LOCAL_MODEL_SLUG],
        )
        self.assertFalse(providers["allow_fallbacks"])
        self.assertEqual(randomization["assignment"]["ratio"], "50/50")
        self.assertTrue(randomization["seed"])
        self.assertEqual(providers["pinned_provider"], LOCAL_PROVIDER)
        self.assertEqual(
            providers["routing_provider_slug"], "nanbeige-llama-cpp-local"
        )
        self.assertIsNone(models["selection"]["selected_model"])
        self.assertEqual(gaia["gaia2_revision"], "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99")
        self.assertEqual(gaia["scenario_ids"], ["scenario_universe_28_2nr5po"])
        self.assertEqual(gaia["row_count"], 160)
        self.assertEqual(
            gaia["data_classification"], "SYNTHETIC_PUBLIC_BENCHMARK"
        )
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
        self.assertEqual(experiment["protocol"]["gaia2_revision"], gaia["gaia2_revision"])
        self.assertEqual(experiment["protocol"]["scenario_ids"], gaia["scenario_ids"])
        model_manifest = LocalModelManifest.from_dict(local["model_manifest"])
        qwen_manifest = LocalModelManifest.from_dict(qwen["model_manifest"])
        candidates = {
            candidate["model_slug"]: candidate
            for candidate in models["candidates"]
        }
        self.assertEqual(
            candidates[LOCAL_MODEL_SLUG]["manifest_sha256"],
            model_manifest.manifest_sha256,
        )
        self.assertEqual(
            candidates["qwen3.6-35b-a3b"]["manifest_sha256"],
            qwen_manifest.manifest_sha256,
        )
        self.assertEqual(
            experiment["manifests"]["model_sha256"][LOCAL_MODEL_SLUG],
            model_manifest.manifest_sha256,
        )
        self.assertEqual(
            experiment["manifests"]["model_sha256"]["qwen3.6-35b-a3b"],
            qwen_manifest.manifest_sha256,
        )
        provider_hash = hashlib.sha256(
            json.dumps(
                local["provider_manifest"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        self.assertEqual(providers["provider_manifest_sha256"], provider_hash)
        self.assertEqual(local["provider_manifest_sha256"], provider_hash)
        self.assertEqual(experiment["manifests"]["provider_sha256"], provider_hash)
        self.assertEqual(experiment["manifests"]["gaia2_sha256"], gaia["manifest_sha256"])
        self.assertEqual(model_manifest.quantization, "Q8_0")
        self.assertEqual(
            local["condition"]["qualification_status"],
            "FAILED_CAPABILITY_FLOOR",
        )
        self.assertEqual(qwen_manifest.quantization, "UD-Q4_K_XL")
        self.assertEqual(
            qwen["condition"]["qualification_status"],
            "FAILED_ACTION_VALIDITY_GATE",
        )
        self.assertEqual(qwen["runtime"]["gpu_layers"], 15)
        self.assertEqual(qwen["runtime"]["threads"], 6)
        self.assertEqual(
            experiment["local_model_manifest_file"],
            "qwen_local_model_manifest.json",
        )
        screen_payload = {
            key: value for key, value in screen.items() if key != "manifest_sha256"
        }
        self.assertEqual(
            screen["manifest_sha256"],
            hashlib.sha256(
                json.dumps(
                    screen_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode()
            ).hexdigest(),
        )
        self.assertEqual(len(screen["scenarios"]), 5)
        self.assertEqual(screen["attempts_per_scenario"], 2)
        self.assertEqual(
            {row["capability"] for row in screen["scenarios"]},
            {"search", "ambiguity", "adaptability", "execution"},
        )
        self.assertEqual(model_manifest.context_length, 32768)
        self.assertEqual(
            model_manifest.llama_cpp_commit,
            "d28da865bf284acaecc98ad18a3c1f607c0fd754",
        )

    def test_scripts_import_without_side_effects(self):
        for name in (
            "pin_upstream",
            "snapshot_openrouter",
            "build_tool_allowlist",
            "qualify_models",
            "run_smoke",
            "setup_are_gui",
            "run_are_gui",
            "probe_local_model",
            "fetch_pinned_gaia2",
            "diagnose_artifact_contract",
            "diagnose_native_artifact_contract",
            "diagnose_native_tool_artifact",
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
