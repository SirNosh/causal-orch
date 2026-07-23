import hashlib
import importlib
import json
import unittest
from unittest.mock import patch

from causal_orch.models.health import (
    FailureClass,
    classify_http_failure,
    qualify_manifest,
    response_qualifies,
)
from causal_orch.models.manifests import (
    MODEL_CANDIDATE_ORDER,
    ModelManifest,
    OpenRouterConfig,
)
from causal_orch.models.openrouter_engine import (
    BackoffPolicy,
    HTTPRequest,
    HTTPResponse,
    OpenRouterLLMEngine,
    OpenRouterProtocolError,
    compute_backoff_delay,
)


def manifest(slug: str = MODEL_CANDIDATE_ORDER[0]) -> ModelManifest:
    return ModelManifest(
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
        provider_data_policy=" ಸ್ಪ ".strip(),
    )


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request: HTTPRequest) -> HTTPResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def response(*, model=MODEL_CANDIDATE_ORDER[0], provider="PinnedProvider", status_code=200):
    return HTTPResponse(
        status_code=status_code,
        headers={"x-request-id": "header-request"},
        body={
            "id": "response-request",
            "model": model,
            "provider": provider,
            "choices": [{"message": {"content": "answer"}}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
            "provider_metadata": {"route": "PinnedProvider", "queue": "free"},
        },
    )


def engine(transport, **kwargs):
    return OpenRouterLLMEngine(
        OpenRouterConfig(MODEL_CANDIDATE_ORDER[0], "PinnedProvider"),
        transport=transport,
        **kwargs,
    )


class ModelIdentityTests(unittest.TestCase):
    def test_manifest_order_hash_and_serialization_are_fixed(self):
        manifests = [manifest(slug) for slug in reversed(MODEL_CANDIDATE_ORDER)]
        self.assertEqual(MODEL_CANDIDATE_ORDER, (
            "openai/gpt-oss-20b:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "google/gemma-4-26b-a4b-it:free",
        ))
        one = manifests[-1]
        self.assertEqual(
            one.manifest_sha256,
            hashlib.sha256(json.dumps(one.to_dict(include_hash=False), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        )
        self.assertEqual(one.serialize(), one.to_json())
        self.assertEqual(ModelManifest.from_dict(one.to_dict()), one)

    def test_health_helpers_qualify_supplied_manifest_and_metadata_only(self):
        catalog_manifest = manifest()
        catalog_manifest = ModelManifest.from_dict({
            **catalog_manifest.to_dict(),
            "returned_canonical_slug": "openai/gpt-oss-20b",
            "manifest_sha256": "",
        })
        self.assertTrue(qualify_manifest(catalog_manifest, "PinnedProvider").qualified)
        self.assertTrue(response_qualifies({
            "returned_model": MODEL_CANDIDATE_ORDER[0],
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "completion_duration": 0.1,
            "request_id": "req",
            "provider": "PinnedProvider",
        }, MODEL_CANDIDATE_ORDER[0], "PinnedProvider").qualified)

    def test_payload_has_exact_sampling_and_pinned_provider(self):
        transport = FakeTransport(response())
        engine(transport).chat_completion([{"role": "user", "content": "hello"}])
        payload = transport.requests[0].json_body
        self.assertEqual(payload["model"], MODEL_CANDIDATE_ORDER[0])
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["provider"], {
            "only": ["PinnedProvider"],
            "allow_fallbacks": False,
            "require_parameters": True,
        })
        self.assertTrue(transport.requests[0].headers["X-Request-ID"])

    def test_reasoning_and_attribution_are_optional_configured_fields(self):
        transport = FakeTransport(response())
        config = OpenRouterConfig(
            MODEL_CANDIDATE_ORDER[0],
            "PinnedProvider",
            reasoning={"effort": "low"},
            attribution_headers={"HTTP-Referer": "https://example.test", "X-Title": "test"},
        )
        OpenRouterLLMEngine(config, transport=transport).chat_completion([])
        self.assertEqual(transport.requests[0].json_body["reasoning"], {"effort": "low"})
        self.assertEqual(transport.requests[0].headers["X-Title"], "test")

    def test_response_metadata_is_extracted_without_raw_reasoning(self):
        text, metadata = engine(FakeTransport(response())).chat_completion([])
        self.assertEqual(text, "answer")
        self.assertEqual(metadata["prompt_tokens"], 11)
        self.assertEqual(metadata["completion_tokens"], 7)
        self.assertEqual(metadata["total_tokens"], 18)
        self.assertEqual(metadata["reasoning_tokens"], 3)
        self.assertGreaterEqual(metadata["completion_duration"], 0)
        self.assertEqual(metadata["request_id"], "response-request")
        self.assertEqual(metadata["returned_model"], MODEL_CANDIDATE_ORDER[0])
        self.assertEqual(metadata["provider"], "PinnedProvider")
        self.assertEqual(metadata["trace_metadata"]["provider_metadata"]["queue"], "free")
        self.assertNotIn("thought", repr(metadata).lower())

    def test_model_or_provider_identity_mismatch_is_deterministic(self):
        with self.assertRaisesRegex(OpenRouterProtocolError, "requested model") as error:
            engine(FakeTransport(response(model=MODEL_CANDIDATE_ORDER[1]))).chat_completion([])
        self.assertEqual(error.exception.error_type, "MODEL_IDENTITY_MISMATCH")
        with self.assertRaisesRegex(OpenRouterProtocolError, "requested provider") as error:
            engine(FakeTransport(response(provider="OtherProvider"))).chat_completion([])
        self.assertEqual(error.exception.error_type, "PROVIDER_IDENTITY_MISMATCH")

    def test_rate_limit_classification_and_backoff_are_bounded(self):
        self.assertEqual(classify_http_failure(429), FailureClass.RATE_LIMIT)
        self.assertEqual(classify_http_failure(503), FailureClass.OUTAGE)
        delays = [compute_backoff_delay(index, random_value=1) for index in range(8)]
        self.assertTrue(all(0 <= delay <= 8 for delay in delays))
        waits = []
        policy = BackoffPolicy(max_retries=1, random_value=lambda: 0, sleeper=waits.append)
        with self.assertRaises(OpenRouterProtocolError) as error:
            engine(FakeTransport(response(status_code=429), response(status_code=429)), backoff=policy).chat_completion([])
        self.assertEqual(error.exception.error_type, "RATE_LIMIT")
        self.assertEqual(len(waits), 1)

    def test_import_does_not_open_a_network_connection(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            module = importlib.import_module("causal_orch.models.openrouter_engine")
            module.OpenRouterLLMEngine(OpenRouterConfig(MODEL_CANDIDATE_ORDER[0], "PinnedProvider"))


if __name__ == "__main__":
    unittest.main()
