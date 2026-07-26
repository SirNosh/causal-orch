import hashlib
import importlib
import json
import unittest
from unittest.mock import patch

from are.simulation.agents.llm.types import MessageRole

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
    GenerationMetadataPollingPolicy,
    HTTPRequest,
    HTTPResponse,
    OpenRouterLLMEngine,
    OpenRouterGenerationMetadataLookup,
    OpenRouterProtocolError,
    compute_backoff_delay,
)
from causal_orch.tracing.events import EventName
from causal_orch.tracing.sink import InMemoryTraceSink


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


class FakeDecodingSchema:
    type = "json"

    def __init__(self, schema):
        self.decoding_schema = schema


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
    def test_api_key_wires_authenticated_generation_lookup_by_default(self):
        configured = OpenRouterLLMEngine(
            OpenRouterConfig(
                MODEL_CANDIDATE_ORDER[0],
                "PinnedProvider",
                api_key="test-key",
            ),
            transport=FakeTransport(response()),
        )
        self.assertIsInstance(
            configured.generation_metadata_lookup,
            OpenRouterGenerationMetadataLookup,
        )
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
            "data_collection": "allow",
        })
        self.assertTrue(transport.requests[0].headers["X-Request-ID"])

    def test_are_tool_response_role_is_openrouter_compatible(self):
        transport = FakeTransport(response())
        engine(transport).chat_completion(
            [
                {"role": MessageRole.ASSISTANT, "content": "Action"},
                {"role": MessageRole.TOOL_RESPONSE, "content": "Observation"},
            ]
        )
        self.assertEqual(
            [message["role"] for message in transport.requests[0].json_body["messages"]],
            ["assistant", "user"],
        )

    def test_payload_routes_by_slug_and_verifies_provider_name(self):
        transport = FakeTransport(response())
        config = OpenRouterConfig(
            MODEL_CANDIDATE_ORDER[0],
            "PinnedProvider",
            routing_provider_slug="pinned-provider",
        )
        OpenRouterLLMEngine(config, transport=transport).chat_completion([])
        self.assertEqual(
            transport.requests[0].json_body["provider"]["only"],
            ["pinned-provider"],
        )

    def test_local_stop_compatibility_preserves_raw_and_truncates_for_parser(self):
        raw = (
            "Thought: complete\n"
            "Action:\n"
            '{"action":"final_answer","action_input":{"answer":"44"}}'
            "<end_action>\n"
            "unwanted trailing text\n"
            "Observation:\n"
            "more unwanted text"
        )
        reply = response()
        reply.body["choices"][0]["message"]["content"] = raw
        transport = FakeTransport(reply)
        sink = InMemoryTraceSink()
        config = OpenRouterConfig(
            MODEL_CANDIDATE_ORDER[1],
            "PinnedProvider",
            routing_provider_slug="pinned-provider",
            provider_stop_forwarded=False,
            preserve_raw_provider_response=True,
            compatibility_condition=(
                "NEMOTRON3_SUPER_OPENROUTER_NVIDIA_LOCAL_STOP_TRUNCATION"
            ),
        )
        reply.body["model"] = config.model_slug
        text, metadata = OpenRouterLLMEngine(
            config,
            trace_sink=sink,
            transport=transport,
        ).chat_completion(
            [{"role": "user", "content": "complete"}],
            stop_sequences=["<end_action>", "Observation:"],
        )

        self.assertEqual(
            text,
            "Thought: complete\n"
            "Action:\n"
            '{"action":"final_answer","action_input":{"answer":"44"}}',
        )
        request = transport.requests[0].json_body
        self.assertNotIn("stop", request)
        self.assertEqual(request["temperature"], 0.2)
        self.assertEqual(request["top_p"], 0.9)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(
            request["provider"],
            {
                "only": ["pinned-provider"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "allow",
            },
        )
        response_event = sink.events[-1]
        self.assertEqual(response_event.provider_slug, "PinnedProvider")
        self.assertEqual(
            response_event.payload["raw_provider_response"],
            raw,
        )
        self.assertFalse(response_event.payload["provider_stop_forwarded"])
        self.assertEqual(
            response_event.payload["local_stop_sequences"],
            ["<end_action>", "Observation:"],
        )
        self.assertEqual(
            metadata["provider_completion_tokens"],
            7,
        )
        self.assertIsInstance(
            metadata["semantic_completion_tokens_before_stop"],
            int,
        )

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

    def test_missing_provider_is_rejected_as_unverifiable(self):
        body = response().body
        body.pop("provider")
        with self.assertRaises(OpenRouterProtocolError) as error:
            engine(FakeTransport(HTTPResponse(200, body))).chat_completion([])
        self.assertEqual(error.exception.error_type, "PROVIDER_IDENTITY_UNVERIFIABLE")

    def test_generation_metadata_lookup_verifies_missing_response_provider(self):
        body = response().body
        body.pop("provider")
        seen = []

        def lookup(request_id):
            seen.append(request_id)
            return {"data": {"id": request_id, "model": MODEL_CANDIDATE_ORDER[0], "provider_name": "PinnedProvider"}}

        text, metadata = OpenRouterLLMEngine(
            OpenRouterConfig(MODEL_CANDIDATE_ORDER[0], "PinnedProvider"),
            transport=FakeTransport(HTTPResponse(200, body)),
            generation_metadata_lookup=lookup,
        ).chat_completion([])
        self.assertEqual(text, "answer")
        self.assertEqual(metadata["trace_metadata"]["provider_identity_source"], "generation_metadata")
        self.assertEqual(seen, ["response-request"])

    def test_generation_header_is_preferred_for_metadata_lookup(self):
        body = response().body
        body.pop("provider")
        seen = []
        lookup = lambda request_id: (
            seen.append(request_id)
            or {"data": {"provider_name": "PinnedProvider"}}
        )
        OpenRouterLLMEngine(
            OpenRouterConfig(MODEL_CANDIDATE_ORDER[0], "PinnedProvider"),
            transport=FakeTransport(
                HTTPResponse(
                    200,
                    body,
                    headers={"X-Generation-Id": "generation-id"},
                )
            ),
            generation_metadata_lookup=lookup,
        ).chat_completion([])
        self.assertEqual(seen, ["generation-id"])

    def test_generation_metadata_lookup_polls_boundedly_until_available(self):
        body = response().body
        body.pop("provider")
        attempts = []
        waits = []

        def lookup(request_id):
            attempts.append(request_id)
            if len(attempts) < 3:
                return HTTPResponse(404, {"error": "not ready"})
            return {"data": {"provider_name": "PinnedProvider"}}

        text, _metadata = OpenRouterLLMEngine(
            OpenRouterConfig(MODEL_CANDIDATE_ORDER[0], "PinnedProvider"),
            transport=FakeTransport(HTTPResponse(200, body)),
            generation_metadata_lookup=lookup,
            generation_metadata_polling=GenerationMetadataPollingPolicy(
                max_attempts=3,
                delay_seconds=0.01,
                sleeper=waits.append,
            ),
        ).chat_completion([])
        self.assertEqual(text, "answer")
        self.assertEqual(attempts, ["response-request"] * 3)
        self.assertEqual(waits, [0.01, 0.01])

    def test_schema_and_trace_tags_are_sent_and_traced_without_credentials(self):
        sink = InMemoryTraceSink()
        transport = FakeTransport(response())
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        engine(transport, trace_sink=sink).chat_completion(
            [{"role": "user", "content": "hello"}],
            schema=schema,
            additional_trace_tags=["action"],
        )
        request = transport.requests[0].json_body
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(request["response_format"]["json_schema"]["schema"], schema)
        self.assertEqual([event.event_type for event in sink.events], [EventName.MODEL_REQUEST, EventName.MODEL_RESPONSE])
        self.assertTrue(sink.events[0].prompt_hash)
        self.assertTrue(sink.events[0].tool_schema_hash)
        self.assertEqual(sink.events[0].payload["trace_tags"], {"tags": ["action"]})
        self.assertEqual(sink.events[1].total_tokens, 18)
        self.assertNotIn("authorization", repr(sink.events).lower())
        json.dumps([event.to_dict() for event in sink.events])

    def test_trace_tags_preserve_mapping_tuple_and_string_shapes(self):
        for tags, expected in (
            ({"phase": "test"}, {"phase": "test"}),
            (("action", "retry"), {"tags": ["action", "retry"]}),
            ("action", {"tags": ["action"]}),
        ):
            sink = InMemoryTraceSink()
            engine(FakeTransport(response()), trace_sink=sink).chat_completion(
                [], additional_trace_tags=tags
            )
            self.assertEqual(sink.events[0].payload["trace_tags"], expected)

    def test_are_decoding_schema_object_uses_structured_output(self):
        transport = FakeTransport(response())
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        engine(transport).chat_completion(schema=FakeDecodingSchema(schema), messages=[])
        self.assertEqual(transport.requests[0].json_body["response_format"]["json_schema"]["schema"], schema)

    def test_model_or_provider_identity_mismatch_is_deterministic(self):
        with self.assertRaisesRegex(OpenRouterProtocolError, "requested model") as error:
            engine(FakeTransport(response(model=MODEL_CANDIDATE_ORDER[1]))).chat_completion([])
        self.assertEqual(error.exception.error_type, "MODEL_IDENTITY_MISMATCH")
        with self.assertRaisesRegex(OpenRouterProtocolError, "requested provider") as error:
            engine(FakeTransport(response(provider="OtherProvider"))).chat_completion([])
        self.assertEqual(error.exception.error_type, "PROVIDER_IDENTITY_MISMATCH")

    def test_health_rejects_missing_or_non_string_pinned_provider(self):
        metadata = {
            "returned_model": MODEL_CANDIDATE_ORDER[0],
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "completion_duration": 0.1,
            "request_id": "req",
        }
        self.assertFalse(response_qualifies(metadata, MODEL_CANDIDATE_ORDER[0], "PinnedProvider").qualified)
        self.assertFalse(response_qualifies({**metadata, "provider": 1}, MODEL_CANDIDATE_ORDER[0], "PinnedProvider").qualified)

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

    def test_every_emitted_request_has_one_terminal_failure_classification(self):
        cases = (
            ("invalid-json", HTTPResponse(200, b"not-json"), "INVALID_JSON_BODY"),
            (
                "no-choice",
                HTTPResponse(
                    200,
                    {
                        "model": MODEL_CANDIDATE_ORDER[0],
                        "provider": "PinnedProvider",
                        "choices": [],
                    },
                ),
                "NO_COMPLETION_CHOICE",
            ),
            (
                "non-text",
                HTTPResponse(
                    200,
                    {
                        "model": MODEL_CANDIDATE_ORDER[0],
                        "provider": "PinnedProvider",
                        "choices": [{"message": {"content": None}}],
                    },
                ),
                "NON_TEXT_COMPLETION",
            ),
            (
                "empty",
                HTTPResponse(
                    200,
                    {
                        "model": MODEL_CANDIDATE_ORDER[0],
                        "provider": "PinnedProvider",
                        "choices": [{"message": {"content": "  "}}],
                    },
                ),
                "EMPTY_COMPLETION",
            ),
            (
                "transport",
                OSError("offline"),
                "TRANSPORT_ERROR",
            ),
        )
        for name, result, expected in cases:
            with self.subTest(name=name):
                sink = InMemoryTraceSink()
                policy = BackoffPolicy(max_retries=0)
                with self.assertRaises(Exception):
                    engine(
                        FakeTransport(result),
                        trace_sink=sink,
                        backoff=policy,
                    ).chat_completion([])
                terminals = [
                    event
                    for event in sink.events
                    if event.event_type
                    in {EventName.MODEL_RESPONSE, EventName.MODEL_CALL_FAILED}
                ]
                self.assertEqual(len(terminals), 1)
                self.assertEqual(terminals[0].event_type, EventName.MODEL_CALL_FAILED)
                self.assertEqual(terminals[0].error_type, expected)
                self.assertEqual(
                    sink.events[0].openrouter_request_id,
                    terminals[0].openrouter_request_id,
                )

    def test_identity_failure_is_followed_by_one_terminal_failure(self):
        sink = InMemoryTraceSink()
        with self.assertRaises(OpenRouterProtocolError):
            engine(
                FakeTransport(response(model=MODEL_CANDIDATE_ORDER[1])),
                trace_sink=sink,
            ).chat_completion([])
        self.assertEqual(
            [event.event_type for event in sink.events],
            [
                EventName.MODEL_REQUEST,
                EventName.MODEL_IDENTITY_MISMATCH,
                EventName.MODEL_CALL_FAILED,
            ],
        )
        self.assertEqual(
            sink.events[-1].error_type,
            "MODEL_IDENTITY_MISMATCH",
        )

    def test_import_does_not_open_a_network_connection(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            module = importlib.import_module("causal_orch.models.openrouter_engine")
            module.OpenRouterLLMEngine(OpenRouterConfig(MODEL_CANDIDATE_ORDER[0], "PinnedProvider"))


if __name__ == "__main__":
    unittest.main()
