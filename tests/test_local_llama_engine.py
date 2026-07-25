import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from causal_orch.models.local_llama_engine import (
    LocalLlamaLLMEngine,
    LocalLlamaProtocolError,
)
from causal_orch.models.manifests import LocalLlamaConfig
from causal_orch.models.openrouter_engine import BackoffPolicy, HTTPResponse
from causal_orch.tracing.events import EventName
from causal_orch.tracing.sink import InMemoryTraceSink


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def completion(message, *, model="Abiray/Nanbeige4.2-3B-GGUF:Q8_0"):
    return HTTPResponse(
        200,
        {
            "id": "local-request",
            "model": model,
            "choices": [{"message": message}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
    )


class LocalLlamaEngineTests(unittest.TestCase):
    def engine(self, directory, transport, sink=None):
        model = b"model"
        server = b"server"
        Path(directory, "model.gguf").write_bytes(model)
        Path(directory, "llama-server.exe").write_bytes(server)
        config = LocalLlamaConfig(
            model_path="model.gguf",
            model_sha256=sha256(model),
            server_binary_path="llama-server.exe",
            server_binary_sha256=sha256(server),
        )
        return LocalLlamaLLMEngine(
            config,
            trace_sink=sink,
            root=directory,
            transport=transport,
            props_lookup=lambda: HTTPResponse(
                200, {"model_alias": config.model_slug}
            ),
            backoff=BackoffPolicy(max_retries=0),
        )

    def test_preserves_stock_are_text_interface(self):
        action = (
            "Thought: I need the age tool.\n"
            'Action:\n{"action":"get_age","action_input":{"name":"Ada"}}'
        )
        transport = FakeTransport(completion({"content": action}))
        sink = InMemoryTraceSink()
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory, transport, sink)
            response, _metadata = engine.chat_completion(
                [
                    {"role": "user", "content": "Find Ada's age."},
                ]
            )

        self.assertEqual(response, action)
        request = transport.requests[0].json_body
        self.assertNotIn("tools", request)
        self.assertNotIn("tool_choice", request)
        self.assertEqual(
            [event.event_type for event in sink.events],
            [
                EventName.MODEL_REQUEST,
                EventName.MODEL_RESPONSE,
            ],
        )

    def test_identity_failure_has_one_terminal_event(self):
        transport = FakeTransport(
            completion({"content": "wrong"}, model="other/model")
        )
        sink = InMemoryTraceSink()
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory, transport, sink)
            with self.assertRaises(LocalLlamaProtocolError):
                engine.chat_completion([{"role": "user", "content": "hello"}])
        terminals = [
            event
            for event in sink.events
            if event.event_type in {EventName.MODEL_RESPONSE, EventName.MODEL_CALL_FAILED}
        ]
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0].error_type, "MODEL_IDENTITY_MISMATCH")


if __name__ == "__main__":
    unittest.main()
