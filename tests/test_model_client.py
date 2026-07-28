import json

import pytest

from causal_orch.model_client import (
    OpenAICompatibleClient,
    StructuredActionError,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_malformed_native_response_retains_raw_provider_payload(monkeypatch):
    raw = {"choices": [{"message": {"content": "plain text"}}]}
    monkeypatch.setattr(
        "causal_orch.model_client.request.urlopen",
        lambda *_args, **_kwargs: Response(raw),
    )
    client = OpenAICompatibleClient(
        base_url="http://localhost/v1",
        api_key="",
        model="model",
    )

    with pytest.raises(StructuredActionError) as exc:
        client.complete([], [])

    assert exc.value.raw_response == raw


def test_max_tokens_is_sent_as_a_hard_request_limit(monkeypatch):
    captured = {}
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "final_answer",
                                "arguments": '{"answer":"44"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }

    def urlopen(req, **_):
        captured.update(json.loads(req.data.decode()))
        return Response(raw)

    monkeypatch.setattr(
        "causal_orch.model_client.request.urlopen", urlopen
    )
    client = OpenAICompatibleClient(
        base_url="http://localhost/v1",
        api_key="",
        model="model",
    )

    client.complete([], [], max_tokens=37)

    assert captured["max_tokens"] == 37


def test_provider_pin_disables_fallbacks(monkeypatch):
    captured = {}
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "final_answer",
                                "arguments": '{"answer":"44"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    def urlopen(req, **_):
        captured.update(json.loads(req.data.decode()))
        return Response(raw)

    monkeypatch.setattr(
        "causal_orch.model_client.request.urlopen", urlopen
    )
    client = OpenAICompatibleClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="key",
        model="model",
        provider="google-ai-studio",
    )

    client.complete([], [])

    assert captured["provider"] == {
        "only": ["google-ai-studio"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_parallel_tool_hint_can_be_omitted(monkeypatch):
    captured = {}
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "final_answer",
                                "arguments": '{"answer":"44"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    def urlopen(req, **_):
        captured.update(json.loads(req.data.decode()))
        return Response(raw)

    monkeypatch.setattr(
        "causal_orch.model_client.request.urlopen", urlopen
    )
    client = OpenAICompatibleClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="key",
        model="model",
        send_parallel_tool_calls=False,
    )

    client.complete([], [])

    assert "parallel_tool_calls" not in captured
