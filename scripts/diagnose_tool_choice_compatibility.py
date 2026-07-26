"""Probe llama.cpp tool-choice compatibility without loading Gaia2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import urllib.request

from causal_orch.agent.worker import (
    native_return_artifact_tool_schema,
    native_worker_prompt,
)
from causal_orch.models.local_llama_engine import LocalLlamaLLMEngine

from diagnose_native_artifact_contract import (
    OBJECTIVE,
    _configuration,
    _payload,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _error_text(body: Any) -> str | None:
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    if not isinstance(error, Mapping):
        return None
    message = error.get("message")
    return message if isinstance(message, str) else None


def _ping_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "ping",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }


class RecordingTransport:
    def __init__(self, transport: Any) -> None:
        self.transport = transport
        self.last_status: int | None = None
        self.last_body: Any = None

    def __call__(self, request: Any) -> Any:
        response = self.transport(request)
        self.last_status = response.status_code
        try:
            body = response.body
            if isinstance(body, bytes):
                body = json.loads(body.decode("utf-8"))
            elif isinstance(body, str):
                body = json.loads(body)
            self.last_body = body
        except Exception:
            self.last_body = None
        return response


def _probe(
    *,
    root: Path,
    config: Any,
    name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: Any,
    chat_template_sha256: str,
    max_tokens: int,
) -> dict[str, Any]:
    from causal_orch.models.openrouter_engine import UrllibTransport

    transport = RecordingTransport(UrllibTransport())
    engine = LocalLlamaLLMEngine(
        config,
        root=root,
        transport=transport,
    )
    generated_tokens = 0
    selected_tools: list[str] = []
    exception_type = None
    try:
        assistant, metadata = engine.native_tool_completion(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )
        value = metadata.get("completion_tokens")
        generated_tokens = value if type(value) is int else 0
        selected_tools = [
            call["function"]["name"]
            for call in assistant.get("tool_calls", [])
            if isinstance(call, Mapping)
            and isinstance(call.get("function"), Mapping)
            and isinstance(call["function"].get("name"), str)
        ]
    except Exception as error:
        exception_type = getattr(error, "error_type", type(error).__name__)
        exception_message = str(error)
    else:
        exception_message = None
    if not generated_tokens and isinstance(transport.last_body, Mapping):
        usage = transport.last_body.get("usage")
        if isinstance(usage, Mapping):
            value = usage.get("completion_tokens")
            generated_tokens = value if type(value) is int else 0
    status = transport.last_status
    generation_reached = status is not None and 200 <= status < 300
    return {
        "probe": name,
        "http_status": status,
        "server_error_text": _error_text(transport.last_body),
        "generated_token_count": generated_tokens,
        "chat_template_sha256": chat_template_sha256,
        "tool_schema_sha256": _sha256_json(tools),
        "tool_choice": tool_choice,
        "grammar_initialization_success": generation_reached,
        "model_generation_reached": generation_reached,
        "selected_tools": selected_tools,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "tool_call_parse_success": exception_type is None,
    }


def run_matrix(*, root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    config, manifest, local = _configuration(root)
    with urllib.request.urlopen(config.props_endpoint, timeout=10) as response:
        props = json.loads(response.read().decode("utf-8"))
    chat_template_sha256 = hashlib.sha256(
        str(props.get("chat_template", "")).encode("utf-8")
    ).hexdigest()
    actual_tool = native_return_artifact_tool_schema()
    actual_messages = [
        {
            "role": "system",
            "content": native_worker_prompt(_payload(2000)),
        },
        {
            "role": "user",
            "content": (
                "Complete the bounded objective using the available native "
                "function tools."
            ),
        },
    ]
    ping_tool = _ping_tool()
    ping_messages = [
        {"role": "user", "content": "Call ping with the value ok."}
    ]
    named_artifact = {
        "type": "function",
        "function": {"name": "return_artifact"},
    }
    named_ping = {
        "type": "function",
        "function": {"name": "ping"},
    }
    probes = [
        _probe(
            root=root,
            config=config,
            name="actual_schema_named_return_artifact",
            messages=actual_messages,
            tools=[actual_tool],
            tool_choice=named_artifact,
            chat_template_sha256=chat_template_sha256,
            max_tokens=2000,
        ),
        *[
            _probe(
                root=root,
                config=config,
                name=name,
                messages=ping_messages,
                tools=[ping_tool],
                tool_choice=choice,
                chat_template_sha256=chat_template_sha256,
                max_tokens=256,
            )
            for name, choice in (
                ("trivial_schema_auto", "auto"),
                ("trivial_schema_named_ping", named_ping),
                ("trivial_schema_required", "required"),
            )
        ],
    ]
    trivial_compatible = all(
        probe["grammar_initialization_success"]
        for probe in probes
        if probe["probe"].startswith("trivial_schema_")
    )
    if trivial_compatible:
        probes.append(
            _probe(
                root=root,
                config=config,
                name="actual_schema_required",
                messages=actual_messages,
                tools=[actual_tool],
                tool_choice="required",
                chat_template_sha256=chat_template_sha256,
                max_tokens=2000,
            )
        )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server_build": props.get("build_info"),
        "model": config.model_slug,
        "gguf_sha256": manifest.gguf_sha256,
        "chat_template_sha256": chat_template_sha256,
        "synthetic_objective": OBJECTIVE,
        "gaia2_loaded": False,
        "probes": probes,
        "actual_named_finalizer_compatible": (
            probes[0]["grammar_initialization_success"]
            and probes[0]["tool_call_parse_success"]
            and probes[0]["selected_tools"] == ["return_artifact"]
        ),
        "trivial_matrix_compatible": trivial_compatible,
        "actual_required_matrix_status": (
            "RUN" if trivial_compatible else "SKIPPED_TRIVIAL_REQUIRED_FAILED"
        ),
        "provider_manifest_sha256": local["provider_manifest_sha256"],
    }
    (output_dir / "tool-choice-compatibility-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default=str(
            root
            / "artifacts"
            / "local"
            / (
                "tool-choice-compatibility-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
        ),
    )
    args = parser.parse_args(argv)
    report = run_matrix(root=root, output_dir=Path(args.artifact_dir))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
