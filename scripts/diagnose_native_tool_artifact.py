"""Run worker-only required-tool plus native-artifact Gaia2 probes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


def _event_name(event: Any) -> str:
    value = getattr(event, "event_type", None)
    return str(getattr(value, "value", value))


def _event_value(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def _exchange_tool_calls(exchange: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calls = exchange.get("normalized_tool_calls")
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, Mapping)]


def _call_name(call: Mapping[str, Any]) -> str | None:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    return name if isinstance(name, str) else None


def _exact_tool_result_replay(exchanges: Sequence[Mapping[str, Any]]) -> bool:
    environment_calls = []
    for exchange_index, exchange in enumerate(exchanges):
        for call in _exchange_tool_calls(exchange):
            if _call_name(call) != "return_artifact":
                environment_calls.append((exchange_index, call))
    if not environment_calls:
        return False
    for exchange_index, call in environment_calls:
        if exchange_index + 1 >= len(exchanges):
            return False
        call_id = call.get("id")
        request = exchanges[exchange_index + 1].get("outgoing_request")
        messages = request.get("messages") if isinstance(request, Mapping) else None
        if not isinstance(call_id, str) or not isinstance(messages, list):
            return False
        assistant_replayed = any(
            isinstance(message, Mapping)
            and message.get("role") == "assistant"
            and any(
                isinstance(replayed_call, Mapping)
                and replayed_call.get("id") == call_id
                for replayed_call in (message.get("tool_calls") or [])
            )
            for message in messages
        )
        result_replayed = any(
            isinstance(message, Mapping)
            and message.get("role") == "tool"
            and message.get("tool_call_id") == call_id
            for message in messages
        )
        if not assistant_replayed or not result_replayed:
            return False
    return True


def run_gate(
    *,
    output_dir: Path,
    attempts: int,
) -> dict[str, Any]:
    if attempts not in {5, 20}:
        raise ValueError("the required-tool gate supports 5 or 20 attempts")
    smoke_path = Path(__file__).with_name("run_smoke.py")
    spec = importlib.util.spec_from_file_location(
        "native_tool_artifact_run_smoke",
        smoke_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load run_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = smoke
    spec.loader.exec_module(smoke)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for attempt_number in range(1, attempts + 1):
        harness = smoke.pinned_gaia2_factory()
        try:
            result = harness.worker_only_delegation()
            events = list(harness.trace_events())
            engine = harness.agent.react_agent.llm_engine
            exchanges = list(getattr(engine, "native_exchanges", ()))
        finally:
            harness.close()
        names = [_event_name(event) for event in events]
        requests = [
            event for event in events if _event_name(event) == "MODEL_REQUEST"
        ]
        terminals = [
            event
            for event in events
            if _event_name(event) in {"MODEL_RESPONSE", "MODEL_CALL_FAILED"}
        ]
        responses = [
            event for event in terminals if _event_name(event) == "MODEL_RESPONSE"
        ]
        terminal_ids = [
            _event_value(event, "openrouter_request_id") for event in terminals
        ]
        terminal_complete = bool(requests) and len(requests) == len(terminals) and all(
            terminal_ids.count(_event_value(request, "openrouter_request_id")) == 1
            for request in requests
        )
        artifact = result.get("artifact") if isinstance(result, Mapping) else None
        completed = (
            isinstance(result, Mapping)
            and result.get("status") == "WORKER_COMPLETED"
            and isinstance(artifact, Mapping)
        )
        tool_success = any(
            _event_name(event) == "WORKER_TOOL_RESULT"
            and _event_value(event, "error_type") is None
            for event in events
        )
        artifact_validations = [
            event
            for event in events
            if _event_name(event) == "ARTIFACT_VALIDATION"
        ]
        artifact_valid = (
            any(
                bool((_event_value(event, "payload", {}) or {}).get("accepted"))
                for event in artifact_validations
            )
            if artifact_validations
            else None
        )
        failure = (
            result.get("failure")
            if isinstance(result, Mapping)
            and isinstance(result.get("failure"), Mapping)
            else {}
        )
        budget_exhausted = any(
            bool(exchange.get("budget_exhausted")) for exchange in exchanges
        ) or failure.get("reason") == "TIMEOUT_OR_BUDGET_EXHAUSTION"
        state_violation = "STATE_CHANGED_DURING_WORKER" in names
        model_verified = (
            all(
                _event_value(event, "requested_model_slug")
                == _event_value(event, "returned_model_slug")
                for event in responses
            )
            if responses
            else None
        )
        provider_verified = (
            all(
                _event_value(event, "provider_slug") == "Nanbeige/llama.cpp"
                for event in responses
            )
            if responses
            else None
        )
        tool_names = [
            name
            for exchange in exchanges
            for call in _exchange_tool_calls(exchange)
            if (name := _call_name(call)) is not None
        ]
        environment_calls = [
            name for name in tool_names if name != "return_artifact"
        ]
        environment_call_signatures = [
            json.dumps(
                call.get("function"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for exchange in exchanges
            for call in _exchange_tool_calls(exchange)
            if _call_name(call) != "return_artifact"
        ]
        repeated_environment_calls = len(environment_call_signatures) - len(
            set(environment_call_signatures)
        )
        plain_text_responses = sum(
            not _exchange_tool_calls(exchange) for exchange in exchanges
        )
        repair_attempts = sum(
            exchange.get("phase") == "ARTIFACT_REPAIR"
            for exchange in exchanges
        )
        exact_id_replay = (
            _exact_tool_result_replay(exchanges) if environment_calls else None
        )
        required_every_request = bool(requests) and (
            not exchanges
            or all(
                isinstance(exchange.get("outgoing_request"), Mapping)
                and exchange["outgoing_request"].get("tool_choice") == "required"
                for exchange in exchanges
            )
        )
        artifact_completion_tokens = next(
            (
                exchange.get("completion_tokens")
                for exchange in exchanges
                if any(
                    _call_name(call) == "return_artifact"
                    for call in _exchange_tool_calls(exchange)
                )
            ),
            None,
        )
        cumulative_generated_tokens = sum(
            value
            for exchange in exchanges
            if type(value := exchange.get("completion_tokens")) is int
        )
        passed = all(
            (
                completed,
                tool_success,
                artifact_valid,
                terminal_complete,
                model_verified,
                provider_verified,
                not budget_exhausted,
                not state_violation,
                exact_id_replay,
                required_every_request,
                plain_text_responses == 0,
                repair_attempts == 0,
            )
        )
        attempt_dir = output_dir / f"attempt-{attempt_number:02d}"
        attempt_dir.mkdir()
        (attempt_dir / "trace.jsonl").write_text(
            "\n".join(
                json.dumps(
                    smoke._redacted_trace_event(event),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for event in events
            )
            + "\n",
            encoding="utf-8",
        )
        (attempt_dir / "native-exchanges.json").write_text(
            json.dumps(exchanges, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        attempt_result = {
            "attempt_number": attempt_number,
            "passed": passed,
            "worker_completed": completed,
            "real_tool_executed": tool_success,
            "artifact_schema_valid": artifact_valid,
            "budget_exhausted": budget_exhausted,
            "terminal_trace_complete": terminal_complete,
            "model_identity_verified": model_verified,
            "provider_identity_verified": provider_verified,
            "state_write_violation": state_violation,
            "tool_choice_required_every_request": required_every_request,
            "exact_tool_call_id_replay": exact_id_replay,
            "plain_text_worker_responses": plain_text_responses,
            "repair_attempts": repair_attempts,
            "first_selected_tool": tool_names[0] if tool_names else None,
            "environment_tool_calls": len(environment_calls),
            "selected_tools": tool_names,
            "repeated_environment_tool_calls": repeated_environment_calls,
            "artifact_completion_tokens": artifact_completion_tokens,
            "cumulative_generated_tokens": cumulative_generated_tokens,
            "model_requests": len(requests),
            "model_call_failure_types": [
                _event_value(event, "error_type")
                for event in terminals
                if _event_name(event) == "MODEL_CALL_FAILED"
            ],
            "native_exchanges": len(exchanges),
            "artifact": artifact,
        }
        (attempt_dir / "result.json").write_text(
            json.dumps(attempt_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(attempt_result)
        print(
            json.dumps(
                {
                    "attempt": attempt_number,
                    "passed": passed,
                    "tool": tool_success,
                    "artifact": artifact_valid,
                    "budget_exhausted": budget_exhausted,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    report = {
        "condition": {
            "interface_label": "NATIVE_TYPED_TOOL_INTERFACE_REQUIRED",
            "scenario_id": "scenario_universe_28_2nr5po",
            "orchestrator_present": False,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "format_retry": False,
            "worker_budget": {
                "max_steps": 8,
                "max_output_tokens": 2000,
            },
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts,
        "passes": sum(result["passed"] for result in results),
        "qualified": all(result["passed"] for result in results),
        "results": results,
    }
    (output_dir / "native-tool-artifact-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    local_manifest = json.loads(
        (root / "configs" / "local_model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    gaia2_manifest = json.loads(
        (root / "configs" / "gaia2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    (output_dir / "runtime-manifests.json").write_text(
        json.dumps(
            {
                "model_manifest": local_manifest["model_manifest"],
                "provider_manifest": local_manifest["provider_manifest"],
                "provider_manifest_sha256": local_manifest[
                    "provider_manifest_sha256"
                ],
                "gaia2_revision": gaia2_manifest["gaia2_revision"],
                "gaia2_manifest_sha256": gaia2_manifest["manifest_sha256"],
                "scenario_sha256": gaia2_manifest["scenario_sha256"][
                    "scenario_universe_28_2nr5po"
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument(
        "--artifact-dir",
        default=str(
            root
            / "artifacts"
            / "local"
            / (
                "native-tool-artifact-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
        ),
    )
    args = parser.parse_args(argv)
    report = run_gate(
        output_dir=Path(args.artifact_dir),
        attempts=args.attempts,
    )
    print(
        json.dumps(
            {
                "qualified": report["qualified"],
                "passes": report["passes"],
                "attempts": report["attempts"],
                "artifact_dir": args.artifact_dir,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
