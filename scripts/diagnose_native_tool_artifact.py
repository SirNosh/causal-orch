"""Run five worker-only real-tool plus native-artifact Gaia2 probes."""

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


def run_gate(
    *,
    output_dir: Path,
    attempts: int,
) -> dict[str, Any]:
    if attempts != 5:
        raise ValueError("the real-tool native-artifact gate requires five attempts")
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
        artifact_valid = any(
            _event_name(event) == "ARTIFACT_VALIDATION"
            and bool((_event_value(event, "payload", {}) or {}).get("accepted"))
            for event in events
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
        model_verified = bool(terminals) and all(
            _event_name(event) != "MODEL_RESPONSE"
            or _event_value(event, "requested_model_slug")
            == _event_value(event, "returned_model_slug")
            for event in terminals
        )
        provider_verified = bool(terminals) and all(
            _event_name(event) != "MODEL_RESPONSE"
            or _event_value(event, "provider_slug") == "Nanbeige/llama.cpp"
            for event in terminals
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
            "model_requests": len(requests),
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
            "interface_label": "NATIVE_TYPED_TOOL_INTERFACE",
            "scenario_id": "scenario_universe_28_2nr5po",
            "orchestrator_present": False,
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
