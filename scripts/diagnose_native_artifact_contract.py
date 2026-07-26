"""Run the strict native typed-tool artifact gate without loading Gaia2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import urllib.request

from causal_orch.agent.schemas import EvidenceReport, diagnose_evidence_report
from causal_orch.agent.worker import (
    FreshReadOnlyWorker,
    NATIVE_TYPED_TOOL_INTERFACE,
    native_return_artifact_tool_schema,
    native_worker_prompt,
)
from causal_orch.models.local_llama_engine import LocalLlamaLLMEngine
from causal_orch.models.manifests import (
    LocalLlamaConfig,
    LocalModelManifest,
    LocalSamplingConfig,
)
from causal_orch.models.openrouter_engine import (
    HTTPRequest,
    UrllibTransport,
    _json_body,
)
from causal_orch.tracing.sink import InMemoryTraceSink


OBJECTIVE = "Identify the most recent email sender."
EVIDENCE_REF = "synthetic_observation:email:0"
OBSERVATION = {
    "emails": [
        {
            "sender": "alice@example.com",
            "timestamp": "2026-07-25T13:00:00Z",
        }
    ],
    "evidence_ref": EVIDENCE_REF,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class CapturingTransport:
    def __init__(self) -> None:
        self.transport = UrllibTransport()
        self.attempt_number = 0
        self.records: list[dict[str, Any]] = []

    def __call__(self, request: HTTPRequest):
        record = {
            "attempt_number": self.attempt_number,
            "outgoing_request": json.loads(json.dumps(request.json_body)),
            "raw_response": None,
            "status_code": None,
        }
        self.records.append(record)
        response = self.transport(request)
        record["status_code"] = response.status_code
        try:
            record["raw_response"] = _json_body(response)
        except Exception:
            record["raw_response"] = {"unparseable_response": True}
        return response


def _configuration(
    root: Path,
) -> tuple[LocalLlamaConfig, LocalModelManifest, dict[str, Any]]:
    local = json.loads(
        (root / "configs" / "local_model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = LocalModelManifest.from_dict(local["model_manifest"])
    runtime = local["runtime"]
    config = LocalLlamaConfig(
        model_slug=manifest.requested_model_slug,
        provider=local["provider_manifest"]["provider_name"],
        model_path=runtime["model_path"],
        model_sha256=manifest.gguf_sha256,
        server_binary_path=runtime["server_binary_path"],
        server_binary_sha256=manifest.server_binary_sha256,
        endpoint=runtime["endpoint"],
        sampling=LocalSamplingConfig(
            temperature=float(manifest.sampling["temperature"]),
            top_p=float(manifest.sampling["top_p"]),
            top_k=int(manifest.sampling["top_k"]),
            max_tokens=int(manifest.sampling["max_tokens"]),
        ),
        reasoning={
            "enable_thinking": True,
            "preserve_thinking": True,
            "tool_call_format": "xml",
        },
    )
    return config, manifest, local


def _payload(budget: int) -> dict[str, Any]:
    return {
        "objective": OBJECTIVE,
        "completion_criterion": "Return the sender from the fixed observation.",
        "context": {"fixed_observation": OBSERVATION},
        "context_refs": [EVIDENCE_REF],
        "evidence_refs": [EVIDENCE_REF],
        "permitted_tools": [],
        "tools": [],
        "budgets": {"max_steps": 8, "max_output_tokens": budget},
    }


def _audit(
    tool: Mapping[str, Any],
    props: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = tool["function"]["parameters"]
    canonical = EvidenceReport.to_json_schema()
    finding = parameters["properties"]["findings"]["items"]
    return {
        "interface_label": NATIVE_TYPED_TOOL_INTERFACE,
        "openai_tools_field_present": True,
        "return_artifact_native_schema_present": True,
        "parameters_equal_canonical_schema": parameters == canonical,
        "top_level_fields": list(parameters["properties"]),
        "required_top_level_fields": list(parameters["required"]),
        "additional_properties_false": (
            parameters.get("additionalProperties") is False
        ),
        "finding_fields": list(finding["properties"]),
        "required_finding_fields": list(finding["required"]),
        "finding_additional_properties_false": (
            finding.get("additionalProperties") is False
        ),
        "artifact_type_enum": parameters["properties"]["artifact_type"]["enum"],
        "status_enum": parameters["properties"]["status"]["enum"],
        "confidence_enum": finding["properties"]["confidence"]["enum"],
        "tool_schema_sha256": _sha256_json([tool]),
        "chat_template_present": bool(props.get("chat_template")),
        "chat_template_sha256": hashlib.sha256(
            str(props.get("chat_template", "")).encode("utf-8")
        ).hexdigest(),
        "chat_template_caps": props.get("chat_template_caps"),
        "server_build_info": props.get("build_info"),
        "model_alias": props.get("model_alias"),
        "model_ftype": props.get("model_ftype"),
        "tool_choice": "auto",
        "tool_call_id_replay": "NOT_APPLICABLE_ARTIFACT_ONLY_SINGLE_TURN",
    }


def run_gate(
    *,
    root: Path,
    output_dir: Path,
    attempts: int,
    budget: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    config, manifest, local = _configuration(root)
    transport = CapturingTransport()
    sink = InMemoryTraceSink()
    engine = LocalLlamaLLMEngine(
        config,
        trace_sink=sink,
        root=root,
        transport=transport,
    )
    with urllib.request.urlopen(config.props_endpoint, timeout=10) as response:
        props = json.loads(response.read().decode("utf-8"))
    tool = native_return_artifact_tool_schema()
    payload = _payload(budget)
    prompt = native_worker_prompt(payload)
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Complete the bounded objective using the available native function tools.",
        },
    ]
    results = []
    for attempt_number in range(1, attempts + 1):
        transport.attempt_number = attempt_number
        completion_tokens = 0
        reasoning_tokens = None
        validator_input = None
        rejections: tuple[str, ...] = ()
        accepted = False
        malformed = False
        error: str | None = None
        terminal_event = "MODEL_CALL_FAILED"
        normalized_tool_call = None
        try:
            assistant, metadata = engine.native_tool_completion(
                messages,
                tools=[tool],
                tool_choice="auto",
                additional_trace_tags={
                    "actor_role": "worker",
                    "diagnostic_attempt": attempt_number,
                },
            )
            terminal_event = "MODEL_RESPONSE"
            completion_tokens = (
                metadata["completion_tokens"]
                if type(metadata.get("completion_tokens")) is int
                else 0
            )
            reasoning_tokens = metadata.get("reasoning_tokens")
            calls = assistant.get("tool_calls")
            if not isinstance(calls, list) or len(calls) != 1:
                rejections = ("MALFORMED_ARGUMENT_JSON",)
                malformed = True
            else:
                normalized_tool_call = calls[0]
                function = calls[0]["function"]
                if function["name"] != "return_artifact":
                    rejections = ("MALFORMED_ARGUMENT_JSON",)
                    malformed = True
                else:
                    validator_input = function["arguments"]
                    rejections = diagnose_evidence_report(
                        validator_input,
                        objective=OBJECTIVE,
                        allowed_evidence_refs={EVIDENCE_REF},
                    )
                    try:
                        FreshReadOnlyWorker._validate_artifact(
                            validator_input,
                            OBJECTIVE,
                            allowed_evidence_refs={EVIDENCE_REF},
                        )
                        accepted = not rejections
                    except (TypeError, ValueError):
                        if not rejections:
                            rejections = ("INVALID_FIELD_TYPE",)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            malformed = (
                getattr(exc, "error_type", None) == "INVALID_TOOL_ARGUMENTS"
            )
            if malformed:
                rejections = ("MALFORMED_ARGUMENT_JSON",)
        budget_exhausted = completion_tokens > budget
        result = {
            "attempt_number": attempt_number,
            "phase": "POST_TOOL_ARTIFACT",
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "budget_before": budget,
            "budget_after": max(0, budget - completion_tokens),
            "budget_exhausted": budget_exhausted,
            "parsed_as_action": normalized_tool_call is not None,
            "action_name": (
                normalized_tool_call["function"]["name"]
                if normalized_tool_call is not None
                else None
            ),
            "normalized_tool_call": normalized_tool_call,
            "validator_input": validator_input,
            "validator_reached": validator_input is not None,
            "validator_accepted": accepted and not budget_exhausted,
            "validator_rejection_categories": list(rejections),
            "repair_attempts_required": 0,
            "malformed_argument_object": malformed,
            "terminal_event": terminal_event,
            "error": error,
        }
        results.append(result)
        if engine.native_exchanges:
            exchange = engine.native_exchanges[-1]
            if exchange.get("outgoing_request", {}).get("messages") == messages:
                exchange["phase"] = result["phase"]
                exchange["budget_before"] = result["budget_before"]
                exchange["budget_after"] = result["budget_after"]
                exchange["budget_exhausted"] = result["budget_exhausted"]
                exchange["validator_input"] = validator_input
                exchange["validator_result"] = {
                    "accepted": result["validator_accepted"],
                    "rejection_categories": list(rejections),
                }
        print(
            json.dumps(
                {
                    "attempt": attempt_number,
                    "accepted": result["validator_accepted"],
                    "tokens": completion_tokens,
                    "budget_exhausted": budget_exhausted,
                    "rejections": list(rejections),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    clean = sum(result["validator_accepted"] for result in results)
    audit = _audit(tool, props)
    report = {
        "condition": {
            "interface_label": NATIVE_TYPED_TOOL_INTERFACE,
            "model": config.model_slug,
            "provider": config.provider,
            "gguf_sha256": manifest.gguf_sha256,
            "llama_cpp_commit": manifest.llama_cpp_commit,
            "budget": budget,
            "thinking": True,
            "sampling": config.sampling.to_dict(),
            "gaia2_loaded": False,
            "synthetic_observation_sha256": _sha256_json(OBSERVATION),
            "validator": "EvidenceReport.from_dict + worker semantic validation",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts,
        "clean_first_attempt_acceptances": clean,
        "repair_attempts_required": 0,
        "unknown_fields": sum(
            "UNKNOWN_FIELD" in result["validator_rejection_categories"]
            for result in results
        ),
        "missing_fields": sum(
            "MISSING_REQUIRED_FIELD" in result["validator_rejection_categories"]
            for result in results
        ),
        "malformed_argument_objects": sum(
            result["malformed_argument_object"] for result in results
        ),
        "budget_exhaustions": sum(
            result["budget_exhausted"] for result in results
        ),
        "qualified": (
            attempts == 20
            and clean == attempts
            and not any(result["validator_rejection_categories"] for result in results)
            and not any(result["budget_exhausted"] for result in results)
            and not any(result["malformed_argument_object"] for result in results)
        ),
        "schema_audit": audit,
        "results": results,
        "model_manifest_sha256": manifest.manifest_sha256,
        "provider_manifest_sha256": local["provider_manifest_sha256"],
    }
    (output_dir / "canonical-tool-schema.json").write_text(
        json.dumps(tool, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "schema-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "native-artifact-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "native-exchanges.json").write_text(
        json.dumps(transport.records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "runtime-manifests.json").write_text(
        json.dumps(
            {
                "model_manifest": local["model_manifest"],
                "provider_manifest": local["provider_manifest"],
                "provider_manifest_sha256": local["provider_manifest_sha256"],
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
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument(
        "--artifact-dir",
        default=str(
            root
            / "artifacts"
            / "local"
            / (
                "native-artifact-only-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
        ),
    )
    args = parser.parse_args(argv)
    if args.attempts < 1 or args.budget < 1:
        parser.error("attempts and budget must be positive")
    report = run_gate(
        root=root,
        output_dir=Path(args.artifact_dir),
        attempts=args.attempts,
        budget=args.budget,
    )
    print(
        json.dumps(
            {
                "qualified": report["qualified"],
                "clean_first_attempt_acceptances": report[
                    "clean_first_attempt_acceptances"
                ],
                "attempts": report["attempts"],
                "budget_exhaustions": report["budget_exhaustions"],
                "artifact_dir": args.artifact_dir,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
