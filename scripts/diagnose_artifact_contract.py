"""Run artifact-only diagnostics without loading Gaia2 or changing the protocol."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import urllib.request

from are.simulation.agents.default_agent.base_agent import BaseAgent, TerminationStep
from are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)

from causal_orch.agent.schemas import (
    EvidenceReport,
    diagnose_evidence_report,
)
from causal_orch.agent.worker import (
    FreshReadOnlyWorker,
    ReturnArtifactTool,
    worker_prompt,
)
from causal_orch.models.local_llama_engine import LocalLlamaLLMEngine
from causal_orch.models.manifests import (
    LocalLlamaConfig,
    LocalModelManifest,
    LocalSamplingConfig,
)
from causal_orch.models.openrouter_engine import HTTPRequest, UrllibTransport
from causal_orch.runtime.budgets import BudgetExceededError, BudgetExceededResult
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


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class CapturingTransport:
    def __init__(self) -> None:
        self.transport = UrllibTransport()
        self.attempt_number = 0
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: HTTPRequest):
        self.requests.append(
            {
                "attempt_number": self.attempt_number,
                "request_number": sum(
                    item["attempt_number"] == self.attempt_number
                    for item in self.requests
                )
                + 1,
                "url": request.url,
                "headers": {
                    key: value
                    for key, value in request.headers.items()
                    if key.lower() != "authorization"
                },
                "body": json.loads(json.dumps(request.json_body)),
            }
        )
        return self.transport(request)


@dataclass
class AttemptState:
    budget: int
    output_tokens_used: int = 0
    budget_exhausted: bool = False
    artifact: EvidenceReport | None = None
    phase: str = "POST_TOOL_ARTIFACT"
    requests: list[dict[str, Any]] = field(default_factory=list)


class DiagnosticBudgetedEngine:
    def __init__(self, engine: LocalLlamaLLMEngine, state: AttemptState) -> None:
        self.engine = engine
        self.state = state

    def __call__(self, *args: Any, **kwargs: Any):
        budget_before = max(0, self.state.budget - self.state.output_tokens_used)
        response = self.engine(*args, **kwargs)
        metadata = response[1] if isinstance(response, tuple) else {}
        completion_tokens = metadata.get("completion_tokens")
        completion_tokens = (
            completion_tokens
            if type(completion_tokens) is int and completion_tokens >= 0
            else 0
        )
        after_raw = self.state.output_tokens_used + completion_tokens
        record = {
            "phase": self.state.phase,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": metadata.get("reasoning_tokens"),
            "budget_before": budget_before,
            "budget_after": max(0, self.state.budget - after_raw),
            "budget_overrun_tokens": max(0, after_raw - self.state.budget),
            "parsed_as_action": False,
            "action_name": None,
            "validator_reached": False,
            "validator_accepted": False,
            "validator_rejection_categories": [],
        }
        self.state.requests.append(record)
        if after_raw > self.state.budget:
            self.state.budget_exhausted = True
            raise BudgetExceededError(
                BudgetExceededResult("max_output_tokens", 0, after_raw)
            )
        self.state.output_tokens_used = after_raw
        return response


class RecordingExecutor(JsonActionExecutor):
    def __init__(self, *, tools: Mapping[str, Any], state: AttemptState) -> None:
        super().__init__(tools=dict(tools))
        self.state = state

    def extract_action(self, *args: Any, **kwargs: Any):
        try:
            return super().extract_action(*args, **kwargs)
        except Exception:
            if self.state.requests:
                self.state.requests[-1][
                    "validator_rejection_categories"
                ] = ["MALFORMED_ARGUMENT_JSON"]
            raise

    def parse_action(self, *args: Any, **kwargs: Any):
        try:
            parsed = super().parse_action(*args, **kwargs)
        except Exception:
            if self.state.requests:
                self.state.requests[-1][
                    "validator_rejection_categories"
                ] = ["MALFORMED_ARGUMENT_JSON"]
            raise
        if self.state.requests:
            self.state.requests[-1]["parsed_as_action"] = True
            self.state.requests[-1]["action_name"] = parsed.tool_name
        return parsed


class RecordingReturnArtifactTool(ReturnArtifactTool):
    def __init__(self, state: AttemptState) -> None:
        super().__init__()
        self.state = state

    def forward(self, artifact: Any) -> dict[str, Any]:
        record = self.state.requests[-1]
        record["validator_reached"] = True
        categories = diagnose_evidence_report(
            artifact,
            objective=OBJECTIVE,
            allowed_evidence_refs={EVIDENCE_REF},
        )
        try:
            value = super().forward(artifact)
            validated = FreshReadOnlyWorker._validate_artifact(
                value,
                OBJECTIVE,
                allowed_evidence_refs={EVIDENCE_REF},
            )
        except Exception:
            record["validator_rejection_categories"] = list(
                categories or ("INVALID_FIELD_TYPE",)
            )
            self.state.phase = "ARTIFACT_REPAIR"
            raise
        record["validator_accepted"] = True
        self.state.artifact = validated
        return validated.to_dict()


def _configuration(
    root: Path,
    *,
    thinking: bool,
    request_max_tokens: int,
) -> tuple[LocalLlamaConfig, LocalModelManifest, dict[str, Any]]:
    local = json.loads(
        (root / "configs" / "local_model_manifest.json").read_text()
    )
    model_manifest = LocalModelManifest.from_dict(local["model_manifest"])
    runtime = local["runtime"]
    config = LocalLlamaConfig(
        model_slug=model_manifest.requested_model_slug,
        provider=local["provider_manifest"]["provider_name"],
        model_path=runtime["model_path"],
        model_sha256=model_manifest.gguf_sha256,
        server_binary_path=runtime["server_binary_path"],
        server_binary_sha256=model_manifest.server_binary_sha256,
        endpoint=runtime["endpoint"],
        sampling=LocalSamplingConfig(
            temperature=float(model_manifest.sampling["temperature"]),
            top_p=float(model_manifest.sampling["top_p"]),
            top_k=int(model_manifest.sampling["top_k"]),
            max_tokens=request_max_tokens,
        ),
        reasoning={
            "enable_thinking": thinking,
            "preserve_thinking": thinking,
            "tool_call_format": "xml",
        },
    )
    return config, model_manifest, local


def _payload(budget: int) -> dict[str, Any]:
    payload = {
        "objective": OBJECTIVE,
        "completion_criterion": "Return the sender from the fixed observation.",
        "context": {"fixed_observation": OBSERVATION},
        "context_refs": [EVIDENCE_REF],
        "evidence_refs": [EVIDENCE_REF],
        "permitted_tools": [],
        "tools": [],
        "budgets": {"max_steps": 8, "max_output_tokens": budget},
    }
    payload["prompt"] = worker_prompt(payload)
    return payload


def _run_attempt(
    engine: LocalLlamaLLMEngine,
    *,
    budget: int,
) -> dict[str, Any]:
    state = AttemptState(budget=budget)
    return_tool = RecordingReturnArtifactTool(state)
    tools = {return_tool.name: return_tool}
    executor = RecordingExecutor(tools=tools, state=state)
    agent = BaseAgent(
        llm_engine=DiagnosticBudgetedEngine(engine, state),
        system_prompts={"system_prompt": _payload(budget)["prompt"]},
        tools=tools,
        action_executor=executor,
        termination_step=TerminationStep(
            condition=lambda agent: (
                state.artifact is not None
                or state.budget_exhausted
                or agent.iterations >= agent.max_iterations
            ),
            function=lambda _agent: state.artifact,
        ),
        max_iterations=8,
        total_iterations=8,
        use_custom_logger=False,
    )
    error: str | None = None
    try:
        agent.run(_payload(budget)["prompt"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    all_rejections = sorted(
        {
            category
            for request in state.requests
            for category in request["validator_rejection_categories"]
        }
    )
    return {
        "validator_accepted": state.artifact is not None,
        "budget_exhausted": state.budget_exhausted,
        "cumulative_generated_tokens": sum(
            request["completion_tokens"] for request in state.requests
        ),
        "requests": state.requests,
        "all_rejections": all_rejections,
        "unknown_field": "UNKNOWN_FIELD" in all_rejections,
        "missing_field": "MISSING_REQUIRED_FIELD" in all_rejections,
        "malformed_argument_object": "MALFORMED_ARGUMENT_JSON" in all_rejections,
        "error": error,
    }


def _request_audit(
    request: Mapping[str, Any],
    props: Mapping[str, Any],
) -> dict[str, Any]:
    body = request["body"]
    serialized = json.dumps(body, sort_keys=True, ensure_ascii=False)
    expected_fields = (
        "artifact_type",
        "objective",
        "status",
        "findings",
        "uncertainties",
        "contradictions",
        "recommended_next_action",
    )
    finding_fields = ("claim", "evidence_refs", "confidence")
    tools = body.get("tools")
    return {
        "interface_label": "STOCK_ARE_REACT_JSON",
        "agent_interface": "stock_are_react_json",
        "openai_tools_field_present": isinstance(tools, list),
        "return_artifact_native_schema_present": bool(tools)
        and "return_artifact" in serialized,
        "top_level_fields_visible": {
            field: field in serialized for field in expected_fields
        },
        "finding_fields_visible": {
            field: field in serialized for field in finding_fields
        },
        "enum_values_visible": {
            value: value in serialized
            for value in (
                "EVIDENCE_REPORT",
                "COMPLETE",
                "PARTIAL",
                "BLOCKED",
                "LOW",
                "MEDIUM",
                "HIGH",
            )
        },
        "required_arrays_present": '"required"' in serialized,
        "additional_properties_false_present": (
            '"additionalProperties": false' in serialized
        ),
        "tool_call_id_replay": "NOT_APPLICABLE_STOCK_ARE_TEXT_INTERFACE",
        "chat_template_present": bool(props.get("chat_template")),
        "chat_template_sha256": hashlib.sha256(
            str(props.get("chat_template", "")).encode()
        ).hexdigest(),
        "chat_template_caps": props.get("chat_template_caps"),
        "server_build_info": props.get("build_info"),
        "model_alias": props.get("model_alias"),
        "model_ftype": props.get("model_ftype"),
        "request_sha256": _sha256_json(body),
    }


def run_diagnostic(
    *,
    root: Path,
    output_dir: Path,
    attempts: int,
    budget: int,
    thinking: bool,
    request_max_tokens: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    config, model_manifest, local = _configuration(
        root,
        thinking=thinking,
        request_max_tokens=request_max_tokens,
    )
    transport = CapturingTransport()
    sink = InMemoryTraceSink()
    engine = LocalLlamaLLMEngine(
        config,
        trace_sink=sink,
        root=root,
        transport=transport,
    )
    with urllib.request.urlopen(config.props_endpoint, timeout=10) as response:
        props = json.loads(response.read().decode())
    results = []
    for attempt_number in range(1, attempts + 1):
        transport.attempt_number = attempt_number
        result = _run_attempt(engine, budget=budget)
        result["attempt_number"] = attempt_number
        results.append(result)
        print(
            json.dumps(
                {
                    "attempt": attempt_number,
                    "validator_accepted": result["validator_accepted"],
                    "budget_exhausted": result["budget_exhausted"],
                    "tokens": result["cumulative_generated_tokens"],
                    "rejections": result["all_rejections"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    exact_request = next(
        request for request in transport.requests if request["attempt_number"] == 1
    )
    audit = _request_audit(exact_request, props)
    exact_accepts = sum(result["validator_accepted"] for result in results)
    report = {
        "condition": {
            "model": config.model_slug,
            "gguf_sha256": model_manifest.gguf_sha256,
            "llama_cpp_commit": model_manifest.llama_cpp_commit,
            "agent_interface": "stock_are_react_json",
            "interface_label": "STOCK_ARE_REACT_JSON",
            "budget": budget,
            "thinking": thinking,
            "sampling": config.sampling.to_dict(),
            "validator": "EvidenceReport.from_dict + worker semantic validation",
            "gaia2_loaded": False,
            "synthetic_observation_sha256": _sha256_json(OBSERVATION),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts,
        "valid_exact_artifacts": exact_accepts,
        "unknown_fields": sum(result["unknown_field"] for result in results),
        "missing_fields": sum(result["missing_field"] for result in results),
        "malformed_argument_objects": sum(
            result["malformed_argument_object"] for result in results
        ),
        "budget_exhaustions": sum(result["budget_exhausted"] for result in results),
        "qualified": (
            exact_accepts == attempts
            and not any(result["all_rejections"] for result in results)
            and not any(result["budget_exhausted"] for result in results)
        ),
        "request_audit": audit,
        "results": results,
        "provider_manifest_sha256": local["provider_manifest_sha256"],
        "model_manifest_sha256": model_manifest.manifest_sha256,
    }
    (output_dir / "exact-request.json").write_text(
        json.dumps(exact_request, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "request-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "artifact-only-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument(
        "--thinking",
        choices=("on", "off"),
        default="on",
    )
    parser.add_argument("--request-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--artifact-dir",
        default=str(
            root
            / "artifacts"
            / "local"
            / (
                "artifact-only-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
        ),
    )
    args = parser.parse_args(argv)
    if args.attempts < 1 or args.budget < 1 or args.request_max_tokens < 1:
        parser.error("attempts, budget, and request-max-tokens must be positive")
    report = run_diagnostic(
        root=root,
        output_dir=Path(args.artifact_dir),
        attempts=args.attempts,
        budget=args.budget,
        thinking=args.thinking == "on",
        request_max_tokens=args.request_max_tokens,
    )
    print(
        json.dumps(
            {
                "qualified": report["qualified"],
                "valid_exact_artifacts": report["valid_exact_artifacts"],
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
