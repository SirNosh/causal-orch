"""The versioned, JSON-safe orchestration event model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Mapping
from uuid import uuid4


class EventName(str, Enum):
    RUN_STARTED = "RUN_STARTED"
    MODEL_MANIFEST_LOCKED = "MODEL_MANIFEST_LOCKED"
    MODEL_REQUEST = "MODEL_REQUEST"
    MODEL_RESPONSE = "MODEL_RESPONSE"
    DIRECT_ACTION = "DIRECT_ACTION"
    ORCHESTRATOR_PROPOSAL = "ORCHESTRATOR_PROPOSAL"
    PROPOSAL_VALIDATION = "PROPOSAL_VALIDATION"
    INTERVENTION_ELIGIBILITY = "INTERVENTION_ELIGIBILITY"
    INTERVENTION_ASSIGNMENT = "INTERVENTION_ASSIGNMENT"
    DELEGATION_EXECUTED = "DELEGATION_EXECUTED"
    DELEGATION_SUPPRESSED = "DELEGATION_SUPPRESSED"
    WORKER_STARTED = "WORKER_STARTED"
    WORKER_TOOL_CALL = "WORKER_TOOL_CALL"
    WORKER_TOOL_RESULT = "WORKER_TOOL_RESULT"
    WORKER_ARTIFACT = "WORKER_ARTIFACT"
    WORKER_STATE_GUARD = "WORKER_STATE_GUARD"
    STATE_CHANGED_DURING_WORKER = "STATE_CHANGED_DURING_WORKER"
    WORKER_COMPLETED = "WORKER_COMPLETED"
    ORCHESTRATOR_RESUMED = "ORCHESTRATOR_RESUMED"
    PROVIDER_RATE_LIMIT = "PROVIDER_RATE_LIMIT"
    PROVIDER_OUTAGE = "PROVIDER_OUTAGE"
    PROVIDER_IDENTITY_MISMATCH = "PROVIDER_IDENTITY_MISMATCH"
    PROVIDER_IDENTITY_UNVERIFIABLE = "PROVIDER_IDENTITY_UNVERIFIABLE"
    MODEL_IDENTITY_MISMATCH = "MODEL_IDENTITY_MISMATCH"
    JUDGE_RESULT = "JUDGE_RESULT"
    RUN_FAILED = "RUN_FAILED"
    RUN_COMPLETED = "RUN_COMPLETED"


EventType = EventName
EVENT_NAMES = tuple(EventName)
_FORBIDDEN_PAYLOAD_KEYS = {"thought", "chain_of_thought", "hidden_reasoning", "private_reasoning"}


def _json_safe(value: Any, path: str = "payload") -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if key.lower() in _FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"{path} contains a hidden chain-of-thought field")
            result[key] = _json_safe(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[]") for item in value]
    raise TypeError(f"{path} is not JSON-compatible")


@dataclass(frozen=True)
class OrchestrationEvent:
    event_type: EventName | str
    schema_version: str = "1.0"
    experiment_version: str = "causal-orch-delegation-v1"
    scenario_id: str | None = None
    universe_id: str | None = None
    capability: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    event_id: str = field(default_factory=lambda: uuid4().hex)
    event_sequence: int | None = None
    correlation_id: str | None = None
    previous_event_id: str | None = None
    causal_parent_ids: tuple[str, ...] = field(default_factory=tuple)
    actor_id: str | None = None
    actor_role: str | None = None
    simulated_timestamp: int | float | str | None = None
    wall_timestamp: int | float | str | None = None
    temporal_batch: str | int | None = None
    randomization_block_key: str | None = None
    block_metadata: Mapping[str, Any] = field(default_factory=dict)
    requested_model_slug: str | None = None
    returned_model_slug: str | None = None
    provider_slug: str | None = None
    openrouter_request_id: str | None = None
    prompt_hash: str | None = None
    tool_schema_hash: str | None = None
    state_before_hash: str | None = None
    state_after_hash: str | None = None
    proposed_action: str | None = None
    executed_action: str | None = None
    eligibility: str | None = None
    eligibility_reason: str | None = None
    treatment_assignment: str | None = None
    assignment_probability: float | None = None
    context_refs: tuple[str, ...] = field(default_factory=tuple)
    artifact_refs: tuple[str, ...] = field(default_factory=tuple)
    tool_refs: tuple[str, ...] = field(default_factory=tuple)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    wall_latency: float | None = None
    simulated_duration: float | None = None
    retry_count: int | None = None
    error_type: str | None = None
    protocol_violation: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventName(self.event_type))
        for field_name in ("event_id", "schema_version", "experiment_version"):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name):
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("causal_parent_ids", "context_refs", "artifact_refs", "tool_refs"):
            values = getattr(self, field_name)
            if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{field_name} must contain non-empty strings")
            object.__setattr__(self, field_name, tuple(values))
        if self.assignment_probability is not None and not 0 <= self.assignment_probability <= 1:
            raise ValueError("assignment_probability must be between zero and one")
        if self.event_sequence is not None and (type(self.event_sequence) is not int or self.event_sequence < 1):
            raise ValueError("event_sequence must be a positive integer when supplied")
        object.__setattr__(self, "payload", _json_safe(deepcopy(dict(self.payload))))
        object.__setattr__(self, "block_metadata", _json_safe(deepcopy(dict(self.block_metadata))))

    @property
    def sequence(self) -> int | None:
        return self.event_sequence

    @property
    def block_key(self) -> str | None:
        return self.randomization_block_key

    @property
    def event_name(self) -> EventName:
        return self.event_type

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "experiment_version": self.experiment_version,
            "event_type": self.event_type.value,
            "scenario_id": self.scenario_id,
            "universe_id": self.universe_id,
            "capability": self.capability,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "correlation_id": self.correlation_id,
            "previous_event_id": self.previous_event_id,
            "causal_parent_ids": list(self.causal_parent_ids),
            "actor_id": self.actor_id,
            "actor_role": self.actor_role,
            "simulated_timestamp": self.simulated_timestamp,
            "wall_timestamp": self.wall_timestamp,
            "temporal_batch": self.temporal_batch,
            "randomization_block_key": self.randomization_block_key,
            "block_metadata": deepcopy(dict(self.block_metadata)),
            "requested_model_slug": self.requested_model_slug,
            "returned_model_slug": self.returned_model_slug,
            "provider_slug": self.provider_slug,
            "openrouter_request_id": self.openrouter_request_id,
            "prompt_hash": self.prompt_hash,
            "tool_schema_hash": self.tool_schema_hash,
            "state_before_hash": self.state_before_hash,
            "state_after_hash": self.state_after_hash,
            "proposed_action": self.proposed_action,
            "executed_action": self.executed_action,
            "eligibility": self.eligibility,
            "eligibility_reason": self.eligibility_reason,
            "treatment_assignment": self.treatment_assignment,
            "assignment_probability": self.assignment_probability,
            "context_refs": list(self.context_refs),
            "artifact_refs": list(self.artifact_refs),
            "tool_refs": list(self.tool_refs),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "wall_latency": self.wall_latency,
            "simulated_duration": self.simulated_duration,
            "retry_count": self.retry_count,
            "error_type": self.error_type,
            "protocol_violation": self.protocol_violation,
            "payload": deepcopy(dict(self.payload)),
        }
        return _json_safe(result)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
