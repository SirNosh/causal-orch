"""Pure validation for the protocol's delegation and evidence schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping
from uuid import UUID


MAX_OBJECTIVE_LENGTH = 512
MAX_CRITERION_LENGTH = 512
MAX_CONTEXT_REFS = 16
MAX_REF_LENGTH = 128
MAX_WORKER_TOOLS = 16
MAX_WORKER_STEPS = 8
MAX_WORKER_OUTPUT_TOKENS = 2_000
MAX_FINDINGS = 32
MAX_FINDING_REFS = 16
MAX_LIST_ITEMS = 32


class ReasonCode(str, Enum):
    INFORMATION_GAP = "INFORMATION_GAP"
    CROSS_APPLICATION_SEARCH = "CROSS_APPLICATION_SEARCH"
    TEMPORAL_DEPENDENCY_UNRESOLVED = "TEMPORAL_DEPENDENCY_UNRESOLVED"
    AMBIGUITY_REQUIRES_INDEPENDENT_ANALYSIS = "AMBIGUITY_REQUIRES_INDEPENDENT_ANALYSIS"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    HIGH_WRITE_RISK = "HIGH_WRITE_RISK"
    CONTEXT_OVERLOAD = "CONTEXT_OVERLOAD"
    SPECIALIZED_CALCULATION = "SPECIALIZED_CALCULATION"
    OTHER_BOUNDED_SUBTASK = "OTHER_BOUNDED_SUBTASK"


class RejectionReason(str, Enum):
    INVALID_SCHEMA = "INVALID_SCHEMA"
    UNBOUNDED_OBJECTIVE = "UNBOUNDED_OBJECTIVE"
    UNKNOWN_CONTEXT_REFERENCE = "UNKNOWN_CONTEXT_REFERENCE"
    WRITE_PERMISSION_REQUESTED = "WRITE_PERMISSION_REQUESTED"
    UNAPPROVED_WORKER_TOOL = "UNAPPROVED_WORKER_TOOL"
    DUPLICATE_OBJECTIVE = "DUPLICATE_OBJECTIVE"
    DELEGATION_ALREADY_DECIDED = "DELEGATION_ALREADY_DECIDED"
    ORACLE_LEAKAGE_RISK = "ORACLE_LEAKAGE_RISK"
    POST_TERMINAL_PROPOSAL = "POST_TERMINAL_PROPOSAL"
    WORKER_BUDGET_EXCEEDED = "WORKER_BUDGET_EXCEEDED"
    SCENARIO_OUT_OF_SCOPE = "SCENARIO_OUT_OF_SCOPE"


class EligibilityStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    REJECTED = "REJECTED"


class ArtifactStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ValidationError(ValueError):
    """A schema error carrying the protocol rejection vocabulary."""

    def __init__(self, message: str, reason: RejectionReason = RejectionReason.INVALID_SCHEMA):
        super().__init__(message)
        self.reason = reason


def _text(value: Any, field: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    value = value.strip()
    if not value and not allow_empty:
        raise ValidationError(f"{field} must not be empty")
    if len(value) > maximum:
        raise ValidationError(f"{field} exceeds its bound", RejectionReason.UNBOUNDED_OBJECTIVE)
    if any(ord(char) < 32 and char not in "\t" for char in value):
        raise ValidationError(f"{field} contains a control character")
    return value


def _string_sequence(value: Any, field: str, maximum_items: int, maximum_length: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValidationError(f"{field} must be a list of strings")
    if len(value) > maximum_items:
        raise ValidationError(f"{field} has too many items")
    result = tuple(_text(item, field, maximum_length) for item in value)
    if len(set(result)) != len(result):
        raise ValidationError(f"{field} must not contain duplicates")
    return result


def _enum(value: Any, enum_type: type[Enum], field: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field} is not in the fixed vocabulary") from exc


@dataclass(frozen=True)
class DelegationProposal:
    proposal_id: str
    objective: str
    reason_code: ReasonCode
    context_refs: tuple[str, ...]
    allowed_read_tools: tuple[str, ...]
    completion_criterion: str
    max_worker_steps: int
    max_worker_output_tokens: int

    def __post_init__(self) -> None:
        try:
            UUID(self.proposal_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValidationError("proposal_id must be a UUID") from exc
        object.__setattr__(self, "objective", _text(self.objective, "objective", MAX_OBJECTIVE_LENGTH))
        object.__setattr__(
            self,
            "completion_criterion",
            _text(self.completion_criterion, "completion_criterion", MAX_CRITERION_LENGTH),
        )
        object.__setattr__(self, "reason_code", _enum(self.reason_code, ReasonCode, "reason_code"))
        object.__setattr__(
            self,
            "context_refs",
            _string_sequence(self.context_refs, "context_refs", MAX_CONTEXT_REFS, MAX_REF_LENGTH),
        )
        object.__setattr__(
            self,
            "allowed_read_tools",
            _string_sequence(self.allowed_read_tools, "allowed_read_tools", MAX_WORKER_TOOLS, MAX_REF_LENGTH),
        )
        if type(self.max_worker_steps) is not int or not 1 <= self.max_worker_steps <= MAX_WORKER_STEPS:
            raise ValidationError("max_worker_steps exceeds its bound", RejectionReason.WORKER_BUDGET_EXCEEDED)
        if (
            type(self.max_worker_output_tokens) is not int
            or not 1 <= self.max_worker_output_tokens <= MAX_WORKER_OUTPUT_TOKENS
        ):
            raise ValidationError(
                "max_worker_output_tokens exceeds its bound", RejectionReason.WORKER_BUDGET_EXCEEDED
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DelegationProposal":
        fields = {
            "proposal_id",
            "objective",
            "reason_code",
            "context_refs",
            "allowed_read_tools",
            "completion_criterion",
            "max_worker_steps",
            "max_worker_output_tokens",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValidationError("proposal fields are missing or unknown")
        return cls(**dict(value))

    @property
    def allowed_worker_tools(self) -> tuple[str, ...]:
        return self.allowed_read_tools

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "objective": self.objective,
            "reason_code": self.reason_code.value,
            "context_refs": list(self.context_refs),
            "allowed_read_tools": list(self.allowed_read_tools),
            "completion_criterion": self.completion_criterion,
            "max_worker_steps": self.max_worker_steps,
            "max_worker_output_tokens": self.max_worker_output_tokens,
        }


@dataclass(frozen=True)
class EligibilityDecision:
    status: EligibilityStatus
    rejection_reason: RejectionReason | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(self.status, EligibilityStatus, "status"))
        if self.rejection_reason is not None:
            object.__setattr__(
                self,
                "rejection_reason",
                _enum(self.rejection_reason, RejectionReason, "rejection_reason"),
            )
        if self.status is EligibilityStatus.ELIGIBLE and self.rejection_reason is not None:
            raise ValueError("eligible proposals cannot have a rejection reason")
        if self.status is EligibilityStatus.REJECTED and self.rejection_reason is None:
            raise ValueError("rejected proposals require a rejection reason")

    @property
    def eligible(self) -> bool:
        return self.status is EligibilityStatus.ELIGIBLE

    @property
    def reason(self) -> str:
        return self.status.value if self.eligible else self.rejection_reason.value  # type: ignore[union-attr]


def _rejected(reason: RejectionReason) -> EligibilityDecision:
    return EligibilityDecision(EligibilityStatus.REJECTED, reason)


def validate_proposal(
    proposal: DelegationProposal | Mapping[str, Any],
    *,
    available_context_refs: Iterable[str] = (),
    allowed_worker_tools: Iterable[str] = (),
    prior_decision: bool = False,
    prior_objectives: Iterable[str] = (),
    objective_completed: bool = False,
    terminal: bool = False,
    oracle_refs: Iterable[str] = (),
    scenario_in_scope: bool = True,
) -> EligibilityDecision:
    """Validate a proposal without reading state or changing the schedule."""
    try:
        candidate = proposal if isinstance(proposal, DelegationProposal) else DelegationProposal.from_dict(proposal)
    except ValidationError as exc:
        return _rejected(exc.reason)

    if prior_decision:
        return _rejected(RejectionReason.DELEGATION_ALREADY_DECIDED)
    if terminal:
        return _rejected(RejectionReason.POST_TERMINAL_PROPOSAL)
    if not scenario_in_scope:
        return _rejected(RejectionReason.SCENARIO_OUT_OF_SCOPE)
    if objective_completed:
        return _rejected(RejectionReason.DUPLICATE_OBJECTIVE)
    if any(candidate.objective == previous for previous in prior_objectives):
        return _rejected(RejectionReason.DUPLICATE_OBJECTIVE)

    known_refs = set(available_context_refs)
    if any(ref not in known_refs for ref in candidate.context_refs):
        return _rejected(RejectionReason.UNKNOWN_CONTEXT_REFERENCE)
    oracle_names = {str(ref).lower() for ref in oracle_refs}
    if any(ref in oracle_names or "oracle" in ref.lower() for ref in candidate.context_refs):
        return _rejected(RejectionReason.ORACLE_LEAKAGE_RISK)

    allowed = set(allowed_worker_tools)
    if any(tool not in allowed for tool in candidate.allowed_read_tools):
        return _rejected(RejectionReason.UNAPPROVED_WORKER_TOOL)
    write_markers = ("write", "delete", "send", "update", "create", "execute")
    if any(any(marker in tool.lower() for marker in write_markers) for tool in candidate.allowed_read_tools):
        return _rejected(RejectionReason.WRITE_PERMISSION_REQUESTED)
    return EligibilityDecision(EligibilityStatus.ELIGIBLE)


@dataclass(frozen=True)
class EvidenceFinding:
    claim: str
    evidence_refs: tuple[str, ...]
    confidence: Confidence

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim", _text(self.claim, "claim", MAX_OBJECTIVE_LENGTH))
        object.__setattr__(
            self,
            "evidence_refs",
            _string_sequence(self.evidence_refs, "evidence_refs", MAX_FINDING_REFS, MAX_REF_LENGTH),
        )
        if not self.evidence_refs:
            raise ValidationError("every finding requires an evidence reference")
        object.__setattr__(self, "confidence", _enum(self.confidence, Confidence, "confidence"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceFinding":
        if not isinstance(value, Mapping) or set(value) != {"claim", "evidence_refs", "confidence"}:
            raise ValidationError("finding fields are missing or unknown")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence.value,
        }


@dataclass(frozen=True)
class EvidenceReport:
    artifact_type: str
    objective: str
    status: ArtifactStatus
    findings: tuple[EvidenceFinding, ...]
    uncertainties: tuple[str, ...]
    contradictions: tuple[str, ...]
    recommended_next_action: str | None

    def __post_init__(self) -> None:
        if self.artifact_type != "EVIDENCE_REPORT":
            raise ValidationError("artifact_type must be EVIDENCE_REPORT")
        object.__setattr__(self, "objective", _text(self.objective, "objective", MAX_OBJECTIVE_LENGTH))
        object.__setattr__(self, "status", _enum(self.status, ArtifactStatus, "status"))
        if not isinstance(self.findings, (list, tuple)) or len(self.findings) > MAX_FINDINGS:
            raise ValidationError("findings has too many items")
        if any(not isinstance(finding, EvidenceFinding) for finding in self.findings):
            raise ValidationError("findings must contain EvidenceFinding values")
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(
            self,
            "uncertainties",
            _string_sequence(self.uncertainties, "uncertainties", MAX_LIST_ITEMS, MAX_OBJECTIVE_LENGTH),
        )
        object.__setattr__(
            self,
            "contradictions",
            _string_sequence(self.contradictions, "contradictions", MAX_LIST_ITEMS, MAX_OBJECTIVE_LENGTH),
        )
        if self.recommended_next_action is not None:
            object.__setattr__(
                self,
                "recommended_next_action",
                _text(self.recommended_next_action, "recommended_next_action", MAX_OBJECTIVE_LENGTH),
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceReport":
        fields = {
            "artifact_type",
            "objective",
            "status",
            "findings",
            "uncertainties",
            "contradictions",
            "recommended_next_action",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValidationError("artifact fields are missing or unknown")
        findings = value["findings"]
        if not isinstance(findings, (list, tuple)):
            raise ValidationError("findings must be a list")
        return cls(
            artifact_type=value["artifact_type"],
            objective=value["objective"],
            status=value["status"],
            findings=tuple(EvidenceFinding.from_dict(item) for item in findings),
            uncertainties=value["uncertainties"],
            contradictions=value["contradictions"],
            recommended_next_action=value["recommended_next_action"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "objective": self.objective,
            "status": self.status.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "uncertainties": list(self.uncertainties),
            "contradictions": list(self.contradictions),
            "recommended_next_action": self.recommended_next_action,
        }
