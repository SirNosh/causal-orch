"""Injected fresh read-only worker boundary; no model loop is implemented here."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Callable, Mapping

from causal_orch.runtime.budgets import BudgetExceededError, BudgetExceededResult, WorkerBudgets
from causal_orch.runtime.read_only_tools import (
    MANUALLY_AUDITED_ALLOWLIST,
    ReadOnlyToolSelection,
    select_read_only_tools,
)
from causal_orch.runtime.state_guard import StateGuard, canonical_json

from .schemas import EvidenceReport, ValidationError


class TreatmentFailureReason(str, Enum):
    MALFORMED_ARTIFACT = "MALFORMED_ARTIFACT"
    TIMEOUT_OR_BUDGET_EXHAUSTION = "TIMEOUT_OR_BUDGET_EXHAUSTION"
    STATE_DIFF_VIOLATION = "STATE_DIFF_VIOLATION"
    RUNNER_ERROR = "RUNNER_ERROR"


@dataclasses.dataclass(frozen=True)
class TreatmentFailure:
    reason: TreatmentFailureReason
    detail: str

    classification: str = "TREATMENT_FAILURE"


@dataclasses.dataclass(frozen=True)
class WorkerResult:
    artifact: EvidenceReport | None
    failure: TreatmentFailure | None
    state_guard: StateGuard
    payload: Mapping[str, Any]

    @property
    def succeeded(self) -> bool:
        return self.artifact is not None and self.failure is None


class FreshReadOnlyWorker:
    """Validate a worker request and delegate one fresh session to an injected runner."""

    def __init__(
        self,
        *,
        environment: Any,
        available_tools: tuple[Any, ...] | list[Any],
        runner: Callable[[Mapping[str, Any]], Any],
        audited_allowlist: frozenset[str] = MANUALLY_AUDITED_ALLOWLIST,
    ) -> None:
        self.environment = environment
        self.available_tools = tuple(available_tools)
        self.runner = runner
        self.audited_allowlist = frozenset(audited_allowlist)

    def _payload(
        self,
        objective: str,
        context: Any,
        selection: ReadOnlyToolSelection,
        budgets: WorkerBudgets,
    ) -> dict[str, Any]:
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be a non-empty string")
        import json

        serialized_context = json.loads(canonical_json(context))
        return {
            "objective": objective.strip(),
            "context": serialized_context,
            "tools": [manifest.to_dict() for manifest in selection.selected_manifests],
            "budgets": budgets.to_dict(),
        }

    @staticmethod
    def _validate_artifact(value: Any, objective: str) -> EvidenceReport:
        if isinstance(value, EvidenceReport):
            artifact = value
        elif isinstance(value, Mapping):
            artifact = EvidenceReport.from_dict(value)
        else:
            raise ValidationError("worker returned a non-mapping artifact")
        if artifact.objective != objective:
            raise ValidationError("worker artifact objective does not match the request")
        return artifact

    def run(
        self,
        *,
        objective: str,
        context: Any,
        requested_tools: tuple[str, ...] | list[str],
        budgets: WorkerBudgets | Mapping[str, Any],
    ) -> WorkerResult:
        worker_budgets = WorkerBudgets.from_mapping(budgets)
        selection = select_read_only_tools(
            self.available_tools,
            requested_tools,
            audited_allowlist=self.audited_allowlist,
        )
        payload = self._payload(objective, context, selection, worker_budgets)
        guard = StateGuard(self.environment)
        value: Any = None
        failure: TreatmentFailure | None = None
        with guard:
            try:
                value = self.runner(payload)
            except (TimeoutError, BudgetExceededError) as exc:
                detail = str(exc)
                failure = TreatmentFailure(TreatmentFailureReason.TIMEOUT_OR_BUDGET_EXHAUSTION, detail)
            except Exception as exc:  # Runner failures are observable treatment outcomes.
                failure = TreatmentFailure(TreatmentFailureReason.RUNNER_ERROR, f"{type(exc).__name__}: {exc}")

        if guard.changed:
            failure = TreatmentFailure(
                TreatmentFailureReason.STATE_DIFF_VIOLATION,
                "application state changed during the worker attempt",
            )
        elif failure is None:
            if isinstance(value, BudgetExceededResult) or (
                isinstance(value, Mapping) and value.get("status") == "BUDGET_EXCEEDED"
            ):
                failure = TreatmentFailure(
                    TreatmentFailureReason.TIMEOUT_OR_BUDGET_EXHAUSTION,
                    "worker exhausted its fixed budget",
                )
            else:
                try:
                    value = self._validate_artifact(value, payload["objective"])
                except (ValidationError, TypeError, ValueError) as exc:
                    failure = TreatmentFailure(TreatmentFailureReason.MALFORMED_ARTIFACT, str(exc))
        return WorkerResult(value if isinstance(value, EvidenceReport) and failure is None else None, failure, guard, payload)
