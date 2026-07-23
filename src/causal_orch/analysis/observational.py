"""Proposal/no-proposal associations, explicitly separate from randomized effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


PRE_TREATMENT_COVARIATES = frozenset({
    "capability", "scenario_id", "universe_id", "model_condition", "temporal_batch",
    "trace_depth", "context_length", "tokens_before_proposal", "read_calls_before_proposal",
    "write_calls_before_proposal", "prior_invalid_calls", "prior_tool_errors",
    "pending_notifications", "simulated_time", "reason_code", "worker_objective_length",
    "requested_worker_tool_count",
})
POST_TREATMENT_MARKERS = frozenset({
    "worker_artifact", "post_treatment", "post_treatment_actions", "token_totals",
    "final_token_count", "post_treatment_trace_length", "artifact_cited", "later_plan_changes",
})


def _value(row: Any, *names: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                return row[name]
    else:
        for name in names:
            if hasattr(row, name):
                return getattr(row, name)
    return default


@dataclass(frozen=True)
class ObservationalAssociation:
    estimand: str
    causal: bool
    proposal_count: int
    no_proposal_count: int
    proposal_success_rate: float
    no_proposal_success_rate: float
    absolute_risk_difference: float
    relative_risk: float | None
    adjusted: bool = False
    covariates: tuple[str, ...] = ()

    @property
    def risk_difference(self) -> float:
        return self.absolute_risk_difference


def _groups(rows: Iterable[Any], proposal_key: str, success_key: str) -> tuple[list[bool], list[bool]]:
    proposed: list[bool] = []
    not_proposed: list[bool] = []
    for index, row in enumerate(rows):
        proposal = _value(row, proposal_key, "proposed", "has_proposal")
        success = _value(row, success_key, "success", "outcome", "binary_success")
        if not isinstance(proposal, bool) or not isinstance(success, bool):
            raise ValueError(f"row {index} requires boolean proposal and success values")
        (proposed if proposal else not_proposed).append(success)
    if not proposed or not not_proposed:
        raise ValueError("both proposal and no-proposal groups must be represented")
    return proposed, not_proposed


def proposal_association(
    rows: Iterable[Any], *, proposal_key: str = "proposed", success_key: str = "success"
) -> ObservationalAssociation:
    """Return a naive association; proposal selection is not randomized."""

    proposed, not_proposed = _groups(rows, proposal_key, success_key)
    proposal_rate = sum(proposed) / len(proposed)
    no_proposal_rate = sum(not_proposed) / len(not_proposed)
    return ObservationalAssociation(
        estimand="observational_proposal_no_proposal_association",
        causal=False,
        proposal_count=len(proposed),
        no_proposal_count=len(not_proposed),
        proposal_success_rate=proposal_rate,
        no_proposal_success_rate=no_proposal_rate,
        absolute_risk_difference=proposal_rate - no_proposal_rate,
        relative_risk=proposal_rate / no_proposal_rate if no_proposal_rate else None,
    )


def adjusted_proposal_association(
    rows: Iterable[Mapping[str, Any]], covariates: Iterable[str], *,
    proposal_key: str = "proposed", success_key: str = "success",
) -> ObservationalAssociation:
    """Standardize stratum-specific associations using pre-treatment covariates only."""

    covariates = tuple(covariates)
    invalid = [name for name in covariates if name not in PRE_TREATMENT_COVARIATES]
    if invalid:
        raise ValueError(f"adjustment covariates must be prespecified pre-treatment fields: {invalid}")
    rows = list(rows)
    overall = proposal_association(rows, proposal_key=proposal_key, success_key=success_key)
    strata: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        strata.setdefault(tuple(row.get(name) for name in covariates), []).append(row)
    adjusted_difference = 0.0
    adjusted_proposal_rate = 0.0
    adjusted_no_proposal_rate = 0.0
    total = len(rows)
    for stratum in strata.values():
        try:
            proposed, not_proposed = _groups(stratum, proposal_key, success_key)
        except ValueError:
            continue
        weight = len(stratum) / total
        proposal_rate = sum(proposed) / len(proposed)
        no_proposal_rate = sum(not_proposed) / len(not_proposed)
        adjusted_difference += weight * (proposal_rate - no_proposal_rate)
        adjusted_proposal_rate += weight * proposal_rate
        adjusted_no_proposal_rate += weight * no_proposal_rate
    return ObservationalAssociation(
        estimand="adjusted_observational_proposal_no_proposal_association",
        causal=False,
        proposal_count=overall.proposal_count,
        no_proposal_count=overall.no_proposal_count,
        proposal_success_rate=adjusted_proposal_rate,
        no_proposal_success_rate=adjusted_no_proposal_rate,
        absolute_risk_difference=adjusted_difference,
        relative_risk=adjusted_proposal_rate / adjusted_no_proposal_rate if adjusted_no_proposal_rate else None,
        adjusted=True,
        covariates=covariates,
    )


summarize_observational = proposal_association

