"""Primary randomized analysis and descriptive secondary summaries."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math
import random
from typing import Any, Iterable, Mapping


def _arm(value: Any) -> str:
    if isinstance(value, bool):
        return "treatment" if value else "control"
    value = getattr(value, "label", value)
    value = getattr(value, "value", value)
    text = str(value).strip().lower()
    if text in {"treatment", "treated", "execute", "1"}:
        return "treatment"
    if text in {"control", "suppress", "suppressed", "0"}:
        return "control"
    raise ValueError(f"unknown assignment label: {value!r}")


def _row_value(row: Any, *names: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                return row[name]
    else:
        for name in names:
            if hasattr(row, name):
                return getattr(row, name)
    return default


def _eligible_rows(rows: Iterable[Any]) -> list[tuple[str, str, bool]]:
    result: list[tuple[str, str, bool]] = []
    for index, row in enumerate(rows):
        if isinstance(row, tuple) and len(row) == 3:
            block, assignment, success = row
            if isinstance(success, bool):
                result.append((str(block), _arm(assignment), success))
                continue
        eligible = _row_value(row, "eligible", "eligibility", default=True)
        if isinstance(eligible, str):
            eligible = eligible.upper() in {"ELIGIBLE", "TRUE"}
        if not eligible:
            continue
        assignment = _row_value(row, "assignment", "arm", "treatment")
        if assignment is None:
            continue
        success = _row_value(row, "success", "outcome", "binary_success")
        if not isinstance(success, bool):
            raise ValueError(f"eligible assigned row {index} lacks a boolean success outcome")
        block = _row_value(row, "block", "randomization_block", default="all")
        result.append((str(block), _arm(assignment), success))
    if not result:
        raise ValueError("no eligible randomized rows with binary outcomes")
    return result


def _risk_difference(assignments: list[str], outcomes: list[bool]) -> float:
    treated = [outcome for arm, outcome in zip(assignments, outcomes) if arm == "treatment"]
    control = [outcome for arm, outcome in zip(assignments, outcomes) if arm == "control"]
    if not treated or not control:
        raise ValueError("both treatment and control must be represented")
    return sum(treated) / len(treated) - sum(control) / len(control)


@dataclass(frozen=True)
class AssignmentBalance:
    treatment: int
    control: int
    by_block: dict[str, tuple[int, int]]

    @property
    def total(self) -> int:
        return self.treatment + self.control

    @property
    def difference(self) -> int:
        return self.treatment - self.control

    @property
    def treatment_count(self) -> int:
        return self.treatment

    @property
    def control_count(self) -> int:
        return self.control

    def __getitem__(self, key: str) -> Any:
        if key == "treatment":
            return self.treatment
        if key == "control":
            return self.control
        if key == "total":
            return self.total
        if key == "difference":
            return self.difference
        if key == "by_block":
            return self.by_block
        raise KeyError(key)


def assignment_balance(rows: Iterable[Any]) -> AssignmentBalance:
    prepared = _eligible_rows(rows)
    blocks: dict[str, list[int]] = {}
    for block, arm, _ in prepared:
        counts = blocks.setdefault(block, [0, 0])
        counts[0 if arm == "treatment" else 1] += 1
    return AssignmentBalance(
        treatment=sum(counts[0] for counts in blocks.values()),
        control=sum(counts[1] for counts in blocks.values()),
        by_block={block: (counts[0], counts[1]) for block, counts in blocks.items()},
    )


@dataclass(frozen=True)
class PrimaryEstimate:
    estimand: str
    treatment_count: int
    control_count: int
    treatment_successes: int
    control_successes: int
    treatment_success_rate: float
    control_success_rate: float
    absolute_risk_difference: float
    relative_risk: float | None
    p_value: float
    p_value_method: str
    monte_carlo_seed: str | int | None = None

    @property
    def risk_difference(self) -> float:
        return self.absolute_risk_difference


@dataclass(frozen=True)
class SecondarySummary:
    metric: str
    treatment_count: int
    control_count: int
    treatment_mean: float | None
    control_mean: float | None
    difference_in_means: float | None


def _enumeration_size(prepared: list[tuple[str, str, bool]]) -> int:
    size = 1
    for block in sorted({row[0] for row in prepared}):
        rows = [row for row in prepared if row[0] == block]
        treated = sum(row[1] == "treatment" for row in rows)
        size *= math.comb(len(rows), treated)
    return size


def _exact_p_value(prepared: list[tuple[str, str, bool]], observed: float) -> float:
    grouped: dict[str, list[tuple[str, bool]]] = {}
    for block, arm, outcome in prepared:
        grouped.setdefault(block, []).append((arm, outcome))
    distributions: list[list[str]] = []
    outcomes: list[bool] = []
    for block in sorted(grouped):
        rows = grouped[block]
        treated = sum(arm == "treatment" for arm, _ in rows)
        distributions.append(["".join("t" if index in chosen else "c" for index in range(len(rows)))
                              for chosen in combinations(range(len(rows)), treated)])
        outcomes.extend(outcome for _, outcome in rows)

    block_sizes = [len(grouped[block]) for block in sorted(grouped)]
    extreme = 0
    total = 0
    for choices in product(*distributions):
        assignments: list[str] = []
        for choice in choices:
            assignments.extend("treatment" if value == "t" else "control" for value in choice)
        statistic = _risk_difference(assignments, outcomes)
        total += 1
        extreme += abs(statistic) >= abs(observed) - 1e-12
    return extreme / total


def _monte_carlo_p_value(
    prepared: list[tuple[str, str, bool]], observed: float, samples: int, seed: str | int
) -> float:
    if type(samples) is not int or samples < 1:
        raise ValueError("monte_carlo_samples must be a positive integer")
    rng = random.Random(seed)
    grouped: dict[str, list[tuple[str, bool]]] = {}
    for block, arm, outcome in prepared:
        grouped.setdefault(block, []).append((arm, outcome))
    extreme = 0
    for _ in range(samples):
        assignments: list[str] = []
        outcomes: list[bool] = []
        for block in sorted(grouped):
            rows = grouped[block]
            treated = sum(arm == "treatment" for arm, _ in rows)
            treated_indexes = set(rng.sample(range(len(rows)), treated))
            assignments.extend("treatment" if index in treated_indexes else "control" for index in range(len(rows)))
            outcomes.extend(outcome for _, outcome in rows)
        extreme += abs(_risk_difference(assignments, outcomes)) >= abs(observed) - 1e-12
    return (extreme + 1) / (samples + 1)


def randomization_inference_p_value(
    rows: Iterable[Any], *, max_exact_assignments: int = 100_000,
    monte_carlo_samples: int = 10_000, seed: str | int | None = None,
) -> float:
    """Two-sided RI p-value preserving each block's observed treated count."""

    prepared = _eligible_rows(rows)
    assignments = [row[1] for row in prepared]
    outcomes = [row[2] for row in prepared]
    observed = _risk_difference(assignments, outcomes)
    space = _enumeration_size(prepared)
    if space <= max_exact_assignments:
        return _exact_p_value(prepared, observed)
    if seed is None:
        raise ValueError("a seed is required for the large-space Monte Carlo fallback")
    return _monte_carlo_p_value(prepared, observed, monte_carlo_samples, seed)


def estimate_primary(
    rows: Iterable[Any], *, max_exact_assignments: int = 100_000,
    monte_carlo_samples: int = 10_000, seed: str | int | None = None,
) -> PrimaryEstimate:
    """Estimate the ITT effect among eligible randomized proposals."""

    prepared = _eligible_rows(rows)
    treated = [outcome for _, arm, outcome in prepared if arm == "treatment"]
    control = [outcome for _, arm, outcome in prepared if arm == "control"]
    if not treated or not control:
        raise ValueError("both treatment and control must be represented")
    treatment_rate = sum(treated) / len(treated)
    control_rate = sum(control) / len(control)
    difference = treatment_rate - control_rate
    relative_risk = treatment_rate / control_rate if control_rate else None
    space = _enumeration_size(prepared)
    p_value = randomization_inference_p_value(
        prepared, max_exact_assignments=max_exact_assignments,
        monte_carlo_samples=monte_carlo_samples, seed=seed,
    )
    return PrimaryEstimate(
        estimand="intent_to_treat_randomized_binary_success",
        treatment_count=len(treated),
        control_count=len(control),
        treatment_successes=sum(treated),
        control_successes=sum(control),
        treatment_success_rate=treatment_rate,
        control_success_rate=control_rate,
        absolute_risk_difference=difference,
        relative_risk=relative_risk,
        p_value=p_value,
        p_value_method="exact_randomization_enumeration" if space <= max_exact_assignments else "seeded_monte_carlo",
        monte_carlo_seed=seed if space > max_exact_assignments else None,
    )


def summarize_secondary(rows: Iterable[Any], value_key: str) -> SecondarySummary:
    """Describe a continuous/count outcome; it never changes the primary estimator."""

    values: dict[str, list[float]] = {"treatment": [], "control": []}
    for row in rows:
        eligible = _row_value(row, "eligible", "eligibility", default=True)
        if isinstance(eligible, str):
            eligible = eligible.upper() in {"ELIGIBLE", "TRUE"}
        if not eligible:
            continue
        arm = _arm(_row_value(row, "assignment", "arm", "treatment"))
        value = _row_value(row, value_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{value_key} must be numeric")
        values[arm].append(float(value))
    treatment = values["treatment"]
    control = values["control"]
    treatment_mean = sum(treatment) / len(treatment) if treatment else None
    control_mean = sum(control) / len(control) if control else None
    return SecondarySummary(
        metric=value_key, treatment_count=len(treatment), control_count=len(control),
        treatment_mean=treatment_mean, control_mean=control_mean,
        difference_in_means=(treatment_mean - control_mean if treatment_mean is not None and control_mean is not None else None),
    )


# Short aliases keep the analysis entry points easy to discover.
primary_estimate = estimate_primary
secondary_summary = summarize_secondary
