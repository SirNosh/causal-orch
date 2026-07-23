"""Small Gaia2 validation adapter; no dataset loading or execution policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Gaia2Evaluation:
    """The unchanged validation result plus its native primary outcome."""

    validation_result: Any
    primary_outcome_label: str = "native_gaia2_binary_success"
    primary_success: bool | None = None

    @property
    def is_primary_outcome(self) -> bool:
        return True


def _success_from_validation(validation_result: Any) -> bool | None:
    if isinstance(validation_result, bool):
        return validation_result
    if isinstance(validation_result, dict) and "success" in validation_result:
        value = validation_result["success"]
    else:
        value = getattr(validation_result, "success", None)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError("scenario.validate(env) success must be bool or None")
    return value


def evaluate_gaia2_scenario(scenario: Any, env: Any) -> Gaia2Evaluation:
    """Call ``scenario.validate(env)`` exactly once and normalize its success flag."""

    validation_result = scenario.validate(env)
    return Gaia2Evaluation(
        validation_result=validation_result,
        primary_success=_success_from_validation(validation_result),
    )
