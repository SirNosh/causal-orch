"""Fixed worker budgets and deterministic accounting."""

from __future__ import annotations

import dataclasses
from typing import Any

from causal_orch.agent.schemas import MAX_WORKER_OUTPUT_TOKENS, MAX_WORKER_STEPS


@dataclasses.dataclass(frozen=True)
class WorkerBudgets:
    max_steps: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= MAX_WORKER_STEPS:
            raise ValueError("max_steps exceeds the fixed worker budget")
        if type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= MAX_WORKER_OUTPUT_TOKENS:
            raise ValueError("max_output_tokens exceeds the fixed worker budget")

    @classmethod
    def from_mapping(cls, value: Any) -> "WorkerBudgets":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("budgets must be WorkerBudgets or a mapping")
        return cls(value["max_steps"], value["max_output_tokens"])

    def to_dict(self) -> dict[str, int]:
        return {"max_steps": self.max_steps, "max_output_tokens": self.max_output_tokens}


@dataclasses.dataclass(frozen=True)
class BudgetExceededResult:
    budget: str
    steps_used: int
    output_tokens_used: int

    status: str = "BUDGET_EXCEEDED"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class BudgetExceededError(RuntimeError):
    def __init__(self, result: BudgetExceededResult):
        self.result = result
        super().__init__(result.status)


@dataclasses.dataclass
class BudgetUsage:
    budgets: WorkerBudgets
    steps_used: int = 0
    output_tokens_used: int = 0

    def _exceeded(self, budget: str) -> BudgetExceededError:
        return BudgetExceededError(BudgetExceededResult(budget, self.steps_used, self.output_tokens_used))

    def consume_step(self) -> None:
        if self.steps_used >= self.budgets.max_steps:
            raise self._exceeded("max_steps")
        self.steps_used += 1

    def consume_output_tokens(self, count: int) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("output token count must be a non-negative integer")
        if self.output_tokens_used + count > self.budgets.max_output_tokens:
            raise self._exceeded("max_output_tokens")
        self.output_tokens_used += count

    def result(self) -> dict[str, int]:
        return {"steps_used": self.steps_used, "output_tokens_used": self.output_tokens_used}


BudgetTracker = BudgetUsage
