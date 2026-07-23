"""Dependency-injected smoke-plan contract with no live scenario assumptions."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class SmokeStep:
    name: str


@dataclass(frozen=True)
class SmokePlan:
    steps: tuple[SmokeStep, ...]
    context: Mapping[str, Any]


SmokeRunner = Callable[[SmokeStep], Any]


def build_smoke_plan(
    checks: Sequence[str] = (),
    *,
    context: Mapping[str, Any] | None = None,
) -> SmokePlan:
    """Construct labels and caller-owned context; it creates no ARE/Gaia2 objects."""

    if any(not isinstance(check, str) or not check for check in checks):
        raise ValueError("smoke check names must be non-empty strings")
    if len(set(checks)) != len(checks):
        raise ValueError("smoke check names must be unique")
    return SmokePlan(tuple(SmokeStep(check) for check in checks), dict(context or {}))


def run_smoke(plan: SmokePlan, runner: SmokeRunner) -> dict[str, Any]:
    """Run only the injected checks and return their supplied results."""

    if not isinstance(plan, SmokePlan):
        raise TypeError("plan must be a SmokePlan")
    if not callable(runner):
        raise TypeError("runner must be callable")
    results = [{"check": step.name, "result": runner(step)} for step in plan.steps]
    return {
        "checks": results,
        "passed": all(bool(item["result"]) for item in results),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checks", nargs="*", default=[])
    args = parser.parse_args(argv)
    plan = build_smoke_plan(args.checks)
    print(json.dumps({"checks": [step.name for step in plan.steps], "context": plan.context}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
