"""Run the frozen five-attempt infrastructure qualification gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from run_smoke import (
    LOCKED_ARE_COMMIT,
    load_configurations,
    load_symbol,
    require_smoke_prerequisites,
    run_route_qualification,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-factory",
        default="scripts.run_smoke:pinned_gaia2_factory",
    )
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--are-revision", default=os.environ.get("ARE_COMMIT"))
    parser.add_argument(
        "--artifact-dir",
        default=str(
            root
            / "artifacts"
            / "qualification"
            / datetime.now(timezone.utc).strftime("qualification-%Y%m%dT%H%M%SZ")
        ),
    )
    args = parser.parse_args()
    experiment, gaia2_manifest, providers = load_configurations(root / "configs")
    require_smoke_prerequisites(
        experiment=experiment,
        gaia2_manifest=gaia2_manifest,
        providers=providers,
        scenario_factory_spec=args.scenario_factory,
        are_revision=args.are_revision,
    )
    report = run_route_qualification(
        load_symbol(args.scenario_factory),
        artifact_root=args.artifact_dir,
        attempts=args.attempts,
        progress=lambda result: print(
            json.dumps(
                {
                    "attempt": result["attempt_number"],
                    "infrastructure_pass": result["infrastructure_pass"],
                    "gaia2_success": result["gaia2_success"],
                    "failure_stage": result["failure_stage"],
                },
                sort_keys=True,
            ),
            flush=True,
        ),
    )
    print(
        json.dumps(
            {
                "qualified": report["qualified"],
                "counts": report["counts"],
                "artifact_dir": args.artifact_dir,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
