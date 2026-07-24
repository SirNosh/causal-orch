#!/usr/bin/env python
"""Materialize and verify the Gaia2 scenario pinned in the repository manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("configs/gaia2_manifest.json")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_manifest_hash(manifest: dict[str, Any]) -> None:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError("manifest_sha256 is missing")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual = sha256_bytes(canonical_json(payload).encode("utf-8"))
    if actual != expected:
        raise ValueError(f"manifest hash mismatch: expected {expected}, got {actual}")


def raw_scenario(value: Any) -> str:
    if isinstance(value, str):
        json.loads(value)
        return value
    return canonical_json(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Gaia2 manifest must contain a JSON object")
    verify_manifest_hash(manifest)

    scenario_ids = manifest.get("scenario_ids")
    row_ids = manifest.get("row_ids")
    if not isinstance(scenario_ids, list) or len(scenario_ids) != 1:
        raise ValueError("the smoke manifest must pin exactly one scenario")
    if not isinstance(row_ids, list) or len(row_ids) != 1:
        raise ValueError("the smoke manifest must pin exactly one row ID")

    scenario_id = scenario_ids[0]
    row_id = row_ids[0]
    output_path = Path(manifest["scenario_files"][scenario_id])
    expected_sha = manifest["scenario_sha256"][scenario_id]

    if output_path.exists():
        actual_sha = sha256_bytes(output_path.read_bytes())
        if actual_sha == expected_sha:
            print(
                json.dumps(
                    {
                        "status": "verified",
                        "scenario_id": scenario_id,
                        "path": output_path.as_posix(),
                        "sha256": actual_sha,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.verify_only:
            raise ValueError(
                f"existing scenario hash mismatch: expected {expected_sha}, got {actual_sha}"
            )
    elif args.verify_only:
        raise FileNotFoundError(output_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required; run with: uv run --with datasets scripts/fetch_pinned_gaia2.py"
        ) from exc

    dataset = load_dataset(
        manifest["source_repo"],
        name=manifest["config"],
        split=manifest["split"],
        revision=manifest["gaia2_revision"],
        streaming=True,
    )

    selected: dict[str, Any] | None = None
    for row in dataset:
        if str(row.get("scenario_id")) == scenario_id:
            selected = dict(row)
            break
    if selected is None:
        raise LookupError(f"scenario not found at pinned revision: {scenario_id}")
    if str(selected.get("id")) != row_id:
        raise ValueError(
            f"row ID mismatch: expected {row_id}, got {selected.get('id')}"
        )

    raw = raw_scenario(selected["data"])
    actual_sha = sha256_bytes(raw.encode("utf-8"))
    if actual_sha != expected_sha:
        raise ValueError(
            f"scenario hash mismatch: expected {expected_sha}, got {actual_sha}"
        )

    parsed = json.loads(raw)
    definition = parsed.get("metadata", {}).get("definition", {})
    if definition.get("scenario_id") != scenario_id:
        raise ValueError("scenario JSON definition does not match the pinned ID")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(raw, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "materialized",
                "scenario_id": scenario_id,
                "path": output_path.as_posix(),
                "sha256": actual_sha,
                "revision": manifest["gaia2_revision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
