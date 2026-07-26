"""Build a reviewed, JSON-safe worker-tool manifest from supplied metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from causal_orch.runtime.read_only_tools import audit_tool


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _metadata_rows(value: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], set[str]]:
    if isinstance(value, Mapping):
        rows = value.get("tools", ())
        names = value.get("reviewed_allowlist", ())
    else:
        rows = value
        names = ()
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("tool metadata must contain a list of tools")
    rows_list = list(rows)
    if any(not isinstance(row, Mapping) for row in rows_list):
        raise ValueError("each tool metadata entry must be an object")
    reviewed = set(names)
    if any(not isinstance(name, str) or not name for name in reviewed):
        raise ValueError("reviewed tool names must be non-empty strings")
    return rows_list, reviewed


def build_reviewed_manifest(
    tool_metadata: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    reviewed_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Audit supplied metadata; unknown write state, including ``None``, is unsafe."""

    rows, embedded_names = _metadata_rows(tool_metadata)
    reviewed = set(embedded_names if reviewed_names is None else reviewed_names)
    if any(not isinstance(name, str) or not name for name in reviewed):
        raise ValueError("reviewed tool names must be non-empty strings")
    records = [
        audit_tool(row, audited_allowlist=reviewed).to_dict()
        for row in rows
    ]
    selected = [
        record["public_name"]
        for record in records
        if record["allowlisted"] and record["exclusion_reason"] is None
    ]
    result: dict[str, Any] = {
        "reviewed_allowlist": sorted(reviewed),
        "tools": records,
        "selected_read_only_tools": sorted(selected),
    }
    result["manifest_sha256"] = hashlib.sha256(_canonical(result).encode("utf-8")).hexdigest()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reviewed-name", action="append", default=[])
    args = parser.parse_args(argv)
    metadata = json.loads(args.input.read_text(encoding="utf-8"))
    manifest = build_reviewed_manifest(metadata, reviewed_names=args.reviewed_name or None)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote reviewed tool manifest with {len(manifest['selected_read_only_tools'])} selected tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
