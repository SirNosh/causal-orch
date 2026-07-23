"""Validate supplied model/provider manifests without making a model decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from causal_orch.models.health import qualify_manifest
from causal_orch.models.manifests import MODEL_CANDIDATE_ORDER, ModelManifest


def _manifest_rows(value: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("manifests"), list):
            value = value["manifests"]
        elif all(isinstance(row, Mapping) for row in value.values()):
            value = value.values()
        else:
            raise ValueError("manifest input must contain a manifests list or slug mapping")
    rows = list(value)
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("each model manifest must be an object")
    return rows


def qualify_models(
    manifests: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    models_config: Mapping[str, Any],
    providers_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return per-candidate validation results; ``selected_model`` is always null."""

    errors: list[str] = []
    order = models_config.get("candidate_order")
    if tuple(order or ()) != MODEL_CANDIDATE_ORDER:
        errors.append("MODEL_CANDIDATE_ORDER_MISMATCH")
    pinned_provider = providers_config.get("pinned_provider", "")
    if not isinstance(pinned_provider, str) or not pinned_provider:
        errors.append("PINNED_PROVIDER_PLACEHOLDER")
    if providers_config.get("allow_fallbacks") is not False:
        errors.append("PROVIDER_FALLBACKS_MUST_BE_DISABLED")
    if providers_config.get("require_parameters") is not True:
        errors.append("PROVIDER_PARAMETER_REQUIREMENT_MUST_BE_TRUE")

    try:
        rows = _manifest_rows(manifests)
    except ValueError as error:
        return {
            "valid": False,
            "candidate_order": list(MODEL_CANDIDATE_ORDER),
            "pinned_provider": pinned_provider,
            "config_errors": errors + [str(error)],
            "candidates": [],
            "selected_model": None,
        }
    by_slug = {row.get("requested_model_slug"): row for row in rows}
    candidates = []
    for slug in MODEL_CANDIDATE_ORDER:
        row = by_slug.get(slug)
        reasons = list(errors)
        if row is None:
            reasons.append("MANIFEST_MISSING")
        elif not row.get("manifest_sha256"):
            reasons.append("MANIFEST_HASH_MISSING")
        else:
            try:
                manifest = ModelManifest.from_dict(row)
                if isinstance(pinned_provider, str) and pinned_provider:
                    result = qualify_manifest(manifest, pinned_provider)
                    reasons.extend(result.reasons)
            except (KeyError, TypeError, ValueError) as error:
                reasons.append(f"INVALID_MANIFEST:{error}")
        candidates.append({"model_slug": slug, "qualified": not reasons, "reasons": reasons})
    return {
        "valid": not errors and any(candidate["qualified"] for candidate in candidates),
        "candidate_order": list(MODEL_CANDIDATE_ORDER),
        "pinned_provider": pinned_provider,
        "config_errors": errors,
        "candidates": candidates,
        "selected_model": None,
    }


def _load(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path)
    parser.add_argument("models_config", type=Path)
    parser.add_argument("providers_config", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = qualify_models(_load(args.manifests), _load(args.models_config), _load(args.providers_config))
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"qualified {sum(item['qualified'] for item in report['candidates'])} supplied candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
