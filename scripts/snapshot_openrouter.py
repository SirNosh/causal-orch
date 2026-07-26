"""Take an OpenRouter model-catalog snapshot, never a completion request."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen


CATALOG_URL = "https://openrouter.ai/api/v1/models"
MODEL_CANDIDATE_ORDER = (
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
)


@dataclass(frozen=True)
class CatalogRequest:
    url: str
    headers: Mapping[str, str]
    method: str = "GET"


@dataclass(frozen=True)
class CatalogResponse:
    status_code: int
    body: Mapping[str, Any]


CatalogTransport = Callable[[CatalogRequest], CatalogResponse]


def _live_catalog_transport(request: CatalogRequest) -> CatalogResponse:
    if request.method != "GET" or request.url != CATALOG_URL:
        raise ValueError("catalog transport accepts only the OpenRouter model catalog GET")
    with urlopen(Request(request.url, headers=dict(request.headers), method=request.method)) as response:
        body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, Mapping):
            raise ValueError("OpenRouter catalog response must be a JSON object")
        return CatalogResponse(response.status, body)


def _rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    data = payload.get("data", ())
    if not isinstance(data, list):
        raise ValueError("catalog response data must be a list")
    return [row for row in data if isinstance(row, Mapping)]


def snapshot_catalog(
    *,
    transport: CatalogTransport | None = None,
    api_key: str | None = None,
    model_slugs: Sequence[str] = MODEL_CANDIDATE_ORDER,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch only catalog metadata through an injected transport or explicit key.

    The returned snapshot intentionally excludes request headers, including the API key.
    """

    if transport is None and not api_key:
        raise ValueError("an injected transport or explicit api_key is required")
    headers = {"Accept": "application/json"}
    if api_key is not None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non-empty string when supplied")
        headers["Authorization"] = f"Bearer {api_key}"
    request = CatalogRequest(CATALOG_URL, headers)
    response = (transport or _live_catalog_transport)(request)
    if response.status_code != 200:
        raise ValueError(f"catalog request failed with HTTP {response.status_code}")
    rows_by_slug = {
        str(row.get("id")): row for row in _rows(response.body) if row.get("id")
    }
    selected = []
    for slug in model_slugs:
        row = rows_by_slug.get(slug)
        selected.append({"requested_model_slug": slug, "available": row is not None, "catalog": row})
    timestamp = now or datetime.now(timezone.utc)
    return {
        "snapshot_timestamp_utc": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "catalog_endpoint": CATALOG_URL,
        "requested_model_order": list(model_slugs),
        "models": selected,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", required=True, help="explicit key; it is not written to the snapshot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    snapshot = snapshot_catalog(api_key=args.api_key)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote catalog snapshot for {len(snapshot['models'])} locked candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
