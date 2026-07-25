"""Verify the pinned local model with a plain text completion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping
import urllib.request


def request_json(url: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = (
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        if payload is not None
        else None
    )
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        value = json.loads(response.read().decode())
    if not isinstance(value, dict):
        raise ValueError("local server response must be a JSON object")
    return value


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        default=str(root / "artifacts" / "local" / "model-probe.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(
        (root / "configs" / "local_model_manifest.json").read_text()
    )
    endpoint = manifest["runtime"]["endpoint"]
    model = manifest["model_manifest"]["requested_model_slug"]
    base_url = endpoint.removesuffix("/v1/chat/completions")
    started = time.monotonic()
    props = request_json(f"{base_url}/props")
    response = request_json(
        endpoint,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly LOCAL_MODEL_READY.",
                }
            ],
            "stream": False,
            **manifest["model_manifest"]["sampling"],
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "tool_call_format": "xml",
            },
        },
    )
    if response.get("model") != model:
        raise ValueError("model identity mismatch")
    content = response["choices"][0]["message"].get("content")
    if not isinstance(content, str) or "LOCAL_MODEL_READY" not in content:
        raise ValueError("plain text completion probe failed")

    report = {
        "status": "passed",
        "model": model,
        "provider": manifest["provider_manifest"]["provider_name"],
        "llama_cpp_commit": manifest["provider_manifest"]["commit"],
        "context_length": manifest["runtime"]["context_length"],
        "model_loaded": bool(props),
        "text_completion_valid": True,
        "completion_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }
    output = Path(args.artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
