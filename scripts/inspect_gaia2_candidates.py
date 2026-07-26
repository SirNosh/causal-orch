#!/usr/bin/env python
"""Download a pinned Gaia2 Search split and rank smoke-scenario candidates.

This script is intentionally selection-only. It never modifies the committed
Gaia2 manifest. It emits a report and a bounded set of raw scenario JSON files
for manual inspection before one scenario is pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import HfApi


DATASET_REPO = "meta-agents-research-environments/gaia2"
QUESTION_WORDS = ("what", "which", "who", "when", "where", "find", "tell", "identify", "determine")
READ_KEYWORDS = ("email", "mail", "inbox", "file", "document", "search", "find", "look up", "which", "who", "what")
WRITE_KEYWORDS = (
    "send ",
    "reply",
    "schedule",
    "book ",
    "order ",
    "update ",
    "delete ",
    "create ",
    "add ",
    "message ",
    "call ",
    "purchase",
    "reserve",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_text(value: Any) -> str:
    if isinstance(value, str):
        json.loads(value)
        return value
    return canonical_json(value)


def iter_strings(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_strings(item, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_strings(item, (*path, str(index)))


def task_candidates(scenario: dict[str, Any]) -> list[str]:
    candidates: list[tuple[int, str]] = []
    events = scenario.get("events", [])
    for index, event in enumerate(events if isinstance(events, list) else []):
        event_blob = canonical_json(event).lower()
        is_user_event = any(
            marker in event_blob
            for marker in (
                "agentuserinterface",
                "user_interface",
                "message_to_agent",
                "send_task",
                "user_message",
            )
        )
        if not is_user_event:
            continue
        for path, text in iter_strings(event, ("events", str(index))):
            cleaned = " ".join(text.split())
            if not 12 <= len(cleaned) <= 1_200:
                continue
            lowered = cleaned.lower()
            score = 0
            if "content" in path or "message" in path or "task" in path or "text" in path:
                score += 5
            if cleaned.endswith("?"):
                score += 4
            if lowered.startswith(QUESTION_WORDS):
                score += 4
            score += 2 * sum(keyword in lowered for keyword in READ_KEYWORDS)
            score -= 3 * sum(keyword in lowered for keyword in WRITE_KEYWORDS)
            if "agentuserinterface" in lowered or lowered in {"user", "assistant"}:
                score -= 10
            candidates.append((score, cleaned))

    # Fall back to task-like metadata strings when event schemas change.
    if not candidates:
        for path, text in iter_strings(scenario):
            cleaned = " ".join(text.split())
            lowered = cleaned.lower()
            if 12 <= len(cleaned) <= 1_200 and any(key in path for key in ("task", "prompt", "content")):
                score = 2 * sum(keyword in lowered for keyword in READ_KEYWORDS)
                score -= 3 * sum(keyword in lowered for keyword in WRITE_KEYWORDS)
                candidates.append((score, cleaned))

    deduped: dict[str, int] = {}
    for score, text in candidates:
        deduped[text] = max(score, deduped.get(text, -10_000))
    return [text for text, _score in sorted(deduped.items(), key=lambda item: (-item[1], len(item[0])))[:8]]


def app_names(scenario: dict[str, Any]) -> list[str]:
    apps = scenario.get("apps", {})
    if isinstance(apps, dict):
        return sorted(str(key) for key in apps)
    if isinstance(apps, list):
        names = []
        for app in apps:
            if isinstance(app, dict):
                name = app.get("name") or app.get("app_name") or app.get("type")
                if name:
                    names.append(str(name))
        return sorted(set(names))
    return []


def event_summary(scenario: dict[str, Any]) -> dict[str, int]:
    events = scenario.get("events", [])
    if not isinstance(events, list):
        return {"total": 0, "oracle_like": 0, "dynamic_non_user": 0}
    oracle_like = 0
    dynamic_non_user = 0
    for event in events:
        blob = canonical_json(event).lower()
        if "oracle" in blob or "validator" in blob:
            oracle_like += 1
        if not any(marker in blob for marker in ("agentuserinterface", "user_message", "message_to_agent")):
            dynamic_non_user += 1
    return {"total": len(events), "oracle_like": oracle_like, "dynamic_non_user": dynamic_non_user}


def score_candidate(tasks: list[str], apps: list[str], events: dict[str, int]) -> int:
    task = tasks[0].lower() if tasks else ""
    score = 0
    score += 12 if tasks else -50
    score += 12 if "email" in task or "mail" in task or "inbox" in task else 0
    score += 8 if "file" in task or "document" in task else 0
    score += 6 if task.endswith("?") else 0
    score += 5 if task.startswith(QUESTION_WORDS) else 0
    score += 3 * sum(keyword in task for keyword in READ_KEYWORDS)
    score -= 7 * sum(keyword in task for keyword in WRITE_KEYWORDS)
    score += 5 if any("email" in app.lower() for app in apps) else 0
    score += 3 if any("file" in app.lower() for app in apps) else 0
    score -= min(events["dynamic_non_user"], 15)
    score -= max(0, len(task) - 400) // 50
    return score


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--output-dir", default="gaia2-inspection")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = HfApi().dataset_info(DATASET_REPO)
    revision = info.sha
    if not revision:
        raise RuntimeError("Hugging Face did not return a dataset revision")

    dataset = load_dataset(
        DATASET_REPO,
        name=args.config,
        split=args.split,
        revision=revision,
    )

    rows: list[dict[str, Any]] = []
    row_digest_records: list[dict[str, str]] = []
    raw_by_id: dict[str, str] = {}

    for row in dataset:
        raw = json_text(row["data"])
        scenario = json.loads(raw)
        scenario_id = str(row["scenario_id"])
        tasks = task_candidates(scenario)
        apps = app_names(scenario)
        events = event_summary(scenario)
        raw_sha = sha256_bytes(raw.encode("utf-8"))
        row_digest_records.append(
            {
                "id": str(row["id"]),
                "scenario_id": scenario_id,
                "data_sha256": raw_sha,
            }
        )
        rows.append(
            {
                "row_id": str(row["id"]),
                "scenario_id": scenario_id,
                "data_sha256": raw_sha,
                "size_bytes": len(raw.encode("utf-8")),
                "score": score_candidate(tasks, apps, events),
                "task_candidates": tasks,
                "apps": apps,
                "events": events,
            }
        )
        raw_by_id[scenario_id] = raw

    rows.sort(key=lambda item: (-item["score"], item["scenario_id"]))
    selected = rows[: args.top]
    dataset_content_sha256 = sha256_bytes(
        "\n".join(
            canonical_json(record)
            for record in sorted(row_digest_records, key=lambda item: (item["scenario_id"], item["id"]))
        ).encode("utf-8")
    )

    report = {
        "source_repo": DATASET_REPO,
        "gaia2_revision": revision,
        "config": args.config,
        "split": args.split,
        "row_count": len(rows),
        "dataset_content_sha256": dataset_content_sha256,
        "selection_method": "deterministic heuristic; manual inspection required",
        "candidates": selected,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for rank, candidate in enumerate(selected, start=1):
        scenario_id = candidate["scenario_id"]
        filename = f"candidate-{rank:02d}-{safe_name(scenario_id)}.json"
        (output_dir / filename).write_text(raw_by_id[scenario_id], encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
