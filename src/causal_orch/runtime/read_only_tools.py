"""Small, explicit audit and selection helpers for worker read tools."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from .state_guard import canonical_json


# This is deliberately exact-name, finite, and boring. Names can be expanded only
# by a reviewed protocol change after the pinned ARE source has been audited.
MANUALLY_AUDITED_ALLOWLIST = frozenset(
    {
        "EmailClient__search_emails",
        "EmailClientApp__search_emails",
        "FileSystem__read_file",
        "SandboxLocalFileSystem__read_document",
    }
)
AUDITED_READ_ONLY_TOOL_NAMES = MANUALLY_AUDITED_ALLOWLIST


class ToolAllowlistError(ValueError):
    """Raised when a requested worker tool is not safe and exactly allowlisted."""


def _get(record: Any, *names: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return record[name]
        return default
    for name in names:
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _callable_name(value: Any) -> str | None:
    if callable(value):
        return getattr(value, "__name__", None)
    return value if isinstance(value, str) else None


def _argument_schema(tool: Any) -> Any:
    direct = _get(tool, "argument_schema", "args_schema", "schema", default=None)
    if direct is not None:
        return direct
    inputs = _get(tool, "inputs", default=None)
    if inputs is not None:
        return inputs
    args = _get(tool, "args", default=None)
    if args is not None:
        result = []
        for arg in args:
            result.append(
                {
                    "name": _get(arg, "name", "arg_name"),
                    "type": _get(arg, "type", "arg_type"),
                    "description": _get(arg, "description"),
                    "has_default": _get(arg, "has_default", default=False),
                    "default": _get(arg, "default"),
                }
            )
        return result
    to_open_ai = getattr(tool, "to_open_ai", None)
    if callable(to_open_ai):
        open_ai = to_open_ai()
        return open_ai.get("function", {}).get("parameters", {})
    return {}


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _exclusion_reason(public_name: str, app_name: str, function_name: str, write_operation: Any) -> str | None:
    combined = " ".join((public_name, app_name, function_name)).lower()
    if "agentuserinterface" in combined:
        return "excluded_agent_user_interface"
    if (
        "systemapp" in combined
        or "environment" in combined
        or any(marker in function_name.lower() for marker in ("wait", "pause", "resume", "advance", "sleep"))
    ):
        return "excluded_waiting_or_environment_control"
    if "reminder" in combined:
        return "excluded_reminder_control"
    if any(marker in combined for marker in ("send", "reply", "message_to_user", "user_communication", "communicat")):
        return "excluded_user_communication"
    if write_operation is not False:
        return "write_operation_unknown_or_unsafe"
    return None


@dataclasses.dataclass(frozen=True)
class ToolManifestRecord:
    public_name: str
    app_name: str
    function_name: str
    write_operation: bool | None
    allowlisted: bool
    exclusion_reason: str | None
    description_hash: str
    argument_schema_hash: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ReadOnlyToolSelection:
    tools: tuple[Any, ...]
    manifests: tuple[ToolManifestRecord, ...]
    selected_manifests: tuple[ToolManifestRecord, ...]

    @property
    def serialized_tools(self) -> tuple[dict[str, Any], ...]:
        return tuple(manifest.to_dict() for manifest in self.selected_manifests)


def audit_tool(tool: Any, *, audited_allowlist: Iterable[str] = MANUALLY_AUDITED_ALLOWLIST) -> ToolManifestRecord:
    public_name = _get(tool, "public_name", "_public_name", "name")
    if not isinstance(public_name, str) or not public_name:
        raise ToolAllowlistError("tool has no public name")
    app_name = _get(tool, "app_name", default="")
    app_name = app_name if isinstance(app_name, str) else str(app_name)
    function = _get(tool, "function_name", "func_name", "function", default="")
    function_name = _callable_name(function) or (function if isinstance(function, str) else "")
    allowlist = frozenset(audited_allowlist)
    write_operation = _get(tool, "write_operation", default=None)
    description = _get(tool, "description", "_public_description", "function_description", default="")
    if description is None:
        description = ""
    if not isinstance(description, str):
        description = str(description)
    allowlisted = public_name in allowlist
    reason = _exclusion_reason(public_name, app_name, function_name, write_operation)
    if reason is None and not allowlisted:
        reason = "not_in_manual_allowlist"
    return ToolManifestRecord(
        public_name=public_name,
        app_name=app_name,
        function_name=function_name,
        write_operation=write_operation if isinstance(write_operation, bool) else None,
        allowlisted=allowlisted,
        exclusion_reason=reason,
        description_hash=_hash(description),
        argument_schema_hash=_hash(_argument_schema(tool)),
    )


def select_read_only_tools(
    tools: Iterable[Any],
    requested_names: Iterable[str],
    *,
    audited_allowlist: Iterable[str] = MANUALLY_AUDITED_ALLOWLIST,
) -> ReadOnlyToolSelection:
    candidates = tuple(tools)
    requested = tuple(requested_names)
    if any(not isinstance(name, str) or not name for name in requested):
        raise ToolAllowlistError("requested tool names must be non-empty strings")
    if len(set(requested)) != len(requested):
        raise ToolAllowlistError("requested tool names must be unique")
    manifests = tuple(audit_tool(tool, audited_allowlist=audited_allowlist) for tool in candidates)
    by_name: dict[str, tuple[Any, ToolManifestRecord]] = {}
    for tool, manifest in zip(candidates, manifests):
        if manifest.public_name in by_name:
            raise ToolAllowlistError(f"duplicate tool public name: {manifest.public_name}")
        by_name[manifest.public_name] = (tool, manifest)
    selected: list[Any] = []
    selected_manifests: list[ToolManifestRecord] = []
    for name in requested:
        pair = by_name.get(name)
        if pair is None:
            raise ToolAllowlistError(f"requested tool is unknown: {name}")
        tool, manifest = pair
        if not manifest.allowlisted or manifest.exclusion_reason is not None:
            reason = manifest.exclusion_reason or "not_in_manual_allowlist"
            raise ToolAllowlistError(f"requested tool rejected: {name} ({reason})")
        selected.append(tool)
        selected_manifests.append(manifest)
    return ReadOnlyToolSelection(tuple(selected), manifests, tuple(selected_manifests))


def validate_read_only_tools(*args: Any, **kwargs: Any) -> ReadOnlyToolSelection:
    """Compatibility spelling for the explicit selection operation."""
    return select_read_only_tools(*args, **kwargs)
