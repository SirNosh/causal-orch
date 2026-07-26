"""Runtime primitives for concealed protocol assignment."""

from .randomization import (
    AssignmentSchedule,
    BlockAssignmentManifest,
    PreallocatedRunSchedule,
    TreatmentAssignment,
    generate_balanced_schedule,
)
from .context_registry import TraceContextRegistry
from .budgets import BudgetExceededError, BudgetExceededResult, BudgetTracker, BudgetUsage, WorkerBudgets
from .read_only_tools import (
    AUDITED_READ_ONLY_TOOL_NAMES,
    MANUALLY_AUDITED_ALLOWLIST,
    ReadOnlyToolSelection,
    ToolAllowlistError,
    ToolManifestRecord,
    audit_tool,
    select_read_only_tools,
)
from .state_guard import StateGuard, application_state_snapshot, canonical_hash, canonical_json

__all__ = [
    "AssignmentSchedule", "BlockAssignmentManifest", "PreallocatedRunSchedule",
    "TreatmentAssignment", "generate_balanced_schedule", "TraceContextRegistry",
    "BudgetExceededError", "BudgetExceededResult", "BudgetTracker", "BudgetUsage", "WorkerBudgets",
    "AUDITED_READ_ONLY_TOOL_NAMES", "MANUALLY_AUDITED_ALLOWLIST", "ReadOnlyToolSelection",
    "ToolAllowlistError", "ToolManifestRecord", "audit_tool", "select_read_only_tools",
    "StateGuard", "application_state_snapshot", "canonical_hash", "canonical_json",
]
