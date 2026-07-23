import hashlib
import unittest
from types import SimpleNamespace

from causal_orch.agent.schemas import EvidenceReport
from causal_orch.agent.worker import FreshReadOnlyWorker, TreatmentFailureReason
from causal_orch.runtime.budgets import (
    BudgetExceededError,
    BudgetUsage,
    WorkerBudgets,
)
from causal_orch.runtime.read_only_tools import (
    ToolAllowlistError,
    audit_tool,
    select_read_only_tools,
)
from causal_orch.runtime.state_guard import StateGuard, canonical_hash, canonical_json


def safe_tool(name="safe_read", **overrides):
    values = {
        "public_name": name,
        "app_name": "RecordsApp",
        "function_name": "read_record",
        "write_operation": False,
        "description": "Read a record.",
        "argument_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
    }
    values.update(overrides)
    return values


def artifact(objective="Find the record."):
    return {
        "artifact_type": "EVIDENCE_REPORT",
        "objective": objective,
        "status": "COMPLETE",
        "findings": [
            {"claim": "The record exists.", "evidence_refs": ["safe_read"], "confidence": "HIGH"}
        ],
        "uncertainties": [],
        "contradictions": [],
        "recommended_next_action": None,
    }


class WorkerReadOnlyTests(unittest.TestCase):
    def test_unknown_none_and_write_tools_are_rejected(self):
        tools = [
            safe_tool(),
            safe_tool("unknown_read", write_operation=False),
            safe_tool("unknown_metadata", write_operation=None),
            safe_tool("write_record", write_operation=True),
        ]
        with self.assertRaises(ToolAllowlistError):
            select_read_only_tools(tools, ["missing"], audited_allowlist={"safe_read"})
        with self.assertRaises(ToolAllowlistError):
            select_read_only_tools(tools, ["unknown_read"], audited_allowlist={"safe_read"})
        with self.assertRaises(ToolAllowlistError):
            select_read_only_tools(tools, ["unknown_metadata"], audited_allowlist={"unknown_metadata"})
        with self.assertRaises(ToolAllowlistError):
            select_read_only_tools(tools, ["write_record"], audited_allowlist={"write_record"})

    def test_exact_allowlist_and_exclusions(self):
        tools = [
            safe_tool(),
            safe_tool("AgentUserInterface__get_messages", app_name="AgentUserInterface"),
            safe_tool("SystemApp__wait", app_name="SystemApp", function_name="wait"),
            safe_tool("ReminderApp__get_all_reminders", app_name="ReminderApp"),
            safe_tool("MessagingApp__send_message", function_name="send_message"),
        ]
        selection = select_read_only_tools(tools, ["safe_read"], audited_allowlist={"safe_read"})
        self.assertEqual([tool["public_name"] for tool in selection.tools], ["safe_read"])
        reasons = {record.public_name: record.exclusion_reason for record in selection.manifests}
        self.assertEqual(reasons["AgentUserInterface__get_messages"], "excluded_agent_user_interface")
        self.assertEqual(reasons["SystemApp__wait"], "excluded_waiting_or_environment_control")
        self.assertEqual(reasons["ReminderApp__get_all_reminders"], "excluded_reminder_control")
        self.assertEqual(reasons["MessagingApp__send_message"], "excluded_user_communication")

    def test_manifest_contains_stable_description_and_schema_hashes(self):
        tool = safe_tool()
        record = audit_tool(tool, audited_allowlist={"safe_read"})
        self.assertTrue(record.allowlisted)
        self.assertEqual(record.write_operation, False)
        self.assertEqual(record.description_hash, hashlib.sha256(canonical_json(tool["description"]).encode()).hexdigest())
        self.assertEqual(
            record.argument_schema_hash,
            hashlib.sha256(canonical_json(tool["argument_schema"]).encode()).hexdigest(),
        )
        self.assertEqual(set(record.to_dict()), {
            "public_name", "app_name", "function_name", "write_operation", "allowlisted",
            "exclusion_reason", "description_hash", "argument_schema_hash",
        })

    def test_canonical_hash_is_order_and_identity_independent_and_ignores_logs(self):
        first = {"apps": {"b": {"value": 2}, "a": {"value": 1}}}
        second = {"apps": {"a": {"value": 1}, "b": {"value": 2}}, "logs": ["different"]}
        self.assertNotEqual(canonical_hash(first), canonical_hash(second))
        self.assertEqual(canonical_hash(first), canonical_hash({"apps": {"a": {"value": 1}, "b": {"value": 2}}}))
        self.assertEqual(canonical_hash(SimpleNamespace(value=1)), canonical_hash(SimpleNamespace(value=1)))

    def test_state_guard_detects_mutation_and_captures_after_on_exception(self):
        state = {"apps": {"record": 1}}
        guard = StateGuard(state)
        with self.assertRaises(RuntimeError):
            with guard:
                state["apps"]["record"] = 2
                raise RuntimeError("runner failed")
        self.assertTrue(guard.changed)
        self.assertIsNotNone(guard.after_hash)

    def test_budget_tracker_enforces_fixed_limits(self):
        usage = BudgetUsage(WorkerBudgets(max_steps=2, max_output_tokens=3))
        usage.consume_step()
        usage.consume_step()
        with self.assertRaises(BudgetExceededError) as error:
            usage.consume_step()
        self.assertEqual(error.exception.result.to_dict()["budget"], "max_steps")
        with self.assertRaises(BudgetExceededError):
            usage.consume_output_tokens(4)

    def test_worker_validates_artifact_and_serializes_only_runner_payload(self):
        calls = []
        worker = FreshReadOnlyWorker(
            environment={"apps": {"record": 1}},
            available_tools=(safe_tool(),),
            runner=lambda payload: calls.append(payload) or artifact(),
            audited_allowlist={"safe_read"},
        )
        result = worker.run(
            objective="Find the record.",
            context={"ref": "task"},
            requested_tools=["safe_read"],
            budgets=WorkerBudgets(2, 3),
        )
        self.assertTrue(result.succeeded)
        self.assertIsInstance(result.artifact, EvidenceReport)
        self.assertEqual(set(calls[0]), {"objective", "context", "tools", "budgets"})
        self.assertNotIn("environment", calls[0])
        self.assertNotIn("safe_tool", repr(calls[0]))

    def test_worker_classifies_malformed_timeout_budget_and_state_diff(self):
        base = {"apps": {"record": 1}}
        malformed = FreshReadOnlyWorker(
            environment=base, available_tools=(safe_tool(),), runner=lambda payload: {"bad": True}, audited_allowlist={"safe_read"}
        ).run(objective="Find the record.", context={}, requested_tools=["safe_read"], budgets={"max_steps": 1, "max_output_tokens": 1})
        self.assertEqual(malformed.failure.reason, TreatmentFailureReason.MALFORMED_ARTIFACT)

        timeout = FreshReadOnlyWorker(
            environment=base, available_tools=(safe_tool(),), runner=lambda payload: (_ for _ in ()).throw(TimeoutError("slow")), audited_allowlist={"safe_read"}
        ).run(objective="Find the record.", context={}, requested_tools=["safe_read"], budgets={"max_steps": 1, "max_output_tokens": 1})
        self.assertEqual(timeout.failure.reason, TreatmentFailureReason.TIMEOUT_OR_BUDGET_EXHAUSTION)

        budget = FreshReadOnlyWorker(
            environment=base, available_tools=(safe_tool(),), runner=lambda payload: {"status": "BUDGET_EXCEEDED"}, audited_allowlist={"safe_read"}
        ).run(objective="Find the record.", context={}, requested_tools=["safe_read"], budgets={"max_steps": 1, "max_output_tokens": 1})
        self.assertEqual(budget.failure.reason, TreatmentFailureReason.TIMEOUT_OR_BUDGET_EXHAUSTION)

        def mutate(_payload):
            base["apps"]["record"] = 2
            return artifact()

        changed = FreshReadOnlyWorker(
            environment=base, available_tools=(safe_tool(),), runner=mutate, audited_allowlist={"safe_read"}
        ).run(objective="Find the record.", context={}, requested_tools=["safe_read"], budgets={"max_steps": 1, "max_output_tokens": 1})
        self.assertEqual(changed.failure.reason, TreatmentFailureReason.STATE_DIFF_VIOLATION)


if __name__ == "__main__":
    unittest.main()
