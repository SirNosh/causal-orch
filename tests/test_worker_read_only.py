import hashlib
import json
import unittest
from types import SimpleNamespace

from causal_orch.agent.schemas import (
    EvidenceReport,
    ValidationError,
    diagnose_evidence_report,
)
from causal_orch.agent.worker import (
    BaseAgentWorkerFactory,
    DelegationWorkerAdapter,
    FreshReadOnlyWorker,
    NativeTypedWorkerRunner,
    PlainTextWorkerRunner,
    ReturnArtifactTool,
    TreatmentFailureReason,
    WorkerEpisode,
    native_return_artifact_tool_schema,
)
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
from causal_orch.runtime.state_guard import StateEventRecord
from causal_orch.tracing.events import EventName
from causal_orch.tracing.sink import InMemoryTraceSink
from are.simulation.types import SimulatedGenerationTimeConfig


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


def artifact(objective="Find the record.", evidence_ref="task"):
    return {
        "artifact_type": "EVIDENCE_REPORT",
        "objective": objective,
        "status": "COMPLETE",
        "findings": [
            {"claim": "The record exists.", "evidence_refs": [evidence_ref], "confidence": "HIGH"}
        ],
        "uncertainties": [],
        "contradictions": [],
        "recommended_next_action": None,
    }


class WorkerReadOnlyTests(unittest.TestCase):
    def test_delegation_adapter_is_json_safe_and_fake_worker_executes_read_tool(self):
        calls = []

        class ReadTool:
            public_name = "FileSystem__read_file"
            app_name = "FileSystem"
            function_name = "read_file"
            write_operation = False
            description = "Read a file."
            argument_schema = {"type": "object"}

            def __call__(self):
                calls.append("read")
                return "record-42"

        proposal = {
            "objective": "Find the record.",
            "reason_code": "INFORMATION_GAP",
            "context_refs": ["task"],
            "allowed_read_tools": ["FileSystem__read_file"],
            "completion_criterion": "Name the record and cite the selected context.",
        }

        def fake_worker(payload, tools):
            self.assertEqual(payload["context"], {"task": {"case": "42"}})
            self.assertEqual(
                payload["budgets"],
                {"max_steps": 8, "max_output_tokens": 2000},
            )
            self.assertIn("completion criterion", payload["prompt"].lower())
            self.assertNotIn("ReadTool", repr(payload))
            self.assertEqual(tools[0](), "record-42")
            return artifact("Find the record.")

        result = DelegationWorkerAdapter(
            environment={"apps": {}},
            available_tools=lambda: [ReadTool()],
            context_resolver=lambda ref: {"case": "42"},
            worker_runner=fake_worker,
        )(proposal)

        self.assertEqual(result["status"], "WORKER_COMPLETED")
        self.assertEqual(result["artifact"]["objective"], "Find the record.")
        self.assertEqual(calls, ["read"])
        self.assertTrue(json.dumps(result))
        self.assertIsInstance(result["payload"]["proposal_id"], str)

    def test_return_artifact_tool_validates_and_base_agent_factory_is_available(self):
        returned = []
        worker_logs = []
        tool = ReturnArtifactTool(returned.append)
        validated = tool(artifact())
        self.assertEqual(validated["artifact_type"], "EVIDENCE_REPORT")
        self.assertEqual(returned[0].objective, "Find the record.")

        factory = BaseAgentWorkerFactory(
            lambda *_args, **_kwargs: "unused",
            log_callback=worker_logs.append,
        )
        agent = factory(
            {
                "objective": "Find the record.",
                "completion_criterion": "cite it",
                "context": {},
                "permitted_tools": [],
                "evidence_refs": [],
                "budgets": {"max_steps": 1, "max_output_tokens": 1},
                "prompt": "worker prompt",
            },
            (),
        )
        self.assertEqual(agent.action_executor.tools["return_artifact"].name, "return_artifact")
        self.assertEqual(agent.log_callback, worker_logs.append)

    def test_native_artifact_schema_and_tool_result_replay_use_canonical_contract(self):
        schema = native_return_artifact_tool_schema()["function"]["parameters"]
        expected = {
            "artifact_type",
            "objective",
            "status",
            "findings",
            "uncertainties",
            "contradictions",
            "recommended_next_action",
        }
        self.assertEqual(set(schema["properties"]), expected)
        self.assertEqual(set(schema["required"]), expected)
        self.assertFalse(schema["additionalProperties"])
        finding = schema["properties"]["findings"]["items"]
        self.assertEqual(
            set(finding["required"]),
            {"claim", "evidence_refs", "confidence"},
        )
        self.assertFalse(finding["additionalProperties"])
        self.assertEqual(
            schema["properties"]["artifact_type"]["enum"],
            ["EVIDENCE_REPORT"],
        )

        class ReadTool:
            name = "FileSystem__read_file"
            description = "Read a record."
            inputs = {"path": {"type": "string", "description": "Path"}}

            def __call__(self, **arguments):
                self.arguments = arguments
                return {"value": "record-42", "evidence_ref": "task"}

        class NativeEngine:
            def __init__(self):
                self.calls = []
                self.native_exchanges = []

            def native_tool_completion(self, messages, *, tools, **kwargs):
                self.calls.append((list(messages), tools, kwargs))
                index = len(self.native_exchanges)
                self.native_exchanges.append({})
                if index == 0:
                    call = {
                        "id": "read-call",
                        "type": "function",
                        "function": {
                            "name": "FileSystem__read_file",
                            "arguments": {"path": "record"},
                        },
                    }
                elif index == 1:
                    self.test_case.assertEqual(messages[-2]["role"], "assistant")
                    self.test_case.assertEqual(messages[-1]["role"], "tool")
                    self.test_case.assertEqual(
                        messages[-1]["tool_call_id"], "read-call"
                    )
                    return (
                        {
                            "role": "assistant",
                            "content": "The record exists.",
                            "tool_calls": [],
                        },
                        {
                            "completion_tokens": 50,
                            "native_exchange_index": index,
                        },
                    )
                else:
                    self.test_case.assertEqual(messages[-1]["role"], "user")
                    self.test_case.assertEqual(
                        kwargs["tool_choice"]["function"]["name"],
                        "return_artifact",
                    )
                    self.test_case.assertEqual(kwargs["max_tokens"], 1900)
                    self.test_case.assertEqual(
                        [tool["function"]["name"] for tool in tools],
                        ["return_artifact"],
                    )
                    call = {
                        "id": "artifact-call",
                        "type": "function",
                        "function": {
                            "name": "return_artifact",
                            "arguments": artifact(),
                        },
                    }
                return (
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    {
                        "completion_tokens": 50,
                        "native_exchange_index": index,
                    },
                )

        engine = NativeEngine()
        engine.test_case = self
        read_tool = ReadTool()
        sink = InMemoryTraceSink()
        runner = NativeTypedWorkerRunner(engine, trace_sink=sink)
        result = runner(
            {
                "objective": "Find the record.",
                "completion_criterion": "Name the record.",
                "context": {"task": "Find the record."},
                "permitted_tools": [],
                "evidence_refs": ["task"],
                "budgets": {"max_steps": 8, "max_output_tokens": 2000},
                "worker_id": "worker-1",
            },
            (read_tool,),
        )

        self.assertIsInstance(result, EvidenceReport)
        self.assertEqual(read_tool.arguments, {"path": "record"})
        self.assertEqual(
            [tool["function"]["name"] for tool in engine.calls[0][1]],
            ["FileSystem__read_file", "return_artifact"],
        )
        self.assertEqual(engine.calls[0][2]["tool_choice"], "auto")
        self.assertEqual(engine.calls[1][2]["tool_choice"], "auto")
        self.assertEqual(
            engine.native_exchanges[2]["validator_result"]["accepted"], True
        )
        self.assertEqual(
            engine.native_exchanges[1]["output_rejection"],
            "PROTOCOL_VIOLATION_PLAIN_TEXT",
        )
        self.assertEqual(sink.events[-1].event_type, EventName.ARTIFACT_VALIDATION)

    def test_native_typed_worker_fails_plain_text_without_repair(self):
        class NativeEngine:
            def __init__(self):
                self.calls = 0
                self.native_exchanges = []

            def native_tool_completion(self, messages, *, tools, **kwargs):
                self.calls += 1
                self.native_exchanges.append({})
                return (
                    {
                        "role": "assistant",
                        "content": "Plain text is not a worker action.",
                        "tool_calls": [],
                    },
                    {
                        "completion_tokens": 10,
                        "native_exchange_index": 0,
                    },
                )

        engine = NativeEngine()
        sink = InMemoryTraceSink()
        runner = NativeTypedWorkerRunner(engine, trace_sink=sink)

        with self.assertRaisesRegex(
            ValidationError,
            "must emit exactly one tool call",
        ):
            runner(
                {
                    "objective": "Find the record.",
                    "completion_criterion": "Name the record.",
                    "context": {"task": "Find the record."},
                    "permitted_tools": [],
                    "evidence_refs": ["task"],
                    "budgets": {"max_steps": 8, "max_output_tokens": 2000},
                    "worker_id": "worker-1",
                },
                (),
            )

        self.assertEqual(engine.calls, 1)
        self.assertEqual(
            sink.events[-1].error_type,
            "PROTOCOL_VIOLATION_NO_TOOL_CALL",
        )

    def test_plain_text_worker_calls_tool_then_returns_harness_episode(self):
        class ReadTool:
            name = "FileSystem__read_file"
            description = "Read a record."
            inputs = {"path": {"type": "string", "description": "Path"}}

            def __call__(self, **arguments):
                self.arguments = arguments
                return {
                    "value": "record-42",
                    "evidence_ref": "worker_tool_result:event-42",
                }

        class NativeEngine:
            def __init__(self):
                self.calls = []

            def native_tool_completion(self, messages, *, tools, **kwargs):
                self.calls.append((list(messages), tools, kwargs))
                if len(self.calls) == 1:
                    return (
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "read-call",
                                    "type": "function",
                                    "function": {
                                        "name": "FileSystem__read_file",
                                        "arguments": {"path": "record"},
                                    },
                                }
                            ],
                        },
                        {"completion_tokens": 20, "completion_duration": 0.25},
                    )
                self.test_case.assertEqual(messages[-1]["role"], "tool")
                self.test_case.assertEqual(
                    messages[-1]["tool_call_id"], "read-call"
                )
                return (
                    {
                        "role": "assistant",
                        "content": "The record is record-42.",
                        "tool_calls": [],
                    },
                    {"completion_tokens": 30, "completion_duration": 0.5},
                )

        engine = NativeEngine()
        engine.test_case = self
        read_tool = ReadTool()
        result = PlainTextWorkerRunner(engine)(
            {
                "objective": "Find the record.",
                "completion_criterion": "Name the record.",
                "context": {"task": "Find the record."},
                "permitted_tools": [],
                "budgets": {"max_steps": 8, "max_output_tokens": 2000},
            },
            (read_tool,),
        )

        self.assertEqual(read_tool.arguments, {"path": "record"})
        self.assertEqual(result.generated_tokens, 50)
        self.assertEqual(result.wall_seconds, 0.75)
        self.assertEqual(result.tool_events[0]["tool_call_id"], "read-call")
        self.assertEqual(
            result.tool_events[0]["result_ref"],
            "worker_tool_result:event-42",
        )
        self.assertEqual(
            engine.calls[0][2]["interface_label"],
            "PLAIN_TEXT_WORKER_EPISODE",
        )
        self.assertNotIn("return_artifact", str(engine.calls[0][1]))

    def test_worker_adapter_accepts_plain_text_episode_without_artifact_validation(self):
        sink = InMemoryTraceSink()
        episode = WorkerEpisode(
            objective="Find the record.",
            final_text="The record is record-42.",
            tool_events=(
                {
                    "tool_name": "FileSystem__read_file",
                    "tool_call_id": "read-call",
                    "result_ref": "worker_tool_result:event-42",
                },
            ),
            generated_tokens=50,
            wall_seconds=0.75,
        )
        result = DelegationWorkerAdapter(
            environment={"apps": {}},
            available_tools=(),
            context_resolver={"task": "Find the record."},
            worker_runner=lambda _payload, _tools: episode,
            trace_sink=sink,
        )(
            {
                "objective": "Find the record.",
                "reason_code": "INFORMATION_GAP",
                "context_refs": ["task"],
                "allowed_read_tools": [],
                "completion_criterion": "Name the record.",
            }
        )

        self.assertEqual(result["status"], "WORKER_COMPLETED")
        self.assertEqual(result["episode"]["final_text"], "The record is record-42.")
        self.assertNotIn("artifact", result)
        self.assertEqual(
            [event.event_type for event in sink.events],
            [
                EventName.WORKER_STARTED,
                EventName.WORKER_STATE_GUARD,
                EventName.WORKER_EPISODE,
                EventName.WORKER_COMPLETED,
            ],
        )

    def test_default_base_agent_worker_emits_events_invokes_read_tool_and_pauses_per_generation(self):
        calls = []
        pause_calls = []
        resume_calls = []
        sink = InMemoryTraceSink()

        class ReadTool:
            public_name = "FileSystem__read_file"
            app_name = "FileSystem"
            function_name = "read_file"
            write_operation = False
            description = "Read a file."
            argument_schema = {"type": "object"}

            def __call__(self):
                calls.append("read")
                return "record-42"

        outputs = iter(
            [
                (
                    'Thought: inspect\nAction: {"action":"FileSystem__read_file","action_input":{}}',
                    {"completion_tokens": 2},
                ),
                (
                    "Thought: report\nAction: "
                    + json.dumps({"action": "return_artifact", "action_input": {"artifact": artifact()}}),
                    {"completion_tokens": 3},
                ),
            ]
        )

        def engine(*_args, **_kwargs):
            return next(outputs)

        proposal = {
            "objective": "Find the record.",
            "reason_code": "INFORMATION_GAP",
            "context_refs": ["task"],
            "allowed_read_tools": ["FileSystem__read_file"],
            "completion_criterion": "Name the record.",
        }
        result = DelegationWorkerAdapter(
            environment={"apps": {}},
            available_tools=lambda: [ReadTool()],
            context_resolver={"task": {"case": "42"}},
            llm_engine=engine,
            trace_sink=sink,
            pause_env=lambda: pause_calls.append("pause"),
            resume_env=lambda offset: resume_calls.append(offset),
            simulated_generation_time_config=SimulatedGenerationTimeConfig(mode="fixed", seconds=5.0),
        )(proposal)

        self.assertEqual(result["status"], "WORKER_COMPLETED")
        self.assertEqual(calls, ["read"])
        self.assertEqual(len(pause_calls), 2)
        self.assertEqual(resume_calls, [5.0, 0.0, 5.0, 0.0])
        event_names = [event.event_type for event in sink.events]
        self.assertEqual(
            event_names,
            [
                EventName.WORKER_STARTED,
                EventName.WORKER_TOOL_CALL,
                EventName.WORKER_TOOL_RESULT,
                EventName.ARTIFACT_VALIDATION,
                EventName.WORKER_STATE_GUARD,
                EventName.WORKER_ARTIFACT,
                EventName.WORKER_COMPLETED,
            ],
        )
        self.assertTrue(all(event.previous_event_id for event in sink.events[1:]))

    def test_artifact_rejection_diagnostics_do_not_change_validation(self):
        value = {
            "artifact_type": "EvidenceReport",
            "objective": "wrong",
            "status": "completed",
            "findings": [
                {
                    "claim": "Found it.",
                    "confidence": 1.0,
                    "summary": "extra",
                }
            ],
            "uncertainties": [],
            "contradictions": [],
            "extra": True,
        }
        categories = set(
            diagnose_evidence_report(
                value,
                objective="Find the record.",
                allowed_evidence_refs={"task"},
            )
        )
        self.assertTrue(
            {
                "MISSING_REQUIRED_FIELD",
                "UNKNOWN_FIELD",
                "INVALID_ENUM_VALUE",
                "INVALID_FINDING_STRUCTURE",
                "INVALID_CONFIDENCE",
                "INVALID_EVIDENCE_REFERENCE",
                "OBJECTIVE_MISMATCH",
            }.issubset(categories)
        )
        with self.assertRaises(ValueError):
            EvidenceReport.from_dict(value)

    def test_default_base_agent_worker_budget_exhaustion_is_not_malformed(self):
        proposal = {
            "objective": "Find the record.",
            "reason_code": "INFORMATION_GAP",
            "context_refs": [],
            "allowed_read_tools": [],
            "completion_criterion": "Name the record.",
        }

        def over_budget_engine(*_args, **_kwargs):
            return (
                'Thought: too large\nAction: {"action":"return_artifact","action_input":{}}',
                {"completion_tokens": 3},
            )

        result = DelegationWorkerAdapter(
            environment={"apps": {}},
            available_tools=[],
            context_resolver={},
            llm_engine=over_budget_engine,
        )(proposal)
        self.assertEqual(
            result["failure"]["reason"],
            TreatmentFailureReason.TIMEOUT_OR_BUDGET_EXHAUSTION.value,
        )

    def test_environment_event_during_worker_is_logged_but_not_a_worker_violation(self):
        class Environment:
            def __init__(self):
                self.apps = {"record": 1}
                self.records = []

            def get_apps_state(self):
                return self.apps

            @property
            def state_records(self):
                return tuple(self.records)

        environment = Environment()

        def eventful_worker(_payload, _tools):
            before = canonical_hash(environment.apps)
            environment.apps["record"] = 2
            environment.records.append(
                StateEventRecord(
                    event_id="scheduled",
                    event_type="ScheduledEvent",
                    actor_role="environment",
                    provenance="environment",
                    state_before_hash=before,
                    state_after_hash=canonical_hash(environment.apps),
                )
            )
            return artifact()

        result = DelegationWorkerAdapter(
            environment=environment,
            available_tools=(),
            context_resolver={"task": {"case": "42"}},
            worker_runner=eventful_worker,
            trace_sink=InMemoryTraceSink(),
        )(
            {
                "objective": "Find the record.",
                "reason_code": "INFORMATION_GAP",
                "context_refs": ["task"],
                "allowed_read_tools": [],
                "completion_criterion": "Name the record.",
            }
        )
        self.assertEqual(result["status"], "WORKER_COMPLETED")
        self.assertTrue(result["state_guard"]["changed"])

    def test_worker_tool_result_handle_is_returned_and_accepted_as_evidence(self):
        class ReadTool:
            public_name = "FileSystem__read_file"
            app_name = "FileSystem"
            function_name = "read_file"
            write_operation = False
            description = "Read a file."
            argument_schema = {"type": "object"}

            def __call__(self):
                return "record-42"

        def worker(payload, tools):
            observation = tools[0]()
            self.assertEqual(observation["value"], "record-42")
            self.assertTrue(
                observation["evidence_ref"].startswith("worker_tool_result:")
            )
            return artifact(
                payload["objective"], evidence_ref=observation["evidence_ref"]
            )

        result = DelegationWorkerAdapter(
            environment={"apps": {}},
            available_tools=(ReadTool(),),
            context_resolver={"task": "Find the record"},
            worker_runner=worker,
            trace_sink=InMemoryTraceSink(),
        )(
            {
                "objective": "Find the record.",
                "reason_code": "INFORMATION_GAP",
                "context_refs": ["task"],
                "allowed_read_tools": ["FileSystem__read_file"],
                "completion_criterion": "Name the record.",
            }
        )
        self.assertEqual(result["status"], "WORKER_COMPLETED")
        cited = result["artifact"]["findings"][0]["evidence_refs"][0]
        self.assertTrue(cited.startswith("worker_tool_result:"))

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
        self.assertTrue(changed.succeeded)
        self.assertTrue(changed.state_guard.changed)


if __name__ == "__main__":
    unittest.main()
