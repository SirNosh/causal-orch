"""Typed actions produced through provider-native function calling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias


@dataclass(frozen=True)
class ToolAction:
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class DelegateAction:
    objective: str


@dataclass(frozen=True)
class FinalAction:
    answer: str


Action: TypeAlias = ToolAction | DelegateAction | FinalAction

DELEGATE_TOOL = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": "Delegate one focused research objective to a read-only worker.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "The specific fact-finding objective.",
                }
            },
            "required": ["objective"],
            "additionalProperties": False,
        },
    },
}

FINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Finish the task with the answer to send to the user.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


def decode_action(
    name: str,
    arguments: Mapping[str, Any],
    ordinary_tool_names: set[str] | frozenset[str],
) -> Action:
    """Convert one native function call into the experiment's action union."""

    if name == "delegate":
        if set(arguments) != {"objective"}:
            raise ValueError("delegate requires exactly objective")
        objective = arguments.get("objective")
        if not isinstance(objective, str):
            raise ValueError("delegate.objective must be a string")
        return DelegateAction(objective=objective)
    if name == "final_answer":
        if set(arguments) != {"answer"}:
            raise ValueError("final_answer requires exactly answer")
        answer = arguments.get("answer")
        if not isinstance(answer, str):
            raise ValueError("final_answer.answer must be a string")
        return FinalAction(answer=answer)
    if name not in ordinary_tool_names:
        raise ValueError(f"unknown action tool: {name}")
    return ToolAction(tool=name, arguments=dict(arguments))
