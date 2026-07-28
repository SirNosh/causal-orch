from causal_orch.actions import (
    DelegateAction,
    FinalAction,
    ToolAction,
    decode_action,
)
import pytest


def test_all_typed_actions_decode_from_native_calls():
    names = {"Emails__list_emails"}

    assert decode_action(
        "Emails__list_emails", {"page": 0}, names
    ) == ToolAction("Emails__list_emails", {"page": 0})
    assert decode_action("delegate", {"objective": "Find the policy"}, names) == (
        DelegateAction("Find the policy")
    )
    assert decode_action("final_answer", {"answer": "44"}, names) == (
        FinalAction("44")
    )


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("delegate", {"objective": "research", "answer": "44"}),
        ("final_answer", {"answer": "44", "objective": "research"}),
        ("delegate", {}),
    ],
)
def test_malformed_or_ambiguous_action_is_rejected(name, arguments):
    with pytest.raises(ValueError):
        decode_action(name, arguments, {"lookup"})
