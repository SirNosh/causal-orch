from causal_orch.actions import (
    DelegateAction,
    FinalAction,
    ToolAction,
    decode_action,
)


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
