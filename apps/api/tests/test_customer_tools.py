"""Integrity of the customer-agent tool catalog.

These are the static guarantees — unique, reachable, well-formed, and each
carrying an explicit boundary. The behavioural guarantee (a given phrasing
routes to exactly one of these) is proven against the live loop in the routing
matrix; this file makes sure the menu itself is sound before anything dispatches
from it.
"""

from waqil_api import customer_tools as ct


def test_every_declared_name_appears_exactly_once() -> None:
    """The exported names and the catalog are one set — no tool is declared but
    missing from the menu (unreachable), and none is in the menu without a name
    the loop and tests can refer to."""
    catalog_names = [tool["name"] for tool in ct.customer_tools()]
    assert catalog_names == list(ct.CUSTOMER_TOOL_NAMES)
    assert len(catalog_names) == len(set(catalog_names)), "duplicate tool name"


def test_write_tools_are_a_subset_of_the_catalog() -> None:
    """The mutate-the-record set can't name a tool that isn't offered."""
    assert ct.CUSTOMER_WRITE_TOOLS <= set(ct.CUSTOMER_TOOL_NAMES)


def test_each_tool_has_the_flat_responses_shape() -> None:
    """Same shape project_tools uses, so the same transports accept it."""
    for tool in ct.customer_tools():
        assert tool["type"] == "function"
        assert isinstance(tool["name"], str) and tool["name"]
        assert isinstance(tool["description"], str) and tool["description"]
        params = tool["parameters"]
        assert params["type"] == "object"
        assert isinstance(params.get("properties", {}), dict)
        # required must reference only declared properties — a required field the
        # model can't see would make the tool impossible to call correctly.
        required = params.get("required", [])
        assert set(required) <= set(params.get("properties", {}))
        assert params.get("additionalProperties") is False


def test_every_description_states_a_boundary() -> None:
    """The anti-overlap discipline: each tool says what it does NOT do, so the
    model has an explicit fence against reaching for the wrong one."""
    for tool in ct.customer_tools():
        assert "NOT" in tool["description"], f"{tool['name']} names no boundary"


def test_write_tools_never_take_an_evidence_body_from_the_model() -> None:
    """The DynaAI line at the schema level: a tool that writes to the record must
    not accept a free-text field the model could paraphrase the user's evidence
    into. file_note carries only a title; record_activity carries nothing; apply
    carries only an id. The note body and activity report come from the host."""
    by_name = {tool["name"]: tool for tool in ct.customer_tools()}

    note_props = set(by_name[ct.FILE_NOTE]["parameters"]["properties"])
    assert note_props == {"title"}, "file_note must not accept a body argument"

    activity_props = set(by_name[ct.RECORD_ACTIVITY]["parameters"]["properties"])
    assert activity_props == set(), "record_activity must read the message from the host"

    apply_props = set(by_name[ct.APPLY_EXTRACTION]["parameters"]["properties"])
    assert apply_props == {"proposal_id"}, "apply_extraction only names which proposal"

    win_props = set(by_name[ct.RECORD_WIN]["parameters"]["properties"])
    assert win_props == {"title"}, "record_win carries only a title, not the evidence"


def test_step_contract_names_stay_in_lockstep_with_the_catalog() -> None:
    """The link that makes reachability hold end to end: a catalog tool the step
    contract can't name is unroutable; a name the contract allows with no catalog
    entry dispatches to nothing. They must be exactly one set."""
    from typing import get_args

    from waqil_api.contracts import CustomerToolCallV1

    allowed = set(get_args(CustomerToolCallV1.model_fields["name"].annotation))
    assert allowed == set(ct.CUSTOMER_TOOL_NAMES)


def test_step_contract_uses_flat_typed_fields_not_a_free_dict() -> None:
    """The routing contract is flat and typed. An earlier free-form `arguments`
    dict made Cohere reject the whole decode (HALLUCINATED_ALL_TOOL_CALLS) and
    silently fall back to answering; the fix was to name the two metadata fields
    the actions actually use. The model sets only the relevant one."""
    from waqil_api.contracts import CustomerToolCallV1

    call = CustomerToolCallV1(name=ct.FILE_NOTE, title="Throttling finding")
    assert call.title == "Throttling finding"
    assert call.proposal_id == ""
    assert not hasattr(call, "arguments")


def test_an_empty_plan_is_valid_and_means_just_answer() -> None:
    """No actionable ask -> no calls -> the router falls back to answering."""
    from waqil_api.contracts import CustomerAgentStepV1

    assert CustomerAgentStepV1.model_validate({}).calls == []
