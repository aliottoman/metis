"""The deterministic repairs that stop a shape mistake costing a step."""
from __future__ import annotations

import pytest

from waqil_api import tool_repair
from waqil_api.model_provider import ModelProviderError, drain_repairs, step_from_function_call


# ── Argument keys ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("tool", "sent", "expected"),
    [
        ("create_file", "filepath", "path"),
        ("create_file", "file_path", "path"),
        ("create_file", "filename", "path"),
        ("create_file", "File Path", "path"),
        ("create_file", "FILEPATH", "path"),
        ("create_file", "contents", "content"),
        ("apply_patch", "old_string", "original"),
        ("apply_patch", "new_string", "replacement"),
        ("search_code", "pattern", "query"),
        ("replace_lines", "start", "start_line"),
    ],
)
def test_a_spelling_the_roster_does_not_use_is_renamed(
    tool: str, sent: str, expected: str
) -> None:
    repaired, notes = tool_repair.repair_arguments(tool, {sent: "x"})
    assert expected in repaired
    assert repaired[expected] == "x"
    assert notes


def test_a_rename_only_happens_toward_a_key_this_tool_accepts() -> None:
    """`original` belongs to apply_patch, so create_file must not gain one.

    Renaming toward a key the tool does not take would turn a refusal the
    workspace explains into a call that looks valid and is not.
    """
    repaired, notes = tool_repair.repair_arguments("create_file", {"old_string": "x"})
    assert repaired == {"old_string": "x"}
    assert notes == []


def test_a_correct_call_is_left_completely_alone() -> None:
    original = {"path": "app/main.py", "content": "print(1)\n"}
    repaired, notes = tool_repair.repair_arguments("create_file", original)
    assert repaired == original
    assert notes == []


def test_an_ambiguous_key_is_never_guessed() -> None:
    """`patch` could be original or replacement, so it stays refused.

    A repair that is right most of the time is worse than a refusal that is
    right every time — the refusal is visible and a wrong repair is not.
    """
    repaired, notes = tool_repair.repair_arguments("apply_patch", {"path": "a.py", "patch": "..."})
    assert "patch" in repaired
    assert "original" not in repaired and "replacement" not in repaired
    assert notes == []


def test_a_repair_never_overwrites_a_correctly_named_value() -> None:
    repaired, _ = tool_repair.repair_arguments(
        "create_file", {"content": "right", "contents": "wrong"}
    )
    assert repaired["content"] == "right"


def test_an_unknown_key_is_kept_so_the_workspace_can_refuse_it_by_name() -> None:
    repaired, _ = tool_repair.repair_arguments("read_file", {"path": "a.py", "wibble": 1})
    assert repaired["wibble"] == 1


# ── Value types ────────────────────────────────────────────────────────────


def test_line_numbers_arriving_as_strings_are_coerced() -> None:
    repaired, notes = tool_repair.repair_arguments(
        "replace_lines",
        {"path": "a.py", "start_line": "12", "end_line": "18", "replacement": "x"},
    )
    assert repaired["start_line"] == 12
    assert repaired["end_line"] == 18
    assert any("start_line" in note for note in notes)


def test_a_boolean_spelled_as_a_string_is_coerced() -> None:
    repaired, _ = tool_repair.repair_arguments(
        "search_code", {"query": "x", "case_sensitive": "true"}
    )
    assert repaired["case_sensitive"] is True


def test_a_single_value_where_a_list_was_declared() -> None:
    repaired, _ = tool_repair.repair_arguments(
        "revise_plan", {"files": "app/main.py", "reason": "it exists already"}
    )
    assert repaired["files"] == ["app/main.py"]


def test_a_json_encoded_list_is_parsed() -> None:
    repaired, _ = tool_repair.repair_arguments(
        "revise_plan", {"files": '["a.py", "b.py"]', "reason": "r"}
    )
    assert repaired["files"] == ["a.py", "b.py"]


def test_a_number_where_a_string_was_declared() -> None:
    repaired, notes = tool_repair.repair_arguments("run_check", {"name": 3})
    assert repaired["name"] == "3"
    assert notes


# ── Paths ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        ("`app/main.py`", "app/main.py"),
        ('"app/main.py"', "app/main.py"),
        ("  app/main.py  ", "app/main.py"),
        ("./app/main.py", "app/main.py"),
        ("app\\static\\index.html", "app/static/index.html"),
        ("app//static//app.js", "app/static/app.js"),
    ],
)
def test_a_decorated_path_is_tidied(sent: str, expected: str) -> None:
    repaired, _ = tool_repair.repair_arguments("read_file", {"path": sent})
    assert repaired["path"] == expected


def test_an_absolute_path_is_left_for_the_workspace_to_refuse() -> None:
    """Escaping the grant is a permission question, never a shape one."""
    repaired, _ = tool_repair.repair_arguments("read_file", {"path": "/etc/passwd"})
    assert repaired["path"] == "/etc/passwd"


# ── JSON text ──────────────────────────────────────────────────────────────


def test_a_fenced_payload_is_unwrapped() -> None:
    text, notes = tool_repair.repair_json_text('```json\n{"path": "a.py"}\n```')
    assert text == '{"path": "a.py"}'
    assert notes


def test_prose_around_one_object_is_dropped() -> None:
    text, _ = tool_repair.repair_json_text('Here is the call: {"path": "a.py"} — done.')
    assert text == '{"path": "a.py"}'


def test_a_trailing_comma_is_removed() -> None:
    text, _ = tool_repair.repair_json_text('{"path": "a.py", "content": "x",}')
    assert text == '{"path": "a.py", "content": "x"}'


def test_python_literals_become_json_ones() -> None:
    text, _ = tool_repair.repair_json_text('{"query": "x", "case_sensitive": True}')
    assert text == '{"query": "x", "case_sensitive": true}'


def test_valid_json_is_returned_untouched() -> None:
    raw = '{"path": "a.py"}'
    text, notes = tool_repair.repair_json_text(raw)
    assert text == raw
    assert notes == []


def test_a_payload_holding_code_is_not_rewritten() -> None:
    """Quote rewriting would corrupt the very content being staged."""
    raw = '{"content": "d = {\'a\': True}"}'
    text, _ = tool_repair.repair_json_text(raw)
    assert "'a'" in text


# ── The decode chokepoint ──────────────────────────────────────────────────


def test_a_fenced_argument_string_now_decodes_instead_of_failing() -> None:
    drain_repairs()
    step = step_from_function_call(
        "read_file", '```json\n{"filepath": "./app/main.py"}\n```', speaker="a hosted model"
    )
    assert step.tool_call is not None
    assert step.tool_call.name == "read_file"
    assert step.tool_call.arguments == {"path": "app/main.py"}
    recorded = drain_repairs()
    assert recorded and recorded[0]["tool"] == "read_file"


def test_genuinely_unparseable_arguments_still_fail() -> None:
    with pytest.raises(ModelProviderError):
        step_from_function_call("read_file", "not json at all {{{", speaker="a hosted model")


def test_an_unsupported_tool_is_still_refused() -> None:
    """Repair renames and re-types. It never widens authority."""
    with pytest.raises(ModelProviderError):
        step_from_function_call("rm_rf", '{"path": "/"}', speaker="a hosted model")


def test_drain_returns_each_repair_once() -> None:
    drain_repairs()
    step_from_function_call("read_file", '{"filepath": "a.py"}', speaker="m")
    assert len(drain_repairs()) == 1
    assert drain_repairs() == []
