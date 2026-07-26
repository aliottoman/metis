"""P11 — the restricted executor + broker frame bridge for model-authored tools.

Runs real `run(inputs, model)` code in a locked-down subprocess and proves:
pure-compute works; model() calls bridge to the host; a tool can catch a rejected
model call and degrade; crashes/timeouts/bad output are contained; disallowed
source never reaches execution.
"""
from __future__ import annotations

import pytest

from waqil_api.authored_code import (
    AuthoredCodeError,
    AuthoredExecutionError,
    execute_authored,
)


def _run(body: str) -> str:
    return "def run(inputs, model):\n" + "\n".join("    " + l for l in body.splitlines())


@pytest.mark.asyncio
async def test_pure_compute_tool_runs() -> None:
    source = _run(
        "import re\n"
        "words = re.findall(r'[a-z]+', str(inputs.get('text','')).lower())\n"
        "return {'count': len(words), 'unique': len(set(words))}"
    )
    output = await execute_authored(source, {"text": "the cat sat on the mat"})
    assert output == {"count": 6, "unique": 5}


@pytest.mark.asyncio
async def test_tool_can_call_the_model_via_bridge() -> None:
    calls: list[dict] = []

    async def on_model_request(params):
        calls.append(params)
        return "SUMMARY: " + str(params.get("text", ""))[:10]

    source = _run("return {'summary': model({'text': inputs.get('text','')})}")
    output = await execute_authored(source, {"text": "hello world"}, on_model_request=on_model_request)
    assert output["summary"] == "SUMMARY: hello worl"
    assert calls == [{"text": "hello world"}]


@pytest.mark.asyncio
async def test_tool_with_no_model_access_gets_a_typed_error() -> None:
    # on_model_request=None → the bridge returns an error frame; an unhandled
    # model() call crashes the tool → contained AuthoredExecutionError.
    source = _run("return {'x': model({'q': 1})}")
    with pytest.raises(AuthoredExecutionError):
        await execute_authored(source, {}, on_model_request=None)


@pytest.mark.asyncio
async def test_tool_can_catch_a_rejected_model_call_and_degrade() -> None:
    async def boom(params):
        raise RuntimeError("broker budget exhausted")

    source = _run(
        "try:\n"
        "    return {'v': model({})}\n"
        "except Exception:\n"
        "    return {'v': 'fallback'}"
    )
    output = await execute_authored(source, {}, on_model_request=boom)
    assert output == {"v": "fallback"}


@pytest.mark.asyncio
async def test_infinite_loop_is_timed_out() -> None:
    source = _run("while True:\n    x = 1\nreturn {}")
    with pytest.raises(AuthoredExecutionError):
        await execute_authored(source, {}, timeout_seconds=2)


@pytest.mark.asyncio
async def test_non_object_output_is_rejected() -> None:
    source = _run("return [1, 2, 3]")
    with pytest.raises(AuthoredExecutionError, match="non-object"):
        await execute_authored(source, {})


@pytest.mark.asyncio
async def test_runtime_crash_is_contained() -> None:
    source = _run("return {'x': 1 // 0}")
    with pytest.raises(AuthoredExecutionError):
        await execute_authored(source, {})


@pytest.mark.asyncio
async def test_disallowed_source_never_executes() -> None:
    # Fails the AST profile before any subprocess is launched.
    with pytest.raises(AuthoredCodeError):
        await execute_authored(_run("import os\n    return {}"), {})


@pytest.mark.asyncio
async def test_allowed_import_works_through_the_safe_importer() -> None:
    source = _run(
        "import math\n"
        "from statistics import mean\n"
        "return {'f': math.factorial(4), 'm': mean([2, 4, 6])}"
    )
    output = await execute_authored(source, {})
    assert output == {"f": 24, "m": 4}
