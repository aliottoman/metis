"""The two rungs that know things the project cannot say about itself.

`AsyncOpenAI(auth=...)` parses, imports, and serves its routes. It was invented
independently by a frontier model and a local one, survived every hand-written
gate, and needed a reference document written by hand to prevent. A type checker
resolving against the installed package rejects it in a second, and a lookup
tool lets the model find that out before writing the line.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from waqil_api.project_lookup import LookupError_, inspect_installed_api
from waqil_api.project_typecheck import staged_static_analysis

pytest.importorskip("openai", reason="the static rungs resolve against real packages")


def _staged(files: dict[str, str]) -> dict[str, dict[str, object]]:
    return {path: {"content": text} for path, text in files.items()}


# ── ruff + mypy over the changeset ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_keyword_argument_the_library_does_not_accept_is_an_error() -> None:
    """The defect this rung exists for. Nothing about the code is malformed —
    only the installed package knows the call cannot work."""
    staged = _staged(
        {
            "app/oci_client.py": (
                "from openai import AsyncOpenAI\n"
                "\n"
                "def build() -> AsyncOpenAI:\n"
                "    return AsyncOpenAI(base_url='https://x', auth=object())\n"
            )
        }
    )

    findings = await staged_static_analysis(staged)
    errors = [item for item in findings if item["severity"] == "error"]

    assert any("auth" in item["error"] for item in errors)
    assert errors[0]["path"] == "app/oci_client.py"


@pytest.mark.asyncio
async def test_correct_code_produces_no_errors() -> None:
    """A rung that flags correct code is noise, and noise is what makes a gate
    get clicked past."""
    staged = _staged(
        {
            "app/oci_client.py": (
                "import httpx\n"
                "from openai import AsyncOpenAI\n"
                "\n"
                "def build() -> AsyncOpenAI:\n"
                "    return AsyncOpenAI(\n"
                "        api_key='not-used',\n"
                "        base_url='https://x',\n"
                "        http_client=httpx.AsyncClient(),\n"
                "    )\n"
            )
        }
    )

    findings = await staged_static_analysis(staged)

    assert [item for item in findings if item["severity"] == "error"] == []


@pytest.mark.asyncio
async def test_an_undefined_name_is_an_error() -> None:
    """Ruff's F821, which the hand-written wiring gate only catches across
    files, not within one."""
    staged = _staged({"app/x.py": "def f() -> int:\n    return missing_name\n"})

    findings = await staged_static_analysis(staged)

    assert any(item["error"].startswith("F821") for item in findings)


@pytest.mark.asyncio
async def test_a_project_without_init_files_is_still_checked() -> None:
    """mypy aborts the whole tree with "Source file found twice" when a package
    has no __init__.py and is handed a directory — it reported a changeset clean
    whose phantom keyword argument it catches instantly when asked directly."""
    staged = _staged(
        {
            "app/config.py": "VALUE: int = 1\n",
            "app/client.py": (
                "from openai import AsyncOpenAI\n"
                "c = AsyncOpenAI(base_url='https://x', auth=1)\n"
            ),
        }
    )

    findings = await staged_static_analysis(staged)

    assert any("auth" in item["error"] for item in findings)


@pytest.mark.asyncio
async def test_style_opinions_never_reach_the_findings() -> None:
    """Only rules meaning "this cannot work as written" are reported. A checker's
    view on line length must never be able to withhold Approve."""
    staged = _staged({"app/x.py": "import os\nx=1\ny  =  2\n" + "z = 1  # " + "n" * 200})

    findings = await staged_static_analysis(staged)

    assert all(item["severity"] == "warning" for item in findings)


@pytest.mark.asyncio
async def test_an_empty_changeset_runs_nothing() -> None:
    assert await staged_static_analysis({}) == []


# ── inspect_api ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_module_reports_its_real_exports() -> None:
    """What a build should have been able to ask instead of guessing."""
    result = await inspect_installed_api("openai")

    assert result["importable"] is True
    assert "AsyncOpenAI" in result["exports"]


@pytest.mark.asyncio
async def test_a_symbol_reports_its_real_signature() -> None:
    """The answer to the question two models got wrong."""
    result = await inspect_installed_api("openai", "AsyncOpenAI")

    assert result["has_symbol"] is True
    assert "api_key" in result["signature"]
    assert "auth=" not in result["signature"]


@pytest.mark.asyncio
async def test_a_symbol_that_does_not_exist_says_so_and_lists_what_does() -> None:
    """A build invented `load_client_config`; the refusal has to point at the
    real names or the next guess is just as likely."""
    result = await inspect_installed_api("json", "load_client_config")

    assert result["has_symbol"] is False
    assert "does not define" in result["hint"]
    assert "loads" in result["exports"]


@pytest.mark.asyncio
async def test_a_project_module_is_refused(tmp_path: Path) -> None:
    """Importing runs top-level code. The project's own files are read_file's
    job — executing model-authored code in the host process is exactly what the
    sandbox exists to prevent."""
    project = tmp_path / "demo"
    (project / "app").mkdir(parents=True)
    (project / "app" / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(LookupError_, match="not an installed package"):
        await inspect_installed_api("app", project_roots=(project,))


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "not a module", "os; import sys", "../etc"])
async def test_a_name_that_is_not_a_module_is_refused(bad: str) -> None:
    with pytest.raises(LookupError_):
        await inspect_installed_api(bad)


@pytest.mark.asyncio
async def test_a_missing_package_names_the_remedy() -> None:
    """"Not installed" is a different problem from "wrong name", and the model
    can act on it: declare it, or pick something available."""
    with pytest.raises(LookupError_, match="not installed here"):
        await inspect_installed_api("definitely_not_a_real_package_xyz")
