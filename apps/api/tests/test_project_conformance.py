"""The rung that asks whether the changeset is the one the turn committed to.

Every case here is a defect taken from a measured build: six builds of the same
request, three models, and each one produced a frontend posting a JSON body to a
handler that reads the query string, while every other rung called it clean.
"""

from __future__ import annotations

from waqil_api.config import Settings
from waqil_api.model_provider import OCIResponsesModelProvider
from waqil_api.project_conformance import (
    json_body_requests,
    native_form_requests,
    staged_conformance_errors,
)


def _staged(**files: str) -> dict[str, dict[str, object]]:
    return {
        path.replace("__", "/").replace("_dot_", "."): {"content": text}
        for path, text in files.items()
    }


# ── The 422 every build shipped ────────────────────────────────────────────


def test_a_json_body_posted_to_a_scalar_handler_is_an_error() -> None:
    """The defect all three models produced. FastAPI reads a bare `str` from the
    query string, so every request from the frontend fails with 422 while the
    project still imports and serves its routes perfectly."""
    staged = _staged(
        app__main_dot_py=(
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "\n"
            "@app.post('/api/jobs')\n"
            "async def submit(prompt: str):\n"
            "    return {'ok': True}\n"
        ),
        app__static__app_dot_js=(
            "await fetch('/api/jobs', {\n"
            "  method: 'POST',\n"
            "  headers: {'Content-Type': 'application/json'},\n"
            "  body: JSON.stringify({ prompt })\n"
            "});\n"
        ),
    )

    errors = staged_conformance_errors(staged)

    assert len(errors) == 1
    assert errors[0]["path"] == "app/main.py"
    assert "422" in errors[0]["error"]
    assert errors[0]["severity"] == "error"


def test_a_handler_taking_a_model_is_accepted() -> None:
    """The correct shape must not be flagged, or the gate is noise."""
    staged = _staged(
        app__main_dot_py=(
            "from fastapi import FastAPI\n"
            "from pydantic import BaseModel\n"
            "app = FastAPI()\n"
            "\n"
            "class JobRequest(BaseModel):\n"
            "    prompt: str\n"
            "\n"
            "@app.post('/api/jobs')\n"
            "async def submit(body: JobRequest):\n"
            "    return {'ok': True}\n"
        ),
        app__static__app_dot_js=(
            "fetch('/api/jobs', {method:'POST', body: JSON.stringify({prompt})});"
        ),
    )

    assert staged_conformance_errors(staged) == []


def test_a_native_form_posted_to_a_json_model_is_an_error() -> None:
    staged = _staged(
        app__main_dot_py=(
            "from fastapi import FastAPI\n"
            "from pydantic import BaseModel\n"
            "app = FastAPI()\n"
            "class Submission(BaseModel):\n"
            "    message: str\n"
            "@app.post('/submit')\n"
            "async def submit(payload: Submission):\n"
            "    return {'echo': payload.message}\n"
        ),
        app__static__index_dot_html=(
            '<form method="post" action="/submit">'
            '<input name="message"><button>Send</button></form>'
        ),
    )

    errors = staged_conformance_errors(staged)

    assert len(errors) == 1
    assert errors[0]["path"] == "app/main.py"
    assert "native form data" in errors[0]["error"]
    assert "422" in errors[0]["error"]


def test_a_native_form_handler_using_form_is_accepted() -> None:
    staged = _staged(
        app__main_dot_py=(
            "from fastapi import FastAPI, Form\n"
            "app = FastAPI()\n"
            "@app.post('/submit')\n"
            "async def submit(message: str = Form(...)):\n"
            "    return {'echo': message}\n"
        ),
        app__static__index_dot_html=(
            "<FORM METHOD='POST' ACTION='/submit?source=browser'>"
            "<input name='message'></FORM>"
        ),
    )

    assert staged_conformance_errors(staged) == []


def test_a_native_form_handler_using_annotated_form_is_accepted() -> None:
    staged = _staged(
        app__main_dot_py=(
            "from typing import Annotated\n"
            "from fastapi import FastAPI, Form\n"
            "app = FastAPI()\n"
            "@app.post('/submit')\n"
            "async def submit(message: Annotated[str, Form()]):\n"
            "    return {'echo': message}\n"
        ),
        app__static__index_dot_html=(
            '<form method="post" action="/submit"><input name="message"></form>'
        ),
    )

    assert staged_conformance_errors(staged) == []


def test_a_native_get_form_matches_fastapi_query_parameters() -> None:
    staged = _staged(
        app__main_dot_py=(
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "@app.get('/search')\n"
            "async def search(query: str):\n"
            "    return {'query': query}\n"
        ),
        app__static__index_dot_html=(
            '<form action="/search"><input name="query"></form>'
        ),
    )

    assert staged_conformance_errors(staged) == []


def test_a_script_intercepted_form_is_left_to_the_json_request_check() -> None:
    staged = _staged(
        app__main_dot_py=(
            "from fastapi import FastAPI\n"
            "from pydantic import BaseModel\n"
            "app = FastAPI()\n"
            "class Submission(BaseModel):\n"
            "    message: str\n"
            "@app.post('/submit')\n"
            "async def submit(payload: Submission):\n"
            "    return {'echo': payload.message}\n"
        ),
        app__static__index_dot_html=(
            '<form method="post" action="/submit">'
            "<script>form.addEventListener('submit', e => e.preventDefault())</script>"
            "</form>"
        ),
    )

    assert staged_conformance_errors(staged) == []


def test_native_form_reader_ignores_external_and_default_get_forms() -> None:
    document = (
        '<form action="https://example.com/submit" method="post"></form>'
        '<form action="/search"></form>'
    )

    assert native_form_requests(document) == {("GET", "/search")}


def test_a_path_parameter_route_still_matches_its_template_literal() -> None:
    """`/api/invoices/${id}/approve` and `/api/invoices/{id}/approve` are the
    same endpoint; both collapse to one placeholder."""
    staged = _staged(
        app__main_dot_py=(
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "\n"
            "@app.post('/api/invoices/{invoice_id}/approve')\n"
            "async def approve(invoice_id: str):\n"
            "    return {'ok': True}\n"
        ),
        app__static__app_dot_js=(
            "fetch(`/api/invoices/${id}/approve`, "
            "{method:'POST', body: JSON.stringify({note})});"
        ),
    )

    errors = staged_conformance_errors(staged)

    assert len(errors) == 1
    assert "approve" in errors[0]["error"]


def test_an_upload_handler_is_not_flagged() -> None:
    """A file upload sends FormData, not JSON, and `UploadFile` is a body
    parameter FastAPI reads correctly. Flagging it would be a false block."""
    staged = _staged(
        app__main_dot_py=(
            "from fastapi import FastAPI, UploadFile\n"
            "app = FastAPI()\n"
            "\n"
            "@app.post('/api/invoices')\n"
            "async def upload(file: UploadFile):\n"
            "    return {'ok': True}\n"
        ),
        app__static__app_dot_js=(
            "const data = new FormData();\n"
            "fetch('/api/invoices', {method:'POST', body: data});\n"
        ),
    )

    assert staged_conformance_errors(staged) == []


def test_a_route_the_changeset_does_not_define_is_left_to_the_wiring_gate() -> None:
    """One rung, one job: an endpoint that does not exist is a different defect
    and reporting it twice inflates the count the user reads first."""
    staged = _staged(
        app__static__app_dot_js=(
            "fetch('/api/missing', {method:'POST', body: JSON.stringify({})});"
        ),
    )

    assert staged_conformance_errors(staged) == []


# ── The file that went missing in six consecutive builds ───────────────────


def test_a_planned_file_that_was_never_written_is_an_error() -> None:
    """The manifest is the turn's own commitment, taken before anything was
    written, so an unmet entry is not a matter of opinion."""
    staged = _staged(app__main_dot_py="x = 1\n")

    errors = staged_conformance_errors(
        staged, planned=["app/main.py", ".env.example", "README.md"]
    )

    assert {error["path"] for error in errors} == {".env.example", "README.md"}
    assert all(error["severity"] == "error" for error in errors)


def test_a_planned_existing_file_must_be_written_in_this_changeset() -> None:
    """Existing bytes cannot satisfy an explicit edit-turn commitment.

    The planner is told not to list existing files unless this turn rewrites
    them. Treating disk presence as completion let a Cline run omit README.md
    while the host reported every planned artifact complete.
    """
    staged = _staged(app__main_dot_py="x = 1\n")

    errors = staged_conformance_errors(
        staged, planned=["app/main.py", "README.md"], on_disk={"README.md"}
    )

    assert len(errors) == 1
    assert errors[0]["path"] == "README.md"
    assert "never written" in errors[0]["error"]


# ── The live-canary failure: a plan that never named a required file ───────


def test_a_file_the_request_required_but_the_plan_never_named_is_an_error() -> None:
    """The exact live-canary shape: the planner's own plan covered only 5 of
    16 explicitly required files, and the 11 missing ones were never even
    `planned` -- so a check against `planned` alone cannot catch them.
    `required` is independent of, and checked whether or not the plan
    agrees."""
    staged = _staged(
        app__main_dot_py="x = 1\n",
        app__config_dot_py="",
        app__db_dot_py="",
        app__models_dot_py="",
        app__repository_dot_py="",
    )
    planned = [
        "app/main.py",
        "app/config.py",
        "app/db.py",
        "app/models.py",
        "app/repository.py",
    ]
    required = [
        *planned,
        "app/extraction.py",
        "app/services.py",
        "requirements.txt",
        "README.md",
        "tests/test_workflows.py",
    ]

    # The verification of the smaller staged subset is clean: nothing planned
    # was left unwritten, because nothing outside the 5 files was ever
    # planned in the first place.
    planned_only_errors = staged_conformance_errors(staged, planned=planned)
    assert planned_only_errors == []

    # The independent required-files check still blocks approval.
    errors = staged_conformance_errors(staged, planned=planned, required=required)
    missing = {error["path"] for error in errors}
    assert missing == {
        "app/extraction.py",
        "app/services.py",
        "requirements.txt",
        "README.md",
        "tests/test_workflows.py",
    }
    assert all(error["severity"] == "error" for error in errors)


def test_a_required_file_already_flagged_as_planned_and_missing_is_not_doubled() -> (
    None
):
    """A required file that was also planned and never written is one defect,
    reported once -- not twice under two different messages."""
    staged = _staged(app__main_dot_py="x = 1\n")

    errors = staged_conformance_errors(
        staged,
        planned=["app/main.py", "README.md"],
        required=["README.md"],
    )

    assert [error["path"] for error in errors] == ["README.md"]


def test_a_required_file_that_was_written_is_not_an_error() -> None:
    staged = _staged(app__main_dot_py="x = 1\n", README_dot_md="# Demo\n")

    errors = staged_conformance_errors(
        staged, planned=["app/main.py"], required=["app/main.py", "README.md"]
    )

    assert errors == []


# ── Settings nobody would know to set ──────────────────────────────────────


def test_an_env_var_absent_from_the_example_file_is_a_warning() -> None:
    """Advice, not a block: the code is correct, the documentation is thin."""
    staged = _staged(
        app__config_dot_py=(
            "import os\n"
            "BASE = os.getenv('OCI_BASE_URL', '')\n"
            "SECRET = os.getenv('OCI_COMPARTMENT_ID')\n"
        ),
        _dot_env_dot_example="OCI_BASE_URL=https://example\n",
    )

    errors = staged_conformance_errors(staged)

    assert len(errors) == 1
    assert "OCI_COMPARTMENT_ID" in errors[0]["error"]
    assert errors[0]["severity"] == "warning"


def test_no_example_file_means_nothing_to_be_inconsistent_with() -> None:
    """Demanding an example env file is a style opinion, not a defect."""
    staged = _staged(app__config_dot_py="import os\nX = os.getenv('ANYTHING')\n")

    assert staged_conformance_errors(staged) == []


# ── The frontend reader itself ─────────────────────────────────────────────


def test_only_json_bodies_count_as_requests() -> None:
    """A GET, and a POST with no body, are not this rung's business."""
    script = (
        "fetch('/api/jobs');\n"
        "fetch('/api/jobs/1/cancel', {method:'POST'});\n"
        "fetch('/api/jobs', {method:'POST', body: JSON.stringify({a:1})});\n"
    )

    assert json_body_requests(script) == {("POST", "/api/jobs")}


def test_a_file_that_does_not_parse_is_left_to_the_syntax_rung() -> None:
    """Nothing here can add to a parse error, and crashing on one would take the
    whole gate down with it."""
    staged = _staged(app__main_dot_py="def broken(:\n    pass\n")

    assert staged_conformance_errors(staged) == []


# ── Making the manifest bind the cloud provider too ────────────────────────


def _oci_provider() -> OCIResponsesModelProvider:
    return OCIResponsesModelProvider(Settings(_env_file=None, allow_test_backends=True))


def test_owed_files_narrow_grok_to_writing_them() -> None:
    """The enum points the model at the files it still owes."""
    tools = {
        tool["name"]: tool
        for tool in _oci_provider()._project_tools([".env.example", "README.md"])
    }

    assert tools["create_file"]["parameters"]["properties"]["path"]["enum"] == [
        ".env.example",
        "README.md",
    ]


def test_finishing_stays_available_while_files_are_owed() -> None:
    """Withholding it was measured and it deadlocks: when the one owed path is
    something the host refuses, a model with no way to finish has no legal move
    at all, and a 19-step build became a 48-step one that still produced
    nothing. The host refuses a premature finish; the model must be able to
    attempt it."""
    tools = {
        tool["name"]: tool for tool in _oci_provider()._project_tools([".env.example"])
    }

    assert "finish_project_task" in tools


def test_nothing_owed_leaves_the_tools_alone() -> None:
    """An edit turn, or a build whose manifest is satisfied, keeps the full set
    — including the ability to say it is finished."""
    tools = {tool["name"]: tool for tool in _oci_provider()._project_tools([])}

    assert "finish_project_task" in tools
    assert "enum" not in tools["create_file"]["parameters"]["properties"]["path"]


def test_narrowing_does_not_mutate_the_shared_tool_definitions() -> None:
    """The narrowed schema is built per step; leaking an enum into the next turn
    would pin a model to files it has already written."""
    provider = _oci_provider()
    provider._project_tools([".env.example"])

    tools = {tool["name"]: tool for tool in provider._project_tools([])}

    assert "enum" not in tools["create_file"]["parameters"]["properties"]["path"]
