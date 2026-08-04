"""Cross-file checks over a staged changeset, and the cases they must NOT flag.

Every rule here gets a negative case as well as a positive one. A false positive
is worse than a missed defect: it spends the model's bounded fix budget rewriting
code that was already correct, and the build ends with less done, not more.
"""
from __future__ import annotations

from waqil_api.project_wiring import ERROR, WARNING, staged_wiring_errors


def _staged(files: dict[str, str]) -> dict[str, dict[str, str]]:
    """Wrap raw file text in the minimal staged-entry shape the checker reads."""
    return {path: {"content": content} for path, content in files.items()}


def _errors(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    """Only the findings the host can prove, ignoring advisory warnings."""
    return [item for item in findings if item["severity"] == ERROR]


# ── Local imports ────────────────────────────────────────────────────────────


def test_flags_an_import_of_a_module_the_project_does_not_have() -> None:
    findings = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/main.py": "from app.agents.base import Agent\n",
                    "app/__init__.py": "",
                }
            )
        )
    )

    assert [item["path"] for item in findings] == ["app/main.py"]
    assert "app.agents.base" in findings[0]["error"]


def test_accepts_an_import_of_a_sibling_staged_in_the_same_turn() -> None:
    findings = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/main.py": "from app.agents.base import Agent\n",
                    "app/agents/base.py": "class Agent:\n    pass\n",
                }
            )
        )
    )

    assert findings == []


def test_accepts_an_import_into_a_file_already_on_disk() -> None:
    """An edit turn imports untouched code; resolution sees the whole project."""
    findings = _errors(
        staged_wiring_errors(
            _staged({"app/main.py": "from app.config import settings\n"}),
            sources={"app/config.py": "settings = {}\n"},
            project_paths=["app/config.py"],
        )
    )

    assert findings == []


def test_accepts_an_implicit_namespace_package_with_no_init_file() -> None:
    """`app/agents/` without __init__.py still imports; calling it broken lies."""
    findings = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/main.py": "import app.agents.base\n",
                    "app/agents/base.py": "class Agent:\n    pass\n",
                }
            )
        )
    )

    assert findings == []


def test_ignores_third_party_imports_when_resolving_local_modules() -> None:
    findings = _errors(
        staged_wiring_errors(
            _staged({"app/main.py": "import fastapi\nfrom pypdf import PdfReader\n"})
        )
    )

    assert findings == []


def test_flags_a_relative_import_that_climbs_past_the_project_root() -> None:
    findings = _errors(
        staged_wiring_errors(_staged({"app/main.py": "from ... import config\n"}))
    )

    assert len(findings) == 1
    assert "above the project root" in findings[0]["error"]


def test_resolves_relative_imports_against_the_files_own_package() -> None:
    findings = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/agents/planner.py": "from .base import Agent\nfrom ..config import URL\n",
                    "app/agents/base.py": "class Agent:\n    pass\n",
                    "app/config.py": "URL = 'x'\n",
                }
            )
        )
    )

    assert findings == []


# ── Imported symbols ─────────────────────────────────────────────────────────


def test_flags_a_name_the_target_module_never_defines() -> None:
    findings = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/main.py": "from app.agents.base import Agent, Missing\n",
                    "app/agents/base.py": "class Agent:\n    pass\n",
                }
            )
        )
    )

    assert len(findings) == 1
    assert "Missing" in findings[0]["error"]


def test_accepts_a_name_re_exported_through_a_package_init() -> None:
    findings = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/main.py": "from app.agents import Agent\n",
                    "app/agents/__init__.py": "from .base import Agent\n",
                    "app/agents/base.py": "class Agent:\n    pass\n",
                }
            )
        )
    )

    assert findings == []


def test_accepts_a_submodule_imported_by_name_from_its_package() -> None:
    findings = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/main.py": "from app.agents import base\n",
                    "app/agents/__init__.py": "",
                    "app/agents/base.py": "class Agent:\n    pass\n",
                }
            )
        )
    )

    assert findings == []


def test_stands_down_on_a_module_whose_exports_it_cannot_enumerate() -> None:
    """A star import or a module __getattr__ hides names from any static reader."""
    star = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/main.py": "from app.registry import ANYTHING\n",
                    "app/registry.py": "from app.base import *\n",
                    "app/base.py": "X = 1\n",
                }
            )
        )
    )
    dynamic = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/main.py": "from app.registry import ANYTHING\n",
                    "app/registry.py": "def __getattr__(name):\n    return name\n",
                }
            )
        )
    )

    assert star == []
    assert dynamic == []


def test_accepts_names_bound_conditionally_or_by_annotation() -> None:
    provider = (
        "from typing import TYPE_CHECKING\n"
        "TIMEOUT: int = 5\n"
        "try:\n"
        "    from fastapi import FastAPI\n"
        "except ImportError:\n"
        "    def FastAPI():\n        return None\n"
        "if TYPE_CHECKING:\n"
        "    class Spec:\n        pass\n"
    )
    findings = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/main.py": "from app.config import TIMEOUT, FastAPI, Spec\n",
                    "app/config.py": provider,
                }
            )
        )
    )

    assert findings == []


# ── Stub modules ─────────────────────────────────────────────────────────────


def test_flags_a_module_where_every_function_is_a_placeholder() -> None:
    """The exact defect a real build shipped: three agents that were all TODO."""
    stub = (
        "class Planner:\n"
        '    """Plans the work."""\n'
        "    def run(self, task):\n"
        "        # TODO: implement\n"
        "        pass\n"
    )
    findings = _errors(staged_wiring_errors(_staged({"app/agents/planner.py": stub})))

    assert [item["path"] for item in findings] == ["app/agents/planner.py"]
    assert "implements none of them" in findings[0]["error"]


def test_does_not_flag_a_deliberately_abstract_interface() -> None:
    base = (
        "from abc import ABC, abstractmethod\n"
        "class Agent(ABC):\n"
        "    @abstractmethod\n"
        "    def run(self, task): ...\n"
    )
    protocol = (
        "from typing import Protocol\n"
        "class Reader(Protocol):\n"
        "    def read(self) -> str: ...\n"
    )

    assert _errors(staged_wiring_errors(_staged({"app/agents/base.py": base}))) == []
    assert _errors(staged_wiring_errors(_staged({"app/ports.py": protocol}))) == []


def test_does_not_flag_a_module_with_one_real_implementation() -> None:
    mixed = (
        "def helper():\n"
        "    pass\n"
        "def summarize(text: str) -> str:\n"
        "    return text[:10]\n"
    )

    assert _errors(staged_wiring_errors(_staged({"app/tools.py": mixed}))) == []


def test_does_not_flag_files_that_define_no_functions_at_all() -> None:
    findings = _errors(
        staged_wiring_errors(
            _staged({"app/__init__.py": "", "app/constants.py": "TIMEOUT = 30\n"})
        )
    )

    assert findings == []


# ── Application wiring ───────────────────────────────────────────────────────


def _app(body: str) -> dict[str, dict[str, str]]:
    """A staged FastAPI entrypoint, which is what turns the wiring rules on."""
    return _staged({"app/main.py": "from fastapi import FastAPI\napp = FastAPI()\n" + body})


def test_flags_static_files_that_are_built_but_never_mounted() -> None:
    """CSS and JS 404 at runtime; nothing about the file itself looks wrong."""
    findings = _errors(
        staged_wiring_errors(
            _app(
                "from fastapi.staticfiles import StaticFiles\n"
                "files = StaticFiles(directory='app/static')\n"
            ),
            # The directory exists, so this isolates the not-mounted rule from the
            # separate check that a mounted directory must be created.
            project_paths=["app/static/logo.svg"],
        )
    )

    assert len(findings) == 1
    assert "404" in findings[0]["error"]


def test_accepts_static_files_that_are_mounted() -> None:
    findings = _errors(
        staged_wiring_errors(
            _app(
                "from fastapi.staticfiles import StaticFiles\n"
                "app.mount('/static', StaticFiles(directory='app/static'))\n"
            ),
            project_paths=["app/static/logo.svg"],
        )
    )

    assert findings == []


def test_accepts_wiring_that_happens_in_another_file() -> None:
    staged = _app("from fastapi.staticfiles import StaticFiles\nfiles = StaticFiles(directory='s')\n")
    findings = _errors(
        staged_wiring_errors(
            staged,
            sources={"app/wiring.py": "def attach(app, files):\n    app.mount('/s', files)\n"},
            project_paths=["app/wiring.py", "s/logo.svg"],
        )
    )

    assert findings == []


def test_flags_a_router_that_is_never_included() -> None:
    findings = _errors(
        staged_wiring_errors(_app("from fastapi import APIRouter\nrouter = APIRouter()\n"))
    )

    assert len(findings) == 1
    assert "include_router" in findings[0]["error"]


def test_stays_quiet_when_the_turn_did_not_stage_the_application() -> None:
    """Without the app object in the changeset the wiring may live in a file
    this turn never touched, so reporting it would be a guess."""
    findings = _errors(
        staged_wiring_errors(
            _staged(
                {
                    "app/routes.py": "from fastapi import APIRouter\nrouter = APIRouter()\n"
                }
            )
        )
    )

    assert findings == []


# ── Declared dependencies ────────────────────────────────────────────────────


def test_warns_about_an_import_the_requirements_never_declare() -> None:
    findings = staged_wiring_errors(
        _staged({"app/extraction.py": "import pypdf\n"}),
        requirements="fastapi==0.115.0\nuvicorn[standard]==0.30.0\n",
    )

    assert [item["severity"] for item in findings] == [WARNING]
    assert "pypdf" in findings[0]["error"]


def test_does_not_warn_about_declared_aliased_or_stdlib_imports() -> None:
    source = "import json\nimport dotenv\nimport fastapi\nfrom bs4 import BeautifulSoup\n"
    findings = staged_wiring_errors(
        _staged({"app/main.py": source}),
        requirements="fastapi==0.115.0\npython-dotenv==1.0.1\nbeautifulsoup4==4.12.3\n",
    )

    assert findings == []


def test_says_nothing_about_dependencies_when_none_are_declared() -> None:
    """A project with no requirements file has not made a claim to contradict."""
    findings = staged_wiring_errors(_staged({"app/main.py": "import pypdf\n"}))

    assert findings == []


# ── Boundaries ───────────────────────────────────────────────────────────────


def test_leaves_unparseable_files_to_the_syntax_gate() -> None:
    findings = staged_wiring_errors(
        _staged({"app/broken.py": "def f(:\n    pass\n", "app/ok.py": "def g():\n    return 1\n"})
    )

    assert findings == []


def test_reports_only_on_files_the_changeset_actually_stages() -> None:
    findings = staged_wiring_errors(
        _staged({"app/main.py": "from app.config import URL\n"}),
        sources={
            "app/config.py": "URL = 'x'\n",
            "app/legacy.py": "from app.gone import thing\n",
        },
        project_paths=["app/config.py", "app/legacy.py"],
    )

    assert findings == []


def test_skips_vendored_directories_when_indexing_modules() -> None:
    findings = staged_wiring_errors(
        _staged({"app/main.py": "import json\n"}),
        project_paths=[".venv/lib/site-packages/thing/__init__.py"],
    )

    assert findings == []


# ── A static mount whose directory the project never creates ─────────────────
# This is the exact defect that made a live qwen build non-functional: the app
# mounts StaticFiles at a directory nothing in the changeset creates, so it
# raises at import. It is environment-independent, so it is a blocking error.


def _static(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in findings if "StaticFiles at directory" in item["error"]]


def test_flags_a_static_mount_at_a_literal_directory_the_project_lacks() -> None:
    findings = staged_wiring_errors(
        _staged({
            "app/main.py": (
                "from fastapi import FastAPI\n"
                "from fastapi.staticfiles import StaticFiles\n"
                "app = FastAPI()\n"
                "app.mount('/static', StaticFiles(directory='app/static'), name='static')\n"
            ),
        }),
    )
    flagged = _static(findings)
    assert len(flagged) == 1
    assert flagged[0]["severity"] == ERROR
    assert "app/static" in flagged[0]["error"]


def test_flags_the_path_file_parent_static_pattern_a_real_build_shipped() -> None:
    # `STATIC = Path(__file__).parent / "static"` — the exact form the live qwen
    # build used, and the reason a literal-only check would have missed it.
    findings = staged_wiring_errors(
        _staged({
            "app/main.py": (
                "from pathlib import Path\n"
                "from fastapi import FastAPI\n"
                "from fastapi.staticfiles import StaticFiles\n"
                "app = FastAPI()\n"
                "STATIC = Path(__file__).parent / 'static'\n"
                "app.mount('/static', StaticFiles(directory=STATIC), name='static')\n"
            ),
        }),
    )
    flagged = _static(findings)
    assert len(flagged) == 1
    assert "app/static" in flagged[0]["error"]


def test_accepts_a_static_mount_whose_directory_the_changeset_creates() -> None:
    findings = staged_wiring_errors(
        _staged({
            "app/main.py": (
                "from pathlib import Path\n"
                "from fastapi import FastAPI\n"
                "from fastapi.staticfiles import StaticFiles\n"
                "app = FastAPI()\n"
                "STATIC = Path(__file__).parent / 'static'\n"
                "app.mount('/static', StaticFiles(directory=STATIC), name='static')\n"
            ),
            "app/static/index.html": "<html></html>\n",
        }),
    )
    assert _static(findings) == []


def test_accepts_a_static_mount_whose_directory_already_exists_on_disk() -> None:
    # An edit turn that touches only main.py must not be blamed for a static
    # directory that is already in the project.
    findings = staged_wiring_errors(
        _staged({
            "app/main.py": (
                "from fastapi import FastAPI\n"
                "from fastapi.staticfiles import StaticFiles\n"
                "app = FastAPI()\n"
                "app.mount('/static', StaticFiles(directory='app/static'), name='static')\n"
            ),
        }),
        project_paths=["app/static/style.css"],
    )
    assert _static(findings) == []


def test_does_not_flag_a_static_mount_with_check_dir_disabled() -> None:
    findings = staged_wiring_errors(
        _staged({
            "app/main.py": (
                "from fastapi import FastAPI\n"
                "from fastapi.staticfiles import StaticFiles\n"
                "app = FastAPI()\n"
                "app.mount('/s', StaticFiles(directory='app/static', check_dir=False), name='s')\n"
            ),
        }),
    )
    assert _static(findings) == []


def test_does_not_flag_a_static_directory_it_cannot_resolve() -> None:
    # A directory that comes from a call the checker cannot read is left alone:
    # an unresolvable path is never a finding, so correct code is never blocked.
    findings = staged_wiring_errors(
        _staged({
            "app/main.py": (
                "from fastapi import FastAPI\n"
                "from fastapi.staticfiles import StaticFiles\n"
                "def pick(): return 'somewhere'\n"
                "app = FastAPI()\n"
                "app.mount('/s', StaticFiles(directory=pick()), name='s')\n"
            ),
        }),
    )
    assert _static(findings) == []
