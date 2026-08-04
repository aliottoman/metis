"""P1.5: a staged frontend may not reference files nobody provides.

The browser resolves these references at page load, so a miss is a broken
page in every environment — checkable from the changeset alone. The negative
space matters as much: external URLs, data URLs, anchors, template
expressions, and unmounted absolute paths (which a route may serve
dynamically) must never false-alarm.
"""

from __future__ import annotations

from waqil_api.project_wiring import staged_wiring_errors

_MAIN = (
    "from fastapi import FastAPI\n"
    "from fastapi.staticfiles import StaticFiles\n\n"
    "app = FastAPI()\n"
    "app.mount('/static', StaticFiles(directory='static'), name='static')\n"
)

_HTML = (
    "<!doctype html>\n"
    "<link rel=\"stylesheet\" href=\"/static/styles.css\">\n"
    "<script src=\"/static/app.js\"></script>\n"
    "<img src=\"https://example.com/logo.png\">\n"
    "<img src=\"data:image/png;base64,AAAA\">\n"
    "<a href=\"#top\">top</a>\n"
    "<img src=\"{{ dynamic_url }}\">\n"
)


def _staged(files: dict[str, str]) -> dict[str, dict[str, str]]:
    return {path: {"content": content} for path, content in files.items()}


def _reference_findings(files: dict[str, str], project_paths: list[str] = []):
    findings = staged_wiring_errors(
        _staged(files), project_paths=project_paths, requirements="fastapi\npython-multipart\n"
    )
    return [f for f in findings if "no file in this project provides" in f["error"] and "references" in f["error"]]


def test_mounted_references_resolve_and_missing_ones_block() -> None:
    complete = {
        "app/main.py": _MAIN,
        "static/index.html": _HTML,
        "static/app.js": "console.log('ok');\n",
        "static/styles.css": "body { color: black; }\n",
    }
    assert _reference_findings(complete) == []

    broken = dict(complete)
    del broken["static/app.js"]
    (finding,) = _reference_findings(broken)
    assert finding["severity"] == "error"
    assert "static/app.js" in finding["error"]
    assert "line 3" in finding["error"]


def test_relative_references_resolve_against_the_referencing_file() -> None:
    files = {
        "static/styles.css": "h1 { background: url('../images/logo.png?v=2'); }\n",
    }
    (finding,) = _reference_findings(files)
    assert "images/logo.png" in finding["error"]
    # Present on disk (not staged) — the project provides it, no finding.
    assert _reference_findings(files, project_paths=["images/logo.png"]) == []


def test_unmounted_absolute_paths_and_externals_never_false_alarm() -> None:
    files = {
        "static/index.html": (
            "<script src=\"/generated/config.js\"></script>\n"  # dynamic route
            "<img src=\"//cdn.example.com/x.png\">\n"
            "<link href=\"mailto:someone@example.com\">\n"
        ),
    }
    assert _reference_findings(files) == []


def test_escaping_the_project_is_skipped_not_crashed() -> None:
    files = {"static/styles.css": "h1 { background: url('../../../etc/passwd'); }\n"}
    assert _reference_findings(files) == []
