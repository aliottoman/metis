"""Deterministic scaffolding for project builds.

Ten reconstructed Ledger builds made the case for this module: every model,
frontier or local, reinvented the OCI client, the config loading and the
upload handling differently, and nine of ten reinventions were broken. The
infrastructure Metis already runs and tests is now written INTO the project
before the model's first step — vendored as a top-level ``appkit`` package —
so a build spends its nondeterminism on the application, never on plumbing.

The vendored files ride the normal overlay: staged by the host, visible on
the approval card, materialized only when the user approves the changeset.
Models may read them and import them; writes under ``appkit/`` are refused.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .project_env import KNOWN_CAPABILITIES, env_documentation, env_example
from .scaffold.appkit import SCAFFOLD_VERSION
from .scaffold.appkit.web import THEME_LINK as _THEME_LINK

__all__ = [
    "SCAFFOLD_VERSION",
    "build_capabilities",
    "scaffold_note",
    "scaffold_prompt",
    "scaffold_sources",
    "wants_oci_responses",
    "wants_web_ui",
]

_APPKIT_DIR = Path(__file__).resolve().parent / "scaffold" / "appkit"

_BASE_MODULES = ("__init__.py", "config.py", "money.py", "uploads.py")
_OCI_MODULE = "oci_responses.py"
# The design language, vendored like any other verified infrastructure. It is
# two files because the theme is only useful if the app can serve it.
_WEB_FILES = ("web.py", "static/theme.css")

# Deliberately narrow: vendoring the adapter into a build that never calls it
# costs an unused file and a documented-but-unread .env.example, so bare
# "image"/"photo" do not qualify — the words here are the ones extraction
# requests actually use. A miss is recoverable: the reference notes still
# teach the correct client, just without the vendored implementation.
_OCI_INTENT = re.compile(
    r"\b(oci\b|grok|responses api|extract\w*|invoice|receipt|ocr|vision|"
    r"multimodal|document intelligence)",
    re.IGNORECASE,
)


# A build that renders anything to a browser. Deliberately broader than the
# OCI pattern: the cost of a false positive is one unused stylesheet, while
# the cost of a miss is an app that invents its own visual language — which
# is the failure this capability exists to end. Words like "api", "service"
# and "endpoint" are absent, so a JSON-only backend stays unthemed.
_WEB_INTENT = re.compile(
    r"\b(ui|user interface|web ?app|web ?site|web ?page|front-?end|html|css|"
    r"dashboard|portal|browser|design|theme|styling|upload form|viewer)\b",
    re.IGNORECASE,
)


def wants_oci_responses(prompt: str) -> bool:
    """Whether a build request needs the OCI Responses adapter vendored."""
    return bool(_OCI_INTENT.search(prompt))


def wants_web_ui(prompt: str) -> bool:
    """Whether a build request renders pages, and so owes the design language."""
    return bool(_WEB_INTENT.search(prompt))


def build_capabilities(prompt: str) -> frozenset[str]:
    """The capabilities a build request declares, read from its text.

    Not every capability projects environment variables — `web_ui` vendors
    files only. The env layer intersects with its own KNOWN_CAPABILITIES, so
    a files-only capability passes through it harmlessly.
    """
    capabilities: set[str] = set()
    if wants_oci_responses(prompt):
        capabilities.add("oci_responses")
    if wants_web_ui(prompt):
        capabilities.add("web_ui")
    return frozenset(capabilities)


def scaffold_sources(capabilities: Iterable[str]) -> dict[str, str]:
    """Path → content for every file the framework writes into a build.

    Contents are read from the canonical modules under ``waqil_api.scaffold``
    — the same files this repo's test suite imports and exercises — so what a
    project receives is exactly what was tested, byte for byte.
    """
    wanted = set(capabilities)
    names = list(_BASE_MODULES)
    if "oci_responses" in wanted:
        names.append(_OCI_MODULE)
    if "web_ui" in wanted:
        names.extend(_WEB_FILES)
    files = {
        f"appkit/{name}": (_APPKIT_DIR / name).read_text(encoding="utf-8")
        for name in names
    }
    # Only capabilities that actually project variables earn the file: a
    # web-only build would otherwise receive a .env.example with a header and
    # nothing under it, which reads as a configuration step that does not exist.
    if wanted & KNOWN_CAPABILITIES:
        files[".env.example"] = env_example(wanted)
    return files


def scaffold_note(*, has_oci: bool, has_web: bool = False) -> str:
    """What the model is told about appkit — names and contracts, no values."""
    lines = [
        f"This project contains appkit/ (Metis-owned scaffold, version "
        f"{SCAFFOLD_VERSION}): verified infrastructure, already staged. Import "
        "it; never rewrite it — writes under appkit/ are refused.",
        "- appkit.config: load_dotenv(), require(name), optional(name, default). "
        "Read configuration at use time; never at import time.",
        "- appkit.money: to_money, sum_money, within_cents, within_percent — "
        "Decimal arithmetic for amounts; a missing value stays None, never zero.",
        "- appkit.uploads exports IMAGE_MIMES, DOCUMENT_MIMES, UploadError, "
        "SavedUpload, sniff_mime(data), and await save_upload(upload, *, "
        "max_bytes=10 * 1024 * 1024, allowed_mimes=IMAGE_MIMES) -> "
        "SavedUpload(path, mime, size). The default is image-only. A document "
        "route MUST call save_upload(upload, allowed_mimes=DOCUMENT_MIMES); "
        "that policy accepts verified images, PDF, and plain UTF-8 text. Catch "
        "UploadError and return HTTP 415, and call saved.remove() in a finally. "
        "The helper sniffs bytes, enforces the size cap, and never trusts client "
        "filenames. FastAPI upload routes also need python-multipart declared "
        "in requirements.",
    ]
    if has_web:
        lines.append(
            "- appkit.web: THE DESIGN LANGUAGE. Every page this app serves wears "
            "it; you never invent a visual style. Mount it once —\n"
            "    from appkit.web import mount_appkit_static\n"
            "    mount_appkit_static(app)\n"
            "  — then link it FIRST in every page's <head>:\n"
            f"    {_THEME_LINK}\n"
            "  Write NO colors, fonts, font sizes, radii, shadows or spacing of "
            "your own: no <style> block redefining them, no second stylesheet, "
            "no CSS framework, no CDN. If you need one small layout rule the "
            "vocabulary lacks, use the CSS variables (var(--ink), var(--line), "
            "var(--radius-md), var(--ease)) rather than literal values.\n"
            "  Compose these classes:\n"
            "    layout   .page (wrap everything; .page-wide for tables/dashboards), "
            ".stack/.stack-sm/.stack-lg, .row, .spread, .grid, .grid-2, .divider\n"
            "    type     .eyebrow (mono uppercase label above a title), .lede, "
            ".muted, .faint, .small, .num, .mono\n"
            "    surfaces .card (glass panel, the default container), .card-tight, "
            ".card-header, .panel\n"
            "    controls .btn, .btn-primary, .btn-ghost, .btn-danger, .btn-sm; "
            ".field + .label + .input/.textarea/.select + .hint\n"
            "    data     .table inside .table-wrap; .chip (+ .chip-ok/-warn/"
            "-danger/-info); .dot (+ .dot-ok/-warn/-danger); .stat + .stat-value "
            "+ .stat-label\n"
            "    states   .dropzone (add .is-active while dragging), .empty, "
            ".note (+ .note-ok/-warn/-danger), .spinner, .skeleton, .rise\n"
            "  appkit.web.page(title, body) returns a complete HTML document "
            "already carrying the charset, viewport, theme link and .page "
            "wrapper — use it for server-rendered pages. A static index.html "
            "must carry the same <link> itself."
        )
    if has_oci:
        lines.append(
            "- appkit.oci_responses: await OciResponses().extract_document(prompt, "
            "image_bytes) and .generate(prompt) return the reply text; "
            "parse_json_output(text) parses it or raises; ExtractionError "
            "carries status and detail. Auth, the required Responses project "
            "and terminal states are already handled. Calls are synchronous — "
            "run them inside FastAPI BackgroundTasks for async workflows; "
            "never pass background=True yourself (signed OCI requests cannot "
            "be re-executed by the service). Declare openai, httpx and "
            "oci-genai-auth in requirements."
        )
        lines.append(
            "Environment this app reads (Metis injects real values at launch; "
            ".env.example documents them — never hard-code values):\n"
            + env_documentation({"oci_responses"})
        )
    return "\n".join(lines)


def scaffold_prompt(
    staged: Mapping[str, Any], prompt_context: Mapping[str, Any]
) -> str:
    """The scaffold note for a step request, or "" when the project has none.

    Presence is read from the overlay and the manifest file tree rather than
    remembered: a follow-up turn on a project whose appkit reached disk long
    ago must describe it exactly as a build turn that staged it seconds ago.
    """
    manifest = prompt_context.get("manifest") or {}
    tree = manifest.get("file_tree") or []
    known = [str(path) for path in (*staged, *tree)]
    if not any(path == "appkit" or path.startswith("appkit/") for path in known):
        return ""
    has_oci = any(path.endswith("appkit/oci_responses.py") for path in known)
    has_web = any(path.endswith("appkit/web.py") for path in known)
    return scaffold_note(has_oci=has_oci, has_web=has_web)
