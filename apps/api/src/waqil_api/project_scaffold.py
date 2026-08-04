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

from .project_env import env_documentation, env_example
from .scaffold.appkit import SCAFFOLD_VERSION

__all__ = [
    "SCAFFOLD_VERSION",
    "build_capabilities",
    "scaffold_note",
    "scaffold_prompt",
    "scaffold_sources",
    "wants_oci_responses",
]

_APPKIT_DIR = Path(__file__).resolve().parent / "scaffold" / "appkit"

_BASE_MODULES = ("__init__.py", "config.py", "money.py", "uploads.py")
_OCI_MODULE = "oci_responses.py"

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


def wants_oci_responses(prompt: str) -> bool:
    """Whether a build request needs the OCI Responses adapter vendored."""
    return bool(_OCI_INTENT.search(prompt))


def build_capabilities(prompt: str) -> frozenset[str]:
    """The capabilities a build request declares, read from its text."""
    return frozenset({"oci_responses"}) if wants_oci_responses(prompt) else frozenset()


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
    files = {
        f"appkit/{name}": (_APPKIT_DIR / name).read_text(encoding="utf-8")
        for name in names
    }
    if wanted:
        files[".env.example"] = env_example(wanted)
    return files


def scaffold_note(*, has_oci: bool) -> str:
    """What the model is told about appkit — names and contracts, no values."""
    lines = [
        f"This project contains appkit/ (Metis-owned scaffold, version "
        f"{SCAFFOLD_VERSION}): verified infrastructure, already staged. Import "
        "it; never rewrite it — writes under appkit/ are refused.",
        "- appkit.config: load_dotenv(), require(name), optional(name, default). "
        "Read configuration at use time; never at import time.",
        "- appkit.money: to_money, sum_money, within_cents, within_percent — "
        "Decimal arithmetic for amounts; a missing value stays None, never zero.",
        "- appkit.uploads: await save_upload(upload) -> SavedUpload(path, mime, "
        "size), .remove() in a finally. Sniffs the real MIME, enforces a size "
        "cap, never trusts client filenames. FastAPI upload routes also need "
        "python-multipart declared in requirements.",
    ]
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
    return scaffold_note(has_oci=has_oci)
