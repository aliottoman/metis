"""What a generated project may know about this machine's configuration.

Generated applications run outside Metis's trust boundary, yet the features
they implement — OCI Responses extraction, for one — need real configuration
to work. Historically that gap was papered over with prose: build specs told
the model about `OCI_COMPARTMENT_ID`, the model invented the rest, and every
generated integration broke differently. This module replaces the prose with
one seam. Canonical ``WAQIL_*`` settings are projected to the stable ``OCI_*``
names a standalone app reads, filtered to the capabilities the project
actually uses, and injected only into a child process environment.

The boundary rule: models and generated files see variable *names and
descriptions*; values reach a launched process and nowhere else — never a
prompt, never a generated file, never a log. The OCI private key never leaves
``~/.oci`` at all; apps receive the config-file/profile pointer and sign for
themselves, exactly as Metis's own provider does.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


@dataclass(frozen=True)
class ProjectedVar:
    """One environment variable a generated app is allowed to receive."""

    name: str  # the name the generated app reads
    description: str  # shown to models and written into .env.example
    required: bool  # missing at launch is a preflight error, not a child crash
    source: str  # the Settings attribute the value is projected from


# Capability -> the variables it justifies. A project gets a variable only by
# using a capability that lists it; there is no "give the app everything".
#
# The Responses surface needs neither region nor compartment: the base URL
# encodes the region, and compartments scope classic GenAI calls that this
# surface never makes. Omitting them here is deliberate — projecting them
# anyway is how compartment/project confusion survived this long.
CAPABILITY_VARS: dict[str, tuple[ProjectedVar, ...]] = {
    "oci_responses": (
        ProjectedVar(
            "OCI_CONFIG_FILE",
            "Path to the OCI config file; empty means the SDK default ~/.oci/config",
            required=False,
            source="oci_config_file",
        ),
        ProjectedVar(
            "OCI_PROFILE",
            "Profile inside the OCI config file used to sign requests",
            required=False,
            source="oci_profile",
        ),
        ProjectedVar(
            "OCI_RESPONSES_BASE_URL",
            "OpenAI-compatible Responses endpoint, e.g. https://inference.generativeai"
            ".us-chicago-1.oci.oraclecloud.com/openai/v1",
            required=True,
            source="oci_responses_base_url",
        ),
        ProjectedVar(
            "OCI_RESPONSES_PROJECT_ID",
            "Responses project id — required by non-OpenAI models (Grok); "
            "this is NOT the compartment OCID",
            required=True,
            source="oci_responses_project_id",
        ),
        ProjectedVar(
            "OCI_RESPONSES_MODEL_ID",
            "Model id the app should request, e.g. xai.grok-4.3",
            required=True,
            source="oci_grok_model",
        ),
    ),
}

KNOWN_CAPABILITIES = frozenset(CAPABILITY_VARS)


# A capability is *used*, not declared: a project has oci_responses when its
# code reaches for the vendored adapter. Detection from the files themselves
# means a follow-up edit that adds or removes the integration changes the
# projection with no manifest to forget to update.
_CAPABILITY_MARKS: dict[str, tuple[re.Pattern[str], ...]] = {
    "oci_responses": (
        re.compile(r"\bappkit\.oci_responses\b"),
        re.compile(r"\bfrom\s+appkit\s+import\s+[^\n]*\boci_responses\b"),
    ),
}


def detect_capabilities(files: Mapping[str, str]) -> frozenset[str]:
    """Which capabilities a project's Python files actually use."""
    found: set[str] = set()
    for path, text in files.items():
        if not str(path).endswith(".py"):
            continue
        for capability, marks in _CAPABILITY_MARKS.items():
            if capability not in found and any(mark.search(text) for mark in marks):
                found.add(capability)
    return frozenset(found)


def capability_blocked(settings: Settings, capability: str) -> str:
    """Why this machine refuses to power a capability, or "" when it will.

    Distinct from a missing variable: a blocked capability is a policy answer
    ("cloud calls are switched off"), and the preflight error should say that
    rather than listing every variable the switch would have populated.
    """
    if capability not in KNOWN_CAPABILITIES:
        return f"unknown capability {capability!r}"
    if capability == "oci_responses" and not settings.allow_oci_responses:
        return (
            "OCI Responses calls are disabled on this machine "
            "(WAQIL_ALLOW_OCI_RESPONSES is false)"
        )
    return ""


def missing_required(settings: Settings, capabilities: Iterable[str]) -> list[str]:
    """Required projected names with no value behind them, for preflight errors."""
    missing: list[str] = []
    for capability in sorted(set(capabilities) & KNOWN_CAPABILITIES):
        for var in CAPABILITY_VARS[capability]:
            value = str(getattr(settings, var.source, "") or "").strip()
            if var.required and not value:
                missing.append(var.name)
    return missing


def project_environment(
    settings: Settings, capabilities: Iterable[str]
) -> dict[str, str]:
    """The exact environment a child process for this project receives.

    Only allowlisted names, only non-empty values, and nothing at all for a
    capability this machine blocks — a launch preflight surfaces the block as
    its own error instead of the child dying on a half-populated environment.
    """
    environment: dict[str, str] = {}
    for capability in sorted(set(capabilities) & KNOWN_CAPABILITIES):
        if capability_blocked(settings, capability):
            continue
        for var in CAPABILITY_VARS[capability]:
            value = str(getattr(settings, var.source, "") or "").strip()
            if value:
                environment[var.name] = value
    return environment


# Directories whose contents can never change what a project *is*.
_SKIPPED_DIRS = frozenset(
    {".git", ".hg", ".metis", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
)


def capabilities_of_tree(root: Path) -> frozenset[str]:
    """Capabilities detected from a project directory on disk, bounded.

    The launch-time twin of `detect_capabilities`: the overlay is gone by the
    time an asset starts, so the applied files are read back — bounded in
    count and size, vendored/VCS directories pruned before descent.
    """
    files: dict[str, str] = {}
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(set(dirs) - _SKIPPED_DIRS)
        for name in sorted(names):
            if not name.endswith(".py") or len(files) >= 400:
                continue
            path = Path(current) / name
            try:
                files[str(path.relative_to(root))] = path.read_text(
                    encoding="utf-8"
                )[:100_000]
            except (OSError, UnicodeDecodeError, ValueError):
                continue
    return detect_capabilities(files)


def asset_environment(settings: Settings, root: Path) -> dict[str, str]:
    """The projected environment a launched asset process should receive."""
    return project_environment(settings, capabilities_of_tree(root))


def env_example(capabilities: Iterable[str]) -> str:
    """A secret-free .env.example for the generated project.

    Placeholders only. This file is written by Metis, not the model, so a
    generation can never leak a real value into the project tree by echoing
    its prompt back out.
    """
    lines = [
        "# Configuration for this application. Copy to .env and fill in values.",
        "# Metis injects these automatically when it launches the app; running",
        "# standalone, export them yourself. Never commit a filled-in .env.",
        "",
    ]
    for capability in sorted(set(capabilities) & KNOWN_CAPABILITIES):
        for var in CAPABILITY_VARS[capability]:
            requirement = "required" if var.required else "optional"
            lines.append(f"# {var.description} ({requirement})")
            lines.append(f"{var.name}=")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def env_documentation(capabilities: Iterable[str]) -> str:
    """Names and descriptions for a build prompt. Never values."""
    lines: list[str] = []
    for capability in sorted(set(capabilities) & KNOWN_CAPABILITIES):
        for var in CAPABILITY_VARS[capability]:
            requirement = "required" if var.required else "optional"
            lines.append(f"- {var.name} ({requirement}): {var.description}")
    return "\n".join(lines)
