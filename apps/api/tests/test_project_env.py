"""The projection seam between Metis settings and generated applications.

The properties under test are the boundary rules: a project receives only the
variables its used capabilities justify, a blocked capability yields a policy
answer rather than a half-populated environment, and nothing value-shaped ever
appears in the artifacts a model or a git tree can see.
"""

from __future__ import annotations

from waqil_api.config import Settings
from waqil_api.project_env import (
    capability_blocked,
    detect_capabilities,
    env_documentation,
    env_example,
    missing_required,
    project_environment,
)

PROJECT_ID = "ocid1.aiproject.oc1.test.example"


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_projection_contains_only_capability_vars_with_values() -> None:
    settings = _settings(
        allow_oci_responses=True, oci_responses_project_id=PROJECT_ID
    )
    environment = project_environment(settings, {"oci_responses"})
    assert environment["OCI_RESPONSES_PROJECT_ID"] == PROJECT_ID
    assert environment["OCI_RESPONSES_MODEL_ID"] == "xai.grok-4.3"
    assert environment["OCI_PROFILE"] == "DEFAULT"
    assert "OCI_RESPONSES_BASE_URL" in environment
    # Empty settings project nothing rather than an empty string.
    assert "OCI_CONFIG_FILE" not in environment
    # Nothing outside the capability's allowlist leaks through.
    assert all(name.startswith("OCI_") for name in environment)


def test_no_capabilities_projects_nothing() -> None:
    settings = _settings(
        allow_oci_responses=True, oci_responses_project_id=PROJECT_ID
    )
    assert project_environment(settings, set()) == {}


def test_blocked_capability_projects_nothing_and_names_the_switch() -> None:
    settings = _settings(
        allow_oci_responses=False, oci_responses_project_id=PROJECT_ID
    )
    assert project_environment(settings, {"oci_responses"}) == {}
    reason = capability_blocked(settings, "oci_responses")
    assert "WAQIL_ALLOW_OCI_RESPONSES" in reason


def test_unknown_capability_is_refused_not_ignored() -> None:
    settings = _settings(allow_oci_responses=True)
    assert "unknown capability" in capability_blocked(settings, "quantum")
    assert project_environment(settings, {"quantum"}) == {}
    assert missing_required(settings, {"quantum"}) == []


def test_missing_required_names_the_projected_variable() -> None:
    settings = _settings(allow_oci_responses=True)  # project id left empty
    assert missing_required(settings, {"oci_responses"}) == [
        "OCI_RESPONSES_PROJECT_ID"
    ]


def test_env_example_lists_names_but_never_values() -> None:
    text = env_example({"oci_responses"})
    assert "OCI_RESPONSES_PROJECT_ID=" in text
    assert "OCI_RESPONSES_MODEL_ID=" in text
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            assert line.endswith("="), f"placeholder line carries a value: {line}"


def test_env_documentation_names_requirements_without_values() -> None:
    text = env_documentation({"oci_responses"})
    assert "OCI_RESPONSES_PROJECT_ID (required)" in text
    assert "OCI_CONFIG_FILE (optional)" in text
    assert PROJECT_ID not in text


def test_detect_capabilities_from_adapter_use() -> None:
    assert detect_capabilities(
        {"app/extract.py": "from appkit.oci_responses import extract_document\n"}
    ) == frozenset({"oci_responses"})
    assert detect_capabilities(
        {"app/extract.py": "from appkit import config, oci_responses\n"}
    ) == frozenset({"oci_responses"})
    assert detect_capabilities(
        {"app/main.py": "import appkit.oci_responses\n\nclient = appkit.oci_responses\n"}
    ) == frozenset({"oci_responses"})


def test_detect_capabilities_ignores_plain_apps_and_non_python() -> None:
    assert detect_capabilities(
        {"app/main.py": "from fastapi import FastAPI\napp = FastAPI()\n"}
    ) == frozenset()
    # The marker inside a README is prose, not a capability.
    assert detect_capabilities(
        {"README.md": "uses appkit.oci_responses under the hood"}
    ) == frozenset()
