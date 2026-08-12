"""The deterministic scaffold: what gets vendored, staged, refused, applied.

The invariants pinned here are the P0 contract: the vendored bytes are the
tested bytes, seeding is idempotent and disk-respecting, models cannot write
under appkit/ while the framework can, and the prompt note describes exactly
what the project actually carries.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from waqil_api.asset_library import AssetManager
from waqil_api.config import Settings
from waqil_api.contracts import ProjectToolCallV1
from waqil_api.model_provider import DeterministicModelProvider
from waqil_api.project_scaffold import (
    build_capabilities,
    scaffold_note,
    scaffold_prompt,
    scaffold_sources,
    wants_oci_responses,
    wants_web_ui,
)
from waqil_api.project_workspace import ProjectWorkspaceError, ProjectWorkspaceService


async def _service(tmp_path: Path) -> tuple[ProjectWorkspaceService, str, Path]:
    projects_root = tmp_path / "Projects"
    project = projects_root / "ledger"
    (project / "app").mkdir(parents=True)
    (project / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n", encoding="utf-8"
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
    )
    assets = AssetManager(
        settings.asset_roots,
        approval_path=settings.asset_approval_path,
        catalog_path=settings.asset_catalog_path,
    )
    discovered = await assets.scan()
    service = ProjectWorkspaceService(settings, assets, DeterministicModelProvider())
    return service, discovered[0].id, project


def test_scaffold_sources_gate_the_oci_module_and_env_example() -> None:
    base = scaffold_sources(frozenset())
    assert set(base) == {
        "appkit/__init__.py",
        "appkit/config.py",
        "appkit/money.py",
        "appkit/uploads.py",
    }
    with_oci = scaffold_sources({"oci_responses"})
    assert "appkit/oci_responses.py" in with_oci
    assert ".env.example" in with_oci
    # The vendored bytes are the tested bytes — and they parse.
    for path, content in with_oci.items():
        if path.endswith(".py"):
            ast.parse(content, filename=path)
    # The env example documents names only, never values.
    for line in with_oci[".env.example"].splitlines():
        if "=" in line and not line.startswith("#"):
            assert line.endswith("=")


def test_wants_oci_responses_reads_extraction_intent() -> None:
    assert wants_oci_responses(
        "Build an app where I upload supplier invoice images and Grok extracts "
        "the structured fields"
    )
    assert wants_oci_responses("Create a tool that does OCR on receipts")
    assert not wants_oci_responses("Build a todo list web app with FastAPI")
    assert not wants_oci_responses("Create a photo gallery site with albums")


@pytest.mark.asyncio
async def test_stage_scaffold_upgrades_owned_files_with_a_drift_guard(
    tmp_path: Path,
) -> None:
    service, asset_id, project = await _service(tmp_path)
    (project / "appkit").mkdir()
    old_init = 'SCAFFOLD_VERSION = "0.1.0"\n'
    old_money = "LEGACY = True\n"
    (project / "appkit" / "__init__.py").write_text(old_init, encoding="utf-8")
    (project / "appkit" / "money.py").write_text(old_money, encoding="utf-8")
    # Optional modules already carried by the scaffold remain capabilities on
    # a later turn, even when this turn's prompt does not mention them.
    (project / "appkit" / "oci_responses.py").write_text(
        "OLD_OCI = True\n", encoding="utf-8"
    )
    (project / "appkit" / "web.py").write_text("OLD_WEB = True\n", encoding="utf-8")
    (project / "appkit" / "extra.py").write_text("mine = True\n", encoding="utf-8")
    (project / ".env.example").write_text("MY_SETTING=\n", encoding="utf-8")

    staged, added = await service.stage_scaffold(asset_id, {}, set())
    assert {
        "appkit/__init__.py",
        "appkit/money.py",
        "appkit/oci_responses.py",
        "appkit/web.py",
        "appkit/static/theme.css",
    }.issubset(added)
    assert ".env.example" not in staged
    assert "appkit/extra.py" not in staged

    entry = staged["appkit/money.py"]
    assert entry["origin"] == "patch"
    assert entry["base_sha256"] == hashlib.sha256(old_money.encode("utf-8")).hexdigest()
    assert "LEGACY" not in entry["content"]
    init_entry = staged["appkit/__init__.py"]
    assert init_entry["origin"] == "patch"
    assert (
        init_entry["base_sha256"]
        == hashlib.sha256(old_init.encode("utf-8")).hexdigest()
    )
    assert 'SCAFFOLD_VERSION = "0.2.0"' in init_entry["content"]
    assert staged["appkit/oci_responses.py"]["origin"] == "patch"
    assert staged["appkit/web.py"]["origin"] == "patch"
    assert staged["appkit/static/theme.css"]["origin"] == "create"
    summary, _, _ = service.staged_summary(staged)
    assert "modify `appkit/money.py`" in summary

    again, added_again = await service.stage_scaffold(asset_id, staged, set())
    assert added_again == []
    assert again == staged

    # Approval only covers the bytes that were visible when staged. A later
    # disk edit is reported and preserved instead of being overwritten.
    drifted = "CHANGED_AFTER_STAGING = True\n"
    (project / "appkit" / "money.py").write_text(drifted, encoding="utf-8")
    outcome = await service.materialize_staged(asset_id, staged)
    assert {item["path"] for item in outcome["skipped"]} == {"appkit/money.py"}
    assert (project / "appkit" / "money.py").read_text(encoding="utf-8") == drifted
    assert (project / ".env.example").read_text(encoding="utf-8") == "MY_SETTING=\n"
    assert (project / "appkit" / "extra.py").read_text(
        encoding="utf-8"
    ) == "mine = True\n"

    retry, retry_added = await service.stage_scaffold(asset_id, {}, set())
    assert retry_added == ["appkit/money.py"]
    assert (await service.materialize_staged(asset_id, retry))["skipped"] == []
    settled, settled_added = await service.stage_scaffold(asset_id, {}, set())
    assert settled_added == []
    assert settled == {}


@pytest.mark.asyncio
async def test_model_writes_under_appkit_are_refused_reads_pass(tmp_path: Path) -> None:
    service, asset_id, _ = await _service(tmp_path)
    staged, _ = await service.stage_scaffold(asset_id, {}, set())

    with pytest.raises(ProjectWorkspaceError, match="Metis-owned"):
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="create_file",
                arguments={"path": "appkit/mine.py", "content": "x = 1\n"},
            ),
            staged,
        )
    with pytest.raises(ProjectWorkspaceError, match="Metis-owned"):
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="apply_patch",
                arguments={
                    "path": "appkit/money.py",
                    "original": "CENT",
                    "replacement": "PENNY",
                },
            ),
            staged,
        )
    # Reading the scaffold is how the model learns its contracts.
    result, overlay = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(name="read_file", arguments={"path": "appkit/money.py"}),
        staged,
    )
    assert overlay is None
    assert "within_percent" in result["content"]


@pytest.mark.asyncio
async def test_materialize_applies_scaffold_entries(tmp_path: Path) -> None:
    service, asset_id, project = await _service(tmp_path)
    staged, added = await service.stage_scaffold(asset_id, {}, set())
    outcome = await service.materialize_staged(asset_id, staged)
    assert sorted(outcome["applied"]) == sorted(added)
    assert outcome["skipped"] == []
    assert (project / "appkit" / "money.py").is_file()
    assert (project / "appkit" / "__init__.py").is_file()


@pytest.mark.asyncio
async def test_inspect_api_reaches_the_vendored_appkit(tmp_path: Path) -> None:
    """The question the model actually asks when it misuses a helper.

    appkit is vendored, not installed, so the probe — which runs with no
    project on its path — could not see it and told the model to "declare it
    in requirements". Measured live: a repair turn asked three times, was
    refused, and never fixed the call it had correctly diagnosed.
    """
    service, asset_id, _ = await _service(tmp_path)
    found = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="inspect_api",
            arguments={"module": "appkit.money", "symbol": "sum_money"},
        ),
        {},
    )
    result = found[0]
    # The real signature, which is what the misuse needed: one argument.
    assert result["has_symbol"] is True
    assert result["signature"].startswith("sum_money(values")
    assert "Decimal" in result["signature"]
    # Reported under the name the project imports, not Metis's internal path.
    assert result["module"] == "appkit.money"
    assert "waqil_api" not in result["module"]


@pytest.mark.asyncio
async def test_inspect_api_still_refuses_the_projects_own_modules(
    tmp_path: Path,
) -> None:
    """The redirect is for appkit only; read_file still owns project code."""
    service, asset_id, _ = await _service(tmp_path)
    with pytest.raises(ProjectWorkspaceError):
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="inspect_api", arguments={"module": "definitely_not_real_pkg"}
            ),
            {},
        )


def test_scaffold_prompt_describes_what_the_project_carries() -> None:
    assert scaffold_prompt({}, {}) == ""
    base_note = scaffold_prompt({"appkit/__init__.py": {}}, {})
    assert "appkit.money" in base_note
    assert "OCI_RESPONSES_PROJECT_ID" not in base_note
    oci_note = scaffold_prompt(
        {}, {"manifest": {"file_tree": ["appkit/oci_responses.py", "app/main.py"]}}
    )
    assert "appkit.oci_responses" in oci_note
    assert "OCI_RESPONSES_PROJECT_ID" in oci_note
    assert "never" in oci_note.lower()


def test_a_page_serving_build_is_vendored_the_design_language() -> None:
    """The complaint this closes: every build invented its own visual style,
    because "follow the frontend design language" shipped no design language
    behind the words. A build that renders anything now receives one."""
    capabilities = build_capabilities(
        "Build a FastAPI web app with a clean UI for uploading diagrams"
    )
    assert "web_ui" in capabilities
    sources = scaffold_sources(capabilities)
    assert "appkit/web.py" in sources
    assert "appkit/static/theme.css" in sources


def test_a_json_only_build_is_not_given_a_stylesheet() -> None:
    """Narrow enough that a backend with no pages stays unthemed."""
    assert not wants_web_ui("Build a REST service with endpoints for invoices")
    assert not wants_web_ui("a CLI tool that renames files")
    sources = scaffold_sources(build_capabilities("Build a CLI that renames files"))
    assert "appkit/static/theme.css" not in sources


def test_a_files_only_capability_earns_no_env_example() -> None:
    """web_ui projects no variables, so a page-only build must not receive a
    .env.example carrying a header and nothing under it — that reads as a
    configuration step the app does not actually have."""
    sources = scaffold_sources(frozenset({"web_ui"}))
    assert ".env.example" not in sources
    assert ".env.example" in scaffold_sources(frozenset({"oci_responses"}))


def test_the_scaffold_note_hands_over_the_theme_and_its_vocabulary() -> None:
    """A vendored stylesheet nothing links is invisible. The note has to carry
    the mount, the link and the class names, or the model composes its own."""
    note = scaffold_note(has_oci=False, has_web=True)
    assert "mount_appkit_static" in note
    assert "/appkit/theme.css" in note
    for klass in (".page", ".eyebrow", ".card", ".btn-primary", ".chip", ".dropzone"):
        assert klass in note, klass
    # And the prohibition, in as many words.
    assert "no CSS framework" in note or "no CDN" in note

    silent = scaffold_note(has_oci=False, has_web=False)
    assert "theme.css" not in silent


def test_the_scaffold_note_gives_the_exact_upload_contract() -> None:
    note = scaffold_note(has_oci=False)
    for public_name in (
        "IMAGE_MIMES",
        "DOCUMENT_MIMES",
        "UploadError",
        "SavedUpload",
        "sniff_mime(data)",
    ):
        assert public_name in note
    assert "max_bytes=10 * 1024 * 1024, allowed_mimes=IMAGE_MIMES" in note
    assert "default is image-only" in note
    assert "save_upload(upload, allowed_mimes=DOCUMENT_MIMES)" in note
    assert "PDF, and plain UTF-8 text" in note
    assert "HTTP 415" in note
    assert "SUPPORTED_MIME_TYPES" not in note
    assert "await save_upload(upload) ->" not in note


def test_the_vendored_theme_is_the_metis_palette_not_a_generic_one() -> None:
    """Pins the identity itself. These exact values are the workspace's own —
    warm greige paper, the one hairline, the lavender-grey muted ink, the
    purple field — so a well-meaning rewrite to some other palette fails here
    rather than shipping quietly into every future build."""
    css = scaffold_sources(frozenset({"web_ui"}))["appkit/static/theme.css"]
    for token in (
        "#f7f4ef",
        "#ede9e1",
        "#211f1d",
        "#7b7789",
        "#dfdacf",
        "#c29ce0",
        "#39594d",
    ):
        assert token in css, token
    # Flat panels: the blur was retired for legibility and must stay retired.
    assert "backdrop-filter" not in css
    # No web fonts and no CDN, so a page renders identically offline. The test
    # is for FETCHES, not for the string "http": the grain's inline SVG carries
    # an xmlns, which is an identifier the browser never requests.
    assert "@import" not in css
    assert "url(http" not in css.replace(" ", "")
    assert "fonts.googleapis" not in css
    assert "prefers-reduced-motion" in css
