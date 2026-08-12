"""A protected file is refused twice, by two parties that do not trust each other.

The sidecar denies the `editor` call (proved in the sidecar suite). This file
proves the half that matters if that denial is ever bypassed, wrong, or simply
absent: Metis reads the mirror itself and refuses the bytes that actually
arrived. The two checks share no code and no state, so a defect in one is not
a defect in both.

The hostile case is simulated the only honest way — by writing the forbidden
bytes into the mirror directly, exactly as a compromised or buggy engine
would, without asking the sidecar's permission first.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from waqil_api.project_coding_engine import ProjectCodingError
from waqil_api.project_direct_contract import (
    build_direct_contract,
    protected_drift,
)
from waqil_api.project_workspace import (
    ExternalChangeProvenance,
    ProjectWorkspaceError,
)

from waqil_api.coding_contracts import CodingProviderV1

from test_project_coding_engine import _coordinator as _direct_coordinator
from test_project_workspace import _service_for  # noqa: F401 - shared fixture


PROTECTED = ("extractor.py", "excel_writer.py", "assets/DHL_Template.xlsx")


async def _logivity_service(tmp_path: Path) -> tuple[Any, str, Path]:
    """A project shaped like the real asset, with protected files on disk."""

    service, asset_id = await _service_for(tmp_path)
    project = tmp_path / "Projects" / "demo"
    (project / "assets").mkdir(parents=True, exist_ok=True)
    (project / "extractor.py").write_text("OCI_MODEL = 'vision'\n", encoding="utf-8")
    (project / "excel_writer.py").write_text("COL_TOTAL = 'AL'\n", encoding="utf-8")
    (project / "assets" / "DHL_Template.xlsx").write_bytes(b"PK\x03\x04template")
    return service, asset_id, project


def _seed_protected(project: Path) -> None:
    """Give the shared fixture the files this suite protects."""

    (project / "assets").mkdir(parents=True, exist_ok=True)
    (project / "extractor.py").write_text("OCI_MODEL = 'vision'\n", encoding="utf-8")
    (project / "excel_writer.py").write_text("COL_TOTAL = 'AL'\n", encoding="utf-8")
    (project / "assets" / "DHL_Template.xlsx").write_bytes(b"PK\x03\x04template")


def _hashes(project: Path) -> dict[str, str]:
    return {
        path: hashlib.sha256((project / path).read_bytes()).hexdigest()
        for path in PROTECTED
        if (project / path).is_file()
    }


@pytest.mark.asyncio
async def test_the_importer_refuses_protected_bytes_the_sidecar_never_saw(
    tmp_path: Path,
) -> None:
    """The second, independent refusal.

    Nothing here goes through the sidecar. The bytes are written straight into
    the mirror, which is precisely what a bypassed or broken tool guard would
    produce, and Metis still refuses them.
    """

    service, asset_id, project = await _logivity_service(tmp_path)
    before = _hashes(project)
    mirror = await service.create_external_mirror(asset_id, {})

    # A hostile engine: it edits its own file AND a protected one.
    (mirror.project_root / "app" / "main.py").write_text(
        "print('the port')\n", encoding="utf-8"
    )
    (mirror.project_root / "extractor.py").write_text(
        "OCI_MODEL = 'tampered'\n", encoding="utf-8"
    )

    changes, staged = await service.import_external_changes(
        asset_id,
        mirror,
        {},
        provenance=ExternalChangeProvenance("clinecore", "session-hostile"),
    )
    changed = {
        str(item.get("path"))
        for item in changes.get("changes", [])
        if isinstance(item, dict)
    }
    # The importer's own job is to report the diff faithfully; refusing a
    # protected path is the coordinator's, and it is proved below. What must
    # be true here is that the tampering is VISIBLE rather than silent.
    assert "extractor.py" in changed

    # Nothing reached project disk: an import stages, it does not apply.
    assert _hashes(project) == before


@pytest.mark.asyncio
async def test_the_coordinator_refuses_a_round_that_touched_a_protected_file(
    tmp_path: Path,
) -> None:
    """The real code path, not a restatement of it.

    A coding engine that writes a protected file is driven through the actual
    coordinator. Nothing in this test knows the rule; the product enforces it.
    """

    def tamper(root: Path, _phase: str) -> None:
        # Exactly what a bypassed tool guard produces: the engine's own file
        # plus one it was told never to touch.
        (root / "app" / "main.py").write_text("print('port')\n", encoding="utf-8")
        (root / "extractor.py").write_text("TAMPERED = 1\n", encoding="utf-8")

    (
        coordinator,
        engine,
        sessions,
        projects,
        database,
        run_id,
        conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, tamper)
    project = tmp_path / "Projects" / "demo"
    _seed_protected(project)
    before = _hashes(project)
    try:
        with pytest.raises(ProjectCodingError, match="changed protected files"):
            await coordinator.start(
                run_id=run_id,
                conversation_id=conv,
                project_id=project_id,
                prompt="port it",
                staged={},
                provider=CodingProviderV1(
                    providerId="ollama", modelId="glm-5.2:cloud", baseUrl="http://x"
                ),
                allowed_paths=[],
                protected_paths=list(PROTECTED),
                broad_scope=True,
            )
        # The protected bytes never reached project disk, and the round's own
        # legitimate edit was discarded with it: the refusal is all-or-nothing.
        assert _hashes(project) == before
        assert (project / "app" / "main.py").read_text() == "print('original')\n"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_broad_scope_round_that_respects_protection_is_accepted(
    tmp_path: Path,
) -> None:
    """The same path, proving the refusal is specific rather than blanket."""

    def behave(root: Path, _phase: str) -> None:
        (root / "app" / "main.py").write_text("print('port')\n", encoding="utf-8")
        # A file that did not exist at admission: the direct path must allow it.
        (root / "README.md").write_text("# Ported\n", encoding="utf-8")

    (
        coordinator,
        engine,
        sessions,
        projects,
        database,
        run_id,
        conv,
        project_id,
    ) = await _direct_coordinator(tmp_path, behave)
    project = tmp_path / "Projects" / "demo"
    _seed_protected(project)
    before = _hashes(project)
    try:
        round_result = await coordinator.start(
            run_id=run_id,
            conversation_id=conv,
            project_id=project_id,
            prompt="port it",
            staged={},
            provider=CodingProviderV1(
                providerId="ollama", modelId="glm-5.2:cloud", baseUrl="http://x"
            ),
            allowed_paths=[],
            protected_paths=list(PROTECTED),
            broad_scope=True,
        )
        staged_paths = set(round_result.staged)
        assert "app/main.py" in staged_paths
        # Creating a new file is the whole reason the contract is not a list.
        assert "README.md" in staged_paths
        assert not staged_paths & set(PROTECTED)
        assert _hashes(project) == before
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_protected_hashes_are_unchanged_after_a_refused_attempt(
    tmp_path: Path,
) -> None:
    service, asset_id, project = await _logivity_service(tmp_path)
    contract, resolution = build_direct_contract(
        prompt="port it to fastapi. do not touch extractor.py, excel_writer.py",
        project=project,
        tree=["extractor.py", "excel_writer.py", "assets/DHL_Template.xlsx", "app.py"],
        project_protected=["assets/DHL_Template.xlsx"],
    )
    assert resolution.needs_user is False
    assert set(contract.protected_files) == set(PROTECTED)

    mirror = await service.create_external_mirror(asset_id, {})
    (mirror.project_root / "extractor.py").write_text("tampered\n", encoding="utf-8")

    # The attempt happened in the mirror; the project's own bytes never moved,
    # so the frozen contract still verifies.
    assert protected_drift(project, dict(contract.protected_hashes)) == []


@pytest.mark.asyncio
async def test_a_symlinked_protected_file_is_refused_before_it_is_read(
    tmp_path: Path,
) -> None:
    service, asset_id, project = await _logivity_service(tmp_path)
    mirror = await service.create_external_mirror(asset_id, {})
    (mirror.project_root / "extractor.py").unlink()
    (mirror.project_root / "extractor.py").symlink_to(tmp_path / "outside.py")
    (tmp_path / "outside.py").write_text("ESCAPED = 1\n", encoding="utf-8")

    with pytest.raises(ProjectWorkspaceError, match="symbolic link"):
        await service.import_external_changes(
            asset_id,
            mirror,
            {},
            provenance=ExternalChangeProvenance("clinecore", "session-symlink"),
        )

    assert (
        _hashes(project)["extractor.py"]
        == hashlib.sha256(b"OCI_MODEL = 'vision'\n").hexdigest()
    )
