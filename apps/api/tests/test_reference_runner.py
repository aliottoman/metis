from __future__ import annotations

import json
import shutil

import pytest

from waqil_api.contracts import (
    ArchitectureBoundaryV1,
    ArchitectureComponentV1,
    ArchitectureEdgeV1,
    ArchitectureSpecV1,
)
from waqil_api.reference_architecture import (
    ReferenceArchitectureRunner,
    ReferenceRunnerError,
)


@pytest.mark.asyncio
async def test_production_runner_has_no_host_fallback(settings, tmp_path) -> None:
    isolated = settings.model_copy(
        update={
            "repo_root": tmp_path / "empty-repository",
            "reference_runner_mode": "podman",
            "allow_test_backends": False,
        }
    )
    isolated.prepare_directories()
    runner = ReferenceArchitectureRunner(isolated)
    spec = ArchitectureSpecV1(
        title="Test",
        components=[ArchitectureComponentV1(id="service", label="Service", kind="service")],
        edges=[],
    )
    with pytest.raises(ReferenceRunnerError, match="sandbox runner unavailable"):
        await runner.run("run_test", "draw it", spec)


@pytest.mark.asyncio
async def test_digest_pinned_image_is_not_re_resolved(settings) -> None:
    runner = ReferenceArchitectureRunner(settings)
    pinned = "localhost/metis/reference-architecture-tool@sha256:" + "a" * 64
    assert await runner.resolve_image_ref(pinned) == pinned


def test_deployment_hash_binds_manifest_infra_and_image(settings) -> None:
    runner = ReferenceArchitectureRunner(settings)
    manifest = runner.portable_manifest()
    assert manifest["input_schema"]["required"] == ["schema_version", "spec"]
    one = runner.bundle_hash("image@sha256:" + "a" * 64)
    two = runner.bundle_hash("image@sha256:" + "b" * 64)
    assert one != two


def test_portable_integrity_ignores_transient_python_caches(settings, tmp_path) -> None:
    source_root = settings.repo_root
    isolated_root = tmp_path / "repo"
    shutil.copytree(
        source_root / "skills" / "reference-architecture-generator",
        isolated_root / "skills" / "reference-architecture-generator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    (isolated_root / "infra" / "sandbox").mkdir(parents=True)
    for name in ("run_reference_architecture.py", "sandbox-policy.json", "Containerfile"):
        shutil.copy2(
            source_root / "infra" / "sandbox" / name,
            isolated_root / "infra" / "sandbox" / name,
        )
    runner = ReferenceArchitectureRunner(settings.model_copy(update={"repo_root": isolated_root}))
    image = "image@sha256:" + "a" * 64
    before = runner.bundle_hash(image)
    cache = runner.settings.reference_skill_dir / "src" / "__pycache__"
    cache.mkdir()
    (cache / "validator.cpython-313.pyc").write_bytes(b"transient bytecode")
    (runner.settings.reference_skill_dir / ".DS_Store").write_bytes(b"transient metadata")
    build = runner.settings.reference_skill_dir / "build"
    build.mkdir()
    (build / "generated.whl").write_bytes(b"transient build output")
    after = runner.bundle_hash(image)
    assert after == before


@pytest.mark.asyncio
async def test_approved_snapshot_survives_live_source_mutation(settings, tmp_path) -> None:
    source_root = settings.repo_root
    isolated_root = tmp_path / "repo"
    shutil.copytree(
        source_root / "skills" / "reference-architecture-generator",
        isolated_root / "skills" / "reference-architecture-generator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    (isolated_root / "infra" / "sandbox").mkdir(parents=True)
    for name in ("run_reference_architecture.py", "sandbox-policy.json", "Containerfile"):
        shutil.copy2(
            source_root / "infra" / "sandbox" / name,
            isolated_root / "infra" / "sandbox" / name,
        )
    isolated_settings = settings.model_copy(update={"repo_root": isolated_root})
    isolated_settings.prepare_directories()
    runner = ReferenceArchitectureRunner(isolated_settings)
    image = "image@sha256:" + "c" * 64
    approved_hash = runner.bundle_hash(image)
    snapshot = await runner.create_snapshot(approved_hash, image)

    live_policy = isolated_root / "infra" / "sandbox" / "sandbox-policy.json"
    live_policy.write_text(live_policy.read_text() + "\n", encoding="utf-8")
    assert runner.bundle_hash(image) != approved_hash
    assert runner.verify_snapshot(str(snapshot), approved_hash, image) == snapshot


@pytest.mark.asyncio
async def test_runner_canonicalizes_model_order_before_exact_spec_validation(settings) -> None:
    runner = ReferenceArchitectureRunner(settings)
    spec = ArchitectureSpecV1(
        title="Unsorted architecture",
        components=[
            ArchitectureComponentV1(id="service", label="Service", kind="service"),
            ArchitectureComponentV1(id="client", label="Client", kind="client"),
            ArchitectureComponentV1(id="database", label="Database", kind="database"),
        ],
        edges=[
            ArchitectureEdgeV1(source="service", target="database", label="SQL"),
            ArchitectureEdgeV1(source="client", target="service", label="HTTPS"),
        ],
        boundaries=[
            ArchitectureBoundaryV1(
                id="application",
                label="Application",
                component_ids=["service", "client"],
            )
        ],
    )
    output = await runner.run("run_order", "draw it", spec)
    artifact = json.loads(
        next(path for path in output.files if path.name == "architecture-spec.json").read_text()
    )["spec"]
    assert [item["id"] for item in artifact["components"]] == [
        "client",
        "database",
        "service",
    ]
    assert artifact["boundaries"][0]["component_ids"] == ["client", "service"]
    assert output.eval_report.static_checks["spec_exact_match"] is True
