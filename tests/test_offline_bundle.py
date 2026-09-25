"""Focused offline-bundle tests for the local Cline coding sidecar."""

from __future__ import annotations

import json
import importlib.util
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "offline_bundle.py"
_SPEC = importlib.util.spec_from_file_location("metis_offline_bundle", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
offline_bundle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(offline_bundle)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_sidecar_contract_requires_exact_sdk_and_compiler_versions() -> None:
    package = json.loads(
        (_repo_root() / offline_bundle.SIDECAR_PACKAGE_FILE).read_text(encoding="utf-8")
    )
    contract = offline_bundle._sidecar_contract_from_package(package)

    assert contract["sdk_version"] == "0.0.86"
    assert contract["typescript_version"] == "6.0.2"
    assert contract["compiled_entrypoint"] == ("apps/cline-sidecar/dist/src/index.js")
    assert contract["offline_compile_verified"] is True

    for dependency, version in (("@cline/sdk", "^0.0.86"), ("typescript", "latest")):
        floating = deepcopy(package)
        section = "dependencies" if dependency == "@cline/sdk" else "devDependencies"
        floating[section][dependency] = version
        with pytest.raises(offline_bundle.BundleError, match="exact version"):
            offline_bundle._sidecar_contract_from_package(floating)

    unsupported_node = deepcopy(package)
    unsupported_node["engines"]["node"] = "*"
    with pytest.raises(offline_bundle.BundleError, match="Node 22"):
        offline_bundle._sidecar_contract_from_package(unsupported_node)


def test_frontend_checkout_and_build_inputs_include_coding_adapters(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    offline_bundle._copy_frontend_checkout(_repo_root(), checkout)
    offline_bundle._copy_sidecar_build_sources(_repo_root(), checkout)
    offline_bundle._copy_cursor_shadow_build_sources(_repo_root(), checkout)

    assert (checkout / "apps/cline-sidecar/package.json").is_file()
    assert (checkout / "apps/cline-sidecar/tsconfig.json").is_file()
    assert (checkout / "apps/cline-sidecar/src/index.ts").is_file()
    assert (checkout / "apps/cline-sidecar/tests/protocol.test.ts").is_file()
    assert not (checkout / "apps/cline-sidecar/dist").exists()
    assert "apps/cline-sidecar/package.json" in offline_bundle.PROJECT_FILES
    assert (checkout / "apps/cursor-shadow/package.json").is_file()
    assert (checkout / "apps/cursor-shadow/tsconfig.json").is_file()
    assert (checkout / "apps/cursor-shadow/src/index.ts").is_file()
    assert (checkout / "apps/cursor-shadow/tests/index.test.ts").is_file()
    assert not (checkout / "apps/cursor-shadow/dist").exists()
    assert "apps/cursor-shadow/package.json" in offline_bundle.PROJECT_FILES


def test_archived_sidecar_contract_must_match_its_package(tmp_path: Path) -> None:
    package_path = _repo_root() / offline_bundle.SIDECAR_PACKAGE_FILE
    package = json.loads(package_path.read_text(encoding="utf-8"))
    member = f"project/{offline_bundle.SIDECAR_PACKAGE_FILE}"
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as output:
        output.writestr(member, json.dumps(package))

    manifest = {
        "members": {member: {"type": "file"}},
        "coding_sidecar": offline_bundle._sidecar_contract_from_package(package),
    }
    with zipfile.ZipFile(archive_path) as archive:
        offline_bundle._verify_sidecar_contract(archive, manifest)
        manifest["coding_sidecar"] = {
            **manifest["coding_sidecar"],
            "sdk_version": "0.0.85",
        }
        with pytest.raises(offline_bundle.BundleError, match="does not match"):
            offline_bundle._verify_sidecar_contract(archive, manifest)


def test_compiled_sidecar_entrypoint_is_required(tmp_path: Path) -> None:
    contract = offline_bundle._sidecar_contract(_repo_root())
    with pytest.raises(offline_bundle.BundleError, match="did not produce"):
        offline_bundle._require_compiled_sidecar(tmp_path, contract)

    entrypoint = tmp_path / contract["compiled_entrypoint"]
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("export {};\n", encoding="utf-8")
    offline_bundle._require_compiled_sidecar(tmp_path, contract)


def test_compiled_cursor_shadow_entrypoint_is_required(tmp_path: Path) -> None:
    with pytest.raises(offline_bundle.BundleError, match="did not produce"):
        offline_bundle._require_compiled_cursor_shadow(tmp_path)

    entrypoint = tmp_path / offline_bundle.CURSOR_SHADOW_ENTRYPOINT
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("export {};\n", encoding="utf-8")
    offline_bundle._require_compiled_cursor_shadow(tmp_path)
