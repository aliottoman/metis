#!/usr/bin/env python3
"""Create, verify, and safely extract a platform-specific Metis offline bundle.

The bundle contains the exact project lockfiles, a populated uv cache, a
populated pnpm store, the pinned uv executable used to populate the cache, and
optionally an OCI archive of the reviewed Podman sandbox image. Every member is
covered by a SHA-256 manifest. Creation validates both dependency caches by
performing an offline install before publishing the archive.

This is an environment bundle, not a source-code or Ollama-model bundle. It is
intended to accompany the exact Metis checkout recorded in its manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE_SCHEMA_VERSION = 1
MANIFEST_NAME = "bundle-manifest.json"
DEFAULT_IMAGE = "localhost/metis/reference-architecture-tool:0.3.0"
PROJECT_FILES = (
    "apps/api/pyproject.toml",
    "apps/api/uv.lock",
    "package.json",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "apps/web/package.json",
)
SANDBOX_PREREQUISITE_FILES = (
    "infra/sandbox/Containerfile",
    "infra/sandbox/containerignore",
    "infra/sandbox/sandbox-policy.json",
    "infra/sandbox/build_reference_architecture_image.sh",
    "skills/reference-architecture-generator/requirements-runtime.lock",
    "skills/reference-architecture-generator/metis.tool.json",
)
ALREADY_COMPRESSED_SUFFIXES = {
    ".bz2",
    ".gz",
    ".png",
    ".tar",
    ".tgz",
    ".whl",
    ".xz",
    ".zip",
    ".zst",
}


class BundleError(RuntimeError):
    """A bundle cannot be safely created, verified, or extracted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(handle: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        size += len(chunk)
        digest.update(chunk)
    return digest.hexdigest(), size


def _bundle_entries(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise BundleError(f"bundle staging root is not a real directory: {root}")
    entries: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            child = current_path / name
            if child.is_symlink():
                _validated_link_target(root, child)
                entries.append(child)
                directories.remove(name)
        for name in names:
            child = current_path / name
            if child.is_symlink():
                _validated_link_target(root, child)
                entries.append(child)
                continue
            if not child.is_file():
                raise BundleError(f"refusing to package non-regular file: {child}")
            mode = stat.S_IMODE(child.stat().st_mode)
            if mode & (stat.S_ISUID | stat.S_ISGID):
                raise BundleError(f"refusing to package privileged file mode: {child}")
            entries.append(child)
    return sorted(entries, key=lambda item: item.relative_to(root).as_posix().encode())


def _validated_link_target(root: Path, link: Path) -> str:
    target = os.readlink(link)
    if not target or os.path.isabs(target) or "\\" in target:
        raise BundleError(f"refusing to package unsafe symlink: {link} -> {target!r}")
    try:
        resolved = (link.parent / target).resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"refusing to package broken symlink: {link}") from exc
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise BundleError(f"symlink escapes bundle staging: {link} -> {target!r}")
    return target


def _copy_regular(source: Path, destination: Path, *, executable: bool = False) -> None:
    if source.is_symlink() or not source.is_file():
        raise BundleError(f"required regular file is unavailable: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as target:
        shutil.copyfileobj(source_handle, target, length=1024 * 1024)
    destination.chmod(0o755 if executable else 0o644)


def _resolve_executable(value: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.parent != Path(".") else None
    if resolved is None or not resolved.is_file():
        found = shutil.which(value)
        if found is None:
            raise BundleError(f"required executable is unavailable: {value}")
        resolved = Path(found).resolve()
    if not os.access(resolved, os.X_OK):
        raise BundleError(f"file is not executable: {resolved}")
    return resolved


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(command)
    print(f"+ {printable}", file=sys.stderr)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise BundleError(
            f"command failed with exit code {completed.returncode}: {printable}\n{detail}"
        )
    return completed


def _version(executable: Path, *arguments: str, cwd: Path) -> str:
    completed = _run([str(executable), *arguments], cwd=cwd)
    output = (completed.stdout or completed.stderr).strip().splitlines()
    if not output:
        raise BundleError(f"could not determine version for {executable}")
    return output[0].strip()


def _copy_project_metadata(project_root: Path, staging: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative_name in PROJECT_FILES:
        source = project_root / relative_name
        member = f"project/{relative_name}"
        destination = staging / member
        _copy_regular(source, destination)
        records[relative_name] = {
            "member": member,
            "sha256": _sha256(destination),
            "size": destination.stat().st_size,
        }
    return records


def _populate_python_cache(
    project_root: Path,
    staging: Path,
    temporary_root: Path,
    uv_binary: Path,
) -> None:
    cache = staging / "python" / "uv-cache"
    cache.mkdir(parents=True)
    environment_path = temporary_root / "python-offline-check"
    environment = os.environ.copy()
    environment.update(
        {
            "UV_CACHE_DIR": str(cache),
            "UV_PROJECT_ENVIRONMENT": str(environment_path),
            "UV_NO_PROGRESS": "1",
        }
    )
    base = [
        str(uv_binary),
        "sync",
        "--project",
        str(project_root / "apps" / "api"),
        "--frozen",
        "--extra",
        "dev",
        "--inexact",
        "--no-python-downloads",
    ]
    _run(base, cwd=project_root, env=environment)
    _run([*base, "--offline", "--check"], cwd=project_root, env=environment)
    if not any(path.is_file() for path in cache.rglob("*")):
        raise BundleError("uv reported success but produced an empty cache")


def _copy_frontend_checkout(source_root: Path, destination_root: Path) -> None:
    for relative_name in (
        "package.json",
        "pnpm-lock.yaml",
        "pnpm-workspace.yaml",
        "apps/web/package.json",
    ):
        _copy_regular(source_root / relative_name, destination_root / relative_name)


def _remove_pnpm_project_links(store: Path) -> None:
    """Remove pnpm's machine-local checkout backlinks from an otherwise portable store."""

    for projects in store.glob("v*/projects"):
        if projects.is_symlink() or not projects.is_dir():
            raise BundleError(f"pnpm projects metadata is unsafe: {projects}")
        for child in projects.iterdir():
            if not child.is_symlink():
                raise BundleError(
                    f"unexpected non-link in pnpm projects metadata: {child}"
                )
            child.unlink()
        projects.rmdir()


def _populate_frontend_store(
    project_root: Path,
    staging: Path,
    temporary_root: Path,
    pnpm_binary: Path,
) -> None:
    store = staging / "frontend" / "pnpm-store"
    store.mkdir(parents=True)
    fetch_checkout = temporary_root / "frontend-fetch-checkout"
    _copy_frontend_checkout(project_root, fetch_checkout)
    _run(
        [
            str(pnpm_binary),
            "fetch",
            "--frozen-lockfile",
            "--store-dir",
            str(store),
        ],
        cwd=fetch_checkout,
    )
    _remove_pnpm_project_links(store)
    checkout = temporary_root / "frontend-offline-check"
    _copy_frontend_checkout(project_root, checkout)
    environment = os.environ.copy()
    environment.update(
        {
            "CI": "true",
            "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0",
            "NPM_CONFIG_OFFLINE": "true",
        }
    )
    _run(
        [
            str(pnpm_binary),
            "install",
            "--offline",
            "--frozen-lockfile",
            "--store-dir",
            str(store),
        ],
        cwd=checkout,
        env=environment,
    )
    _remove_pnpm_project_links(store)
    store_files = [path for path in store.rglob("*") if path.is_file()]
    if len(store_files) < 2:
        raise BundleError("pnpm reported success but produced an incomplete store")


def _container_base_image(containerfile: Path) -> str:
    for line in containerfile.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*FROM\s+(\S+)", line, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    raise BundleError("sandbox Containerfile has no FROM instruction")


def _repository_name(image: str) -> str:
    without_digest = image.split("@", 1)[0]
    final_slash = without_digest.rfind("/")
    final_colon = without_digest.rfind(":")
    if final_colon > final_slash:
        return without_digest[:final_colon]
    return without_digest


def _sandbox_prerequisites(project_root: Path, image: str) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for name in SANDBOX_PREREQUISITE_FILES:
        source = project_root / name
        if source.is_symlink() or not source.is_file():
            raise BundleError(f"sandbox prerequisite is unavailable or unsafe: {name}")
        records[name] = {
            "sha256": _sha256(source),
            "size": source.stat().st_size,
        }
    machine = platform.machine().lower()
    architecture = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64"}.get(
        machine, machine
    )
    return {
        "schema_version": 1,
        "requested_image": image,
        "expected_platform": os.environ.get(
            "WAQIL_SANDBOX_PLATFORM", f"linux/{architecture}"
        ),
        "base_image": _container_base_image(
            project_root / "infra" / "sandbox" / "Containerfile"
        ),
        "build_command": "make sandbox-image",
        "sources": records,
        "bundled": False,
        "resolved_image": None,
        "archive_member": None,
    }


def _bundle_podman_image(
    project_root: Path,
    staging: Path,
    image: str,
    prerequisites: dict[str, Any],
) -> None:
    podman = _resolve_executable("podman")
    inspected = _run(
        [str(podman), "image", "inspect", "--format=json", image], cwd=project_root
    )
    try:
        payload = json.loads(inspected.stdout)
        record = payload[0]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise BundleError("Podman returned an invalid image inspection record") from exc
    digest = str(record.get("Digest", ""))
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise BundleError("sandbox image has no immutable repository digest")
    resolved = f"{_repository_name(image)}@{digest}"
    image_user = str((record.get("Config") or {}).get("User", ""))
    if image_user in {"", "0", "0:0", "root", "root:root"}:
        raise BundleError("sandbox image does not declare a non-root user")
    archive = staging / "podman" / "reference-architecture.oci.tar"
    archive.parent.mkdir(parents=True)
    _run(
        [
            str(podman),
            "save",
            "--format",
            "oci-archive",
            "--output",
            str(archive),
            image,
        ],
        cwd=project_root,
    )
    if not archive.is_file() or archive.stat().st_size == 0:
        raise BundleError("Podman did not create the requested OCI archive")
    prerequisites.update(
        {
            "bundled": True,
            "resolved_image": resolved,
            "archive_member": "podman/reference-architecture.oci.tar",
            "image_id": record.get("Id"),
            "image_user": image_user,
            "platform": f"{record.get('Os')}/{record.get('Architecture')}",
        }
    )


def _member_records(staging: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in _bundle_entries(staging):
        relative = path.relative_to(staging).as_posix()
        if relative == MANIFEST_NAME:
            continue
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8")
            records[relative] = {
                "type": "symlink",
                "target": payload.decode("utf-8"),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "mode": 0o777,
            }
        else:
            mode = stat.S_IMODE(path.stat().st_mode)
            records[relative] = {
                "type": "file",
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "mode": mode,
            }
    return records


def _zip_info(
    name: str, mode: int, compression: int, *, entry_type: str = "file"
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = compression
    file_type = stat.S_IFLNK if entry_type == "symlink" else stat.S_IFREG
    info.external_attr = (file_type | mode) << 16
    return info


def _write_archive(staging: Path, output: Path) -> None:
    partial = output.with_name(f".{output.name}.partial")
    if partial.exists():
        raise BundleError(f"stale partial bundle exists: {partial}")
    try:
        with zipfile.ZipFile(
            partial,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for source in _bundle_entries(staging):
                name = source.relative_to(staging).as_posix()
                is_link = source.is_symlink()
                compression = (
                    zipfile.ZIP_STORED
                    if is_link or source.suffix.lower() in ALREADY_COMPRESSED_SUFFIXES
                    else zipfile.ZIP_DEFLATED
                )
                mode = 0o777 if is_link else stat.S_IMODE(source.stat().st_mode)
                info = _zip_info(
                    name,
                    mode,
                    compression,
                    entry_type="symlink" if is_link else "file",
                )
                if is_link:
                    archive.writestr(info, os.readlink(source).encode("utf-8"))
                else:
                    with source.open("rb") as source_handle, archive.open(info, "w") as target:
                        shutil.copyfileobj(source_handle, target, length=1024 * 1024)
        partial.replace(output)
    finally:
        if partial.exists():
            partial.unlink()


def create_bundle(
    project_root: Path,
    output: Path,
    *,
    uv_value: str,
    pnpm_value: str,
    image: str,
    include_image: bool,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    output = output.resolve()
    if project_root.is_symlink() or not project_root.is_dir():
        raise BundleError(f"project root is unavailable or unsafe: {project_root}")
    if output.suffix.lower() != ".zip":
        raise BundleError("offline bundle output must use the .zip suffix")
    if output.exists():
        raise BundleError(f"refusing to overwrite existing bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    uv_binary = _resolve_executable(uv_value)
    pnpm_binary = _resolve_executable(pnpm_value)

    with tempfile.TemporaryDirectory(prefix="metis-offline-bundle-") as temporary_name:
        temporary_root = Path(temporary_name)
        staging = temporary_root / "bundle"
        staging.mkdir()
        project_records = _copy_project_metadata(project_root, staging)
        _copy_regular(uv_binary, staging / "tooling" / "uv", executable=True)

        uv_version = _version(uv_binary, "--version", cwd=project_root)
        pnpm_version = _version(pnpm_binary, "--version", cwd=project_root)
        node_binary = _resolve_executable("node")
        node_version = _version(node_binary, "--version", cwd=project_root)

        _populate_python_cache(project_root, staging, temporary_root, uv_binary)
        _populate_frontend_store(
            project_root, staging, temporary_root, pnpm_binary
        )

        prerequisites = _sandbox_prerequisites(project_root, image)
        if include_image:
            _bundle_podman_image(project_root, staging, image, prerequisites)
        prerequisites_path = staging / "podman" / "prerequisites.json"
        prerequisites_path.parent.mkdir(parents=True, exist_ok=True)
        prerequisites_path.write_text(
            json.dumps(prerequisites, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        manifest: dict[str, Any] = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_kind": "metis-platform-specific-offline-environment",
            "compatibility": {
                "sys_platform": sys.platform,
                "machine": platform.machine().lower(),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "node": node_version,
                "pnpm": pnpm_version,
                "uv": uv_version,
            },
            "project_files": project_records,
            "python": {
                "lockfile": "project/apps/api/uv.lock",
                "cache_prefix": "python/uv-cache/",
                "uv_executable": "tooling/uv",
                "install": (
                    "UV_CACHE_DIR=<bundle>/python/uv-cache "
                    "UV_PROJECT_ENVIRONMENT=<checkout>/.venv <bundle>/tooling/uv "
                    "sync --offline --project <checkout>/apps/api --frozen --extra dev "
                    "--inexact --no-python-downloads"
                ),
            },
            "frontend": {
                "lockfile": "project/pnpm-lock.yaml",
                "store_prefix": "frontend/pnpm-store/",
                "install": (
                    "pnpm install --offline --frozen-lockfile "
                    "--store-dir <bundle>/frontend/pnpm-store"
                ),
            },
            "sandbox": prerequisites,
        }
        manifest["members"] = _member_records(staging)
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_archive(staging, output)
    return manifest


def _safe_member_name(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if (
        not name
        or member.is_absolute()
        or ".." in member.parts
        or "" in member.parts
        or "\\" in name
    ):
        raise BundleError(f"unsafe bundle member name: {name!r}")
    return member


def _member_kind_and_mode(info: zipfile.ZipInfo) -> tuple[str, int]:
    raw = info.external_attr >> 16
    file_type = stat.S_IFMT(raw)
    if file_type in {0, stat.S_IFREG}:
        kind = "file"
    elif file_type == stat.S_IFLNK:
        kind = "symlink"
    else:
        raise BundleError(f"bundle contains a non-regular member: {info.filename}")
    mode = stat.S_IMODE(raw) or 0o644
    if mode & (stat.S_ISUID | stat.S_ISGID):
        raise BundleError(f"bundle member has a privileged mode: {info.filename}")
    return kind, mode


def _validate_archived_link(
    name: str, target: str, member_names: set[str]
) -> None:
    if not target or posixpath.isabs(target) or "\\" in target:
        raise BundleError(f"unsafe archived symlink: {name} -> {target!r}")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
    if resolved == ".." or resolved.startswith("../") or resolved.startswith("/"):
        raise BundleError(f"archived symlink escapes the bundle: {name} -> {target!r}")
    if resolved == "." or not any(
        candidate == resolved or candidate.startswith(f"{resolved}/")
        for candidate in member_names
    ):
        raise BundleError(f"archived symlink target is missing: {name} -> {target!r}")


def _read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    try:
        payload = archive.read(MANIFEST_NAME)
        manifest = json.loads(payload)
    except KeyError as exc:
        raise BundleError(f"bundle has no {MANIFEST_NAME}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError("bundle manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleError("unsupported offline bundle schema")
    if manifest.get("bundle_kind") != "metis-platform-specific-offline-environment":
        raise BundleError("offline bundle kind is invalid")
    if not isinstance(manifest.get("members"), dict):
        raise BundleError("offline bundle manifest has no member map")
    return manifest


def _verify_project_files(manifest: dict[str, Any], project_root: Path) -> None:
    records = manifest.get("project_files")
    members = manifest.get("members", {})
    if not isinstance(records, dict) or set(records) != set(PROJECT_FILES):
        raise BundleError("project-file manifest is incomplete")
    project_root = project_root.resolve()
    for relative_name, record in records.items():
        if not isinstance(record, dict):
            raise BundleError(f"invalid project-file record: {relative_name}")
        member = record.get("member")
        archived_record = members.get(member) if isinstance(member, str) else None
        if (
            not isinstance(archived_record, dict)
            or archived_record.get("type") != "file"
            or archived_record.get("sha256") != record.get("sha256")
            or archived_record.get("size") != record.get("size")
        ):
            raise BundleError(f"project-file member record is inconsistent: {relative_name}")
        source = project_root / relative_name
        if source.is_symlink() or not source.is_file():
            raise BundleError(f"checkout is missing recorded project file: {relative_name}")
        if _sha256(source) != record.get("sha256"):
            raise BundleError(
                f"checkout does not match the offline bundle: {relative_name}"
            )


def verify_bundle(
    path: Path,
    *,
    project_root: Path | None,
    require_image: bool,
) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"offline bundle is unavailable or unsafe: {path}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise BundleError("offline bundle contains duplicate member names")
        for info in infos:
            _safe_member_name(info.filename)
            _member_kind_and_mode(info)
        manifest = _read_manifest(archive)
        records = manifest["members"]
        if set(names) != set(records) | {MANIFEST_NAME}:
            raise BundleError("offline bundle members do not match the manifest")
        for name, record in records.items():
            if not isinstance(record, dict):
                raise BundleError(f"invalid member record: {name}")
            info = archive.getinfo(name)
            kind, mode = _member_kind_and_mode(info)
            if kind != record.get("type") or mode != record.get("mode"):
                raise BundleError(f"mode mismatch: {name}")
            with archive.open(info) as handle:
                digest, size = _sha256_stream(handle)
            if size != record.get("size"):
                raise BundleError(f"size mismatch: {name}")
            if digest != record.get("sha256"):
                raise BundleError(f"SHA-256 mismatch: {name}")
            if kind == "symlink":
                try:
                    target = archive.read(info).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise BundleError(f"symlink target is not UTF-8: {name}") from exc
                if target != record.get("target"):
                    raise BundleError(f"symlink target mismatch: {name}")
                _validate_archived_link(name, target, set(names))

        python_prefix = manifest.get("python", {}).get("cache_prefix")
        frontend_prefix = manifest.get("frontend", {}).get("store_prefix")
        if not isinstance(python_prefix, str) or not any(
            name.startswith(python_prefix) for name in records
        ):
            raise BundleError("offline bundle has no populated uv cache")
        if not isinstance(frontend_prefix, str) or not any(
            name.startswith(frontend_prefix) for name in records
        ):
            raise BundleError("offline bundle has no populated pnpm store")
        if "tooling/uv" not in records or "podman/prerequisites.json" not in records:
            raise BundleError("offline bundle is missing required tooling metadata")
        if (
            records["tooling/uv"].get("type") != "file"
            or not (int(records["tooling/uv"].get("mode", 0)) & stat.S_IXUSR)
        ):
            raise BundleError("bundled uv executable is not a regular executable")

        sandbox = manifest.get("sandbox")
        if not isinstance(sandbox, dict):
            raise BundleError("offline bundle has no sandbox record")
        try:
            prerequisite_payload = json.loads(
                archive.read("podman/prerequisites.json")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BundleError("sandbox prerequisites are not valid UTF-8 JSON") from exc
        if prerequisite_payload != sandbox:
            raise BundleError("sandbox prerequisite record does not match the manifest")
        bundled = sandbox.get("bundled") is True
        archive_member = sandbox.get("archive_member")
        if bundled:
            if archive_member != "podman/reference-architecture.oci.tar":
                raise BundleError("sandbox OCI archive member is invalid")
            if archive_member not in records:
                raise BundleError("sandbox OCI archive is missing")
            if not re.fullmatch(
                r"[^@]+@sha256:[a-f0-9]{64}", str(sandbox.get("resolved_image", ""))
            ):
                raise BundleError("sandbox image is not digest-pinned")
        elif archive_member is not None:
            raise BundleError("prerequisite-only bundle declares an unexpected image archive")
        if require_image and not bundled:
            raise BundleError("offline bundle does not contain the sandbox OCI image")

    if project_root is not None:
        _verify_project_files(manifest, project_root)
    return manifest


def extract_bundle(
    path: Path,
    destination: Path,
    *,
    project_root: Path | None,
    require_image: bool,
) -> dict[str, Any]:
    manifest = verify_bundle(
        path, project_root=project_root, require_image=require_image
    )
    destination = destination.resolve()
    if destination.exists():
        raise BundleError(f"refusing to overwrite extraction destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise BundleError(f"stale extraction directory exists: {temporary}")
    temporary.mkdir()
    try:
        with zipfile.ZipFile(path.resolve()) as archive:
            infos = sorted(archive.infolist(), key=lambda item: item.filename)
            for info in infos:
                member = _safe_member_name(info.filename)
                kind, mode = _member_kind_and_mode(info)
                if kind == "symlink":
                    continue
                target = temporary.joinpath(*member.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                target.chmod(mode)
            for info in infos:
                member = _safe_member_name(info.filename)
                kind, _ = _member_kind_and_mode(info)
                if kind != "symlink":
                    continue
                target = archive.read(info).decode("utf-8")
                _validate_archived_link(
                    info.filename, target, {item.filename for item in infos}
                )
                link_path = temporary.joinpath(*member.parts)
                link_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(target, link_path)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _compatibility_check(manifest: dict[str, Any], project_root: Path) -> None:
    expected = manifest.get("compatibility", {})
    actual = {
        "sys_platform": sys.platform,
        "machine": platform.machine().lower(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise BundleError(
                f"bundle host mismatch for {key}: expected {expected.get(key)!r}, got {value!r}"
            )
    node = _resolve_executable("node")
    pnpm = _resolve_executable("pnpm")
    if _version(node, "--version", cwd=project_root) != expected.get("node"):
        raise BundleError("Node.js version does not match the bundle creator")
    if _version(pnpm, "--version", cwd=project_root) != expected.get("pnpm"):
        raise BundleError("pnpm version does not match the bundle creator")


def smoke_install(path: Path, project_root: Path, *, require_image: bool) -> dict[str, Any]:
    project_root = project_root.resolve()
    manifest = verify_bundle(
        path, project_root=project_root, require_image=require_image
    )
    _compatibility_check(manifest, project_root)
    with tempfile.TemporaryDirectory(prefix="metis-offline-install-check-") as temporary_name:
        temporary = Path(temporary_name)
        extracted = temporary / "bundle"
        extract_bundle(
            path,
            extracted,
            project_root=project_root,
            require_image=require_image,
        )
        uv = extracted / "tooling" / "uv"
        uv_environment = os.environ.copy()
        uv_environment.update(
            {
                "UV_CACHE_DIR": str(extracted / "python" / "uv-cache"),
                "UV_PROJECT_ENVIRONMENT": str(temporary / "python-environment"),
                "UV_NO_PROGRESS": "1",
            }
        )
        _run(
            [
                str(uv),
                "sync",
                "--offline",
                "--project",
                str(project_root / "apps" / "api"),
                "--frozen",
                "--extra",
                "dev",
                "--inexact",
                "--no-python-downloads",
            ],
            cwd=project_root,
            env=uv_environment,
        )
        frontend_checkout = temporary / "frontend-checkout"
        _copy_frontend_checkout(extracted / "project", frontend_checkout)
        pnpm = _resolve_executable("pnpm")
        frontend_environment = os.environ.copy()
        frontend_environment.update(
            {
                "CI": "true",
                "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0",
                "NPM_CONFIG_OFFLINE": "true",
            }
        )
        _run(
            [
                str(pnpm),
                "install",
                "--offline",
                "--frozen-lockfile",
                "--store-dir",
                str(extracted / "frontend" / "pnpm-store"),
            ],
            cwd=frontend_checkout,
            env=frontend_environment,
        )
    return manifest


def _summary(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    sandbox = manifest["sandbox"]
    return {
        "path": str(path.resolve()),
        "members": len(manifest["members"]),
        "sandbox_image_bundled": sandbox["bundled"],
        "sandbox_image": sandbox.get("resolved_image") or sandbox["requested_image"],
        "compatibility": manifest["compatibility"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="create a verified offline bundle")
    create_parser.add_argument("--output", type=Path, required=True)
    create_parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    create_parser.add_argument("--uv-bin", default=".venv/bin/uv")
    create_parser.add_argument("--pnpm-bin", default="pnpm")
    create_parser.add_argument("--image", default=os.environ.get("WAQIL_REFERENCE_RUNNER_IMAGE", DEFAULT_IMAGE))
    create_parser.add_argument(
        "--without-image",
        action="store_true",
        help="record exact sandbox prerequisites but do not include an OCI image archive",
    )

    verify_parser = subparsers.add_parser("verify", help="verify hashes, locks, and compatibility metadata")
    verify_parser.add_argument("path", type=Path)
    verify_parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    verify_parser.add_argument("--no-project-check", action="store_true")
    verify_parser.add_argument("--require-image", action="store_true")
    verify_parser.add_argument("--smoke-install", action="store_true")

    extract_parser = subparsers.add_parser("extract", help="verify and safely extract a bundle")
    extract_parser.add_argument("path", type=Path)
    extract_parser.add_argument("--destination", type=Path, required=True)
    extract_parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    extract_parser.add_argument("--no-project-check", action="store_true")
    extract_parser.add_argument("--require-image", action="store_true")

    arguments = parser.parse_args()
    try:
        if arguments.command == "create":
            manifest = create_bundle(
                arguments.project_root,
                arguments.output,
                uv_value=arguments.uv_bin,
                pnpm_value=arguments.pnpm_bin,
                image=arguments.image,
                include_image=not arguments.without_image,
            )
            path = arguments.output
        elif arguments.command == "verify":
            project_root = None if arguments.no_project_check else arguments.project_root
            if arguments.smoke_install:
                if project_root is None:
                    raise BundleError("--smoke-install requires checkout verification")
                manifest = smoke_install(
                    arguments.path, project_root, require_image=arguments.require_image
                )
            else:
                manifest = verify_bundle(
                    arguments.path,
                    project_root=project_root,
                    require_image=arguments.require_image,
                )
            path = arguments.path
        else:
            project_root = None if arguments.no_project_check else arguments.project_root
            manifest = extract_bundle(
                arguments.path,
                arguments.destination,
                project_root=project_root,
                require_image=arguments.require_image,
            )
            path = arguments.destination
    except (BundleError, FileNotFoundError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"offline bundle error: {exc}\n")
    print(json.dumps(_summary(path, manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
