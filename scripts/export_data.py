#!/usr/bin/env python3
"""Create or verify a portable, integrity-checked Metis data export.

SQLite databases are copied through SQLite's online backup API so WAL-backed
databases remain consistent while Metis is running. Blob and immutable tool
bundle files are copied without following symlinks, then every archived member
is recorded in a SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


EXPORT_VERSION = 1
DATABASES = ("waqil.db", "checkpoints.db")
DATA_DIRECTORIES = ("blobs", "tool-bundles")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_regular_tree(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"export source must be a real directory: {root}")
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"refusing to export symlinked directory: {child}")
        for name in names:
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"refusing to export symlinked file: {child}")
            if not child.is_file():
                raise ValueError(f"refusing to export non-regular file: {child}")
            files.append(child)
    return sorted(files)


def _backup_database(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"required database is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"file:{source.resolve()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source_db:
        with sqlite3.connect(destination) as destination_db:
            source_db.backup(destination_db)
            status = destination_db.execute("PRAGMA integrity_check").fetchone()
            if status is None or status[0] != "ok":
                raise RuntimeError(f"SQLite integrity check failed for {source.name}")


def create_export(data_dir: Path, output: Path) -> dict[str, object]:
    data_dir = data_dir.resolve()
    output = output.resolve()
    if not data_dir.is_dir() or data_dir.is_symlink():
        raise ValueError(f"data directory does not exist or is unsafe: {data_dir}")
    if output.suffix.lower() != ".zip":
        raise ValueError("export output must use the .zip suffix")
    if output == data_dir or data_dir in output.parents:
        raise ValueError("export output must be outside the Metis data directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="metis-export-") as temporary:
        staging = Path(temporary) / "metis-export"
        staging.mkdir()
        for name in DATABASES:
            _backup_database(data_dir / name, staging / name)

        for directory_name in DATA_DIRECTORIES:
            source_root = data_dir / directory_name
            for source in _assert_regular_tree(source_root):
                relative = source.relative_to(data_dir)
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as source_handle, destination.open("xb") as target:
                    for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                        target.write(chunk)

        members = {
            path.relative_to(staging).as_posix(): {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in _assert_regular_tree(staging)
        }
        manifest: dict[str, object] = {
            "schema_version": EXPORT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "members": members,
        }
        manifest_path = staging / "export-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        temporary_output = output.with_name(f".{output.name}.partial")
        if temporary_output.exists():
            temporary_output.unlink()
        try:
            with zipfile.ZipFile(
                temporary_output,
                "x",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for source in _assert_regular_tree(staging):
                    archive.write(source, source.relative_to(staging).as_posix())
            temporary_output.replace(output)
        finally:
            if temporary_output.exists():
                temporary_output.unlink()
    return manifest


def verify_export(path: Path) -> dict[str, object]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate member names")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe archive member: {name}")
        try:
            manifest = json.loads(archive.read("export-manifest.json"))
        except KeyError as exc:
            raise ValueError("archive has no export-manifest.json") from exc
        if manifest.get("schema_version") != EXPORT_VERSION:
            raise ValueError("unsupported export schema version")
        expected = manifest.get("members")
        if not isinstance(expected, dict):
            raise ValueError("manifest members must be an object")
        actual_names = set(names) - {"export-manifest.json"}
        if actual_names != set(expected):
            raise ValueError("archive members do not match the manifest")
        for name, record in expected.items():
            if not isinstance(record, dict):
                raise ValueError(f"invalid manifest record: {name}")
            payload = archive.read(name)
            if len(payload) != record.get("size"):
                raise ValueError(f"size mismatch: {name}")
            digest = hashlib.sha256(payload).hexdigest()
            if digest != record.get("sha256"):
                raise ValueError(f"SHA-256 mismatch: {name}")
        for database in DATABASES:
            if database not in expected:
                raise ValueError(f"export is missing {database}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("create", help="create a new export")
    export_parser.add_argument("--data-dir", type=Path, default=Path(".data"))
    export_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify", help="verify an existing export")
    verify_parser.add_argument("path", type=Path)
    arguments = parser.parse_args()

    if arguments.command == "create":
        manifest = create_export(arguments.data_dir, arguments.output)
        print(
            json.dumps(
                {
                    "output": str(arguments.output.resolve()),
                    "members": len(manifest["members"]),
                }
            )
        )
    else:
        manifest = verify_export(arguments.path)
        print(
            json.dumps(
                {
                    "verified": str(arguments.path.resolve()),
                    "members": len(manifest["members"]),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
