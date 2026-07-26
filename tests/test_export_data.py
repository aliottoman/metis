from __future__ import annotations

import sqlite3
import zipfile
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_data.py"
_SPEC = importlib.util.spec_from_file_location("metis_export_data", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
create_export = _MODULE.create_export
verify_export = _MODULE.verify_export


def _data_directory(root: Path) -> Path:
    data = root / "data"
    data.mkdir()
    for name in ("waqil.db", "checkpoints.db"):
        with sqlite3.connect(data / name) as database:
            database.execute("CREATE TABLE sample(value TEXT NOT NULL)")
            database.execute("INSERT INTO sample VALUES ('preserved')")
    blob = data / "blobs" / "ab" / "payload"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"artifact")
    bundle = data / "tool-bundles" / "hash" / "SKILL.md"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("# capability\n", encoding="utf-8")
    return data


def test_export_round_trip_and_database_integrity(tmp_path: Path) -> None:
    data = _data_directory(tmp_path)
    output = tmp_path / "export.zip"

    created = create_export(data, output)
    verified = verify_export(output)

    assert created["members"] == verified["members"]
    assert set(verified["members"]) == {
        "waqil.db",
        "checkpoints.db",
        "blobs/ab/payload",
        "tool-bundles/hash/SKILL.md",
    }
    with zipfile.ZipFile(output) as archive:
        restored = tmp_path / "restored.db"
        restored.write_bytes(archive.read("waqil.db"))
    with sqlite3.connect(restored) as database:
        assert database.execute("SELECT value FROM sample").fetchone() == ("preserved",)


def test_export_refuses_overwrite_and_symlink(tmp_path: Path) -> None:
    data = _data_directory(tmp_path)
    output = tmp_path / "export.zip"
    create_export(data, output)

    with pytest.raises(FileExistsError):
        create_export(data, output)

    output.unlink()
    (data / "blobs" / "unsafe").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        create_export(data, output)


def test_verify_detects_archive_tampering(tmp_path: Path) -> None:
    data = _data_directory(tmp_path)
    output = tmp_path / "export.zip"
    create_export(data, output)

    with zipfile.ZipFile(output, "a") as archive:
        archive.writestr("unexpected", b"tampered")

    with pytest.raises(ValueError, match="members do not match"):
        verify_export(output)
