from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any, Protocol

from .contracts import ToolSourceFileV1, ToolVersionEvidenceV1


class SnapshotVerifier(Protocol):
    def verify_snapshot(
        self, snapshot_path: str, expected_hash: str, image_ref: str
    ) -> Path: ...


_MAX_FILE_BYTES = 100_000
_MAX_TOTAL_BYTES = 600_000
_MAX_DIFF_CHARACTERS = 240_000
_ROOT_FILES = {"SKILL.md", "metis.tool.json", "workflow.yaml", "requirements-runtime.lock"}
_SOURCE_SUFFIXES = {".py", ".json", ".jsonl", ".yaml", ".yml", ".md", ".lock"}


def _is_reviewable(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if parts[0] == "skills" and len(parts) >= 3:
        skill_relative = Path(*parts[2:])
        return (
            skill_relative.as_posix() in _ROOT_FILES
            or skill_relative.parts[0] in {"src", "tests", "evals"}
            and skill_relative.suffix.lower() in _SOURCE_SUFFIXES
        )
    return (
        parts[0] == "infra"
        and len(parts) >= 3
        and parts[1] == "sandbox"
        and relative.suffix.lower() in {".py", ".json"}
    )


def _read_files(root: Path) -> tuple[list[ToolSourceFileV1], bool]:
    files: list[ToolSourceFileV1] = []
    total = 0
    truncated = False
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or any(parent.is_symlink() for parent in path.parents if parent != root):
            raise ValueError("tool evidence contains an unsafe symbolic link")
        relative = resolved.relative_to(root)
        if not _is_reviewable(relative):
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) > _MAX_FILE_BYTES or total + len(raw) > _MAX_TOTAL_BYTES:
            truncated = True
            content = "[Content omitted because this evidence file exceeds the review limit.]"
        else:
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                truncated = True
                content = "[Binary content omitted from review evidence.]"
            else:
                total += len(raw)
        files.append(
            ToolSourceFileV1(
                path=relative.as_posix(),
                sha256=digest,
                size=len(raw),
                content=content,
            )
        )
    return files, truncated


def _source_diff(
    prior: list[ToolSourceFileV1], current: list[ToolSourceFileV1]
) -> tuple[str, bool]:
    before = {item.path: item.content for item in prior}
    after = {item.path: item.content for item in current}
    chunks: list[str] = []
    for path in sorted(before.keys() | after.keys()):
        if before.get(path) == after.get(path):
            continue
        chunks.extend(
            difflib.unified_diff(
                before.get(path, "").splitlines(keepends=True),
                after.get(path, "").splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        if sum(len(chunk) for chunk in chunks) > _MAX_DIFF_CHARACTERS:
            return "".join(chunks)[:_MAX_DIFF_CHARACTERS], True
    return "".join(chunks), False


def build_tool_version_evidence(
    record: dict[str, Any],
    *,
    verifier: SnapshotVerifier,
    compared_record: dict[str, Any] | None = None,
) -> ToolVersionEvidenceV1:
    manifest = record["manifest"]
    image_ref = manifest.get("runner_image")
    if not image_ref:
        raise ValueError("tool version has no pinned runner image")
    root = verifier.verify_snapshot(
        record["bundle_path"], record["content_hash"], image_ref
    )
    files, truncated = _read_files(root)
    compared_to_version_id: str | None = None
    source_diff = ""
    if compared_record and compared_record["id"] != record["id"]:
        prior_manifest = compared_record["manifest"]
        prior_image = prior_manifest.get("runner_image")
        if not prior_image:
            raise ValueError("comparison version has no pinned runner image")
        prior_root = verifier.verify_snapshot(
            compared_record["bundle_path"],
            compared_record["content_hash"],
            prior_image,
        )
        prior_files, prior_truncated = _read_files(prior_root)
        source_diff, diff_truncated = _source_diff(prior_files, files)
        compared_to_version_id = compared_record["id"]
        truncated = truncated or prior_truncated or diff_truncated

    return ToolVersionEvidenceV1(
        tool_id=record["tool_id"],
        version_id=record["id"],
        state=record["state"],
        content_hash=record["content_hash"],
        manifest=manifest,
        eval_report=record.get("eval_report"),
        bundle_verified=True,
        files=files,
        evidence_truncated=truncated,
        compared_to_version_id=compared_to_version_id,
        source_diff=source_diff,
    )
