"""Explicit, bounded project workspaces for Metis chat.

The saved Asset catalog is the grant boundary: selecting a project never scans
for new siblings. A deterministic local manifest and an evolving METIS.md live
inside the project. Models can inspect the grant through narrow tools; exact
mutations are executed only after the control plane records user approval.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .asset_library import AssetLibraryError, AssetManager
from .config import Settings
from .contracts import ProjectBootstrapV1, ProjectToolCallV1, ProjectWorkspaceV1


_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".next",
        ".nuxt",
        ".turbo",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "coverage",
        "target",
        "__pycache__",
    }
)
_TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".css",
        ".go",
        ".h",
        ".html",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".md",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".scss",
        ".sh",
        ".sql",
        ".swift",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_LANGUAGE_BY_SUFFIX = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".php": "PHP",
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".css": "CSS",
    ".scss": "SCSS",
    ".html": "HTML",
    ".md": "Markdown",
}
_PRIORITY_FILES = (
    "README.md",
    "README",
    "AGENTS.md",
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "docker-compose.yml",
    "compose.yml",
)
_SECRETISH = re.compile(
    r"(?i)(password|passwd|secret|private[_ -]?key|api[_ -]?key|access[_ -]?token)"
)
_SENSITIVE_SUFFIXES = frozenset({".key", ".pem", ".p12", ".pfx", ".jks"})
_MANAGED_HEADER = "<!-- metis-project-context:v1 -->"
_LEARNINGS_START = "<!-- metis-learnings:start -->"
_LEARNINGS_END = "<!-- metis-learnings:end -->"


class ProjectWorkspaceError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _bounded_line(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())[:limit]


def _is_text_file(path: Path) -> bool:
    return path.name in _PRIORITY_FILES or path.suffix.casefold() in _TEXT_SUFFIXES


class ProjectWorkspaceService:
    def __init__(
        self,
        settings: Settings,
        assets: AssetManager,
        bootstrap_model: Any,
    ) -> None:
        self.settings = settings
        self.assets = assets
        self.bootstrap_model = bootstrap_model
        self._lock = asyncio.Lock()

    async def list(self) -> list[ProjectWorkspaceV1]:
        projects: list[ProjectWorkspaceV1] = []
        for asset in await self.assets.list():
            initialized = False
            revision = 0
            file_count = 0
            updated_at = None
            try:
                root = await self.assets.project_path(asset.id)
                manifest = self._read_manifest(root)
                initialized = bool(manifest)
                revision = int(manifest.get("revision", 0)) if manifest else 0
                file_count = int(manifest.get("file_count", 0)) if manifest else 0
                raw_updated = manifest.get("updated_at") if manifest else None
                updated_at = raw_updated if isinstance(raw_updated, str) else None
            except (AssetLibraryError, OSError, ValueError, TypeError):
                pass
            projects.append(
                ProjectWorkspaceV1(
                    id=asset.id,
                    name=asset.name,
                    summary=asset.summary,
                    framework=asset.framework,
                    initialized=initialized,
                    manifest_revision=revision,
                    file_count=file_count,
                    updated_at=updated_at,
                )
            )
        return projects

    async def open(self, asset_id: str) -> ProjectWorkspaceV1:
        """Create or refresh local context; Grok is called only on first access."""
        async with self._lock:
            root = await self.assets.project_path(asset_id)
            metadata = await self._asset_metadata(asset_id)
            metis_dir = root / ".metis"
            manifest_path = metis_dir / "project-context.json"
            notes_path = metis_dir / "METIS.md"
            if metis_dir.exists() and metis_dir.is_symlink():
                raise ProjectWorkspaceError("the project's .metis folder may not be a symlink")
            if manifest_path.is_symlink() or notes_path.is_symlink():
                raise ProjectWorkspaceError("Metis project context files may not be symlinks")
            metis_dir.mkdir(parents=True, exist_ok=True)

            prior = self._read_manifest(root)
            if manifest_path.exists() and not prior:
                raise ProjectWorkspaceError(
                    ".metis/project-context.json already exists but is not a valid Metis manifest"
                )
            snapshot, sample = await asyncio.to_thread(self._snapshot, root)
            bootstrap_raw = prior.get("bootstrap") if prior else None
            bootstrap = (
                ProjectBootstrapV1.model_validate(bootstrap_raw)
                if isinstance(bootstrap_raw, dict)
                else None
            )
            if bootstrap is None:
                available = getattr(self.bootstrap_model, "available", None)
                if available is None and hasattr(self.bootstrap_model, "oci"):
                    available = getattr(self.bootstrap_model.oci, "available", False)
                if available is False:
                    raise ProjectWorkspaceError(
                        "Grok project bootstrap requires OCI Responses to be configured"
                    )
                bootstrap = await self.bootstrap_model.bootstrap_project(
                    {
                        "project": metadata,
                        "manifest": snapshot,
                        "bounded_file_samples": sample,
                    }
                )
                self._write_initial_notes(notes_path, metadata, bootstrap)
            elif not notes_path.is_file():
                self._write_initial_notes(notes_path, metadata, bootstrap)

            timestamp = _now()
            revision = int(prior.get("revision", 0)) + 1 if prior else 1
            manifest = {
                "schema_version": "1",
                "project_id": asset_id,
                "project_name": metadata["name"],
                "root_name": root.name,
                "revision": revision,
                "created_at": prior.get("created_at", timestamp) if prior else timestamp,
                "updated_at": timestamp,
                "bootstrap_provider": "oci-grok",
                "bootstrap_model": self.settings.oci_grok_model,
                **snapshot,
                "bootstrap": bootstrap.model_dump(mode="json"),
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return ProjectWorkspaceV1(
                id=asset_id,
                name=metadata["name"],
                summary=metadata["summary"],
                framework=metadata.get("framework"),
                initialized=True,
                manifest_revision=revision,
                file_count=int(snapshot["file_count"]),
                updated_at=timestamp,
            )

    async def context(self, asset_id: str) -> dict[str, Any]:
        root = await self.assets.project_path(asset_id)
        manifest = self._read_manifest(root)
        notes_path = root / ".metis" / "METIS.md"
        if not manifest or not notes_path.is_file() or notes_path.is_symlink():
            raise ProjectWorkspaceError("open this project once before starting a project chat")
        notes = notes_path.read_text(encoding="utf-8")[:40_000]
        return {
            "project_id": asset_id,
            "project_name": manifest.get("project_name", root.name),
            "manifest": manifest,
            "metis_md": notes,
        }

    async def preview(self, asset_id: str, call: ProjectToolCallV1) -> dict[str, str]:
        root = await self.assets.project_path(asset_id)
        if call.name not in {"apply_patch", "create_file"}:
            raise ProjectWorkspaceError("that project tool does not mutate files")
        relative = _bounded_line(call.arguments.get("path"), 1_000)
        target = self._safe_target(root, relative, write=True)
        if call.name == "apply_patch":
            original = str(call.arguments.get("original", ""))
            replacement = str(call.arguments.get("replacement", ""))
            detail = (
                f"Replace {len(original)} characters with {len(replacement)} characters "
                f"in {relative}."
            )
        else:
            content = str(call.arguments.get("content", ""))
            detail = f"Create {relative} with {len(content.encode('utf-8'))} bytes."
        digest = hashlib.sha256(
            json.dumps(call.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {"path": str(target.relative_to(root)), "summary": detail, "digest": digest}

    async def execute(self, asset_id: str, call: ProjectToolCallV1) -> dict[str, Any]:
        root = await self.assets.project_path(asset_id)
        if call.name == "list_files":
            return await asyncio.to_thread(self._list_files, root, call.arguments)
        if call.name == "search_code":
            return await asyncio.to_thread(self._search_code, root, call.arguments)
        if call.name == "read_file":
            return await asyncio.to_thread(self._read_file, root, call.arguments)
        if call.name == "apply_patch":
            async with self._lock:
                result = await asyncio.to_thread(self._apply_patch, root, call.arguments)
                await asyncio.to_thread(self._record_mutation, root, call, result)
                return result
        if call.name == "create_file":
            async with self._lock:
                result = await asyncio.to_thread(self._create_file, root, call.arguments)
                await asyncio.to_thread(self._record_mutation, root, call, result)
                return result
        raise ProjectWorkspaceError(f"unsupported project tool: {call.name}")

    async def record_learnings(
        self, asset_id: str, run_id: str, learnings: list[str]
    ) -> None:
        clean = []
        for raw in learnings[:16]:
            value = _bounded_line(raw, 600)
            if value and not _SECRETISH.search(value):
                clean.append(value)
        if not clean:
            return
        async with self._lock:
            root = await self.assets.project_path(asset_id)
            path = root / ".metis" / "METIS.md"
            if not path.is_file() or path.is_symlink():
                return
            text = path.read_text(encoding="utf-8")
            start = text.find(_LEARNINGS_START)
            end = text.find(_LEARNINGS_END)
            if start < 0 or end < start:
                return
            existing = text[start + len(_LEARNINGS_START) : end]
            known = {line[2:].strip().casefold() for line in existing.splitlines() if line.startswith("- ")}
            additions = [item for item in clean if item.casefold() not in known]
            if not additions:
                return
            body = existing.rstrip() + "\n" + "\n".join(f"- {item}" for item in additions) + "\n"
            text = text[: start + len(_LEARNINGS_START)] + body + text[end:]
            log = f"\n- {_now()[:10]} · `{run_id}` · captured {len(additions)} durable learning(s).\n"
            text = text.rstrip() + log
            path.write_text(text, encoding="utf-8")

    async def _asset_metadata(self, asset_id: str) -> dict[str, Any]:
        for asset in await self.assets.list():
            if asset.id == asset_id:
                return {
                    "id": asset.id,
                    "name": asset.name,
                    "summary": asset.summary,
                    "framework": asset.framework,
                    "entrypoint": asset.entrypoint,
                    "tags": asset.tags,
                }
        raise AssetLibraryError("project is not in the saved Asset catalog")

    def _manifest_path(self, root: Path) -> Path:
        return root / ".metis" / "project-context.json"

    def _read_manifest(self, root: Path) -> dict[str, Any]:
        path = self._manifest_path(root)
        try:
            if path.is_symlink():
                return {}
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) and value.get("schema_version") == "1" else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _iter_files(self, root: Path):
        count = 0
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            dirs[:] = sorted(
                item
                for item in dirs
                if item not in _IGNORE_DIRS
                and not item.startswith(".")
                and not (current_path / item).is_symlink()
            )
            for name in sorted(files):
                path = current_path / name
                if (
                    path.is_symlink()
                    or name.startswith(".env")
                    or path.suffix.casefold() in _SENSITIVE_SUFFIXES
                ):
                    continue
                relative = path.relative_to(root)
                if relative.parts[:1] == (".metis",):
                    continue
                yield path, relative
                count += 1
                if count >= self.settings.project_manifest_max_files:
                    return

    def _snapshot(self, root: Path) -> tuple[dict[str, Any], str]:
        paths: list[str] = []
        languages: Counter[str] = Counter()
        total_bytes = 0
        digest = hashlib.sha256()
        for path, relative in self._iter_files(root):
            rel = relative.as_posix()
            paths.append(rel)
            try:
                stat = path.stat()
            except OSError:
                continue
            total_bytes += stat.st_size
            digest.update(rel.encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            language = _LANGUAGE_BY_SUFFIX.get(path.suffix.casefold())
            if language:
                languages[language] += 1

        samples: list[str] = []
        remaining = self.settings.project_manifest_sample_chars
        priority = list(_PRIORITY_FILES)
        priority.extend(path for path in paths if path.endswith(("/AGENTS.md", "/README.md")))
        for relative in dict.fromkeys(priority):
            if remaining <= 0:
                break
            path = root / relative
            if not path.is_file() or path.is_symlink():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            excerpt = content[: min(remaining, 16_000)]
            samples.append(f"--- {relative} ---\n{excerpt}")
            remaining -= len(excerpt)
        snapshot = {
            "file_count": len(paths),
            "total_bytes": total_bytes,
            "tree_digest": digest.hexdigest(),
            "languages": dict(languages.most_common()),
            "key_files": [item for item in paths if Path(item).name in _PRIORITY_FILES][:80],
            "file_tree": paths[:2_000],
            "truncated": len(paths) >= self.settings.project_manifest_max_files,
        }
        return snapshot, "\n\n".join(samples)

    def _initial_notes(
        self, metadata: dict[str, Any], bootstrap: ProjectBootstrapV1
    ) -> str:
        def bullets(values: list[str], empty: str) -> str:
            cleaned = [_bounded_line(item, 600) for item in values if _bounded_line(item, 600)]
            return "\n".join(f"- {item}" for item in cleaned) or f"- {empty}"

        return (
            f"{_MANAGED_HEADER}\n"
            f"# {metadata['name']} — Metis project context\n\n"
            "> Metis created this local working map on first access. Keep durable "
            "project facts here; never place credentials or secret values in this file.\n\n"
            "## Project overview\n\n"
            f"{bootstrap.summary.strip()}\n\n"
            "## Architecture\n\n"
            f"{bullets(bootstrap.architecture, 'Inspect the project further as work requires.')}\n\n"
            "## Important paths\n\n"
            f"{bullets(bootstrap.important_paths, 'No important paths recorded yet.')}\n\n"
            "## Working conventions\n\n"
            f"{bullets(bootstrap.conventions, 'Follow the repository’s existing patterns.')}\n\n"
            "## Verification\n\n"
            f"{bullets(bootstrap.verification, 'Confirm the relevant checks before finishing a change.')}\n\n"
            "## Known risks and open questions\n\n"
            f"{bullets(bootstrap.risks, 'No project-specific risks recorded yet.')}\n\n"
            "## Durable learnings\n\n"
            f"{_LEARNINGS_START}\n"
            "- Initial structure mapped by Grok through OCI Responses.\n"
            f"{_LEARNINGS_END}\n\n"
            "## Work log\n\n"
            f"- {_now()[:10]} · Initial manifest and working context created.\n"
        )

    def _write_initial_notes(
        self,
        path: Path,
        metadata: dict[str, Any],
        bootstrap: ProjectBootstrapV1,
    ) -> None:
        generated = self._initial_notes(metadata, bootstrap)
        if path.is_file():
            existing = path.read_text(encoding="utf-8")
            if _MANAGED_HEADER in existing:
                # A missing/corrupt JSON manifest must not erase a surviving
                # project memory file. Its managed markers remain usable.
                return
            generated = (
                existing.rstrip()
                + "\n\n---\n\n"
                + generated
            )
        path.write_text(generated, encoding="utf-8")

    def _record_mutation(
        self, root: Path, call: ProjectToolCallV1, result: dict[str, Any]
    ) -> None:
        """Keep the deterministic map and work log current after an approved edit."""
        manifest = self._read_manifest(root)
        if manifest:
            snapshot, _ = self._snapshot(root)
            manifest.update(snapshot)
            manifest["revision"] = int(manifest.get("revision", 0)) + 1
            manifest["updated_at"] = _now()
            history = list(manifest.get("recent_changes", []))[-19:]
            history.append(
                {
                    "at": manifest["updated_at"],
                    "tool": call.name,
                    "path": _bounded_line(result.get("path"), 1_000),
                }
            )
            manifest["recent_changes"] = history
            self._manifest_path(root).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        notes_path = root / ".metis" / "METIS.md"
        if notes_path.is_file() and not notes_path.is_symlink():
            text = notes_path.read_text(encoding="utf-8").rstrip()
            path = _bounded_line(result.get("path"), 1_000)
            notes_path.write_text(
                f"{text}\n- {_now()[:10]} · Approved `{call.name}` on `{path}`.\n",
                encoding="utf-8",
            )

    def _safe_target(self, root: Path, relative: str, *, write: bool = False) -> Path:
        if not relative or len(relative) > 1_000:
            raise ProjectWorkspaceError("project paths must be non-empty and bounded")
        candidate_path = Path(relative)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            raise ProjectWorkspaceError("project paths must stay inside the selected project")
        if candidate_path.parts and candidate_path.parts[0] in {".git", ".metis"}:
            raise ProjectWorkspaceError("models cannot directly change Metis or source-control internals")
        if (
            candidate_path.name.startswith(".env")
            or candidate_path.suffix.casefold() in _SENSITIVE_SUFFIXES
        ):
            raise ProjectWorkspaceError("environment and secret files are not exposed to project tools")
        candidate = (root / candidate_path).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ProjectWorkspaceError("project path escaped the selected project") from exc
        if candidate.exists() and candidate.is_symlink():
            raise ProjectWorkspaceError("project tools do not follow symbolic links")
        if write and candidate.exists() and not candidate.is_file():
            raise ProjectWorkspaceError("project mutation target must be a regular file")
        return candidate

    def _list_files(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        prefix = _bounded_line(arguments.get("path", ""), 1_000)
        limit = min(max(int(arguments.get("limit", 200)), 1), 500)
        files = []
        for _, relative in self._iter_files(root):
            rel = relative.as_posix()
            if not prefix or rel == prefix or rel.startswith(prefix.rstrip("/") + "/"):
                files.append(rel)
                if len(files) >= limit:
                    break
        return {"files": files, "truncated": len(files) >= limit}

    def _read_file(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = _bounded_line(arguments.get("path"), 1_000)
        path = self._safe_target(root, relative)
        if not path.is_file() or not _is_text_file(path):
            raise ProjectWorkspaceError("project file is unavailable or not readable text")
        start = min(max(int(arguments.get("start_line", 1)), 1), 1_000_000)
        end = min(max(int(arguments.get("end_line", start + 399)), start), start + 999)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ProjectWorkspaceError("project file could not be decoded as UTF-8") from exc
        selected = lines[start - 1 : end]
        numbered = "\n".join(
            f"{number:>6}  {line}" for number, line in enumerate(selected, start=start)
        )
        return {
            "path": relative,
            "start_line": start,
            "end_line": start + len(selected) - 1,
            "total_lines": len(lines),
            "content": numbered[: self.settings.project_tool_result_chars],
        }

    def _search_code(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        query = _bounded_line(arguments.get("query"), 300)
        if not query:
            raise ProjectWorkspaceError("search_code requires a query")
        limit = min(max(int(arguments.get("limit", 80)), 1), 200)
        case_sensitive = bool(arguments.get("case_sensitive", False))
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, Any]] = []
        for path, relative in self._iter_files(root):
            if not _is_text_file(path):
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle in haystack:
                    matches.append(
                        {"path": relative.as_posix(), "line": number, "text": line[:500]}
                    )
                    if len(matches) >= limit:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def _apply_patch(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = _bounded_line(arguments.get("path"), 1_000)
        original = str(arguments.get("original", ""))
        replacement = str(arguments.get("replacement", ""))
        if not original:
            raise ProjectWorkspaceError("apply_patch requires a non-empty exact original block")
        if len(replacement.encode("utf-8")) > self.settings.project_max_write_bytes:
            raise ProjectWorkspaceError("project replacement exceeds the write limit")
        path = self._safe_target(root, relative, write=True)
        if not path.is_file() or not _is_text_file(path):
            raise ProjectWorkspaceError("apply_patch target must be an existing text file")
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(original)
        if occurrences != 1:
            raise ProjectWorkspaceError(
                f"exact patch context matched {occurrences} times; refresh the file and narrow it"
            )
        updated = text.replace(original, replacement, 1)
        if len(updated.encode("utf-8")) > self.settings.project_max_write_bytes:
            raise ProjectWorkspaceError("updated project file exceeds the write limit")
        path.write_text(updated, encoding="utf-8")
        return {
            "path": relative,
            "changed": True,
            "before_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "after_sha256": hashlib.sha256(updated.encode("utf-8")).hexdigest(),
        }

    def _create_file(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = _bounded_line(arguments.get("path"), 1_000)
        content = str(arguments.get("content", ""))
        encoded = content.encode("utf-8")
        if not content or len(encoded) > self.settings.project_max_write_bytes:
            raise ProjectWorkspaceError("new project file is empty or exceeds the write limit")
        path = self._safe_target(root, relative, write=True)
        if path.exists():
            raise ProjectWorkspaceError("create_file refuses to overwrite an existing file")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "path": relative,
            "created": True,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
