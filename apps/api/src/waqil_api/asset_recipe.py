"""Drafting .metis/asset.json for an asset that has none.

The asset scanner refuses to guess how a discovered folder runs — that is a
launch recipe's job, and until now writing one meant a whole project-mode
conversation. This module is the one-click path: a bounded look at the
project, one structured Command A+ call, and a candidate manifest that must
survive the scanner's own parser before it may touch disk.

What this deliberately does NOT change: a generated recipe arrives exactly as
untrusted as a hand-written one. It lands as `launch_configured` and NOT
`launch_approved`, so the existing "Trust this exact recipe" review — with
the command shown in full — still stands between the model's draft and
anything executing on this machine.
"""
from __future__ import annotations

import json
from pathlib import Path

from .asset_library import manifest_metadata_from_body
from .contracts import AssetRecipeV1

# Directories that describe how a project is built, not what it is.
_SKIP_DIRS = frozenset({
    ".git", ".metis", ".venv", "venv", "node_modules", "__pycache__",
    ".next", "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})
# The manifests these files carry are what the model actually reasons from.
_CONFIG_FILES = (
    "requirements.txt", "pyproject.toml", "package.json", "Procfile",
    "setup.py", "environment.yml", "uv.lock",
)
_ENTRY_CANDIDATES = (
    "app.py", "main.py", "run.py", "server.py", "streamlit_app.py",
    "api.py", "index.js", "server.js",
)
_MAX_TREE_ENTRIES = 120
_HEAD_CHARS = 1_200
_README_CHARS = 2_000


class RecipeError(RuntimeError):
    """A recipe could not be drafted or written; the message is user-facing."""


def _safe_head(path: Path, limit: int) -> str | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return None


def gather_recipe_context(project: Path) -> dict:
    """A bounded, read-only description of the project for the model.

    Two levels of file listing and the heads of the files that declare how
    the thing runs. Nothing here reads a real `.env`, and nothing follows a
    symlink — the same posture the scanner takes.
    """
    tree: list[str] = []
    try:
        for child in sorted(project.iterdir(), key=lambda item: item.name.casefold()):
            if child.name in _SKIP_DIRS or child.name.startswith("."):
                continue
            if len(tree) >= _MAX_TREE_ENTRIES:
                break
            if child.is_dir() and not child.is_symlink():
                tree.append(child.name + "/")
                try:
                    for grandchild in sorted(child.iterdir(), key=lambda item: item.name.casefold()):
                        if grandchild.name in _SKIP_DIRS or grandchild.name.startswith("."):
                            continue
                        if len(tree) >= _MAX_TREE_ENTRIES:
                            break
                        tree.append(f"{child.name}/{grandchild.name}" + ("/" if grandchild.is_dir() else ""))
                except OSError:
                    continue
            elif child.is_file():
                tree.append(child.name)
    except OSError as exc:
        raise RecipeError("the project folder could not be read") from exc

    configs = {
        name: head
        for name in _CONFIG_FILES
        if (head := _safe_head(project / name, _HEAD_CHARS)) is not None
    }
    entry_heads = {}
    for name in _ENTRY_CANDIDATES:
        if len(entry_heads) >= 3:
            break
        if (head := _safe_head(project / name, _HEAD_CHARS)) is not None:
            entry_heads[name] = head

    readme = None
    for name in ("README.md", "readme.md", "README.txt"):
        if (readme := _safe_head(project / name, _README_CHARS)) is not None:
            break

    return {
        "folder_name": project.name,
        "files": tree,
        "readme_head": readme,
        "config_files": configs,
        "entry_file_heads": entry_heads,
    }


def write_recipe(project: Path, recipe: AssetRecipeV1) -> dict:
    """Validate a drafted recipe through the scanner's parser, then write it.

    Refuses to overwrite: an existing asset.json is someone's reviewed work,
    and "regenerate" should be a deliberate delete, not a button side effect.
    """
    body: dict = {
        "schema_version": "1",
        "launch": {"command": list(recipe.launch_command)},
    }
    if recipe.entrypoint:
        body["entrypoint"] = recipe.entrypoint
    if recipe.launch_path:
        body["launch"]["path"] = recipe.launch_path
    if recipe.env_keys:
        body["env_keys"] = sorted(set(recipe.env_keys))

    # The scanner is the judge. If ITS parser drops the command — bad token,
    # reserved env key, oversized argv — the draft is unusable, and writing
    # it would produce an asset that looks configured and refuses to launch.
    parsed = manifest_metadata_from_body(body)
    if parsed.command is None:
        raise RecipeError(
            "the drafted recipe did not survive validation "
            "(bad command tokens or reserved environment keys)"
        )

    metis_dir = project / ".metis"
    manifest_path = metis_dir / "asset.json"
    if metis_dir.is_symlink() or manifest_path.is_symlink():
        raise RecipeError("the project's .metis location may not be a symlink")
    if manifest_path.exists():
        raise RecipeError(
            "this asset already has .metis/asset.json — delete it first to regenerate"
        )
    try:
        metis_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise RecipeError("the recipe could not be written") from exc
    return body
