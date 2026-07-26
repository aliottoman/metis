#!/usr/bin/env python3
"""Install Metis launch manifests into an explicitly selected projects folder.

The installer is intentionally manual and refuses to overwrite a project's own
manifest. It exists so the reviewed starter recipes remain auditable alongside
Metis while each project still carries the manifest used at launch time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    root = arguments.root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise SystemExit("projects root is not a directory")

    recipe_path = Path(__file__).with_name("asset-manifests.json")
    recipes = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipes, list):
        raise SystemExit("manifest bundle must be a JSON list")

    planned = 0
    unchanged = 0
    conflicts = 0
    missing = 0
    for recipe in recipes:
        if not isinstance(recipe, dict):
            raise SystemExit("invalid manifest bundle entry")
        folder = recipe.get("folder")
        manifest = recipe.get("manifest")
        if (
            not isinstance(folder, str)
            or not folder
            or Path(folder).name != folder
            or folder.startswith(".")
            or not isinstance(manifest, dict)
        ):
            raise SystemExit("invalid manifest bundle entry")

        project = (root / folder).resolve(strict=False)
        if project.parent != root or not project.is_dir() or project.is_symlink():
            print(f"MISSING   {folder}")
            missing += 1
            continue

        destination_dir = project / ".metis"
        destination = destination_dir / "asset.json"
        desired = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        if destination.exists() or destination.is_symlink():
            try:
                same = (
                    not destination.is_symlink()
                    and json.loads(destination.read_text(encoding="utf-8")) == manifest
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                same = False
            if same:
                print(f"UNCHANGED {folder}")
                unchanged += 1
            else:
                print(f"CONFLICT  {folder}")
                conflicts += 1
            continue

        print(f"{'INSTALL' if arguments.apply else 'WOULD ADD'} {folder}")
        planned += 1
        if not arguments.apply:
            continue
        if destination_dir.is_symlink():
            raise SystemExit(f"refusing symbolic-link manifest directory: {folder}")
        destination_dir.mkdir(mode=0o755, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(desired, encoding="utf-8")
            temporary.chmod(0o644)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    verb = "installed" if arguments.apply else "planned"
    print(
        f"{verb}={planned} unchanged={unchanged} "
        f"conflicts={conflicts} missing={missing}"
    )
    return 1 if conflicts or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
