"""Contract tests for the launcher's incremental sidecar build gate."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "metis"


def _function_source(name: str) -> str:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index(f"{name}() {{")
    end = source.index("\n}\n", start) + len("\n}\n")
    return source[start:end]


def _needs_sidecar_build(root: Path) -> bool:
    check = subprocess.run(
        [
            "/bin/zsh",
            "-c",
            f'ROOT="$1"\n{_function_source("needs_sidecar_build")}\n'
            "needs_sidecar_build",
            "metis-build-state-test",
            str(root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode in {0, 1}, check.stderr
    return check.returncode == 0


def test_sidecar_build_gate_detects_missing_stale_and_fresh_output(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "apps" / "cline-sidecar"
    source = sidecar / "src" / "index.ts"
    package = sidecar / "package.json"
    config = sidecar / "tsconfig.json"
    entrypoint = sidecar / "dist" / "src" / "index.js"
    source.parent.mkdir(parents=True)
    source.write_text("export {};\n", encoding="utf-8")
    package.write_text("{}\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")

    assert _needs_sidecar_build(tmp_path) is True

    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("export {};\n", encoding="utf-8")
    older = 1_700_000_000
    newer = older + 10
    for path in (source, package, config, source.parent):
        os.utime(path, (older, older))
    os.utime(entrypoint, (newer, newer))
    assert _needs_sidecar_build(tmp_path) is False

    os.utime(source, (newer + 10, newer + 10))
    assert _needs_sidecar_build(tmp_path) is True


def test_launcher_builds_and_validates_sidecar_before_the_web_build() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    sidecar_gate = source.index("if needs_sidecar_build; then")
    web_gate = source.index("if needs_build; then")
    assert sidecar_gate < web_gate
    assert 'pnpm --dir "$ROOT/apps/cline-sidecar" build' in source
    assert '[[ ! -s "$ROOT/apps/cline-sidecar/dist/src/index.js" ]]' in source
