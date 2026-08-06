"""replace_lines: the forgiving edit primitive.

apply_patch demands a byte-exact quote of the current text, and the measured
repair turns show that is exactly what weak models cannot produce — 0 of 8
attempted fixes landed, every miss on whitespace. replace_lines takes line
coordinates instead, straight from read_file's own start_line/end_line, so a
repair no longer depends on quoting. These tests pin the splice, the guards
that make a mis-aimed range refuse loudly, and the overlay behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from waqil_api.asset_library import AssetManager
from waqil_api.config import Settings
from waqil_api.contracts import ProjectToolCallV1
from waqil_api.model_provider import DeterministicModelProvider
from waqil_api.project_workspace import (
    ProjectWorkspaceError,
    ProjectWorkspaceService,
    _splice_lines,
)


def test_splice_replaces_an_inclusive_range() -> None:
    text = "one\ntwo\nthree\nfour\n"
    updated = _splice_lines(
        text, {"start_line": 2, "end_line": 3, "replacement": "TWO\nTHREE\n"}
    )
    assert updated == "one\nTWO\nTHREE\nfour\n"


def test_splice_covers_whole_file_rewrite_and_deletion() -> None:
    text = "a\nb\n"
    assert _splice_lines(
        text, {"start_line": 1, "end_line": 2, "replacement": "c\n"}
    ) == "c\n"
    # An empty replacement deletes the range.
    assert _splice_lines(text, {"start_line": 1, "end_line": 1, "replacement": ""}) == "b\n"


def test_splice_repairs_a_replacement_missing_its_newline() -> None:
    """A replacement that stops mid-line must not glue onto the next line and
    manufacture a syntax error the model never wrote."""
    text = "a\nb\nc\n"
    updated = _splice_lines(text, {"start_line": 2, "end_line": 2, "replacement": "B"})
    assert updated == "a\nB\nc\n"


def test_out_of_range_and_inverted_ranges_refuse() -> None:
    with pytest.raises(ProjectWorkspaceError, match="past the end"):
        _splice_lines("a\n", {"start_line": 5, "end_line": 6, "replacement": "x"})
    with pytest.raises(ProjectWorkspaceError, match="1 <= start_line"):
        _splice_lines("a\n", {"start_line": 3, "end_line": 2, "replacement": "x"})
    # An end past the file is clamped, not refused: the model asked for "to
    # the end", and refusing that teaches nothing.
    assert _splice_lines(
        "a\nb\n", {"start_line": 2, "end_line": 99, "replacement": "B\n"}
    ) == "a\nB\n"


def test_the_expect_guard_shows_what_the_range_actually_holds() -> None:
    with pytest.raises(ProjectWorkspaceError) as refused:
        _splice_lines(
            "def alpha():\n    pass\n\ndef beta():\n    pass\n",
            {
                "start_line": 1,
                "end_line": 2,
                "replacement": "x",
                "expect": "def beta",
            },
        )
    message = str(refused.value)
    assert "do not contain" in message
    assert "def alpha" in message  # the refusal teaches, not just refuses


@pytest.mark.asyncio
async def test_replace_lines_stages_through_the_overlay(tmp_path: Path) -> None:
    """The staged path: line edits land in the overlay, layer on staged
    content, and keep the parse gate — a splice that breaks the file refuses."""
    projects_root = tmp_path / "Projects"
    project = projects_root / "demo"
    project.mkdir(parents=True)
    (project / "main.py").write_text("VALUE = 1\nPRICE = 2\n", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[projects_root],
        model_backend="deterministic",
        allow_test_backends=True,
    )
    assets = AssetManager(
        settings.asset_roots,
        approval_path=settings.asset_approval_path,
        catalog_path=settings.asset_catalog_path,
    )
    discovered = await assets.scan()
    service = ProjectWorkspaceService(settings, assets, DeterministicModelProvider())
    asset_id = discovered[0].id

    result, staged = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="replace_lines",
            arguments={
                "path": "main.py",
                "start_line": 1,
                "end_line": 1,
                "replacement": "VALUE = 10\n",
                "expect": "VALUE",
            },
        ),
        {},
    )
    assert result["staged"] is True
    assert staged is not None
    assert staged["main.py"]["content"] == "VALUE = 10\nPRICE = 2\n"

    # A second edit sees the overlay text, not the disk text.
    result, staged = await service.execute_staged(
        asset_id,
        ProjectToolCallV1(
            name="replace_lines",
            arguments={
                "path": "main.py",
                "start_line": 2,
                "end_line": 2,
                "replacement": "PRICE = 20\n",
            },
        ),
        staged,
    )
    assert staged["main.py"]["content"] == "VALUE = 10\nPRICE = 20\n"

    # The parse gate holds: a splice that breaks the file refuses and stages
    # nothing, exactly like the other write tools.
    with pytest.raises(ProjectWorkspaceError, match="does not parse"):
        await service.execute_staged(
            asset_id,
            ProjectToolCallV1(
                name="replace_lines",
                arguments={
                    "path": "main.py",
                    "start_line": 1,
                    "end_line": 1,
                    "replacement": "def broken(:\n",
                },
            ),
            staged,
        )
