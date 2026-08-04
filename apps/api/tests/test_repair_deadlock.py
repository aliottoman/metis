"""A file must never become both unpatchable and unreadable in one turn.

Measured on a live repair turn: three near-miss apply_patch calls on the same
file closed the write target, and the re-reads that would have supplied the
exact bytes were classified as repeats — closing the read target too. The
model held a file it could neither patch nor read, and spent 39 of 48 steps
against that door without fixing a two-line defect it had correctly named.

The two rules that reopen it: a read chasing a refused write is recovery, not
repetition; and a successful read reopens writing to that file.
"""

from __future__ import annotations

from typing import Any

from waqil_api.contracts import ProjectToolCallV1
from waqil_api.control_plane import _repeated_project_call, _write_failed_since

READ = ProjectToolCallV1(name="read_file", arguments={"path": "app/rules.py"})


def _read_entry(ok: bool = True) -> dict[str, Any]:
    return {
        "tool": "read_file",
        "arguments": {"path": "app/rules.py"},
        "result": {"ok": ok, "output": {"content": "x = 1\n"}},
    }


def _patch_entry(path: str = "app/rules.py", ok: bool = False) -> dict[str, Any]:
    return {
        "tool": "apply_patch",
        "arguments": {"path": path, "original": "a", "replacement": "b"},
        "result": {"ok": ok, "error": "" if ok else "exact patch context matched 0 times"},
    }


def test_a_plain_repeat_is_still_refused() -> None:
    """The original guard survives: nothing happened, so re-reading is waste."""
    state = {"project_trace": [_read_entry()]}
    refusal = _repeated_project_call(state, READ)
    assert refusal is not None
    assert refusal["ok"] is False
    assert "same read_file call" in refusal["error"]


def test_a_read_chasing_a_refused_patch_is_allowed() -> None:
    """The deadlock case: the re-read is how the model gets patchable bytes."""
    state = {"project_trace": [_read_entry(), _patch_entry()]}
    assert _repeated_project_call(state, READ) is None


def test_a_refused_patch_on_another_file_does_not_reopen_this_read() -> None:
    state = {"project_trace": [_read_entry(), _patch_entry(path="app/store.py")]}
    assert _repeated_project_call(state, READ) is not None


def test_a_successful_patch_does_not_reopen_the_read() -> None:
    """Success changes the file, so the overlay read differs — and the guard
    compares against the *earlier* answer, which is now stale but harmless.
    Only a refusal signals the model still needs the exact current bytes."""
    state = {"project_trace": [_read_entry(), _patch_entry(ok=True)]}
    assert _repeated_project_call(state, READ) is not None


def test_write_failed_since_scans_only_after_the_read() -> None:
    trace = [_patch_entry(), _read_entry()]
    # The refusal came BEFORE the read, so the read already accounts for it.
    assert _write_failed_since(trace, 2, "app/rules.py") is False
    assert _write_failed_since(trace, 0, "app/rules.py") is True
    assert _write_failed_since(trace, 0, "") is False
