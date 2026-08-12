"""What a direct run is allowed to touch, decided without a model.

The manifest experiment failed because a model was asked to enumerate files
before the work began. Enumerating the existing tree instead would fail the
same way from the other end: it cannot authorize a file that does not exist
yet, and every real task creates one.

So a direct run is admitted under a contract — write anywhere in the project
except here — and the "except here" is resolved deterministically: persistent
per-project settings, plus explicit protection wording in the request matched
against the actual tree. Where that cannot be done safely the run stops and
asks rather than guessing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from waqil_api.project_direct_contract import (
    build_direct_contract,
    freeze_protected_hashes,
    is_binary_path,
    is_framework_path,
    protected_drift,
    resolve_task_protections,
)


# The real Logivity tree, and the exact words the user typed.
LOGIVITY_TREE = [
    "app.py",
    "config.py",
    "excel_writer.py",
    "extractor.py",
    "surcharge_mapping.py",
    "assets/DHL_Template.xlsx",
    "README.md",
    "requirements.txt",
]

LOGIVITY_REQUEST = (
    "revamp the logivity invoice extractor so it uses the metis design "
    "language. move it off streamlit to fastapi with a static frontend, same "
    "as my other apps. dont touch extractor.py, excel_writer.py, "
    "surcharge_mapping.py or the DHL template, the extraction and excel "
    "output have to stay exactly the same."
)

LOGIVITY_PROTECTED = (
    "assets/DHL_Template.xlsx",
    "excel_writer.py",
    "extractor.py",
    "surcharge_mapping.py",
)


def _project(tmp_path: Path, tree: list[str] = LOGIVITY_TREE) -> Path:
    project = tmp_path / "logivity"
    for relative in tree:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"contents of {relative}\n".encode())
    return project


# ── The wording the user actually used ─────────────────────────────────────


def test_the_logivity_wording_resolves_every_intended_file() -> None:
    resolution = resolve_task_protections(LOGIVITY_REQUEST, LOGIVITY_TREE)

    assert resolution.resolved == LOGIVITY_PROTECTED
    assert resolution.needs_user is False


def test_a_filename_dot_does_not_end_the_protection_clause() -> None:
    """The bug this test exists for.

    A sentence-ending dot is followed by whitespace; an extension dot is
    followed by a character. Reading them the same way ended the clause at the
    first filename and protected one file out of four.
    """

    resolution = resolve_task_protections(
        "do not touch extractor.py, excel_writer.py", LOGIVITY_TREE
    )

    assert set(resolution.resolved) == {"extractor.py", "excel_writer.py"}


def test_prose_after_the_file_list_is_not_read_as_a_failed_protection() -> None:
    # "the extraction and excel output have to stay exactly the same" names no
    # file. It is a sentence, not a protection Metis failed to resolve.
    resolution = resolve_task_protections(LOGIVITY_REQUEST, LOGIVITY_TREE)

    assert resolution.unmatched == ()
    assert resolution.ambiguous == ()


def test_glob_like_patterns_resolve_against_the_tree() -> None:
    resolution = resolve_task_protections(
        "do not modify excel*writer.py and surcharge*mapping.py", LOGIVITY_TREE
    )

    assert set(resolution.resolved) == {"excel_writer.py", "surcharge_mapping.py"}


def test_an_exact_subdirectory_path_resolves() -> None:
    resolution = resolve_task_protections(
        "preserve unchanged assets/DHL_Template.xlsx", LOGIVITY_TREE
    )

    assert resolution.resolved == ("assets/DHL_Template.xlsx",)


def test_bare_words_resolve_only_when_the_filename_carries_all_of_them() -> None:
    assert resolve_task_protections(
        "do not touch the DHL template", LOGIVITY_TREE
    ).resolved == ("assets/DHL_Template.xlsx",)
    # "template" alone still identifies exactly one file here.
    assert resolve_task_protections(
        "do not touch the template", LOGIVITY_TREE
    ).resolved == ("assets/DHL_Template.xlsx",)


# ── Refusing to guess ──────────────────────────────────────────────────────


def test_an_unmatched_explicit_protection_stops_before_inference() -> None:
    resolution = resolve_task_protections("do not modify payments.py", LOGIVITY_TREE)

    assert resolution.unmatched == ("payments.py",)
    assert resolution.needs_user is True
    assert "cannot find" in resolution.question()
    assert "payments.py" in resolution.question()


def test_an_unmatched_glob_also_stops() -> None:
    resolution = resolve_task_protections("do not touch billing_*.py", LOGIVITY_TREE)

    assert resolution.unmatched == ("billing_*.py",)
    assert resolution.needs_user is True


def test_a_term_matching_too_much_is_ambiguous_rather_than_included() -> None:
    tree = [f"module_{index}.py" for index in range(12)]

    resolution = resolve_task_protections("do not touch the module", tree)

    assert resolution.resolved == ()
    assert resolution.ambiguous == ("the module",)
    assert "do not want to guess" in resolution.question()


def test_wording_that_is_not_an_explicit_protection_protects_nothing() -> None:
    # "be careful with" is not a protection. Deciding that it might be is the
    # judgment this resolver refuses to make.
    resolution = resolve_task_protections(
        "be careful with extractor.py and try to avoid excel_writer.py",
        LOGIVITY_TREE,
    )

    assert resolution.resolved == ()
    assert resolution.needs_user is False


# ── Freezing, drift and the union ──────────────────────────────────────────


def test_hashes_are_frozen_at_admission_and_drift_is_rejected(tmp_path: Path) -> None:
    project = _project(tmp_path)
    frozen = freeze_protected_hashes(project, LOGIVITY_PROTECTED)

    assert set(frozen) == set(LOGIVITY_PROTECTED)
    assert protected_drift(project, frozen) == []

    (project / "extractor.py").write_text("tampered\n", encoding="utf-8")
    assert protected_drift(project, frozen) == ["extractor.py"]

    # A protected file that disappears is drift too, not an absence to ignore.
    (project / "excel_writer.py").unlink()
    assert protected_drift(project, frozen) == ["excel_writer.py", "extractor.py"]


def test_a_later_disk_change_cannot_inherit_a_stale_hash(tmp_path: Path) -> None:
    """Persistent settings store identities; every run re-freezes the bytes."""

    project = _project(tmp_path)
    first = freeze_protected_hashes(project, ["extractor.py"])
    (project / "extractor.py").write_text("edited between runs\n", encoding="utf-8")
    second = freeze_protected_hashes(project, ["extractor.py"])

    assert first != second
    assert protected_drift(project, second) == []
    assert (
        second["extractor.py"]
        == hashlib.sha256((project / "extractor.py").read_bytes()).hexdigest()
    )


def test_persistent_and_task_protections_are_unioned(tmp_path: Path) -> None:
    project = _project(tmp_path)

    contract, resolution = build_direct_contract(
        prompt=LOGIVITY_REQUEST,
        project=project,
        tree=LOGIVITY_TREE,
        project_protected=["config.py"],
    )

    assert resolution.needs_user is False
    assert contract.task_protected == LOGIVITY_PROTECTED
    assert contract.project_protected == ("config.py",)
    assert set(contract.protected_files) == set(LOGIVITY_PROTECTED) | {"config.py"}
    # Every one of them is frozen, including the persistent addition.
    assert set(contract.protected_hashes) == set(contract.protected_files)


def test_the_contract_authorizes_files_that_do_not_exist_yet(tmp_path: Path) -> None:
    """The whole reason the contract is not a file list."""

    project = _project(tmp_path)
    contract, _ = build_direct_contract(
        prompt=LOGIVITY_REQUEST, project=project, tree=LOGIVITY_TREE
    )

    assert contract.writable_roots == (".",)
    scope = contract.as_state()["authorized_scope"]
    assert "do not exist yet" in scope
    # Nothing in the contract enumerates the tree, so a new path is not
    # excluded by omission.
    state = contract.as_state()
    assert "app/main.py" not in state["protected_files"]
    assert "README.md" not in state["protected_files"]


def test_the_state_shape_round_trips_the_whole_contract(tmp_path: Path) -> None:
    project = _project(tmp_path)
    contract, _ = build_direct_contract(
        prompt=LOGIVITY_REQUEST,
        project=project,
        tree=LOGIVITY_TREE,
        project_protected=["config.py"],
        max_iterations=60,
        check_budget=12,
    )

    state = contract.as_state()

    assert state["writable_roots"] == ["."]
    assert state["max_iterations"] == 60
    assert state["check_budget"] == 12
    assert state["approval_required"] is True
    assert state["protected_hashes"]["extractor.py"]


# ── Standard exclusions the contract never widens ──────────────────────────


def test_framework_and_control_paths_are_never_the_sessions_to_write() -> None:
    for path in (
        "appkit",
        "appkit/web.py",
        "appkit/static/theme.css",
        ".git/config",
        ".metis/project-context.json",
    ):
        assert is_framework_path(path) is True, path
    for path in ("app/main.py", "README.md", "src/appkitchen.py"):
        assert is_framework_path(path) is False, path


def test_binary_paths_are_recognised() -> None:
    for path in ("assets/DHL_Template.xlsx", "logo.png", "vendor/lib.so", "data.db"):
        assert is_binary_path(path) is True, path
    for path in ("app/main.py", "README.md", "styles.css"):
        assert is_binary_path(path) is False, path


# ── Durability: the contract a continuation resumes under ──────────────────


def test_a_continuation_restores_the_identical_frozen_contract(
    tmp_path: Path,
) -> None:
    """A follow-up must not silently re-resolve into a different envelope.

    The contract is frozen at admission and carried in the checkpoint. Re-
    resolving on a continuation would let a disk change between rounds widen
    or narrow what the second round may touch.
    """

    project = _project(tmp_path)
    first, _ = build_direct_contract(
        prompt=LOGIVITY_REQUEST,
        project=project,
        tree=LOGIVITY_TREE,
        project_protected=["config.py"],
        max_iterations=60,
        check_budget=12,
    )
    carried = first.as_state()

    # Between rounds a protected file changes on disk, and a new file appears.
    (project / "extractor.py").write_text("edited between rounds\n", encoding="utf-8")
    (project / "new_module.py").write_text("NEW = 1\n", encoding="utf-8")

    # The continuation restores the carried contract rather than resolving
    # again: byte-identical paths and byte-identical hashes.
    assert carried["protected_files"] == list(first.protected_files)
    assert carried["protected_hashes"] == dict(first.protected_hashes)

    # And the drift is therefore visible rather than absorbed.
    assert protected_drift(project, carried["protected_hashes"]) == ["extractor.py"]

    # Re-resolving instead would have produced a different hash for the same
    # path -- which is exactly the substitution the frozen contract prevents.
    re_resolved, _ = build_direct_contract(
        prompt=LOGIVITY_REQUEST,
        project=project,
        tree=LOGIVITY_TREE,
        project_protected=["config.py"],
    )
    assert (
        re_resolved.protected_hashes["extractor.py"]
        != carried["protected_hashes"]["extractor.py"]
    )


def test_the_contract_state_is_json_round_trippable(tmp_path: Path) -> None:
    """It rides in a checkpoint, so it has to survive serialization exactly."""

    import json

    project = _project(tmp_path)
    contract, _ = build_direct_contract(
        prompt=LOGIVITY_REQUEST,
        project=project,
        tree=LOGIVITY_TREE,
        project_protected=["config.py"],
        max_iterations=60,
    )

    restored = json.loads(json.dumps(contract.as_state()))

    assert restored == contract.as_state()
    assert restored["protected_hashes"] == dict(contract.protected_hashes)
    assert restored["max_iterations"] == 60
