"""Dependency slices catch foundation defects before their consumers exist."""

from waqil_api.project_slices import (
    dependency_slice_prefix,
    next_vertical_slice,
    vertical_build_slices,
)


MERIDIAN_PLAN = [
    "app/__init__.py",
    "app/config.py",
    "app/db.py",
    "app/models.py",
    "app/repository.py",
    "app/extraction.py",
    "app/services.py",
    "app/routes/__init__.py",
    "app/routes/documents.py",
    "app/routes/questions.py",
    "app/main.py",
    "app/static/index.html",
    "app/static/app.js",
    "requirements.txt",
    "README.md",
    "tests/test_workflows.py",
]


def test_meridian_is_checked_at_bounded_dependency_boundaries() -> None:
    staged: set[str] = set()
    verified = 0
    checkpoints = 0
    boundaries: list[str] = []

    for path in MERIDIAN_PLAN:
        staged.add(path)
        prefix = dependency_slice_prefix(
            MERIDIAN_PLAN,
            staged,
            verified_count=verified,
            checkpoints=checkpoints,
        )
        if prefix:
            verified = len(prefix)
            checkpoints += 1
            boundaries.append(prefix[-1])

    assert boundaries == [
        "app/extraction.py",
        "app/routes/questions.py",
        "app/static/app.js",
        "README.md",
    ]


def test_a_later_file_never_makes_a_dependency_hole_verifiable() -> None:
    plan = ["app/models.py", "app/services.py", "app/main.py"]
    staged = {"app/models.py", "app/main.py"}

    assert dependency_slice_prefix(plan, staged) == []


def test_a_one_file_directory_transition_does_not_start_the_sandbox() -> None:
    plan = ["app/models.py", "app/routes/items.py", "app/main.py"]

    assert dependency_slice_prefix(plan, {"app/models.py"}) == []


def test_final_verification_is_owned_by_the_existing_completion_gate() -> None:
    plan = ["app/models.py", "app/main.py"]

    assert dependency_slice_prefix(plan, set(plan)) == []


def test_declared_vertical_slices_are_an_exact_manifest_partition() -> None:
    plan = ["app/models.py", "app/routes/items.py", "tests/test_items.py"]
    declared = [
        {
            "name": "Items workflow",
            "outcome": "A user can create and list items.",
            "files": plan,
            "scenario_names": ["create item"],
        }
    ]
    # A declared slice that never named owned_files/integration_files falls
    # back to "the whole slice is its own, exclusive work" -- today's
    # behavior, preserved exactly.
    expected = [{**declared[0], "owned_files": plan, "integration_files": []}]

    assert vertical_build_slices(plan, declared) == expected
    assert next_vertical_slice(plan, declared, verified_count=0) == expected[0]
    assert next_vertical_slice(plan, declared, verified_count=len(plan)) is None


def test_malformed_slice_scope_falls_back_without_dropping_files() -> None:
    plan = [f"app/file_{index}.py" for index in range(8)]
    malformed = [
        {
            "name": "Wrong order",
            "outcome": "Invalid planner output.",
            "files": [plan[1], plan[0]],
        }
    ]

    slices = vertical_build_slices(plan, malformed)

    assert [path for item in slices for path in item["files"]] == plan
    assert [len(item["files"]) for item in slices] == [6, 2]
