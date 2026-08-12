"""Judge a planner's slice topology before any model is asked to build it.

A live qualification run produced a five-file narrow edit split into four
slices: a UI shell, its JavaScript, a tests-only slice, and a docs-only
slice. Every existing gate passed it — the file scope was exact, protected
files were untouched, and the acceptance scenarios were strict — because the
only structural rule in the contract (a bootstrap slice must contain an
entrypoint) applies to whole-app builds. The plan was nonetheless horizontal:
"write the tests" and "write the README" are not outcomes a user can observe,
and slicing that way spends a fresh coding session on work that belongs
beside the feature it describes.

This module is the general rule. It is pure — a plan in, findings out — so
the topology can be tested without a model, and its findings are written to
be handed straight back to a planner as a correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .contracts import MAX_PLAN_SCENARIOS, MAX_SLICE_FILES


# Files that describe, validate, or configure a feature rather than being one.
# None of these, on their own, is something a user can run or observe, so a
# slice made only of them has no independently checkable outcome.
_TEST_DIRECTORIES = ("tests/", "test/", "spec/", "__tests__/")
_DOC_SUFFIXES = (".md", ".rst", ".txt", ".adoc")
_DOC_DIRECTORIES = ("docs/", "doc/")
_SUPPORT_BASENAMES = frozenset(
    {
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "poetry.lock",
        "dockerfile",
        "containerfile",
        "makefile",
        ".gitignore",
        ".dockerignore",
        ".env.example",
        "tsconfig.json",
        "pytest.ini",
        "tox.ini",
    }
)


def support_kind(path: str) -> str:
    """ "test", "documentation", "dependency", or "" for a runtime file."""

    candidate = PurePosixPath(str(path))
    lowered = str(candidate).casefold()
    basename = candidate.name.casefold()
    if lowered.startswith(_TEST_DIRECTORIES) or any(
        f"/{segment}" in lowered for segment in _TEST_DIRECTORIES
    ):
        return "test"
    if (
        basename.startswith("test_")
        or basename.endswith("_test.py")
        or ".test." in basename
        or ".spec." in basename
    ):
        return "test"
    # Named manifests win over their extension: requirements.txt is a
    # dependency file, not a document, and calling it one would let a
    # dependency-only slice pass as prose.
    if basename in _SUPPORT_BASENAMES:
        return "dependency"
    if (
        lowered.startswith(_DOC_DIRECTORIES)
        or candidate.suffix.casefold() in _DOC_SUFFIXES
    ):
        return "documentation"
    return ""


@dataclass(frozen=True, slots=True)
class PlanValidation:
    """Whether a slice topology may be built, and precisely why not."""

    ok: bool
    findings: tuple[str, ...] = ()
    # Machine-readable companion to `findings`, so telemetry can count defect
    # classes without parsing prose.
    codes: tuple[str, ...] = field(default=())

    def correction_text(self) -> str:
        """The findings, phrased as an instruction a planner can act on."""

        if self.ok:
            return ""
        lines = "\n".join(f"- {item}" for item in self.findings)
        return (
            "Your previous plan was rejected before any code was written. Fix "
            "exactly these problems and return a corrected plan with the same "
            "file scope:\n"
            f"{lines}\n"
            "Every slice must deliver one independently runnable or "
            "user-observable outcome. Tests, documentation and dependency "
            "files are not outcomes on their own: put each one in the slice "
            "whose feature it validates or explains. A later slice that edits "
            "a file an earlier slice created must list it in that slice's "
            "integration_files."
        )


def _clean(values: Any) -> list[str]:
    return [str(item) for item in values or [] if str(item).strip()]


def validate_build_plan(
    files: Sequence[str],
    slices: Sequence[Mapping[str, Any]],
    *,
    scenarios: Sequence[Mapping[str, Any]] = (),
) -> PlanValidation:
    """Check one planner-authored slice topology against the general rules.

    Applies to every scope, narrow edits included — the qualification failure
    this exists for was a narrow edit. Deliberately does NOT impose a slice
    count: how many outcomes a request contains is the planner's judgment,
    and a specific count is a property of one evaluation scenario, not of
    Metis. An empty slice list is not judged here; ``validate_effective_plan``
    is what covers that case, because the host's deterministic partition can
    itself end in a support-only chunk.
    """

    manifest = _clean(files)
    if not manifest or not slices:
        return PlanValidation(ok=True)

    findings: list[str] = []
    codes: list[str] = []

    def fail(code: str, message: str) -> None:
        codes.append(code)
        findings.append(message)

    owned_sequence: list[str] = []
    seen_files: set[str] = set()
    runtime_in_plan = [path for path in manifest if not support_kind(path)]

    for index, item in enumerate(slices, 1):
        name = str(item.get("name") or f"slice {index}")
        slice_files = _clean(item.get("files"))
        owned = _clean(item.get("owned_files")) or slice_files
        integration = _clean(item.get("integration_files"))
        owned_sequence.extend(owned)

        if not owned:
            fail(
                "slice_owns_nothing",
                f"slice {index} ({name!r}) owns no files, so it cannot deliver "
                "an outcome of its own",
            )
        # The core rule. A slice whose entire WRITE SCOPE is tests, docs or
        # dependency files describes or validates work that lives somewhere
        # else; it is not an outcome. Scope, not ownership, is what counts: a
        # slice that owns the tests and docs for a feature while extending
        # that feature's runtime file through integration_files is doing real
        # runtime work and is exactly the shape we want. This is only a defect
        # when the plan HAS runtime files to attach the support work to -- a
        # request that genuinely only edits a README must stay plannable.
        elif runtime_in_plan:
            scope = owned + integration
            kinds = {support_kind(path) for path in scope}
            if "" not in kinds:
                kind = ", ".join(sorted(kind for kind in kinds if kind))
                fail(
                    "support_only_slice",
                    f"slice {index} ({name!r}) writes only {kind} files "
                    f"({', '.join(scope)}) and so delivers nothing runnable; "
                    "attach them to the slice whose feature they validate or "
                    "explain instead of giving them a slice of their own",
                )

        # A later slice may re-open an earlier slice's file, but only by
        # naming it: an undeclared re-edit is how a "vertical" plan quietly
        # becomes two models writing the same file.
        reopened = (set(slice_files) & seen_files) - set(integration)
        if reopened:
            fail(
                "undeclared_reedit",
                f"slice {index} ({name!r}) edits {', '.join(sorted(reopened))}, "
                "which an earlier slice already owns, without declaring them "
                "in integration_files",
            )
        for path in integration:
            if path not in seen_files:
                fail(
                    "forward_integration",
                    f"slice {index} ({name!r}) declares {path} as an "
                    "integration file, but no earlier slice owns it",
                )
        seen_files |= set(slice_files)

    duplicates = sorted(
        {path for path in owned_sequence if owned_sequence.count(path) > 1}
    )
    if duplicates:
        fail(
            "owned_more_than_once",
            f"{', '.join(duplicates)} are owned by more than one slice; each "
            "file must be owned exactly once",
        )
    missing = [path for path in manifest if path not in set(owned_sequence)]
    if missing:
        fail(
            "unowned_file",
            f"{', '.join(missing)} appear in the plan but no slice owns them",
        )
    extra = [path for path in owned_sequence if path not in set(manifest)]
    if extra:
        fail(
            "owned_outside_manifest",
            f"{', '.join(sorted(set(extra)))} are owned by a slice but are not "
            "in the planned file list",
        )
    if not duplicates and not missing and not extra and owned_sequence != manifest:
        fail(
            "dependency_order",
            "the slices' owned files do not follow the planned dependency "
            f"order; expected {', '.join(manifest)}",
        )

    declared = {str(item.get("name") or "") for item in scenarios}
    if declared:
        for index, item in enumerate(slices, 1):
            unknown = [
                name
                for name in _clean(item.get("scenario_names"))
                if name not in declared
            ]
            if unknown:
                fail(
                    "unknown_scenario",
                    f"slice {index} names acceptance scenarios that the plan "
                    f"never declares: {', '.join(unknown)}",
                )

    return PlanValidation(ok=not findings, findings=tuple(findings), codes=tuple(codes))


def validate_effective_plan(
    files: Sequence[str],
    slices: Sequence[Mapping[str, Any]],
    *,
    scenarios: Sequence[Mapping[str, Any]] = (),
) -> PlanValidation:
    """Judge the slices that will actually be built, however they arose.

    A planner that declares no slices does not escape review. What the host
    does in that case is now a single synthesized slice owning the whole
    manifest, and that slice is judged by exactly the same rules as a
    planner-authored one. Validating the effective plan is what makes "no plan
    bypasses topology validation" true rather than nearly true.

    Declared slices are judged first so the findings name what the planner
    actually did; the effective plan is judged either way.
    """

    declared = validate_build_plan(files, slices, scenarios=scenarios)
    if not declared.ok:
        return declared

    # Imported here: project_slices is the partitioner, and importing it at
    # module scope would tie this leaf rule module to it for callers that only
    # want the declared-plan check.
    from .project_slices import synthesize_single_slice, vertical_build_slices

    if not slices:
        synthesized = synthesize_single_slice(files, scenarios)
        # Too large to synthesize is not a topology finding -- the control
        # plane asks the planner again and then refuses. Judging the retired
        # mechanical partition here would report faults in a plan nothing
        # will ever build.
        if synthesized is None:
            return PlanValidation(ok=True)
        return validate_build_plan(files, [synthesized], scenarios=scenarios)

    effective = vertical_build_slices(files, slices)
    if not effective:
        return PlanValidation(ok=True)
    # The planner's own slices were accepted above, so anything wrong here came
    # out of how they resolve, not out of a plan nobody wrote.
    return validate_build_plan(files, effective, scenarios=scenarios)


# The shared slice bound, re-exported so normalization cannot produce a slice
# the sidecar's write-scope wire format would refuse.
MAX_NORMALIZED_SLICE_FILES = MAX_SLICE_FILES
# Scenarios are bounded by the PLAN's limit, shared with the contract. A
# separate, smaller normalization cap refused a correct Atlas plan for naming
# six of its own six scenarios -- a merged slice carrying every scenario the
# plan declared is exactly right, not excess.
MAX_NORMALIZED_SCENARIOS = MAX_PLAN_SCENARIOS


@dataclass(frozen=True, slots=True)
class PlanNormalization:
    """The result of canonicalizing a plan, and the evidence for it."""

    slices: tuple[dict[str, Any], ...] = ()
    applied: bool = False
    codes: tuple[str, ...] = ()
    moved: tuple[dict[str, Any], ...] = ()
    rejected: str = ""

    @property
    def ok(self) -> bool:
        return not self.rejected


def _slice_scope(item: Mapping[str, Any]) -> list[str]:
    owned = _clean(item.get("owned_files")) or _clean(item.get("files"))
    return owned + _clean(item.get("integration_files"))


def _is_support_only(item: Mapping[str, Any]) -> bool:
    scope = _slice_scope(item)
    return bool(scope) and all(support_kind(path) for path in scope)


def normalize_build_plan(
    files: Sequence[str],
    slices: Sequence[Mapping[str, Any]],
) -> PlanNormalization:
    """Fold a TRAILING tests/docs slice into the outcome it describes.

    Two live planner attempts produced the same predictable shape: correct
    runtime slices, then the tests and the README given a slice of their own.
    Trailing support work has exactly one possible owner -- the last outcome
    in the plan -- so folding it there canonicalizes a structural difference
    without inventing anything.

    Everything else is refused rather than guessed:

    * a support slice before or between runtime slices could belong to either
      neighbour, and choosing would silently reassign the planner's work;
    * a dependency file is never moved at all, because where a requirements
      manifest sits is a build-ordering decision, not tidying;
    * a merge that would breach the six-file bound, reorder the manifest, or
      drop a scenario the planner committed to.

    Pure and model-free: no inference, no new paths, no removed commitments.
    """

    manifest = _clean(files)
    working = [dict(item) for item in slices]
    if not manifest or not working:
        return PlanNormalization(slices=tuple(working))
    if not any(not support_kind(path) for path in manifest):
        # A request that is genuinely only documentation or tests: there is no
        # runtime outcome to attach anything to, and nothing to normalize.
        return PlanNormalization(slices=tuple(working))

    moved: list[dict[str, Any]] = []
    codes: list[str] = []
    merged_into: dict[int, int] = {}

    # Only a TRAILING run of support slices is canonicalizable. After the last
    # slice that delivers an outcome, the tests and docs can only belong to
    # that outcome -- there is nothing after them to describe. Anywhere else
    # their owner is a guess: a support slice sitting between two runtime
    # slices might validate either, and picking one would silently reassign
    # work the planner is responsible for placing.
    runtime_indexes = [
        index for index, item in enumerate(working) if not _is_support_only(item)
    ]
    last_runtime = runtime_indexes[-1] if runtime_indexes else None

    for index, item in enumerate(working):
        if not _is_support_only(item):
            continue
        if last_runtime is None:
            return PlanNormalization(
                rejected=(
                    f"slice {index + 1} "
                    f"({str(item.get('name') or index + 1)!r}) writes only support "
                    "files and no slice in this plan delivers an outcome for them "
                    "to belong to"
                ),
                codes=("support_slice_before_runtime",),
            )
        if index < last_runtime:
            # Ambiguous by construction, so refused rather than approximated.
            code = (
                "support_slice_before_runtime"
                if not any(earlier < index for earlier in runtime_indexes)
                else "support_slice_between_runtime"
            )
            placement = (
                "no earlier slice delivers an outcome for them to belong to"
                if code == "support_slice_before_runtime"
                else (
                    "it sits between slices that deliver outcomes, so which one "
                    "it belongs to is ambiguous; place it in that slice yourself"
                )
            )
            return PlanNormalization(
                rejected=(
                    f"slice {index + 1} "
                    f"({str(item.get('name') or index + 1)!r}) writes only support "
                    f"files and {placement}"
                ),
                codes=(code,),
            )
        # Dependency files are never moved automatically. Where a requirements
        # or package manifest belongs is a build-ordering decision with real
        # consequences -- an install step that runs before the dependency it
        # needs is a broken build, not a tidy-up -- so the planner must place
        # it, or the plan is refused.
        kinds = {support_kind(path) for path in _slice_scope(item)}
        if "dependency" in kinds:
            return PlanNormalization(
                rejected=(
                    f"slice {index + 1} "
                    f"({str(item.get('name') or index + 1)!r}) writes dependency "
                    "files, which are never moved automatically; place them in "
                    "the slice that needs them"
                ),
                codes=("dependency_slice_not_movable",),
            )
        merged_into[index] = last_runtime
        host = working[last_runtime]
        owned = _clean(host.get("owned_files")) or _clean(host.get("files"))
        incoming = _clean(item.get("owned_files")) or _clean(item.get("files"))
        integration = [
            path
            for path in dict.fromkeys(
                _clean(host.get("integration_files"))
                + _clean(item.get("integration_files"))
            )
            # A file the merged slice now owns cannot also be one of its
            # integration points; the contract forbids the overlap.
            if path not in set(owned) | set(incoming)
        ]
        combined_owned = owned + [path for path in incoming if path not in set(owned)]
        if len(combined_owned) + len(integration) > MAX_NORMALIZED_SLICE_FILES:
            return PlanNormalization(
                rejected=(
                    f"merging slice {index + 1} into slice {last_runtime + 1} would give "
                    f"it {len(combined_owned) + len(integration)} files, past the "
                    f"{MAX_NORMALIZED_SLICE_FILES}-file slice limit"
                ),
                codes=("merge_exceeds_slice_limit",),
            )
        scenarios = list(
            dict.fromkeys(
                _clean(host.get("scenario_names")) + _clean(item.get("scenario_names"))
            )
        )
        if len(scenarios) > MAX_NORMALIZED_SCENARIOS:
            return PlanNormalization(
                rejected=(
                    f"merging slice {index + 1} into slice {last_runtime + 1} would name "
                    f"{len(scenarios)} acceptance scenarios, past the "
                    f"{MAX_NORMALIZED_SCENARIOS}-scenario limit, and dropping one "
                    "would discard a planner commitment"
                ),
                codes=("merge_exceeds_scenario_limit",),
            )
        host["owned_files"] = combined_owned
        host["integration_files"] = integration
        host["files"] = combined_owned + integration
        host["scenario_names"] = scenarios
        moved.append(
            {
                "paths": incoming,
                "from_slice": str(item.get("name") or f"slice {index + 1}"),
                "into_slice": str(host.get("name") or f"slice {last_runtime + 1}"),
                "kinds": sorted({support_kind(path) for path in incoming}),
            }
        )
        codes.append("support_slice_merged")

    if not merged_into:
        return PlanNormalization(slices=tuple(working))

    normalized = [
        item for index, item in enumerate(working) if index not in merged_into
    ]

    # Nothing invented, nothing dropped, order intact -- checked rather than
    # trusted, because this runs on a plan a model wrote.
    before = [path for item in slices for path in _clean(item.get("files"))]
    after = [path for item in normalized for path in _clean(item.get("files"))]
    if sorted(before) != sorted(after):
        return PlanNormalization(
            rejected="normalization would change the plan's file set",
            codes=("merge_changed_file_set",),
        )
    owned_sequence = [
        path for item in normalized for path in _clean(item.get("owned_files"))
    ]
    if owned_sequence != manifest:
        return PlanNormalization(
            rejected=(
                "normalization would break the planned dependency order; "
                f"expected {', '.join(manifest)}"
            ),
            codes=("merge_breaks_dependency_order",),
        )
    return PlanNormalization(
        slices=tuple(normalized),
        applied=True,
        codes=tuple(dict.fromkeys(codes)),
        moved=tuple(moved),
    )
