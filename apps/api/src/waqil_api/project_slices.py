"""Deterministic verification boundaries for dependency-ordered build plans.

The planner's file order is executable: a file may depend on earlier paths,
never on a later one. That makes every contiguous staged prefix dependency
closed. Verifying every file would repeatedly start the sandbox; verifying only
the complete manifest lets one bad foundation contaminate every dependent.
This module chooses a small number of useful boundaries between those extremes.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import AbstractSet, Any, Mapping, Sequence

from .contracts import MAX_PLAN_SCENARIOS, MAX_SLICE_FILES


__all__ = [
    "MAX_SLICE_FILES",
    "MAX_SLICE_CHECKPOINTS",
    "MAX_UNVERIFIED_FILES",
    "MAX_LEGACY_CHUNK_FILES",
    "synthesize_single_slice",
    "vertical_build_slices",
    "next_vertical_slice",
    "dependency_slice_prefix",
]


MAX_SLICE_CHECKPOINTS = 4
# Verification cadence is a separate question from how large one slice may be.
# Widening the slice bound to eight should not quietly halve how often a long
# legacy build stops to check itself, so the "how many unverified files may
# accumulate" rule keeps the number it was tuned with.
MAX_UNVERIFIED_FILES = 6
# The retired mechanical partition's width. Frozen at six so a run checkpointed
# before slice synthesis existed resolves to exactly the boundaries it was
# already verified against.
MAX_LEGACY_CHUNK_FILES = 6


def synthesize_single_slice(
    planned: Sequence[str],
    scenarios: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """One outcome slice owning a whole small manifest, or ``None``.

    A planner that returns a manifest and no slices used to be handed to a
    mechanical six-file partitioner, which is how a seven-file Logivity plan
    became "files 1-6" plus a second slice holding only ``README.md`` and
    ``requirements.txt`` -- a support-only slice, with an outcome sentence the
    host wrote, and an integration boundary nobody designed.

    Splitting is an architectural decision. When the host has to act without
    one it should take the position that invents least: for a manifest small
    enough to build in a single pass, that is one slice that owns all of it
    and carries every acceptance scenario the plan declared. Anything larger
    goes back to the planner, because there the split genuinely matters.
    """

    ordered = [str(path) for path in planned]
    if not ordered or len(ordered) > MAX_SLICE_FILES:
        return None
    names = list(
        dict.fromkeys(
            str(item.get("name") or "")
            for item in scenarios
            if str(item.get("name") or "")
        )
    )[:MAX_PLAN_SCENARIOS]
    return {
        "name": "Complete implementation",
        "outcome": ("Deliver every planned file as one working, verifiable increment."),
        "files": list(ordered),
        "owned_files": list(ordered),
        "integration_files": [],
        "scenario_names": names,
    }


def vertical_build_slices(
    planned: Sequence[str],
    declared: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return a complete, ordered set of bounded implementation slices.

    Planner-authored slices are accepted only when they are an exact contiguous
    partition of the host manifest. A malformed or legacy plan falls back to
    deterministic six-file chunks; scope is never widened or silently dropped.
    """

    ordered = [str(path) for path in planned]
    if not ordered:
        return []
    accepted: list[dict[str, Any]] = []
    flattened: list[str] = []
    for index, raw in enumerate(declared):
        files = [str(path) for path in raw.get("files") or []]
        if not files or len(files) > MAX_SLICE_FILES:
            accepted = []
            break
        # owned_files/integration_files are the outcome-based split beneath
        # `files` (ProjectBuildPlanV1.bind_slice_ownership already validated
        # that, together, they name exactly `files`, and that every
        # integration_files reference points at a strictly earlier slice's
        # owned file). A legacy/scripted caller that only ever knew `files`
        # falls back to "the whole slice is its own, exclusive work" — the
        # same default the contract itself fills in.
        owned = [str(path) for path in raw.get("owned_files") or []] or list(files)
        integration = [str(path) for path in raw.get("integration_files") or []]
        if not owned:
            # A slice that owns nothing can never advance the manifest
            # frontier -- next_vertical_slice would reselect it forever at
            # the same verified_count. Only owned_files may form the exact
            # partition below; a slice needs at least one to contribute to it.
            accepted = []
            break
        accepted.append(
            {
                "name": str(raw.get("name") or f"Slice {index + 1}")[:120],
                "outcome": str(raw.get("outcome") or "")[:600],
                "files": files,
                "owned_files": owned,
                "integration_files": integration,
                # Bounded by the PLAN's scenario limit, not a smaller
                # per-slice one: a slice may legitimately carry every
                # scenario the plan declared, and the old five-item trim
                # silently dropped planner commitments here as well as in
                # the contract.
                "scenario_names": list(
                    dict.fromkeys(
                        str(name)[:120] for name in raw.get("scenario_names") or []
                    )
                )[:MAX_PLAN_SCENARIOS],
            }
        )
        # The exact contiguous partition is formed by owned_files alone.
        # integration_files intentionally re-lists a strictly earlier slice's
        # already-partitioned path -- counting it here too would make this
        # path appear twice and break the partition-equality check below,
        # even though re-touching a declared integration point is exactly
        # what this field exists to allow.
        flattened.extend(owned)
    if accepted and flattened == ordered:
        return accepted
    # The mechanical partition below is retained for checkpoint and legacy
    # compatibility only: an in-flight run planned before slice synthesis
    # existed must keep resolving to the same boundaries it was verified
    # against. New ClineCore turns never reach it -- the control plane
    # synthesizes one slice, or asks the planner again, before coding.
    return [
        {
            "name": f"Slice {index // MAX_LEGACY_CHUNK_FILES + 1}",
            "outcome": "Complete and verify this dependency-ordered project increment.",
            "files": ordered[index : index + MAX_LEGACY_CHUNK_FILES],
            # The deterministic fallback has no concept of a declared
            # integration point: every file in a mechanical chunk is that
            # chunk's own, exclusive work, exactly as before this field
            # existed.
            "owned_files": ordered[index : index + MAX_LEGACY_CHUNK_FILES],
            "integration_files": [],
            "scenario_names": [],
        }
        for index in range(0, len(ordered), MAX_LEGACY_CHUNK_FILES)
    ]


def next_vertical_slice(
    planned: Sequence[str],
    declared: Sequence[Mapping[str, Any]],
    *,
    verified_count: int,
) -> dict[str, Any] | None:
    """Select the exact slice starting at the durable verified frontier."""

    slices = vertical_build_slices(planned, declared)
    frontier = max(0, min(int(verified_count), len(planned)))
    consumed = 0
    for item in slices:
        # The frontier advances by owned_files -- the exact partition -- not
        # by the full write scope, which may re-list an earlier slice's
        # already-consumed integration file.
        owned = list(item["owned_files"])
        if consumed == frontier:
            return item
        consumed += len(owned)
    return None


def _family(path: str) -> str:
    """The directory whose files normally form one implementation layer."""

    parent = str(PurePosixPath(path).parent)
    return "" if parent == "." else parent


def dependency_slice_prefix(
    planned: Sequence[str],
    staged_paths: AbstractSet[str],
    *,
    verified_count: int = 0,
    checkpoints: int = 0,
) -> list[str]:
    """Return the next dependency-closed prefix worth verifying, or ``[]``.

    Boundaries are host-observable and model-independent:

    * at most ``MAX_UNVERIFIED_FILES`` new files may accumulate without a
      check;
    * a transition between directories closes a layer after at least two files;
    * the complete manifest is left to the existing final acceptance gate.

    ``checkpoints`` caps intermediate sandbox work. Final verification remains
    mandatory and is intentionally outside this helper.
    """

    ordered = [str(path) for path in planned]
    if not ordered or checkpoints >= MAX_SLICE_CHECKPOINTS:
        return []
    prefix: list[str] = []
    for path in ordered:
        if path not in staged_paths:
            break
        prefix.append(path)
    if len(prefix) == len(ordered):
        return []
    start = max(0, min(int(verified_count), len(prefix)))
    fresh = prefix[start:]
    if not fresh:
        return []
    current = prefix[-1]
    next_path = ordered[len(prefix)]
    at_layer_boundary = _family(current) != _family(next_path) and len(fresh) >= 2
    if len(fresh) >= MAX_UNVERIFIED_FILES or at_layer_boundary:
        return prefix
    return []
