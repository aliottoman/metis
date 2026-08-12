"""Route one acceptance-level defect to the exact slice that must repair it.

A build that reaches its approval card and then fails a final acceptance
probe used to be repaired by replaying the whole request: the follow-up run
reset the verified frontier to zero, so every already-clean slice was planned
and coded again to fix one file. That is expensive, and worse, it puts
verified bytes back under a model's pen for no reason -- a live two-slice
canary spent a second full build to correct one seed string.

The slice plan already says who owns what. This module turns that into a
routing decision: which single slice owns (or, having explicitly integrated
it, most recently wrote) the attributed file; exactly what that repair may
write; and which previously passing scenarios must be replayed afterwards so
the repair cannot quietly break what earlier slices proved. Everything here
is pure -- state in, decision out -- so the routing rules are testable
without a model, a sandbox, or a graph.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# A repair that is allowed to touch everything is not a repair. Even a slice
# with a wide declared scope stays bounded by its own plan entry.
MAX_REPAIR_PATHS = 6


@dataclass(frozen=True, slots=True)
class RepairRoute:
    """Where one acceptance finding must be repaired, and under what scope."""

    target: str = ""
    slice_index: int = -1
    slice_name: str = ""
    slice_outcome: str = ""
    authorized_paths: tuple[str, ...] = ()
    owned_files: tuple[str, ...] = ()
    integration_files: tuple[str, ...] = ()
    rerun_scenarios: tuple[str, ...] = ()
    attribution: str = ""
    reason: str = ""
    findings: tuple[Mapping[str, Any], ...] = field(default=())

    @property
    def routed(self) -> bool:
        """Whether a model repair may start at all.

        False is an honest stop, never a speculative retry: a finding with no
        exact app-owned path cannot be attributed to a slice, and guessing
        would put unrelated verified files back in a write scope.
        """

        return self.slice_index >= 0 and bool(self.authorized_paths)


def _clean_paths(values: Any) -> list[str]:
    return [str(path) for path in values or [] if str(path).strip()]


def slice_ownership(
    slices: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Index a slice plan by who owns each file and who last integrates it.

    Ownership is exclusive and comes from ``owned_files`` (the exact manifest
    partition). Integration is additive: a later slice that explicitly names
    an earlier slice's file in ``integration_files`` is the most recent author
    of those bytes, so it -- not the original owner -- is where a defect in
    that file belongs.
    """

    owner: dict[str, int] = {}
    latest_integrator: dict[str, int] = {}
    for index, item in enumerate(slices):
        owned = _clean_paths(item.get("owned_files")) or _clean_paths(item.get("files"))
        for path in owned:
            owner.setdefault(path, index)
        for path in _clean_paths(item.get("integration_files")):
            # Later wins: the last slice to declare it is the last to write it.
            latest_integrator[path] = index
    return owner, latest_integrator


def cumulative_scenarios(
    slices: Sequence[Mapping[str, Any]],
    *,
    through_index: int,
) -> list[str]:
    """Every scenario proven at or before one slice, in stable order.

    A repair to slice N may regress anything slices 0..N already established,
    and anything a later slice built on top of it. The caller replays this
    whole set rather than only the repaired slice's own scenarios, which is
    what turns "the repair compiled" into "the repair did not cost us
    something that used to work".
    """

    names: list[str] = []
    for item in list(slices)[: max(0, through_index) + 1]:
        for name in _clean_paths(item.get("scenario_names")):
            if name not in names:
                names.append(name)
    return names


def route_acceptance_finding(
    *,
    findings: Sequence[Mapping[str, Any]],
    slices: Sequence[Mapping[str, Any]],
    staged: Mapping[str, Any],
) -> RepairRoute:
    """Attribute the first exact acceptance defect to one repairing slice.

    ``findings`` arrive already filtered to blocking, app-owned, existing
    paths (the verifier resolves a route/runtime failure to its own source
    file before this point, so an exact ``path`` here is the resolved path).
    A finding without such a path is not routed: the caller stops honestly
    for review rather than pointing a model at a guess.
    """

    exact = [
        item
        for item in findings
        if str(item.get("path") or "").strip() and str(item.get("path")) in staged
    ]
    if not exact:
        return RepairRoute(
            reason=(
                "the acceptance failure names no exact staged project file, so "
                "no slice owns it and no repair was started"
            ),
            findings=tuple(dict(item) for item in findings[:12]),
        )
    if not slices:
        return RepairRoute(
            reason=(
                "no vertical slice plan was carried forward, so the failing file "
                "cannot be attributed to an owning slice"
            ),
            findings=tuple(dict(item) for item in exact[:12]),
        )

    target = str(exact[0]["path"])
    targets = {str(item["path"]) for item in exact}
    owner, latest_integrator = slice_ownership(slices)

    # "Owns or most recently integrates": a file an earlier slice created and
    # a later slice explicitly reopened belongs to the later one, whose work
    # is what is actually in those bytes now.
    integrator_index = latest_integrator.get(target)
    owner_index = owner.get(target)
    if integrator_index is not None and (
        owner_index is None or integrator_index > owner_index
    ):
        index, attribution = integrator_index, "integration_file"
    elif owner_index is not None:
        index, attribution = owner_index, "owned_file"
    else:
        return RepairRoute(
            target=target,
            reason=(
                f"{target} is not owned or integrated by any planned slice, so "
                "the repair has no attributable scope"
            ),
            findings=tuple(dict(item) for item in exact[:12]),
        )

    item = slices[index]
    authorized = _clean_paths(item.get("files")) or (
        _clean_paths(item.get("owned_files"))
        + _clean_paths(item.get("integration_files"))
    )
    # Every other exact defect that this same slice already authorizes rides
    # along; anything it does not authorize waits for its own routed repair
    # rather than silently widening this one's scope.
    authorized = [path for path in dict.fromkeys(authorized)][:MAX_REPAIR_PATHS]
    if target not in authorized:
        return RepairRoute(
            target=target,
            reason=(
                f"the slice attributed to {target} does not authorize writing it, "
                "which means the carried slice plan is inconsistent"
            ),
            findings=tuple(dict(item) for item in exact[:12]),
        )
    in_scope = [entry for entry in exact if str(entry["path"]) in set(authorized)]
    held = sorted(targets - set(authorized))
    relation = (
        "an integration file most recently written by"
        if attribution == "integration_file"
        else "owned by"
    )
    reason = (
        f"{target} is {relation} slice {index + 1}; only that slice's declared "
        "scope may be rewritten"
    )
    if held:
        reason += f"; deferred to their own routed repairs: {', '.join(held)}"
    return RepairRoute(
        target=target,
        slice_index=index,
        slice_name=str(item.get("name") or f"Slice {index + 1}")[:120],
        slice_outcome=str(item.get("outcome") or "")[:600],
        authorized_paths=tuple(authorized),
        owned_files=tuple(_clean_paths(item.get("owned_files")) or authorized),
        integration_files=tuple(_clean_paths(item.get("integration_files"))),
        # Every scenario the whole plan ever proved: the repaired slice's own,
        # everything earlier that it could regress, and every downstream
        # scenario built on top of it.
        rerun_scenarios=tuple(
            cumulative_scenarios(slices, through_index=len(slices) - 1)
        ),
        attribution=attribution,
        reason=reason,
        findings=tuple(dict(entry) for entry in (in_scope or exact)[:12]),
    )


def staged_file_hashes(
    staged: Mapping[str, Mapping[str, Any]],
    *,
    exclude: Sequence[str] = (),
) -> dict[str, str]:
    """SHA-256 of every staged file a repair is NOT authorized to touch.

    Recorded before the repair session starts and compared after it imports,
    so "the repair changed only what it was allowed to change" is evidence
    rather than an assumption about a write allowlist that has already been
    bypassed once in this codebase's history.
    """

    skip = {str(path) for path in exclude}
    hashes: dict[str, str] = {}
    for path, entry in staged.items():
        if str(path) in skip:
            continue
        content = entry.get("content") if isinstance(entry, Mapping) else None
        if not isinstance(content, str):
            continue
        hashes[str(path)] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return hashes


def changed_unaffected_paths(
    before: Mapping[str, str],
    staged: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Unaffected files whose bytes moved, or that vanished, during a repair."""

    after = staged_file_hashes(staged)
    changed: list[str] = []
    for path, digest in before.items():
        if after.get(path) != digest:
            changed.append(str(path))
    return sorted(changed)
