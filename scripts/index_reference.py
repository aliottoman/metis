#!/usr/bin/env python3
"""Register and index reference/ as a corpus source against a running Metis.

The reference library is only useful once it is retrievable: project build
turns pull from the corpus, so an unindexed reference/ is a directory the
model never sees. This registers it, grants consent, and indexes it.

Idempotent — re-run after editing a reference file to pick up the change.
Registration is skipped if the source already exists; indexing always runs.

Run (with the API up):
    python scripts/index_reference.py
    python scripts/index_reference.py --base-url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

LABEL = "Coding reference"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/api/v1"
CONSENT_REASON = "Index the repo's own verified coding reference library"


def call(base_url: str, method: str, path: str, body: dict | None = None) -> object:
    """One JSON request against the Metis API, returning the decoded response."""
    request = urllib.request.Request(
        f"{base_url}{path}",
        method=method,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else None


def find_source(base_url: str, root: Path) -> dict | None:
    """The already-registered source for this path, if there is one."""
    sources = call(base_url, "GET", "/corpus/sources") or []
    assert isinstance(sources, list)
    for source in sources:
        if Path(str(source.get("root_path", ""))).resolve() == root:
            return source
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "reference",
        help="Directory to index (default: the repo's reference/)",
    )
    args = parser.parse_args()

    root = args.path.resolve()
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1

    try:
        source = find_source(args.base_url, root)
        if source is None:
            source = call(
                args.base_url,
                "POST",
                "/corpus/sources",
                {"root_path": str(root), "label": LABEL, "kind": "notes"},
            )
            print(f"registered {LABEL} → {root}")
        else:
            print(f"already registered: {source['id']}")

        assert isinstance(source, dict)
        source_id = str(source["id"])
        if not source.get("consent"):
            call(
                args.base_url,
                "POST",
                f"/corpus/sources/{source_id}/consent",
                {"consent": True, "reason": CONSENT_REASON},
            )
            print("consent granted")

        result = call(args.base_url, "POST", f"/corpus/sources/{source_id}/reindex")
        assert isinstance(result, dict)
        print(
            f"indexed: {result.get('files_indexed', '?')} file(s), "
            f"{result.get('chunks', '?')} chunk(s), "
            f"status {result.get('status', '?')}"
        )
        if result.get("message"):
            print(result["message"])
    except urllib.error.HTTPError as exc:
        print(f"{exc.code} {exc.reason}: {exc.read().decode('utf-8')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"could not reach Metis at {args.base_url}: {exc.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
