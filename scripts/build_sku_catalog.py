#!/usr/bin/env python3
"""Vendor Oracle's UCM Service Descriptions PDF into a SKU catalog.

The PDF is Oracle's authoritative list of what can be billed against Universal
Credits: every service's official name, its part number, and — critically — the
*metric* it is billed by ("GPU Per Hour", "OCPU Per Hour", "Gigabyte Storage
Capacity Per Month"). It carries no prices at all; the only dollar figures in
its 330 pages are worked examples and contract minimums. So this script gives
the estimator the vocabulary (which SKU, in which unit) and `rates.json`
supplies the money, separately and under the user's control.

Run:
    python scripts/build_sku_catalog.py path/to/ucm.pdf

Requires `pdftotext` (poppler) on PATH; the layout mode is load-bearing,
because the plain mode collapses the four-column table into unordered runs.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PART = re.compile(r"\bB\d{5,6}\b")
PAGE_FOOTER = re.compile(r"Oracle UCM\s+[Vv]\d+|Page \d+ of \d+")
# The Note column is a bare footnote marker: "2", "1, 3".
NOTE = re.compile(r"^\d(?:\s*,\s*\d)*$")

# Category boundaries, keyed by the first page of each section in the table of
# contents. Anchoring on page number rather than pattern-matching headings is
# deliberate: the headings repeat inside body prose, the page footers do not.
SECTIONS: list[tuple[int, str]] = [
    (49, "Analytics"),
    (55, "Application Development"),
    (76, "Content Management"),
    (86, "Data Integration"),
    (94, "Data Management"),
    (158, "Enterprise Integration"),
    (166, "Management"),
    (182, "Security and Identity"),
    (195, "Compute"),
    (208, "Network"),
    (222, "GPU"),
    (223, "Storage"),
    (230, "Data and AI"),
    (238, "Not Discount Eligible"),
    (245, "Roving Edge Infrastructure"),
    (249, "Cloud Success Protection"),
    (256, "Cloud Success Assurance"),
    (260, "Optional Subscription"),
    (265, "Retired 6/1/18"),
    (273, "Retired"),
    (294, "Appendix"),
]
# Anything from here on describes SKUs Oracle no longer sells. They stay in the
# catalog so an old note that names one can still be recognised, but they are
# flagged so the estimator never proposes one for a new deal.
FIRST_RETIRED_PAGE = 265


def category_for(page: int) -> str:
    name = "General"
    for start, label in SECTIONS:
        if page >= start:
            name = label
    return name


def extract_pages(pdf: Path) -> list[str]:
    if not shutil.which("pdftotext"):
        sys.exit("pdftotext (poppler) is required: brew install poppler")
    with tempfile.TemporaryDirectory() as work:
        out = Path(work) / "ucm.txt"
        subprocess.run(
            ["pdftotext", "-layout", str(pdf), str(out)],
            check=True, capture_output=True,
        )
        # pdftotext separates pages with a form feed, so the page number of every
        # row is known exactly rather than parsed out of the footer text.
        return out.read_text(encoding="utf-8", errors="replace").split("\f")


def clean(text: str) -> str:
    """Normalise one name fragment as it appears in the table cell."""
    text = PAGE_FOOTER.sub("", text)
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    text = re.sub(r"^[\s•·*]+", "", text)
    return re.sub(r"\s+", " ", text).strip(" -•")


def parse_row(line: str) -> tuple[str, str, str, str] | None:
    """Split one table line into (name fragment, part number, note, metric)."""
    match = PART.search(line)
    if match is None:
        return None
    name = clean(line[: match.start()])
    tail = clean(line[match.end() :])
    note = ""
    # The note column, when present, sits between the part number and the
    # metric; splitting on the run of spaces that separates the columns keeps a
    # metric like "10,000 Requests Per Month" intact.
    columns = [part for part in re.split(r"\s{2,}", tail) if part]
    if columns and NOTE.match(columns[0]):
        note = columns.pop(0).replace(" ", "")
    return name, match.group(0), note, " ".join(columns).strip()


# Structural headings that follow the last row of a table and sit inside the
# name column, so geometry alone would read them as part of that row's name.
HEADINGS = {
    "DESCRIPTION", "DESCRIPTIONS", "NOTE", "NOTES",
    "SERVICE ACTIVATION, MEASUREMENT AND USAGE", "MEASUREMENT AND USAGE",
}


def _is_tail(fragment: str) -> bool:
    """Whether a line inside the name column continues the name above it.

    Every SKU name in this document opens with "Oracle", so a line that starts
    that way is the next entry of a two-column list rather than the tail of the
    entry before it — which is how names ended up concatenated in pairs.
    """
    if fragment.upper() in HEADINGS or fragment.endswith("."):
        return False
    return not fragment.startswith("Oracle ")


def parse(pages: list[str]) -> list[dict[str, Any]]:
    """Read the four-column tables, following names that wrap onto later lines.

    A wrapped name is identified by *geometry*, not by wording: the tail of a
    name cell is text that stays inside the name column, ending before the
    column where its row's part number began. Judging it by length or casing
    instead both drops real tails ("H100", "MI300X" read as headings) and
    swallows the body paragraph that follows the last row of a table.
    """
    rows: list[dict[str, Any]] = []
    for index, page in enumerate(pages, start=1):
        pending: dict[str, Any] | None = None
        name_column = 0
        wrapped = 0
        for line in page.splitlines():
            if PAGE_FOOTER.search(line):
                continue
            parsed = parse_row(line)
            if parsed is None:
                fragment = clean(line)
                if not fragment:
                    continue
                fits_column = len(line.rstrip()) <= name_column
                if pending and wrapped < 2 and fits_column and _is_tail(fragment):
                    pending["name"] = f"{pending['name']} {fragment}".strip()
                    wrapped += 1
                else:
                    pending = None
                continue
            name, part, note, metric = parsed
            name_column = PART.search(line).start()  # type: ignore[union-attr]
            wrapped = 0
            pending = {
                "part_number": part,
                "name": name,
                "metric": metric,
                "note": note,
                "category": category_for(index),
                "page": index,
                "retired": index >= FIRST_RETIRED_PAGE,
                # Part numbers are also cited mid-sentence in the body text.
                # Column gaps only exist in a laid-out table, so they separate a
                # real catalog row from a passing mention of the same SKU.
                "tabular": bool(re.search(r"\s{2,}", line.strip())),
            }
            rows.append(pending)
    return rows


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per part number, preferring the richest non-retired row.

    A part number legitimately appears several times — once in its own table,
    again in the "draws down against these SKUs" lists of services built on it.
    Those later mentions carry no metric, so the row that names a metric wins.
    """
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["name"]:
            continue
        current = best.get(row["part_number"])
        if current is None:
            best[row["part_number"]] = row
            continue
        if _rank(row) > _rank(current):
            best[row["part_number"]] = row
    return sorted(best.values(), key=lambda row: (row["category"], row["name"]))


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["tabular"],
        row["name"].startswith("Oracle"),
        not row["retired"],
        bool(row["metric"]),
        len(row["name"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Oracle UCM Service Descriptions PDF")
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "apps/api/src/waqil_api/data/skus/catalog.json",
    )
    args = parser.parse_args()
    if not args.pdf.exists():
        sys.exit(f"no such PDF: {args.pdf}")

    pages = extract_pages(args.pdf)
    entries = dedupe(parse(pages))
    for entry in entries:
        entry.pop("page", None)
        entry.pop("tabular", None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": args.pdf.name,
                "document": "Oracle PaaS and IaaS Universal Credits Service Descriptions",
                "effective_date": "2025-02-06",
                "version": "v020625",
                "notes": [
                    "Names, part numbers, and billing metrics only. This document",
                    "publishes no prices — see rates.json for the dollar figures,",
                    "which are user-owned and separately verified.",
                ],
                "entries": entries,
            },
            indent=2, ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metered = sum(1 for entry in entries if entry["metric"])
    retired = sum(1 for entry in entries if entry["retired"])
    print(
        f"{len(entries)} SKUs → {args.out} "
        f"({metered} with a billing metric, {retired} retired)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
