"""Contracts between files — the defects where every file is fine and the
changeset is broken.

The case on record: a revamp rewrote eleven React components and never wrote the
stylesheet they were written for. Every file parsed, every Python import
resolved, and the approval card said the static checks passed. 121 of 187 CSS
classes were defined nowhere and the application rendered unstyled.
"""
from __future__ import annotations

from waqil_api.project_contracts import (
    cross_file_findings,
    json_findings,
    local_import_findings,
    package_findings,
    style_contract_findings,
)


# ── The style contract ─────────────────────────────────────────────────────


def test_the_argus_failure_is_caught() -> None:
    """Components written against a stylesheet that was never written."""
    staged = {
        "web/src/App.jsx": {
            "content": '<div className="shell"><main className="main" /></div>'
        },
        "web/src/components/Rail.jsx": {
            "content": '<nav className="rail rail-item" style={{color: "var(--sage)"}} />'
        },
    }
    # The stylesheet as it actually was: untouched, knowing nothing of the new markup.
    stylesheets = {"web/src/styles.css": ".shell { display: flex; }\n:root { --ink: #222; }"}
    findings = style_contract_findings(staged, stylesheets)
    reported = {finding["path"] for finding in findings}
    assert reported == {"web/src/App.jsx", "web/src/components/Rail.jsx"}
    rail = next(f for f in findings if f["path"].endswith("Rail.jsx"))
    assert "--sage" in rail["error"]
    # An undeclared custom property is unambiguous, so it blocks.
    assert rail["severity"] == "error"


def test_a_staged_stylesheet_satisfies_the_contract() -> None:
    """The overlay is what approval applies, so a stylesheet written in the SAME
    turn is the definition — this is the run that should pass."""
    staged = {
        "web/src/App.jsx": {"content": '<div className="shell main" />'},
        "web/src/styles.css": {"content": ".shell{}\n.main{}\n"},
    }
    assert style_contract_findings(staged, {"web/src/styles.css": ".shell{}\n.main{}\n"}) == []


def test_a_few_unknown_classes_advise_rather_than_block() -> None:
    """A class can legitimately come from a library, so a small share is a
    warning — the check might be the thing that is wrong."""
    staged = {"web/src/App.jsx": {"content": '<div className="a b c d e f g h stranger" />'}}
    sheets = {"s.css": ".a{}.b{}.c{}.d{}.e{}.f{}.g{}.h{}"}
    findings = style_contract_findings(staged, sheets)
    assert len(findings) == 1
    assert findings[0]["severity"] == "warning"


def test_a_utility_framework_is_left_alone() -> None:
    """Tailwind generates its vocabulary at build time; judging a project
    against the stylesheet on disk would report every file as broken."""
    staged = {"src/App.jsx": {"content": '<div className="flex gap-4 rounded-xl bg-slate-50" />'}}
    assert style_contract_findings(staged, {"src/index.css": "@tailwind utilities;"}) == []


def test_a_project_with_no_stylesheet_is_not_judged() -> None:
    """With no stylesheet anywhere there is no contract, and every class would
    be reported."""
    staged = {"src/App.jsx": {"content": '<div className="anything at all" />'}}
    assert cross_file_findings(staged, on_disk=["src/App.jsx"], disk_sources={}) == []


# ── Local imports ──────────────────────────────────────────────────────────


def test_a_relative_import_out_of_a_subdirectory_resolves() -> None:
    """`components/../api.js` is `api.js`. Path keeps `..` as a literal
    segment, so without normalising, every such import was reported missing."""
    staged = {"web/src/components/Panel.jsx": {"content": "import { get } from '../api.js';"}}
    assert local_import_findings(staged, ["web/src/api.js"]) == []


def test_an_extensionless_import_resolves_through_its_candidates() -> None:
    staged = {"src/App.jsx": {"content": "import x from './lib/util';"}}
    assert local_import_findings(staged, ["src/lib/util.ts"]) == []
    assert local_import_findings(staged, ["src/lib/util/index.js"]) == []
    assert len(local_import_findings(staged, ["src/lib/other.js"])) == 1


def test_a_file_staged_this_turn_satisfies_an_import() -> None:
    staged = {
        "src/App.jsx": {"content": "import { Rail } from './Rail.jsx';"},
        "src/Rail.jsx": {"content": "export function Rail() {}"},
    }
    assert local_import_findings(staged, []) == []


def test_a_package_import_is_not_a_local_import() -> None:
    staged = {"src/App.jsx": {"content": "import React from 'react';"}}
    assert local_import_findings(staged, []) == []


# ── JSON ───────────────────────────────────────────────────────────────────


def test_broken_json_is_caught_before_it_reaches_disk() -> None:
    """Python has a stage-time parse gate; presets, fixtures and manifests had
    no equivalent and failed at runtime instead of at review."""
    findings = json_findings({"presets/high_risk.json": {"content": '{"a": 1,}'}})
    assert len(findings) == 1
    assert findings[0]["severity"] == "error"
    assert json_findings({"presets/ok.json": {"content": '{"a": 1}'}}) == []
    # An empty file is a placeholder, not a defect.
    assert json_findings({"data/empty.json": {"content": "  "}}) == []


# ── Packages ───────────────────────────────────────────────────────────────


def test_an_undeclared_package_advises() -> None:
    """Aliases and transitive imports are legitimate, so this can only advise —
    the same call the undeclared-Python-dependency check makes."""
    staged = {"src/App.jsx": {"content": "import axios from 'axios';\nimport React from 'react';"}}
    manifests = {"package.json": '{"dependencies": {"react": "^19.0.0"}}'}
    findings = package_findings(staged, manifests)
    assert len(findings) == 1
    assert findings[0]["severity"] == "warning"
    assert "axios" in findings[0]["error"]
    assert "react" not in findings[0]["error"]


def test_a_scoped_package_declares_by_its_scope_and_name() -> None:
    staged = {"src/App.jsx": {"content": "import { z } from '@scope/pkg/sub';"}}
    assert package_findings(staged, {"package.json": '{"dependencies": {"@scope/pkg": "1"}}'}) == []


def test_no_manifest_means_no_opinion() -> None:
    staged = {"src/App.jsx": {"content": "import axios from 'axios';"}}
    assert package_findings(staged, {}) == []


# ── The repair turn: a staged stylesheet judged against markup on disk ──────


def test_a_stylesheet_repair_is_judged_against_the_markup_already_on_disk() -> None:
    """A turn that stages ONLY a stylesheet finds nothing in the one-directional
    check — the markup is on disk and already broken. A 51-byte edit passed as a
    clean one-file change because of exactly this."""
    staged = {"web/src/styles.css": {"content": ".shell{}\n/* tweaked */\n"}}
    disk = {
        "web/src/styles.css": ".shell{}\n",
        "web/src/App.jsx": '<div className="shell rail dash" style={{color:"var(--sage)"}} />',
    }
    findings = cross_file_findings(
        staged, on_disk=list(disk), disk_sources=disk
    )
    blocking = [f for f in findings if f["severity"] == "error"]
    assert len(blocking) == 1
    # Reported against the stylesheet — the file this turn is changing.
    assert blocking[0]["path"] == "web/src/styles.css"
    assert "still undefined" in blocking[0]["error"]
    assert "--sage" in blocking[0]["error"]


def test_a_complete_stylesheet_repair_passes() -> None:
    staged = {"web/src/styles.css": {"content": ".shell{}\n.rail{}\n.dash{}\n:root{--sage:#6a7}\n"}}
    disk = {
        "web/src/styles.css": ".shell{}\n",
        "web/src/App.jsx": '<div className="shell rail dash" style={{color:"var(--sage)"}} />',
    }
    assert cross_file_findings(staged, on_disk=list(disk), disk_sources=disk) == []


def test_the_host_can_state_the_whole_job_as_a_list() -> None:
    """Two regexes in milliseconds, versus asking the model to re-derive it
    across eleven files inside one output budget."""
    from waqil_api.project_contracts import style_gaps

    gaps = style_gaps(
        {"s.css": ".shell{}\n:root{--ink:#222}\n"},
        {"App.jsx": '<div className="shell rail" style={{color:"var(--sage)"}} />'},
    )
    assert gaps["classes"] == ["rail"]
    assert gaps["variables"] == ["--sage"]


def test_a_utility_project_reports_no_gaps() -> None:
    from waqil_api.project_contracts import style_gaps

    gaps = style_gaps({"s.css": "@tailwind utilities;"}, {"App.jsx": '<div className="flex" />'})
    assert gaps == {"classes": [], "variables": []}
