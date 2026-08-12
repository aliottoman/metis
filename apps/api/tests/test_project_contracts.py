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
    stylesheets = {
        "web/src/styles.css": ".shell { display: flex; }\n:root { --ink: #222; }"
    }
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
    assert (
        style_contract_findings(staged, {"web/src/styles.css": ".shell{}\n.main{}\n"})
        == []
    )


def test_a_few_unknown_classes_advise_rather_than_block() -> None:
    """A class can legitimately come from a library, so a small share is a
    warning — the check might be the thing that is wrong."""
    staged = {
        "web/src/App.jsx": {"content": '<div className="a b c d e f g h stranger" />'}
    }
    sheets = {"s.css": ".a{}.b{}.c{}.d{}.e{}.f{}.g{}.h{}"}
    findings = style_contract_findings(staged, sheets)
    assert len(findings) == 1
    assert findings[0]["severity"] == "warning"


def test_a_utility_framework_is_left_alone() -> None:
    """Tailwind generates its vocabulary at build time; judging a project
    against the stylesheet on disk would report every file as broken."""
    staged = {
        "src/App.jsx": {
            "content": '<div className="flex gap-4 rounded-xl bg-slate-50" />'
        }
    }
    assert (
        style_contract_findings(staged, {"src/index.css": "@tailwind utilities;"}) == []
    )


def test_a_project_with_no_stylesheet_is_not_judged() -> None:
    """With no stylesheet anywhere there is no contract, and every class would
    be reported."""
    staged = {"src/App.jsx": {"content": '<div className="anything at all" />'}}
    assert cross_file_findings(staged, on_disk=["src/App.jsx"], disk_sources={}) == []


def test_inline_style_definitions_satisfy_the_document_that_owns_them() -> None:
    """A self-contained HTML app has a real stylesheet even when it has no
    external ``.css`` file. The browser applies these definitions directly."""
    staged = {
        "app/static/index.html": {
            "content": """
                <style>
                  :root { --surface: #fff; --radius-sm: 4px; }
                  .workspace { background: var(--surface); }
                  .evidence-card { border-radius: var(--radius-sm); }
                </style>
                <main class="workspace">
                  <article class="evidence-card"></article>
                </main>
            """
        }
    }

    assert cross_file_findings(staged, on_disk=[], disk_sources={}) == []


def test_inline_styles_apply_to_markup_from_another_staged_component() -> None:
    """A page-shell style element participates in the document cascade, so it
    can legitimately own classes rendered by a separate JSX component."""
    staged = {
        "web/index.html": {"content": "<style>.evidence-card { display:grid }</style>"},
        "web/src/EvidenceCard.jsx": {
            "content": '<article className="evidence-card"></article>'
        },
    }

    assert cross_file_findings(staged, on_disk=[], disk_sources={}) == []


def test_inline_style_lookalikes_do_not_hide_genuinely_missing_symbols() -> None:
    """Comments and script strings are not CSS sources; a whole-document regex
    would incorrectly accept both as definitions."""
    staged = {
        "app/static/index.html": {
            "content": """
                <!-- <style>.missing { --radius-sm: 4px; }</style> -->
                <script>const example = '<style>.missing{--radius-sm:4px}</style>';</script>
                <style>.defined { display: block; }</style>
                <main class="defined missing" style="border-radius:var(--radius-sm)"></main>
            """
        }
    }

    findings = cross_file_findings(staged, on_disk=[], disk_sources={})

    assert len(findings) == 1
    assert findings[0]["path"] == "app/static/index.html"
    assert "missing" in findings[0]["error"]
    assert "--radius-sm" in findings[0]["error"]


def test_a_sole_staged_app_stylesheet_is_the_deterministic_repair_target() -> None:
    """When an authored stylesheet remains incomplete, direct repair to it,
    not to the markup that merely exposes the missing custom property."""
    staged = {
        "app/static/index.html": {
            "content": '<main class="workspace" style="border-radius:var(--radius-sm)"></main>'
        },
        "app/static/style.css": {"content": ".workspace { display: grid; }"},
        "appkit/static/theme.css": {"content": ":root { --surface: #fff; }"},
    }

    findings = style_contract_findings(
        staged,
        {
            "app/static/style.css": ".workspace { display: grid; }",
            "appkit/static/theme.css": ":root { --surface: #fff; }",
        },
    )

    assert len(findings) == 1
    assert findings[0]["path"] == "app/static/style.css"
    assert "--radius-sm" in findings[0]["error"]


def test_multiple_staged_stylesheets_do_not_guess_a_repair_target() -> None:
    staged = {
        "app/static/index.html": {"content": '<main class="missing"></main>'},
        "app/static/layout.css": {"content": ".page {}"},
        "app/static/components.css": {"content": ".card {}"},
    }

    findings = style_contract_findings(
        staged,
        {
            "app/static/layout.css": ".page {}",
            "app/static/components.css": ".card {}",
        },
    )

    assert len(findings) == 1
    assert findings[0]["path"] == "app/static/index.html"


def test_reverse_check_does_not_guess_between_multiple_app_stylesheets() -> None:
    staged = {
        "app/static/layout.css": {"content": ".page {}"},
        "app/static/components.css": {"content": ".card {}"},
    }
    disk = {
        "app/static/index.html": '<main class="missing"></main>',
        "app/static/layout.css": "",
        "app/static/components.css": "",
    }

    findings = cross_file_findings(staged, on_disk=list(disk), disk_sources=disk)

    assert len(findings) == 1
    assert findings[0]["path"] == "app/static/index.html"


def test_inline_definitions_also_satisfy_a_stylesheet_repair_turn() -> None:
    """The reverse-direction check reads existing markup. Its inline CSS must
    count too when this turn stages only an external stylesheet."""
    staged = {"app/static/style.css": {"content": ".external-only {}"}}
    disk = {
        "app/static/index.html": """
            <STYLE>:root { --radius-sm: 4px } .workspace { display: grid }</STYLE>
            <main class="workspace" style="border-radius:var(--radius-sm)"></main>
        """,
        "app/static/style.css": "",
    }

    assert cross_file_findings(staged, on_disk=list(disk), disk_sources=disk) == []


# ── Local imports ──────────────────────────────────────────────────────────


# Script asset wiring: a real external behavior source must reach the page.


def test_app_html_loads_its_sibling_app_script_from_public_static_url() -> None:
    staged = {
        "app/static/index.html": {
            "content": '<main></main><script src="/static/app.js?v=1"></script>'
        },
        "app/static/app.js": {"content": "window.appReady = true;"},
    }

    assert cross_file_findings(staged, on_disk=[], disk_sources={}) == []


def test_app_html_loads_its_sibling_app_script_by_relative_url() -> None:
    staged = {
        "app/static/index.html": {
            "content": '<main></main><script defer src="./app.js"></script>'
        },
        "app/static/app.js": {"content": "window.appReady = true;"},
    }

    assert cross_file_findings(staged, on_disk=[], disk_sources={}) == []


def test_html_repair_is_checked_against_an_existing_sibling_app_script() -> None:
    staged = {
        "app/static/index.html": {
            "content": '<main></main><script src="app.js"></script>'
        }
    }

    assert (
        cross_file_findings(
            staged,
            on_disk=(path for path in ["app/static/index.html", "app/static/app.js"]),
            disk_sources={},
        )
        == []
    )


def test_staged_app_script_rechecks_existing_sibling_html() -> None:
    staged = {
        "app/static/app.js": {"content": "window.appReady = true;"},
    }
    disk = {
        "app/static/index.html": (
            "<main></main><script>window.appReady = true;</script>"
        ),
        "app/static/app.js": "window.oldApp = true;",
    }

    findings = cross_file_findings(staged, on_disk=list(disk), disk_sources=disk)

    assert len(findings) == 1
    assert findings[0]["path"] == "app/static/index.html"
    assert "app/static/app.js" in findings[0]["error"]


def test_staged_app_script_accepts_an_existing_wired_sibling_html() -> None:
    staged = {
        "app/static/app.js": {"content": "window.appReady = true;"},
    }
    disk = {
        "app/static/index.html": (
            '<main></main><script defer src="/static/app.js"></script>'
        ),
        "app/static/app.js": "window.oldApp = true;",
    }

    assert cross_file_findings(staged, on_disk=list(disk), disk_sources=disk) == []


def test_unrelated_staged_javascript_does_not_recheck_existing_html() -> None:
    staged = {
        "app/static/chart.js": {"content": "window.drawChart = () => {};"},
    }
    disk = {
        "app/static/index.html": "<main></main><script>inlineApp()</script>",
        "app/static/app.js": "window.oldApp = true;",
    }

    assert cross_file_findings(staged, on_disk=list(disk), disk_sources=disk) == []


def test_inline_javascript_does_not_connect_a_sibling_app_script() -> None:
    staged = {
        "app/static/index.html": {
            "content": "<main></main><script>window.appReady = true;</script>"
        },
        "app/static/app.js": {"content": "window.appReady = true;"},
    }

    findings = cross_file_findings(staged, on_disk=[], disk_sources={})

    assert len(findings) == 1
    assert findings[0]["path"] == "app/static/index.html"
    assert findings[0]["severity"] == "error"
    assert "app/static/app.js" in findings[0]["error"]
    assert "<script src=...>" in findings[0]["error"]


def test_script_src_examples_in_comments_and_strings_do_not_fake_wiring() -> None:
    staged = {
        "app/static/index.html": {
            "content": """
                <!-- <script src="/static/app.js"></script> -->
                <script>
                  const example = '<script src="/static/app.js"></script>';
                </script>
            """
        },
        "app/static/app.js": {"content": "window.appReady = true;"},
    }

    findings = cross_file_findings(staged, on_disk=[], disk_sources={})

    assert len(findings) == 1
    assert findings[0]["path"] == "app/static/index.html"


def test_html_without_a_sibling_app_script_has_no_script_contract() -> None:
    staged = {
        "app/static/index.html": {
            "content": "<main></main><script>window.appReady = true;</script>"
        },
        "app/static/chart.js": {"content": "window.drawChart = () => {};"},
    }

    assert cross_file_findings(staged, on_disk=[], disk_sources={}) == []


def test_framework_owned_appkit_script_is_not_an_app_contract() -> None:
    staged = {
        "appkit/static/index.html": {"content": "<main></main>"},
        "appkit/static/app.js": {"content": "window.frameworkReady = true;"},
    }

    assert cross_file_findings(staged, on_disk=[], disk_sources={}) == []


# Local JavaScript imports.


def test_a_relative_import_out_of_a_subdirectory_resolves() -> None:
    """`components/../api.js` is `api.js`. Path keeps `..` as a literal
    segment, so without normalising, every such import was reported missing."""
    staged = {
        "web/src/components/Panel.jsx": {"content": "import { get } from '../api.js';"}
    }
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
    staged = {
        "src/App.jsx": {
            "content": "import axios from 'axios';\nimport React from 'react';"
        }
    }
    manifests = {"package.json": '{"dependencies": {"react": "^19.0.0"}}'}
    findings = package_findings(staged, manifests)
    assert len(findings) == 1
    assert findings[0]["severity"] == "warning"
    assert "axios" in findings[0]["error"]
    assert "react" not in findings[0]["error"]


def test_a_scoped_package_declares_by_its_scope_and_name() -> None:
    staged = {"src/App.jsx": {"content": "import { z } from '@scope/pkg/sub';"}}
    assert (
        package_findings(
            staged, {"package.json": '{"dependencies": {"@scope/pkg": "1"}}'}
        )
        == []
    )


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
    findings = cross_file_findings(staged, on_disk=list(disk), disk_sources=disk)
    blocking = [f for f in findings if f["severity"] == "error"]
    assert len(blocking) == 1
    # Reported against the stylesheet — the file this turn is changing.
    assert blocking[0]["path"] == "web/src/styles.css"
    assert "still undefined" in blocking[0]["error"]
    assert "--sage" in blocking[0]["error"]


def test_a_complete_stylesheet_repair_passes() -> None:
    staged = {
        "web/src/styles.css": {
            "content": ".shell{}\n.rail{}\n.dash{}\n:root{--sage:#6a7}\n"
        }
    }
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

    gaps = style_gaps(
        {"s.css": "@tailwind utilities;"}, {"App.jsx": '<div className="flex" />'}
    )
    assert gaps == {"classes": [], "variables": []}


# ── Template literals: fragments are never class names ─────────────────────


def test_an_interpolated_suffix_is_not_a_class_name() -> None:
    """`className={`rail-btn${x}`}` was read as the class `rail-btn$` and
    reported missing. Three such phantoms reached a live approval card."""
    from waqil_api.project_contracts import _tokens

    assert _tokens("<nav className={`rail-btn${active}`} />") == set()
    assert _tokens('<i className={`tb-status${ok ? "-on" : "-off"}`} />') == set()
    # No `$`, no brace, no sentinel ever survives as a name.
    for source in (
        "<div className={`a${b}`} />",
        "<div className={`${b}`} />",
        "<div className={`x-${b}-y`} />",
    ):
        assert not any(ch in name for name in _tokens(source) for ch in "${}")


def test_a_space_before_an_interpolation_keeps_the_literal_class() -> None:
    """The space is the whole signal: `rail-btn` is complete, so it is a real
    class and a real finding if no stylesheet defines it."""
    from waqil_api.project_contracts import _tokens

    assert _tokens("<nav className={`rail-btn ${extra}`} />") == {"rail-btn"}
    assert _tokens("<nav className={`rail-btn ${a} ${b}`} />") == {"rail-btn"}


def test_literal_classes_after_an_interpolation_are_still_found() -> None:
    """The old pattern stopped at the first `{`, losing every class beyond it —
    a false negative sitting behind the false positive."""
    from waqil_api.project_contracts import _tokens

    assert _tokens("<b className={`a ${x} b ${y} c`} />") == {"a", "b", "c"}


def test_every_ordinary_class_form_still_reads() -> None:
    from waqil_api.project_contracts import _tokens

    assert _tokens('<div className="shell main" />') == {"shell", "main"}
    assert _tokens("<div className='rail' />") == {"rail"}
    assert _tokens('<div className={"dash"} />') == {"dash"}
    assert _tokens('<main class="workspace card" />') == {"workspace", "card"}
    assert _tokens("<div className={`plain-template`} />") == {"plain-template"}


def test_a_dynamic_class_never_becomes_a_finding() -> None:
    """End to end: an interpolated component must not block a changeset."""
    staged = {
        "web/src/components/Rail.jsx": {
            "content": "<nav className={`rail-btn${active}`}><b className={`rail ${x}`}/></nav>"
        }
    }
    findings = style_contract_findings(staged, {"s.css": ".rail{}"})
    # `rail` is defined and `rail-btn${...}` is unknowable, so nothing is wrong.
    assert findings == []
