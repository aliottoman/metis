"""The ranked symbol map that replaces reading the tree to find out what exists."""
from __future__ import annotations

from waqil_api import repo_map


def test_python_symbols_carry_their_line_numbers() -> None:
    facts = repo_map.extract(
        "app/main.py",
        "import json\n"
        "\n"
        "class App:\n"
        "    def run(self):\n"
        "        return json.dumps({})\n"
        "\n"
        "def create_app():\n"
        "    return App()\n",
    )
    by_name = {symbol.name: symbol for symbol in facts.symbols}
    assert by_name["App"].kind == "class"
    assert by_name["App"].line == 3
    assert by_name["run"].kind == "method"
    assert by_name["create_app"].line == 7
    # The module node is the heading, not a symbol of its own.
    assert "main" not in by_name
    assert "json" in facts.references
    assert "App" in facts.references  # create_app calls it


def test_a_syntax_error_contributes_nothing_and_never_raises() -> None:
    facts = repo_map.extract("app/broken.py", "def oops(:\n")
    assert facts.symbols == ()
    assert facts.references == ()


def test_typescript_declarations_and_components() -> None:
    facts = repo_map.extract(
        "web/components/panel.tsx",
        'import { getSession } from "@/lib/api";\n'
        "\n"
        "export type PanelProps = { open: boolean };\n"
        "\n"
        "export function ModelPanel(props: PanelProps) {\n"
        "  return null;\n"
        "}\n"
        "\n"
        "const shorten = (id: string) => id.split(':')[0];\n",
    )
    by_name = {symbol.name: symbol.kind for symbol in facts.symbols}
    # A capitalised function is reported as a component, because "which file
    # owns the panel" is the question a UI task actually asks.
    assert by_name["ModelPanel"] == "component"
    assert by_name["PanelProps"] == "type"
    assert by_name["shorten"] == "function"
    assert "getSession" in facts.references
    assert "api" in facts.references


def test_html_and_css_contribute_anchors_selectors_and_tokens() -> None:
    page = repo_map.extract(
        "app/static/index.html",
        '<link href="/static/theme.css" rel="stylesheet">\n'
        '<main id="root"></main>\n'
        '<form action="/api/notes"></form>\n',
    )
    assert any(symbol.name == "#root" for symbol in page.symbols)
    assert "theme" in page.references

    sheet = repo_map.extract(
        "app/static/theme.css",
        ":root {\n  --ink: #222;\n}\n.card {\n  color: var(--ink);\n}\n",
    )
    names = {symbol.name for symbol in sheet.symbols}
    assert "--ink" in names
    assert ".card" in names


def test_an_unknown_suffix_is_listed_but_holds_no_symbols() -> None:
    assert repo_map.extract("Dockerfile", "FROM python:3.12\n").symbols == ()


def test_focus_terms_bridge_camel_case_and_prose() -> None:
    terms = repo_map.focus_terms("Please rework the createApp entry point")
    assert {"createapp", "create", "app", "rework"} <= terms
    # Two-letter noise never becomes a term.
    assert "to" not in terms


def test_rank_prefers_the_file_everything_depends_on() -> None:
    sources = [
        ("app/models.py", "class Note:\n    pass\n"),
        ("app/api.py", "from app.models import Note\n\ndef list_notes():\n    return Note()\n"),
        ("app/web.py", "from app.models import Note\n\ndef page():\n    return Note()\n"),
        ("app/unused.py", "def nothing():\n    return 1\n"),
    ]
    facts = [repo_map.extract(path, text) for path, text in sources]
    scores = repo_map.rank(facts)
    assert scores["app/models.py"] > scores["app/unused.py"]
    assert scores["app/models.py"] == max(scores.values())


def test_the_request_changes_the_ranking() -> None:
    sources = [
        ("app/billing.py", "def charge():\n    return 1\n"),
        ("app/reporting.py", "def report():\n    return 2\n"),
    ]
    facts = [repo_map.extract(path, text) for path, text in sources]
    neutral = repo_map.rank(facts)
    asked = repo_map.rank(facts, repo_map.focus_terms("fix the charge flow"))
    assert abs(neutral["app/billing.py"] - neutral["app/reporting.py"]) < 1e-9
    assert asked["app/billing.py"] > asked["app/reporting.py"]


def test_render_states_what_it_left_out() -> None:
    sources = [
        (f"app/mod{index}.py", f"def fn{index}():\n    return {index}\n")
        for index in range(12)
    ]
    facts = [repo_map.extract(path, text) for path, text in sources]
    text = repo_map.render(facts, repo_map.rank(facts), max_chars=120)
    assert "not shown" in text
    # The budget is honoured, with the one deliberate exception that at least
    # the top-ranked file is always named.
    assert len(text) < 400
    assert "app/mod" in text


def test_render_names_files_that_declare_nothing() -> None:
    facts = [
        repo_map.extract("app/main.py", "def go():\n    return 1\n"),
        repo_map.extract("Dockerfile", "FROM python:3.12\n"),
    ]
    text = repo_map.render(facts, repo_map.rank(facts), max_chars=4_000)
    assert "Other files: Dockerfile" in text


def test_build_renders_lines_that_can_be_checked() -> None:
    text = repo_map.build(
        [("app/main.py", "class App:\n    pass\n\ndef create_app():\n    return App()\n")],
        request="create_app",
        max_chars=2_000,
    )
    assert "app/main.py" in text
    assert "class App" in text
    assert "def create_app" in text
    # Every symbol carries the line that proves it, so the model can verify.
    assert "    1  class App" in text
    assert "    4  def create_app" in text


def test_a_zero_budget_sends_no_map() -> None:
    assert repo_map.build([("a.py", "def f():\n    pass\n")], max_chars=0) == ""


def test_a_method_call_is_not_a_vote_for_whoever_declares_that_name() -> None:
    """The bug that put a tool registry above the module everything imports.

    `self.db.get(...)` says nothing about which class was meant, so counting it
    handed the whole ranking to the one file with a top-level `get`.
    """
    sources = [
        ("app/registry.py", "def get(key):\n    return key\n"),
        ("app/core.py", "class Core:\n    def run(self):\n        return self.db.get('x')\n"),
        ("app/edge.py", "class Edge:\n    def go(self):\n        return self.cache.get('y')\n"),
        ("app/util.py", "class Util:\n    def do(self):\n        return self.store.get('z')\n"),
    ]
    facts = [repo_map.extract(path, text) for path, text in sources]
    # Three attribute calls to `.get()` produce no edges at all, so nothing
    # outranks anything: with no real dependencies, the ranking stays flat.
    assert all("get" not in entry.references for entry in facts[1:])
    scores = repo_map.rank(facts)
    assert all(abs(score - scores["app/core.py"]) < 1e-9 for score in scores.values())


def test_an_import_of_a_module_resolves_to_that_file() -> None:
    """Imports name the MODULE, which is not a symbol inside it."""
    sources = [
        ("lib/theme.ts", "export const TOKENS = 1;\n"),
        ("app/one.tsx", 'import { TOKENS } from "@/lib/theme";\n'),
        ("app/two.tsx", 'import { TOKENS } from "@/lib/theme";\n'),
    ]
    facts = [repo_map.extract(path, text) for path, text in sources]
    assert "theme" in facts[1].references
    scores = repo_map.rank(facts)
    assert scores["lib/theme.ts"] > scores["app/one.tsx"]


def test_only_top_level_declarations_are_extracted() -> None:
    """Locals inside a component are not the file's surface, and counting them
    both drowned the real declarations and corrupted the ranking."""
    facts = repo_map.extract(
        "web/panel.tsx",
        "export function Panel() {\n"
        "  const body = 1;\n"
        "  const preview = 2;\n"
        "  let router = 3;\n"
        "  return body + preview + router;\n"
        "}\n",
    )
    assert [symbol.name for symbol in facts.symbols] == ["Panel"]


def test_a_closure_is_not_part_of_a_python_file_surface() -> None:
    facts = repo_map.extract(
        "app/main.py",
        "def outer():\n"
        "    def on_token(value):\n"
        "        return value\n"
        "    return on_token\n"
        "\n"
        "class Thing:\n"
        "    def method(self):\n"
        "        def nested():\n"
        "            return 1\n"
        "        return nested\n",
    )
    names = {symbol.name for symbol in facts.symbols}
    assert names == {"outer", "Thing", "method"}


def test_references_are_distinct_per_file_not_per_call_site() -> None:
    facts = repo_map.extract(
        "app/caller.py",
        "from app.helpers import build\n\n"
        "def run():\n"
        "    return build(), build(), build()\n",
    )
    assert facts.references.count("build") == 1


def test_no_single_file_may_eat_the_whole_budget() -> None:
    wide = "\n".join(f"def fn{index}():\n    return {index}\n" for index in range(60))
    facts = [
        repo_map.extract("app/wide.py", wide),
        repo_map.extract("app/narrow.py", "def only():\n    return 1\n"),
    ]
    text = repo_map.render(facts, repo_map.rank(facts), max_chars=900)
    # The second file still gets named, however much the first one holds.
    assert "app/narrow.py" in text
    assert "more in this file" in text


def test_a_stylesheet_names_its_tokens_without_enumerating_them() -> None:
    """135 custom properties took the entire top of a real project's map."""
    sheet = "\n".join(f"  --token-{index}: #000;" for index in range(60))
    facts = [
        repo_map.extract("app/theme.css", ":root {\n" + sheet + "\n}\n"),
        repo_map.extract("app/main.py", "def create_app():\n    return 1\n"),
    ]
    text = repo_map.render(facts, repo_map.rank(facts), max_chars=4_000)
    assert "--token-0" in text          # named …
    assert "more in this file" in text  # … but counted, not listed
    assert text.count("--token-") <= repo_map._MAX_LEAF_SYMBOLS
    # And the file with real definitions still gets its full listing.
    assert "def create_app" in text
