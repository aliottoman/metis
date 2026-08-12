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


def test_python_interfaces_carry_exact_signatures_bindings_and_imports() -> None:
    facts = repo_map.extract(
        "app/client.py",
        "from pathlib import Path\n"
        "import json as wire\n"
        "\n"
        "PUBLIC_LIMIT = 3\n"
        "_private_value = 4\n"
        "\n"
        "class Client:\n"
        "    endpoint: str\n"
        "\n"
        "    def __init__(self, endpoint: str = 'local') -> None:\n"
        "        self.endpoint = endpoint\n"
        "\n"
        "    @classmethod\n"
        "    def from_env(cls) -> 'Client':\n"
        "        return cls()\n"
        "\n"
        "    async def send(self, payload: dict[str, object], *, timeout: float = 3.0) -> str:\n"
        "        return wire.dumps(payload)\n"
        "\n"
        "    def _secret(self):\n"
        "        return None\n"
        "\n"
        "async def save(path: Path, /, *, replace: bool = False) -> str:\n"
        "    return str(path)\n",
    )
    rendered = {symbol.name: symbol.render_interface() for symbol in facts.symbols}
    assert rendered["__init__"] == (
        "def Client.__init__(self, endpoint: str = 'local') -> None"
    )
    assert rendered["from_env"] == ("@classmethod def Client.from_env(cls) -> 'Client'")
    assert rendered["send"] == (
        "async def Client.send(self, payload: dict[str, object], *, "
        "timeout: float = 3.0) -> str"
    )
    assert rendered["save"] == (
        "async def save(path: Path, /, *, replace: bool = False) -> str"
    )
    assert [item.statement for item in facts.imports] == [
        "from pathlib import Path",
        "import json as wire",
    ]

    interface_map = repo_map.render_interfaces(
        [facts], target_path="app/client.py", max_chars=4_000
    )
    assert "current target contract; import app.client" in interface_map
    assert "async def Client.send" in interface_map
    assert "from pathlib import Path" in interface_map
    assert "_secret" not in interface_map
    assert "_private_value" not in interface_map


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


def test_typescript_interface_map_uses_exports_and_callable_signatures() -> None:
    facts = repo_map.extract(
        "web/panel.tsx",
        'import type { Invoice } from "./contracts";\n'
        'import { send } from "@/lib/api";\n'
        "export interface Props { amount: number }\n"
        "export async function submit(invoice: Invoice, retry = false): Promise<string> {\n"
        "  return send(invoice);\n"
        "}\n"
        "const local = (value: number): number => value;\n"
        "export const Panel = (props: Props) => null;\n",
    )
    text = repo_map.render_interfaces(
        [facts], target_path="web/panel.tsx", max_chars=4_000
    )
    assert "type Props { amount: number }" in text
    assert (
        "async function submit(invoice: Invoice, retry = false): Promise<string>"
        in text
    )
    assert "component Panel(props: Props)" in text
    assert 'import { send } from "@/lib/api";' in text
    assert "local" not in text


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
        (
            "app/api.py",
            "from app.models import Note\n\ndef list_notes():\n    return Note()\n",
        ),
        (
            "app/web.py",
            "from app.models import Note\n\ndef page():\n    return Note()\n",
        ),
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


def test_target_interface_map_includes_dependencies_and_appkit_with_a_hard_bound() -> (
    None
):
    facts = [
        repo_map.extract(
            "app/contracts.py",
            "class Extraction:\n"
            "    value: str\n\n"
            "def validate(item: Extraction, *, strict: bool = True) -> bool:\n"
            "    return strict\n",
        ),
        repo_map.extract(
            "appkit/uploads.py",
            "IMAGE_MIMES = frozenset({'image/png'})\n"
            "class UploadError(ValueError):\n"
            "    pass\n\n"
            "async def save_upload(upload, *, max_bytes: int = 10) -> str:\n"
            "    return 'saved'\n\n"
            "def _delete_everything():\n"
            "    pass\n",
        ),
    ]
    text = repo_map.render_interfaces(
        facts,
        target_path="app/service.py",
        dependency_paths=["app/contracts.py"],
        max_chars=2_000,
    )
    assert "Target is new" in text
    assert "app/contracts.py [earlier dependency; import app.contracts]" in text
    assert "def validate(item: Extraction, *, strict: bool = True) -> bool" in text
    assert "appkit/uploads.py [appkit public API; import appkit.uploads]" in text
    assert "async def save_upload(upload, *, max_bytes: int = 10) -> str" in text
    assert "_delete_everything" not in text
    assert len(text) <= 2_000

    tiny = repo_map.render_interfaces(
        facts,
        target_path="app/service.py",
        dependency_paths=["app/contracts.py"],
        max_chars=80,
    )
    assert len(tiny) <= 80


def test_build_renders_lines_that_can_be_checked() -> None:
    text = repo_map.build(
        [
            (
                "app/main.py",
                "class App:\n    pass\n\ndef create_app():\n    return App()\n",
            )
        ],
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
        (
            "app/core.py",
            "class Core:\n    def run(self):\n        return self.db.get('x')\n",
        ),
        (
            "app/edge.py",
            "class Edge:\n    def go(self):\n        return self.cache.get('y')\n",
        ),
        (
            "app/util.py",
            "class Util:\n    def do(self):\n        return self.store.get('z')\n",
        ),
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
    assert "--token-0" in text  # named …
    assert "more in this file" in text  # … but counted, not listed
    assert text.count("--token-") <= repo_map._MAX_LEAF_SYMBOLS
    # And the file with real definitions still gets its full listing.
    assert "def create_app" in text
