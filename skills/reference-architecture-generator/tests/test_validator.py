from __future__ import annotations

from pathlib import Path
import sys
import unittest


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

from architecture_tool import generate_diagram_source  # noqa: E402
from validator import (  # noqa: E402
    SourceValidationError,
    validate_generated_source,
    validate_generated_source_v2,
    validate_source_against_spec,
    validate_source_against_spec_v2,
)


class GeneratedSourcePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = {
            "title": "Safe system",
            "provider": "generic",
            "direction": "LR",
            "components": [
                {"id": "api", "label": "API", "kind": "service"},
                {"id": "database", "label": "Database", "kind": "database"},
            ],
            "edges": [{"source": "api", "target": "database", "label": "SQL"}],
            "boundaries": [
                {"id": "application", "label": "Application", "component_ids": ["api", "database"]}
            ],
            "assumptions": [],
            "unresolved_ambiguities": [],
        }

    def test_generated_source_passes_allowlist(self) -> None:
        source = generate_diagram_source(self.spec, ["svg", "png"])
        evidence = validate_generated_source(source)
        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(evidence["component_assignments"], 2)
        semantic = validate_source_against_spec(source, self.spec, ["svg", "png"])
        self.assertEqual(semantic["status"], "passed")

    def test_malicious_label_remains_a_string_literal(self) -> None:
        self.spec["components"][0]["label"] = "'); __import__('os').system('id'); #"
        source = generate_diagram_source(self.spec, ["svg"])
        self.assertIn("__import__", source)
        self.assertEqual(validate_generated_source(source)["status"], "passed")

    def test_rejects_process_execution(self) -> None:
        source = generate_diagram_source(self.spec, ["svg"])
        source += "import subprocess\nsubprocess.run(['id'])\n"
        with self.assertRaises(SourceValidationError):
            validate_generated_source(source)

    def test_rejects_file_write(self) -> None:
        source = generate_diagram_source(self.spec, ["svg"])
        source += "open('/tmp/payload', 'w')\n"
        with self.assertRaises(SourceValidationError):
            validate_generated_source(source)

    # ── diagrams-draw-v2 policy ──────────────────────────────────────────────
    def _v2_source(self) -> str:
        return (
            "from pathlib import Path\n"
            "from diagrams import Cluster, Diagram, Edge\n"
            "from diagrams.generic.blank import Blank\n\n"
            'OUTPUT_STEM = str(Path(__file__).resolve().parent / "architecture")\n\n'
            "with Diagram('Safe system', filename=OUTPUT_STEM, "
            "outformat=['svg', 'png'], show=False, direction='LR', "
            "graph_attr={'splines': 'ortho', 'nodesep': '0.7'}, "
            "node_attr={'fontsize': '13'}):\n"
            "    with Cluster('Application'):\n"
            "        node_000 = Blank('API\\n[service]')\n"
            "        node_001 = Blank('Database\\n[database]')\n"
            "    node_000 >> Edge(label='SQL') >> node_001\n"
        )

    def test_v2_accepts_styled_source(self) -> None:
        source = self._v2_source()
        self.assertEqual(validate_generated_source_v2(source)["policy"], "allowlisted-ast-v2")
        semantic = validate_source_against_spec_v2(source, self.spec, ["svg", "png"])
        self.assertEqual(semantic["status"], "passed")
        self.assertEqual(semantic["edges"], 1)

    def test_v1_policy_rejects_v2_layout_attrs(self) -> None:
        # The stricter v1 policy must still reject graph_attr — proving the two
        # profiles are genuinely different guarantees, not a widened v1.
        with self.assertRaises(SourceValidationError):
            validate_generated_source(self._v2_source())

    def test_v2_rejects_process_execution(self) -> None:
        source = self._v2_source() + "    node_000 = str(Path('/etc'))\n"
        source = self._v2_source().replace(
            "node_000 = Blank('API\\n[service]')",
            "node_000 = Blank(__import__('os').getcwd())",
        )
        with self.assertRaises(SourceValidationError):
            validate_generated_source_v2(source)

    def test_v2_rejects_missing_component(self) -> None:
        source = self._v2_source().replace(
            "        node_001 = Blank('Database\\n[database]')\n", ""
        )
        with self.assertRaises(SourceValidationError):
            validate_source_against_spec_v2(source, self.spec, ["svg", "png"])

    def test_rejects_dynamic_control_flow(self) -> None:
        source = generate_diagram_source(self.spec, ["svg"])
        source += "while True:\n    pass\n"
        with self.assertRaises(SourceValidationError):
            validate_generated_source(source)


if __name__ == "__main__":
    unittest.main()
