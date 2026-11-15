"""The open-capability list reports every capability unit the support gate does not find supported."""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import check_open_capabilities as check
import gate_support


def unit(ident, problems):
    made = gate_support.Unit("capability", ident)
    made.problems.extend(problems)
    return made


class GateUnsupportedCapabilities(unittest.TestCase):
    def test_lists_unsupported_units_with_the_gates_reasons(self):
        result = {
            "capabilities": [
                unit("a.supported", []),
                unit("b.no_provenance", ["no provenance: names nothing", "no provenance: names nothing"]),
                unit("c.no_check", ["no independent check: no truth or parity fixture"]),
            ]
        }
        with patch.object(gate_support, "evaluate", return_value=result) as evaluate:
            items = check.gate_unsupported_capabilities()
        evaluate.assert_called_once_with(evidence_only=True)
        self.assertEqual(
            items,
            [
                ("capability", "b.no_provenance", "no provenance: names nothing"),
                ("capability", "c.no_check", "no independent check: no truth or parity fixture"),
            ],
        )

    def test_a_declared_open_record_is_not_listed_twice(self):
        declared = {"id": "x.declared", "status": "draft", "guarantee": "g"}
        real = check.load_toml

        def fake(path):
            if path.name == "capability_records.toml":
                return {"record": [declared]}
            return real(path)

        with patch.object(check, "load_toml", side_effect=fake):
            items = check.open_items(
                gate_items=[("capability", "x.gate_only", "no provenance"), ("capability", "x.declared", "dup")]
            )
        ids = [i[1] for i in items if i[0] == "capability"]
        self.assertEqual(ids.count("x.declared"), 1)
        self.assertEqual(ids.count("x.gate_only"), 1)

    def test_require_zero_fails_on_a_gate_unsupported_unit(self):
        out = io.StringIO()
        with patch.object(check, "open_items", return_value=[("capability", "u", "no provenance")]):
            with redirect_stdout(out):
                self.assertEqual(check.main(["--require-zero"]), 1)
                self.assertEqual(check.main([]), 0)
        self.assertIn("capability   u: no provenance", out.getvalue())

    def test_require_zero_passes_when_nothing_is_open(self):
        with patch.object(check, "open_items", return_value=[]), redirect_stdout(io.StringIO()):
            self.assertEqual(check.main(["--require-zero"]), 0)


if __name__ == "__main__":
    unittest.main()
