"""Coverage figure checks cover every sentence and actual printed precision."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import coverage_citations as gate


class CoverageFigures(unittest.TestCase):
    def test_structured_row_does_not_hide_an_extra_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            licensed, records, backlog = (root / name for name in ("licensed.toml", "records.toml", "backlog.txt"))
            licensed.write_text('[[cell]]\nquery="q"\ngraph_class="g"\nstructure="s"\ninference="i"\nvalidation="v"\nlimitations="Coverage 0.9457 (`cov.a`). Coverage 0.8123."\ncoverage_figures=[{cites=["cov.a"]}]\n', encoding="utf-8")
            records.write_text('[[record]]\nid="cov.a"\nobserved=0.9457\n', encoding="utf-8")
            with patch.object(gate, "LICENSED", licensed), patch.object(gate, "RECORDS", records), patch.object(gate, "BACKLOG", backlog):
                self.assertTrue(any("0.8123" in problem and "no record" in problem for problem in gate.check()))

    def test_precision_and_every_sentence_are_checked(self):
        records = {"cov.a": {"observed": 0.9457}}
        text = "Coverage 0.9453 (`cov.a`). Coverage 0.8123."
        problems = gate._legacy_figure_problems(text, "fixture", records)
        self.assertTrue(any("0.9453" in p and "does not match" in p for p in problems))
        self.assertTrue(any("0.8123" in p and "no record" in p for p in problems))
        self.assertLess(gate.figure_tolerance("0.9453"), 0.0001)
        self.assertGreater(gate.figure_tolerance("0.945"), 0.00049)

    def test_figure_after_citation_is_not_covered_and_nominal_is_not_a_result(self):
        records = {"cov.a": {"observed": 0.9457}}
        text = "Nominal coverage 0.950; coverage 0.9457 (`cov.a`) and coverage 0.8123."
        problems = gate._legacy_figure_problems(text, "fixture", records)
        self.assertTrue(any("0.8123" in p and "no record" in p for p in problems))
        self.assertFalse(any("0.950" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
