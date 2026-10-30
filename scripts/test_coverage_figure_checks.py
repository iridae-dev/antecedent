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

    def test_context_does_not_hide_measured_coverage(self):
        for text in ("Coverage is at 0.868.", "Mean coverage = 0.8123.",
                     "Coverage effect 0.8123.", "Total coverage 0.8123.",
                     "Coverage = 0.8123."):
            with self.subTest(text=text):
                self.assertTrue(gate._legacy_figure_problems(text, "case", {}))
        for text in ("Coverage at 0.87.", "Mean coverage = 0.87.", "Total coverage = 87%."):
            with self.subTest(text=text):
                self.assertTrue(gate.wide_figures(text, 0, len(text)))

    def test_nominal_values_are_checked_when_explicitly_observed(self):
        for text in ("Observed coverage 95%.", "Measured coverage = 0.95.",
                     "Coverage measured at 90%.", "Coverage 95% observed.",
                     "Mean coverage 95%.", "Coverage was 0.95."):
            with self.subTest(text=text):
                self.assertEqual(len(gate.wide_figures(text, 0, len(text))), 1)
        for text in ("Nominal coverage 95%.", "The 95% interval has coverage.",
                     "Nominal coverage at 0.95.", "The nominal-90% coverage is checked.", "Coverage of the 95% credible interval.",
                     "Coverage at confidence level 0.95."):
            with self.subTest(text=text):
                self.assertEqual(gate.wide_figures(text, 0, len(text)), [])

    def test_explicit_construction_level_and_distinct_observed_result(self):
        text = "400-replicate 90% coverage 0.882 (`cov.a`)."
        self.assertEqual(gate.wide_figures(text, 0, len(text)), [])
        self.assertEqual([m.group() for m in gate.figures(text, 0, len(text))], ["0.882"])
        text = "Band: 90% coverage 0.8850 (record `cov.a`)."
        self.assertEqual(gate.wide_figures(text, 0, len(text)), [])
        text = "400-replicate 95% coverage of P(Y = 1) is measured (the registry records hold the rates)."
        self.assertEqual(gate.wide_figures(text, 0, len(text)), [])
        for text in ("400-replicate 90% coverage was observed.", "Measured coverage 90%."):
            self.assertEqual(len(gate.wide_figures(text, 0, len(text))), 1)
        for text in ("Coverage of the 95% logit-scale delta-method interval.",
                     "Nominal 0.90 and 0.95 coverage was measured.",
                     "Coverage has asymptotic probability at least 90%."):
            self.assertEqual(gate.wide_figures(text, 0, len(text)), [])

    def test_explicit_model_and_nominal_context_stays_exempt(self):
        for text in ("Nominal coverage 0.950.", "Coverage at rho = 0.950.",
                     "Coverage at x = 0.350.", "Coverage 0.851 at 0.965.",
                     "Coverage with path effect 0.125, total 0.356.",
                     "Coverage with outcome mean 0.350 and bias 0.123."):
            with self.subTest(text=text):
                actual = [m.group() for m in gate.figures(text, 0, len(text))]
                self.assertEqual(actual, ["0.851"] if "0.851" in text else [])

    def test_figure_after_citation_is_not_covered_and_nominal_is_not_a_result(self):
        records = {"cov.a": {"observed": 0.9457}}
        text = "Nominal coverage 0.950; coverage 0.9457 (`cov.a`) and coverage 0.8123."
        problems = gate._legacy_figure_problems(text, "fixture", records)
        self.assertTrue(any("0.8123" in p and "no record" in p for p in problems))
        self.assertFalse(any("0.950" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
