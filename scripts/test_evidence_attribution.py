"""Regression checks for the graph-imputation transport probe and multi-unit assertions."""
import unittest

from evidence_residue import credits


class AttributionTests(unittest.TestCase):
    def test_only_the_measured_transport_result_is_credited(self):
        cell = ("matrix", "AverageEffect/Dag/explicit/Bayesian/none")
        measured = ("capability", "graph_imputation.compiled_candidate")
        link = {"executed": [measured, cell, ("capability", "composition.propagation_chain")],
                "returned": [measured]}
        self.assertTrue(credits(link, measured))
        self.assertFalse(credits(link, cell))

    def test_multiple_kinds_without_a_result_require_an_assertion(self):
        a, b = ("matrix", "a"), ("capability", "b")
        self.assertTrue(credits({"executed": [a]}, a))
        self.assertFalse(credits({"executed": [a, b]}, a))
        self.assertFalse(credits({"executed": [a], "refused": [b]}, a))
        self.assertFalse(credits({"executed": [a, b], "returned": [a, b], "declared": [b]}, a))
        self.assertTrue(credits({"executed": [a, b], "returned": [a, b]}, b))


if __name__ == "__main__":
    unittest.main()
