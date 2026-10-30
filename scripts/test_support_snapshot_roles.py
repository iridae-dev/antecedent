"""Support generation retains evidence roles without reparsing for every cell."""
import unittest
from unittest.mock import patch

import generate_support_matrix_docs as generator


class SupportSnapshotRoles(unittest.TestCase):
    def test_one_registry_read_preserves_estimator_ownership_and_order(self):
        records = [
            {"id": "cov.linear_adjustment.gated", "role": "gated"},
            {"id": "cov.aipw.diagnostic", "role": "reported_level"},
            {"id": "cov.propensity_matching.boundary", "role": "named_boundary"},
        ]
        cell = dict(query="AverageEffect", graph_class="Dag", structure="explicit",
                    inference="frequentist", validation="none", estimators=["explicit"],
                    calibration=[row["id"] for row in records])
        with patch.object(generator, "load", return_value={"record": records}) as load:
            with patch.object(generator.external_evidence, "routes", return_value={}):
                snapshot = generator.support_snapshot([], [], [], [cell] * 2696)
        load.assert_called_once_with("parity/coverage_records.toml")
        self.assertEqual(len(snapshot["licensed"]), 2696)
        for result in snapshot["licensed"]:
            self.assertEqual(result["estimators"],
                             ["explicit", "linear.adjustment.ate", "propensity.matching"])

    def test_an_empty_registry_is_not_reloaded_for_each_cell(self):
        cell = dict(query="AverageEffect", graph_class="Dag", structure="explicit",
                    inference="frequentist", validation="none", calibration=[],
                    estimators=["explicit"])
        with patch.object(generator, "load", return_value={"record": []}) as load:
            with patch.object(generator.external_evidence, "routes", return_value={}):
                snapshot = generator.support_snapshot([], [], [], [cell] * 8)
        load.assert_called_once()
        self.assertEqual(snapshot["licensed"][0]["estimators"], ["explicit"])


if __name__ == "__main__":
    unittest.main()
