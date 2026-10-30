"""Historical validation proofs are tied to their exact measured sources."""
import unittest
from copy import deepcopy

import coverage_validation_bindings as bindings


class HistoricalValidationBindingsTest(unittest.TestCase):
    def setUp(self):
        sha = "a" * 40
        self.record = {"id": "cov.example", "calibration_sha": sha,
                       "test": "crates/example/tests/case.rs::test_case", "surface_list_blob": "d" * 40}
        self.row = {"record_id": self.record["id"], "calibration_sha": sha,
                    "test": self.record["test"], "validation": "none",
                    "test_source_blob": "b" * 40, "bridge_source_blob": "c" * 40,
                    "surface_list_blob": "d" * 40,
                    "evidence": "crates/example/tests/case.rs:50",
                    "review_basis": "the scored study explicitly uses None", "review_status": "approved"}
        self.doc = {"schema_version": 1, "binding": [self.row]}
        self.blobs = {"crates/example/tests/case.rs": "b" * 40,
                      bindings.BRIDGE: "c" * 40,
                      "scripts/calibration_surface.list": "d" * 40}

    def resolve(self, doc=None):
        return bindings.checked_bindings([self.record], doc or self.doc,
                                         lambda sha, path: self.blobs[path])

    def test_additional_historical_dependencies_are_bound(self):
        document = deepcopy(self.doc)
        document["binding"][0]["source_blobs"] = ["crates/example/tests/case.rs@" + "b" * 40]
        self.assertEqual(len(self.resolve(document)), 1)
        document["binding"][0]["source_blobs"] = ["crates/example/tests/case.rs@" + "e" * 40]
        with self.assertRaises(ValueError):
            self.resolve(document)

    def test_approval_is_per_record(self):
        other = deepcopy(self.record)
        other["id"] = "cov.other"
        row = deepcopy(self.row)
        row["record_id"] = other["id"]
        row["review_status"] = "pending_review"
        document = {"schema_version": 1, "binding": [self.row, row]}
        result = bindings.checked_bindings([self.record, other], document,
                                           lambda sha, path: self.blobs[path])
        self.assertEqual([binding["record_id"] for binding in result], [self.record["id"]])

    def test_only_reviewed_exact_proof_is_exported(self):
        self.assertEqual(self.resolve()[0]["validation"], "none")
        pending = deepcopy(self.doc)
        pending["binding"][0]["review_status"] = "pending_review"
        self.assertEqual(self.resolve(pending), [])
        for field, value in (("record_id", "other"), ("calibration_sha", "e" * 40),
                             ("test", "crates/example/tests/other.rs::test_case"),
                             ("test_source_blob", "e" * 40), ("bridge_source_blob", "e" * 40),
                             ("surface_list_blob", "e" * 40), ("validation", "unknown"),
                             ("evidence", "elsewhere:50"), ("review_status", "unrecognized")):
            with self.subTest(field=field):
                doc = deepcopy(self.doc)
                doc["binding"][0][field] = value
                with self.assertRaises(ValueError):
                    self.resolve(doc)
        duplicate = deepcopy(self.doc)
        duplicate["binding"].append(deepcopy(duplicate["binding"][0]))
        with self.assertRaises(ValueError):
            self.resolve(duplicate)


if __name__ == "__main__":
    unittest.main()
