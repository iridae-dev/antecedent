"""The global attestation check is lazy and scoped to one coordinate audit."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_interval_coordinates as coordinates


class AttestationInvocationTests(unittest.TestCase):
    def check_fixture(self, attested, *, present=True):
        records = [
            {
                "id": f"owner.{name}",
                "status": "carried_forward",
                "inference_claim": "calibrated",
                "coverage_records": [f"cov.fixture.{name}"],
            }
            for name in ("first", "second")
        ]
        files = {
            "parity/promotion.toml": {"record": records},
            "parity/coverage_records.toml": {
                "record": [{"id": row["coverage_records"][0]} for row in records] if present else []
            },
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(coordinates, "ROOT", Path(directory)),
            patch.object(coordinates, "load", side_effect=lambda path: files.get(path, {})),
            patch.object(coordinates, "emitted_by_a_test", return_value=True),
            patch.object(coordinates.cai, "attested", attested),
        ):
            return coordinates.check()

    def test_failed_global_check_rejects_every_owner_with_one_call(self):
        from unittest.mock import Mock

        attested = Mock(return_value=(False, "source changed"))
        result = self.check_fixture(attested)
        attested.assert_called_once_with()
        self.assertEqual(result["intervals"], "FAIL")
        for name in ("first", "second"):
            self.assertEqual(result["calibration"][f"owner.{name}"]["status"], "FAIL")
            self.assertIn(
                f"owner.{name}: coverage records present but not attested: source changed",
                result["errors"],
            )

    def test_next_invocation_recomputes_after_state_changes(self):
        from unittest.mock import Mock

        attested = Mock(side_effect=[(False, "pending"), (True, "current")])
        first = self.check_fixture(attested)
        second = self.check_fixture(attested)
        self.assertEqual(attested.call_count, 2)
        self.assertEqual(first["intervals"], "FAIL")
        self.assertEqual(second["intervals"], "PASS")
        self.assertTrue(all(row["status"] == "PASS" for row in second["calibration"].values()))

    def test_missing_records_do_not_start_global_attestation(self):
        from unittest.mock import Mock

        attested = Mock(side_effect=AssertionError("unnecessary attestation"))
        result = self.check_fixture(attested, present=False)
        attested.assert_not_called()
        self.assertTrue(
            all(row["status"] == "PENDING_CALIBRATION" for row in result["calibration"].values())
        )


if __name__ == "__main__":
    unittest.main()
