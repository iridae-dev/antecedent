"""Native measurement fields on the Criterion regression result. No extension import."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import bench_regression as bench  # noqa: E402

ESTIMATE = {
    "mean": {
        "point_estimate": 0.01,
        "confidence_interval": {"lower_bound": -0.02, "upper_bound": 0.04},
    }
}

NATIVE = {
    "python_peak_bytes": 99,
    "python_retained_allocations": 4,
    "peak_rss_unsupported": "peak RSS is unsupported here: antecedent-core does not depend on libc",
    "allocation_bytes": 256,
    "copy_bytes": 256,
    "borrow_bytes": 256,
    "source_passes": 1,
    "fit_count": 2,
    "test_count": 1,
    "retained_outputs": 1,
    "spill_bytes": 0,
    "threads": 1,
    "device": "gpu0",
    "stage_times_ns": [{"name": "estimate", "nanos": 100}],
    "cancelled": False,
    "failed": True,
    "variability": {"allocation_bytes": {"count": 2, "min": 10, "median": 20, "max": 30}},
}


class MeasurementSchemaTest(unittest.TestCase):
    def test_native_fields_round_trip_and_python_allocator_keys_do_not(self) -> None:
        result = bench.comparison_result(ESTIMATE, "gather", NATIVE)
        measurement = result["measurement"]
        self.assertNotIn("python_peak_bytes", measurement)
        self.assertNotIn("python_retained_allocations", measurement)
        self.assertEqual(measurement["allocation_bytes"], 256)
        self.assertEqual(measurement["copy_bytes"], 256)
        self.assertEqual(measurement["borrow_bytes"], 256)
        self.assertEqual(measurement["source_passes"], 1)
        self.assertEqual(measurement["fit_count"], 2)
        self.assertEqual(measurement["test_count"], 1)
        self.assertEqual(measurement["retained_outputs"], 1)
        self.assertEqual(measurement["spill_bytes"], 0)
        self.assertEqual(measurement["threads"], 1)
        self.assertEqual(measurement["device"], "gpu0")
        self.assertEqual(measurement["stage_times_ns"], [{"name": "estimate", "nanos": 100}])
        self.assertFalse(measurement["cancelled"])
        self.assertTrue(measurement["failed"])
        self.assertEqual(
            measurement["variability"]["allocation_bytes"],
            {"count": 2, "min": 10, "median": 20, "max": 30},
        )
        self.assertIn("libc", measurement["peak_rss_unsupported"])
        self.assertNotIn("peak_rss_bytes", measurement)

    def test_absent_measurement_omits_the_object(self) -> None:
        result = bench.comparison_result(ESTIMATE, "gather", None)
        self.assertNotIn("measurement", result)
        python_only = bench.comparison_result(ESTIMATE, "gather", {"python_peak_bytes": 1})
        self.assertNotIn("measurement", python_only)

    def test_cancellation_and_failure_flags_round_trip(self) -> None:
        result = bench.comparison_result(
            ESTIMATE, "gather", {"cancelled": True, "failed": False}
        )
        self.assertTrue(result["measurement"]["cancelled"])
        self.assertFalse(result["measurement"]["failed"])

    def test_a_python_bool_is_not_a_byte_count(self) -> None:
        with self.assertRaises(ValueError):
            bench.measurement_fields({"allocation_bytes": True})

    def test_measurement_file_does_not_change_the_criterion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            change = root / "gather" / "change"
            change.mkdir(parents=True)
            (change / "estimates.json").write_text(json.dumps(ESTIMATE), encoding="utf-8")
            (root / "gather" / "measurement.json").write_text(json.dumps(NATIVE), encoding="utf-8")
            loaded = bench.measurement_beside(change / "estimates.json")
            self.assertEqual(loaded["allocation_bytes"], 256)
            attached = bench.comparison_result(ESTIMATE, "gather", loaded)["measurement"]
            self.assertNotIn("python_peak_bytes", attached)
            buffer = StringIO()
            with redirect_stdout(buffer):
                code = bench.main([str(root)])
            self.assertEqual(code, 0)
            self.assertIn("1 benchmark(s) compared", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
