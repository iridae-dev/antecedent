"""Native measurement fields on the Criterion regression result. No extension import.

Every payload here comes from work that ran: a child process that really touched memory,
measured by the operating system, and the summary the Rust measurement record wrote after
timing real allocator and engine activity. Nothing is a hand-written counter.

The Rust side is `crates/antecedent-kernels/tests/native_probe.rs`. Run it with
`ANTECEDENT_MEASUREMENT_OUT=<dir>` set, then point this test at the same directory:

    ANTECEDENT_MEASUREMENT_OUT=target/native_measurement \\
        cargo test -p antecedent-kernels --test native_probe
    ANTECEDENT_MEASUREMENT_OUT=target/native_measurement \\
        ANTECEDENT_REQUIRE_NATIVE_MEASUREMENT=1 python3 scripts/test_measurement_record.py

Without the variable the Rust-emitted cases are skipped (and fail when
`ANTECEDENT_REQUIRE_NATIVE_MEASUREMENT=1`).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from _repo import REPO_ROOT, ensure_scripts_on_path

ensure_scripts_on_path()
sys.path.insert(0, str(REPO_ROOT / "parity" / "performance_lanes"))
import bench_regression as bench  # noqa: E402
import measure  # noqa: E402

ESTIMATE = {
    "mean": {
        "point_estimate": 0.01,
        "confidence_interval": {"lower_bound": -0.02, "upper_bound": 0.04},
    }
}
MIB = 1 << 20
NATIVE_DIR = os.environ.get("ANTECEDENT_MEASUREMENT_OUT")
REQUIRE_NATIVE = os.environ.get("ANTECEDENT_REQUIRE_NATIVE_MEASUREMENT") == "1"


def touch_in_child(megabytes: int) -> dict:
    """Run a child that allocates and touches `megabytes`; the kernel measures it."""
    code = (
        f"buf = bytearray({megabytes} * 1024 * 1024)\n"
        "for i in range(0, len(buf), 4096):\n"
        "    buf[i] = 1\n"
    )
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        run = measure.run_measured(
            [sys.executable, "-c", code],
            env={"PATH": "/usr/bin:/bin"},
            cwd=root,
            stdout_path=root / "out",
            stderr_path=root / "err",
            timeout=120,
        )
    if run["outcome"] != "completed":
        raise AssertionError(run["stderr_tail"])
    return run


def payload_from(run: dict) -> dict:
    """The measurement object for one measured process."""
    return {
        "peak_rss_bytes": run["peak_rss_bytes"],
        "stage_times_ns": [{"name": "process_wall", "nanos": run["wall_ns"]}],
        "cancelled": False,
        "failed": run["returncode"] != 0,
        # A Python-side trace is not native work and must never reach the report.
        "python_peak_bytes": 1,
    }


class ObservedWorkTest(unittest.TestCase):
    def test_peak_rss_attached_to_the_result_is_what_the_process_reached(self) -> None:
        small, large = touch_in_child(8), touch_in_child(128)
        small_fields = bench.comparison_result(ESTIMATE, "touch", payload_from(small))["measurement"]
        large_fields = bench.comparison_result(ESTIMATE, "touch", payload_from(large))["measurement"]
        self.assertNotIn("python_peak_bytes", large_fields)
        self.assertGreaterEqual(large_fields["peak_rss_bytes"], 128 * MIB)
        self.assertGreater(large_fields["peak_rss_bytes"], small_fields["peak_rss_bytes"] + 90 * MIB)
        self.assertEqual(large_fields["stage_times_ns"][0]["name"], "process_wall")
        self.assertGreater(large_fields["stage_times_ns"][0]["nanos"], 0)
        self.assertFalse(large_fields["failed"])
        self.assertNotIn("peak_rss_unsupported", large_fields)

    def test_a_failed_run_is_attached_as_failed_not_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            run = measure.run_measured(
                [sys.executable, "-c", "raise SystemExit(2)"],
                env={"PATH": "/usr/bin:/bin"},
                cwd=root,
                stdout_path=root / "out",
                stderr_path=root / "err",
            )
        fields = bench.comparison_result(ESTIMATE, "touch", payload_from(run))["measurement"]
        self.assertTrue(fields["failed"])
        self.assertGreater(fields["peak_rss_bytes"], 0)

    def test_absent_measurement_omits_the_object(self) -> None:
        self.assertNotIn("measurement", bench.comparison_result(ESTIMATE, "gather", None))
        python_only = bench.comparison_result(ESTIMATE, "gather", {"python_peak_bytes": 1})
        self.assertNotIn("measurement", python_only)

    def test_a_python_bool_is_not_a_byte_count(self) -> None:
        with self.assertRaises(ValueError):
            bench.measurement_fields({"allocation_bytes": True})
        with self.assertRaises(ValueError):
            bench.measurement_fields({"peak_rss_bytes": -1})

    def test_measurement_file_does_not_change_the_criterion_gate(self) -> None:
        run = touch_in_child(16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            change = root / "gather" / "change"
            change.mkdir(parents=True)
            (change / "estimates.json").write_text(json.dumps(ESTIMATE), encoding="utf-8")
            (root / "gather" / "measurement.json").write_text(
                json.dumps(payload_from(run)), encoding="utf-8"
            )
            loaded = bench.measurement_beside(change / "estimates.json")
            self.assertEqual(loaded["peak_rss_bytes"], run["peak_rss_bytes"])
            buffer = StringIO()
            with redirect_stdout(buffer):
                code = bench.main([str(root)])
            self.assertEqual(code, 0)
            self.assertIn("1 benchmark(s) compared", buffer.getvalue())


class RustEmittedMeasurementTest(unittest.TestCase):
    """The summary `antecedent-kernels` writes after timing real allocation and gathers."""

    def load(self) -> dict:
        if not NATIVE_DIR:
            if REQUIRE_NATIVE:
                self.fail("ANTECEDENT_MEASUREMENT_OUT is not set")
            self.skipTest("ANTECEDENT_MEASUREMENT_OUT is not set")
        path = Path(NATIVE_DIR) / "native_probe_scaling" / "measurement.json"
        if not path.is_file():
            if REQUIRE_NATIVE:
                self.fail(f"{path} was not written by the native_probe test")
            self.skipTest(f"{path} does not exist")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_native_summary_is_accepted_and_shows_the_work_that_ran(self) -> None:
        payload = self.load()
        fields = bench.measurement_fields(payload)
        variability = fields["variability"]
        allocated = variability["allocation_bytes"]
        # Three buffers of 1, 2 and 8 MiB were built in the timed windows.
        self.assertEqual(allocated["count"], 3)
        self.assertGreaterEqual(allocated["min"], MIB)
        self.assertGreaterEqual(allocated["max"], 8 * MIB)
        self.assertGreater(allocated["max"], allocated["min"])
        self.assertLessEqual(allocated["min"], allocated["median"])
        self.assertLessEqual(allocated["median"], allocated["max"])
        self.assertEqual(fields["allocation_bytes"], allocated["median"])
        self.assertGreaterEqual(fields["peak_live_bytes"], MIB)
        self.assertGreater(fields["allocation_count"], 0)
        stages = {stage["name"]: stage["nanos"] for stage in fields["stage_times_ns"]}
        self.assertIn("total", stages)
        # Peak RSS is a byte count on 64-bit Unix hosts and an explicit reason elsewhere.
        if sys.platform != "win32":
            self.assertGreater(fields["peak_rss_bytes"], 8 * MIB)
        else:
            self.assertTrue(fields["peak_rss_unsupported"])
        # Counters this workload never crossed are named, not reported as zero.
        for name in ("fit_count", "test_count", "spill_bytes"):
            self.assertIn(name, payload["unobserved"])
        self.assertNotIn("allocation_bytes", payload["unobserved"])

    def test_the_native_summary_rides_on_a_criterion_comparison(self) -> None:
        payload = self.load()
        result = bench.comparison_result(ESTIMATE, "native_probe_scaling", payload)
        self.assertEqual(
            result["measurement"]["variability"]["allocation_bytes"]["count"], 3
        )


if __name__ == "__main__":
    unittest.main()
