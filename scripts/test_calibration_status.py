#!/usr/bin/env python3
"""Calibration status: current by measurement, current by probe, probe owed, measurement owed."""

from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_probe as probe  # noqa: E402
import calibration_status as status  # noqa: E402
from test_calibration_probe import MEASURED, PROBED, grid_entry, probe_table  # noqa: E402

NOW = "a1" * 32
OLD = "b2" * 32


def record(rid: str, digests: bool = True, **extra) -> dict:
    rec = {
        "id": rid,
        "nominal": 0.95,
        "calibration_sha": MEASURED,
        "test": f"crates/c/tests/t.rs::{rid}",
        "grid": [grid_entry(p) for p in probe.GRID_POINTS],
    }
    if digests:
        rec["probe"] = probe_table()
    return rec | extra


def row(rid: str, **extra) -> dict:
    return {
        "record_id": rid,
        "calibration_sha": MEASURED,
        "sha": PROBED,
        "fingerprint": NOW,
        "k": 100,
        "bound": 3 / 100,
        "result": "pass",
    } | extra


class ClassifyTests(unittest.TestCase):
    def classify(self, records, rows=(), now=NOW):
        return status.classify(records, list(rows), {str(r["id"]): now for r in records})

    def test_a_stored_fingerprint_equal_to_the_tree_is_current_by_measurement(self):
        (s,) = self.classify([record("a", fingerprint=NOW)]).values()
        self.assertEqual((s.status, s.via), (status.CURRENT, "measurement"))

    def test_a_passing_probe_at_the_tree_fingerprint_is_current_by_probe(self):
        (s,) = self.classify([record("a", fingerprint=OLD)], [row("a")]).values()
        self.assertEqual((s.status, s.via), (status.CURRENT, "probe"))

    def test_a_probe_of_another_tree_does_not_count(self):
        (s,) = self.classify([record("a", fingerprint=OLD)], [row("a", fingerprint=OLD)]).values()
        self.assertEqual(s.status, status.PROBE_OWED)

    def test_failed_and_fingerprintless_probes_do_not_count(self):
        legacy = row("a")
        del legacy["fingerprint"]
        for candidate in (row("a", result="mismatch"), legacy, row("a", calibration_sha="d" * 40)):
            with self.subTest(candidate=candidate):
                (s,) = self.classify([record("a", fingerprint=OLD)], [candidate]).values()
                self.assertEqual(s.status, status.PROBE_OWED)

    def test_a_changed_tree_with_digests_owes_a_probe(self):
        (s,) = self.classify([record("a", fingerprint=OLD)]).values()
        self.assertEqual(s.status, status.PROBE_OWED)

    def test_a_changed_tree_without_digests_owes_a_measurement(self):
        (s,) = self.classify([record("a", digests=False, fingerprint=OLD)]).values()
        self.assertEqual(s.status, status.MEASUREMENT_OWED)

    def test_a_record_without_a_stored_fingerprint_is_never_current_by_measurement(self):
        (s,) = self.classify([record("a", digests=False)]).values()
        self.assertEqual(s.status, status.MEASUREMENT_OWED)
        (s,) = self.classify([record("a")]).values()
        self.assertEqual(s.status, status.PROBE_OWED)

    def test_a_test_with_no_computable_fingerprint_is_not_current(self):
        (s,) = self.classify([record("a", fingerprint=NOW)], now=None).values()
        self.assertNotEqual(s.status, status.CURRENT)

    def test_summary_counts(self):
        records = [record("a", fingerprint=NOW), record("b", fingerprint=OLD), record("c", digests=False)]
        counts = status.summary(self.classify(records, [row("b")]))
        self.assertEqual(
            (counts[status.CURRENT], counts["current_by_probe"], counts[status.MEASUREMENT_OWED]),
            (2, 1, 1),
        )


class CommandTests(unittest.TestCase):
    def run_status(self, *argv):
        records = [record("a", fingerprint=NOW), record("b", fingerprint=OLD)]
        out = io.StringIO()
        with (
            patch.object(status, "load_records", return_value=records),
            patch.object(status, "load_attestations", return_value=[]),
            patch.object(status, "tree_fingerprinter"),
            patch.object(status, "tree_fingerprints", return_value={"a": NOW, "b": NOW}),
            redirect_stdout(out),
        ):
            code = status.main(["status", *argv])
        return code, out.getvalue()

    def test_owed_records_never_fail_the_command(self):
        code, text = self.run_status()
        self.assertEqual(code, 0)
        self.assertIn("1 current", text)
        self.assertIn("probe_owed: b", text)

    def test_owed_lists_only_the_ids(self):
        self.assertEqual(self.run_status("--owed")[1].split(), ["b"])

    def test_json_carries_counts_and_per_record_status(self):
        data = json.loads(self.run_status("--json")[1])
        self.assertEqual(data["records"], {"a": status.CURRENT, "b": status.PROBE_OWED})
        self.assertEqual(data["counts"][status.CURRENT], 1)


class IntervalStateTests(unittest.TestCase):
    def state(self, allocated, records):
        return status.interval_state(allocated, {r["id"]: r for r in records})

    def test_no_allocated_record_present_is_not_measured(self):
        self.assertEqual(self.state([], []), status.NOT_MEASURED)
        self.assertEqual(self.state(["a"], [{"id": "b"}]), status.NOT_MEASURED)

    def test_every_record_present_and_passing_is_calibrated(self):
        self.assertEqual(self.state(["a", "b"], [{"id": "a"}, {"id": "b"}]), status.CALIBRATED)

    def test_a_missing_allocated_record_is_outside_the_measured_scope(self):
        self.assertEqual(self.state(["a", "b"], [{"id": "a"}]), status.OUTSIDE_MEASURED_SCOPE)

    def test_a_boundary_record_is_measured_below_nominal(self):
        self.assertEqual(
            self.state(["a", "b"], [{"id": "a", "boundary": True}, {"id": "b"}]),
            status.MEASURED_BELOW_NOMINAL,
        )

    def test_only_supported_interval_routes_are_annotated(self):
        capability = [
            {
                "id": "r",
                "inference_claim": "interval",
                "coverage_records": ["a"],
                "routes": [
                    {"name": "est", "status": "supported"},
                    {"name": "closed", "status": "closed"},
                    {"name": "point", "status": "supported", "claim": "point_only"},
                ],
            }
        ]
        rows = status.interval_routes(capability, [{"id": "a"}])
        self.assertEqual([(r["route"], r["state"]) for r in rows], [("est", status.CALIBRATED)])


class ShapeTests(unittest.TestCase):
    def test_the_shipped_records_are_well_formed(self):
        self.assertEqual(status.shape_problems(), [])

    def test_a_malformed_record_is_reported_and_exit_is_nonzero(self):
        out = io.StringIO()
        with (
            patch.object(status, "shape_problems", return_value=["coverage_records.toml x: bad"]),
            redirect_stdout(out),
        ):
            self.assertEqual(status.main(["--check"]), 1)
        self.assertIn("bad", out.getvalue())

    def test_informational_subcommands_exit_zero(self):
        for command in ("intervals", "backlog", "plan"):
            with self.subTest(command=command), redirect_stdout(io.StringIO()):
                self.assertEqual(status.main([command]), 0)


if __name__ == "__main__":
    unittest.main()
