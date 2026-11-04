#!/usr/bin/env python3
"""Probe sizing, digest comparison and attestation rows.

Pure Python: nothing here builds or runs a replicate.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_probe as probe  # noqa: E402
import collect_coverage_records as collector  # noqa: E402

MEASURED = "a" * 40
PROBED = "b" * 40
HEAD = "c" * 40
RECORD_ID = "cov.example"
FINGERPRINT = "f" * 64


def digest(point: int, k: int) -> str:
    return f"{point:02x}{k:030x}"


def probe_table(prefixes=probe.PREFIXES) -> dict:
    return {f"p{p}": {str(k): digest(p, k) for k in prefixes} for p in probe.GRID_POINTS}


def grid_entry(point: int, observed=0.95, replicates=400, boundary=False, role="gated") -> dict:
    mcse = math.sqrt(observed * (1 - observed) / replicates)
    return {
        "point": point,
        "n_min": 100 * (point + 1),
        "n_max": 100 * (point + 1),
        "observed": observed,
        "mcse": mcse,
        "replicates": replicates,
        "boundary": boundary,
        "role": role,
    }


def record(**extra) -> dict:
    base = {
        "id": RECORD_ID,
        "nominal": 0.95,
        "calibration_sha": MEASURED,
        "test": "crates/antecedent/tests/example_calibration.rs::example",
        "dgp": "crates/antecedent/tests/example_calibration.rs::dgp",
        "grid": [grid_entry(p) for p in probe.GRID_POINTS],
        "probe": probe_table(),
    }
    return base | extra


def row(**extra) -> dict:
    base = {
        "record_id": RECORD_ID,
        "calibration_sha": MEASURED,
        "sha": PROBED,
        "fingerprint": FINGERPRINT,
        "k": 100,
        "bound": 3 / 100,
        "result": "pass",
    }
    return base | extra


class SizingTests(unittest.TestCase):
    def test_acceptance_floor_is_the_band_then_the_precision_floor(self) -> None:
        mcse400 = math.sqrt(0.95 * 0.05 / 400)
        self.assertAlmostEqual(probe.acceptance_floor(0.95, 400), 0.95 - 3 * mcse400)
        mcse2000 = math.sqrt(0.95 * 0.05 / 2000)
        # From 1000 replicates the 2-MCSE floor is tighter than the 3-MCSE band.
        self.assertAlmostEqual(probe.acceptance_floor(0.95, 2000), 0.95 - 2 * mcse2000)

    def test_margin_is_observed_minus_floor_at_the_worst_point(self) -> None:
        rec = record(grid=[grid_entry(0, 0.95), grid_entry(1, 0.94), grid_entry(2, 0.96)])
        floor = probe.acceptance_floor(0.95, 400)
        self.assertAlmostEqual(probe.record_margin(rec), 0.94 - floor)

    def test_boundary_point_margin_is_three_mcse(self) -> None:
        entry = grid_entry(0, 0.80, boundary=True, role="named_boundary")
        rec = record(grid=[entry, grid_entry(1), grid_entry(2)])
        self.assertAlmostEqual(probe.point_margin(rec, entry), 3 * entry["mcse"])

    def test_bounded_k_is_the_smallest_stored_prefix_at_least_three_over_margin(self) -> None:
        rec = record()
        margin = probe.record_margin(rec)
        need = math.ceil(3 / margin)
        self.assertEqual(probe.required_k(rec), need)
        expected = next(k for k in probe.PREFIXES if k >= need)
        self.assertEqual(probe.bounded_k(rec), expected)
        self.assertGreaterEqual(3 / expected, 0)
        self.assertLessEqual(3 / expected, margin)

    def test_bounded_k_uses_a_prefix_stored_at_every_grid_point(self) -> None:
        table = probe_table()
        del table["p1"]["100"]
        rec = record(probe=table)
        self.assertNotIn(100, probe.stored_prefixes(rec))
        self.assertEqual(probe.bounded_k(rec), 320)

    def test_a_margin_beyond_the_longest_prefix_is_owed(self) -> None:
        rec = record(grid=[grid_entry(p, 0.9185) for p in probe.GRID_POINTS])
        self.assertGreater(probe.required_k(rec), probe.PREFIXES[-1])
        self.assertIsNone(probe.bounded_k(rec))

    def test_a_record_at_or_below_its_floor_is_owed(self) -> None:
        rec = record(grid=[grid_entry(p, 0.90) for p in probe.GRID_POINTS])
        self.assertLessEqual(probe.record_margin(rec), 0)
        self.assertIsNone(probe.required_k(rec))
        self.assertIsNone(probe.bounded_k(rec))

    def test_a_record_without_digests_is_owed_without_a_run(self) -> None:
        rec = record()
        del rec["probe"]
        self.assertEqual(probe.stored_prefixes(rec), [])
        for mode in ("quick", "bounded"):
            (item,) = probe.plan([rec], mode)
            self.assertIsNone(item.k)
            self.assertIn("no digests", item.owed_reason)

    def test_plan_modes(self) -> None:
        rec = record()
        (quick,) = probe.plan([rec], "quick")
        (bounded,) = probe.plan([rec], "bounded")
        self.assertEqual(quick.k, probe.QUICK_K)
        self.assertEqual(bounded.k, probe.bounded_k(rec))
        hard = record(grid=[grid_entry(p, 0.9185) for p in probe.GRID_POINTS])
        (owed,) = probe.plan([hard], "bounded")
        self.assertIsNone(owed.k)
        self.assertIn("longest stored prefix", owed.owed_reason)


class CompareTests(unittest.TestCase):
    def observed(self, k: int) -> dict:
        return {
            p: {str(j): digest(p, j) for j in probe.PREFIXES if j <= k} for p in probe.GRID_POINTS
        }

    def test_identical_digests_match(self) -> None:
        self.assertEqual(probe.compare(record(), self.observed(100), 100), [])

    def test_a_changed_prefix_is_located(self) -> None:
        seen = self.observed(100)
        seen[1]["32"] = "0" * 32
        problems = probe.compare(record(), seen, 100)
        self.assertEqual(problems, ["grid point 1: the first 32 replicates differ"])

    def test_a_late_divergence_shows_only_at_the_long_prefix(self) -> None:
        seen = self.observed(100)
        seen[2]["100"] = "f" * 32
        self.assertEqual(
            probe.compare(record(), seen, 100), ["grid point 2: the first 100 replicates differ"]
        )
        # The same run judged at k = 32 cannot see it.
        self.assertEqual(probe.compare(record(), seen, 32), [])

    def test_a_missing_point_or_a_short_run_does_not_match(self) -> None:
        seen = self.observed(100)
        del seen[0]
        self.assertTrue(any("grid point 0" in p for p in probe.compare(record(), seen, 100)))
        short = self.observed(32)
        problems = probe.compare(record(), short, 100)
        self.assertEqual(len(problems), 3)
        self.assertTrue(all("fewer than 100" in p for p in problems))

    def test_a_prefix_the_record_never_stored_cannot_match(self) -> None:
        self.assertTrue(probe.compare(record(), self.observed(100), 50))


class AttestationTests(unittest.TestCase):
    def test_a_passing_row_of_enough_replicates_covers(self) -> None:
        self.assertTrue(probe.covers(record(), row()))

    def test_rejections(self) -> None:
        rec = record()
        need = probe.bounded_k(rec)
        cases = {
            "mismatch result": row(result="mismatch"),
            "another record": row(record_id="cov.other"),
            "another measurement": row(calibration_sha="d" * 40),
            "too few replicates": row(k=10, bound=0.3) if need > 10 else None,
            "a bound that is not 3/k": row(bound=0.5),
            "a k that was never stored": row(k=50, bound=3 / 50),
            "a boolean bound": row(bound=True),
        }
        for label, candidate in cases.items():
            if candidate is None:
                continue
            with self.subTest(label):
                self.assertFalse(probe.covers(rec, candidate))

    def test_a_record_with_no_stored_digests_is_never_covered(self) -> None:
        rec = record()
        del rec["probe"]
        self.assertFalse(probe.covers(rec, row()))

    def test_problems_flag_malformed_and_orphaned_rows(self) -> None:
        rec = record()
        self.assertEqual(probe.attestation_problems([row()], [rec]), [])
        self.assertTrue(probe.attestation_problems([{"record_id": RECORD_ID}], [rec]))
        self.assertTrue(probe.attestation_problems([row(result="maybe")], [rec]))
        self.assertTrue(probe.attestation_problems([row(k=7, bound=3 / 7)], [rec]))
        self.assertTrue(probe.attestation_problems([row(bound=0.5)], [rec]))
        self.assertTrue(probe.attestation_problems([row(sha="short")], [rec]))
        self.assertTrue(probe.attestation_problems([row(fingerprint="short")], [rec]))
        legacy = row()
        del legacy["fingerprint"]
        self.assertTrue(probe.attestation_problems([legacy], [rec]))
        self.assertTrue(probe.attestation_problems([row(record_id="cov.gone")], [rec]))
        self.assertTrue(probe.attestation_problems([row(calibration_sha="e" * 40)], [rec]))

    def test_merge_keeps_the_longest_pass_and_a_mismatch_erases_the_rest(self) -> None:
        bounded = row(k=100, bound=0.03)
        quick = row(k=10, bound=0.3)
        self.assertEqual(probe.merge_results([bounded], [quick]), [bounded])
        longer = row(k=320, bound=3 / 320)
        self.assertEqual(probe.merge_results([bounded], [longer]), [longer])
        later = row(sha=HEAD, k=10, bound=0.3)
        self.assertEqual(len(probe.merge_results([bounded], [later])), 2)
        miss = row(sha=HEAD, result="mismatch")
        self.assertEqual(probe.merge_results([bounded, later], [miss]), [miss])
        other = row(record_id="cov.other")
        self.assertIn(other, probe.merge_results([other], [miss]))

    def test_rows_round_trip_through_the_file(self) -> None:
        import tomllib

        rows = [row(), row(record_id="cov.b", k=320, bound=3 / 320, result="mismatch")]
        parsed = tomllib.loads(probe.render_attestations(rows))["attestation"]
        self.assertEqual(
            sorted(parsed, key=lambda r: r["record_id"]), sorted(rows, key=lambda r: r["record_id"])
        )
        mine = [r for r in parsed if r["record_id"] == RECORD_ID]
        self.assertEqual(probe.attestation_problems(mine, [record()]), [])

    def test_pruning_drops_rows_of_replaced_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attestations.toml"
            path.write_text(probe.render_attestations([row(), row(record_id="cov.b")]), encoding="utf-8")
            dropped = probe.prune_attestations([record()], path)
            self.assertEqual(dropped, 1)
            self.assertIn("cov.example", path.read_text(encoding="utf-8"))
            self.assertNotIn("cov.b", path.read_text(encoding="utf-8"))
            self.assertEqual(probe.prune_attestations([record(calibration_sha="f" * 40)], path), 1)

    def test_log_parsing(self) -> None:
        digests = {"10": digest(1, 10)}
        line = (
            'calibration-record {"id": "cov.example", "grid_point": 1, "prefix_digests": '
            + str(digests).replace("'", '"')
            + "}"
        )
        seen, measured = probe.parse_log(f"noise\n{line}\ncalibration-measured-at {HEAD}\n")
        self.assertEqual(seen, {("cov.example", 1): digests})
        self.assertEqual(measured, HEAD)


class JudgeTests(unittest.TestCase):
    class Group:
        safe = "suite__test"
        label = "suite: test"

    def run_judge(self, observed_for) -> tuple[list[dict], dict]:
        rec = record()
        (item,) = probe.plan([rec], "bounded")
        tasks = [probe.Task(self.Group(), point, item.k, [rec]) for point in probe.GRID_POINTS]
        results = {
            task.stem: {
                "status": 0,
                "measured_at": HEAD,
                "observed": {(RECORD_ID, task.point): observed_for(task.point, item.k)},
            }
            for task in tasks
        }
        return probe.judge([item], tasks, results, HEAD, {RECORD_ID: FINGERPRINT})

    def test_matching_digests_write_a_covering_pass_row(self) -> None:
        rows, verdicts = self.run_judge(
            lambda p, k: {str(j): digest(p, j) for j in probe.PREFIXES if j <= k}
        )
        (written,) = rows
        self.assertEqual((written["result"], written["sha"]), ("pass", HEAD))
        self.assertEqual(written["fingerprint"], FINGERPRINT)
        self.assertTrue(probe.covers(record(), written))
        self.assertTrue(verdicts[RECORD_ID].startswith("pass"))

    def test_a_mismatch_writes_a_mismatch_row_that_covers_nothing(self) -> None:
        rows, verdicts = self.run_judge(
            lambda p, k: {str(j): digest(p, j + (p == 2)) for j in probe.PREFIXES if j <= k}
        )
        (written,) = rows
        self.assertEqual(written["result"], "mismatch")
        self.assertFalse(probe.covers(record(), written))
        self.assertTrue(verdicts[RECORD_ID].startswith("owed (mismatch)"))


class CollectorTests(unittest.TestCase):
    def points(self, table) -> list[dict]:
        return [{"grid_point": p, "prefix_digests": table(p)} for p in probe.GRID_POINTS]

    def test_probe_table_keeps_each_points_digests(self) -> None:
        table = collector.probe_table(
            RECORD_ID, self.points(lambda p: {str(k): "ab" * 16 for k in (100, 10)})
        )
        self.assertEqual(sorted(table), ["p0", "p1", "p2"])
        self.assertEqual(list(table["p0"]), ["10", "100"])

    def test_lines_without_digests_leave_the_record_without_a_probe(self) -> None:
        self.assertIsNone(collector.probe_table(RECORD_ID, self.points(lambda p: None)))

    def test_malformed_digests_are_refused(self) -> None:
        for bad in ({"7": "ab" * 16}, {"10": "xyz"}, {}):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                collector.probe_table(RECORD_ID, self.points(lambda p, bad=bad: bad))

    def test_the_probe_is_written_after_the_role_as_a_nested_table(self) -> None:
        text = collector.toml_value(
            "probe", {"p0": {"10": "ab" * 16}, "p1": {"10": "cd" * 16}, "p2": {"10": "ef" * 16}}
        )
        import tomllib

        parsed = tomllib.loads(f"probe = {text}")["probe"]
        self.assertEqual(parsed["p1"]["10"], "cd" * 16)
        self.assertIn("probe", collector.OPTIONAL_FIELDS)
        self.assertIn("prefix_digests", collector.POINT_FIELDS)


if __name__ == "__main__":
    unittest.main()
