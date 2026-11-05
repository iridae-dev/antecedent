#!/usr/bin/env python3
"""Calibration status: current by measurement, current by probe, probe owed, measurement owed."""

from __future__ import annotations

import io
import json
import sys
import tempfile
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
    def run_status(self, *argv, routes=()):
        records = [record("a", fingerprint=NOW), record("b", fingerprint=OLD)]
        out = io.StringIO()
        with (
            patch.object(status, "load_records", return_value=records),
            patch.object(status, "interval_routes", return_value=list(routes)),
            patch.object(status, "load_route_map", return_value={}),
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
        self.assertIn("fingerprint_changed: b", text)

    def test_owed_lists_only_the_ids(self):
        self.assertEqual(self.run_status("--owed")[1].split(), ["b"])

    def test_owed_lists_every_owed_record_and_route(self):
        routes = [
            route_row("r", "mapped", NOT_MEASURED, groups=["g_file: filter"]),
            route_row("r", "slot", NOT_MEASURED, slots=["cov.slot"]),
            route_row("r", "done", status.CALIBRATED),
        ]
        self.assertEqual(
            self.run_status("--owed", routes=routes)[1].split(),
            ["b", "route:r/mapped", "cov.slot"],
        )

    def test_the_summary_counts_the_owed_items_by_reason(self):
        routes = [route_row("r", "mapped", NOT_MEASURED, groups=["g_file"])]
        text = self.run_status(routes=routes)[1]
        self.assertIn("owed: 2 item(s): 1 no_record, 1 fingerprint_changed, 0 probe_mismatch", text)
        self.assertIn("(0 record(s) never stored a fingerprint)", text)

    def test_json_lists_the_owed_items_with_their_reasons(self):
        routes = [route_row("r", "mapped", NOT_MEASURED, groups=["g_file: f"])]
        data = json.loads(self.run_status("--json", routes=routes)[1])
        self.assertEqual(data["owed_by_reason"], {"no_record": 1, "fingerprint_changed": 1, "probe_mismatch": 0})
        self.assertEqual(
            [(i["id"], i["reason"]) for i in data["owed"]],
            [("b", "fingerprint_changed"), ("route:r/mapped", "no_record")],
        )
        self.assertEqual(data["owed"][1]["groups"], ["g_file: f"])

    def test_json_carries_counts_and_per_record_status(self):
        data = json.loads(self.run_status("--json")[1])
        self.assertEqual(data["records"], {"a": status.CURRENT, "b": status.PROBE_OWED})
        self.assertEqual(data["counts"][status.CURRENT], 1)


NOT_MEASURED = status.NOT_MEASURED


def route_row(record_id, route, state, groups=(), slots=()):
    return {
        "record": record_id,
        "route": route,
        "allocated": len(slots),
        "measured": 0,
        "mapped_groups": list(groups),
        "missing_slots": list(slots),
        "state": state,
    }


class OwedItemTests(unittest.TestCase):
    def owed(self, records, rows=(), routes=(), fingerprints=None):
        fingerprints = fingerprints or {str(r["id"]): NOW for r in records}
        return status.owed_items(records, list(rows), fingerprints, list(routes))

    def test_a_current_record_is_not_owed(self):
        self.assertEqual(self.owed([record("a", fingerprint=NOW)]), [])

    def test_a_changed_or_never_stored_fingerprint_is_fingerprint_changed(self):
        items = self.owed([record("a", fingerprint=OLD), record("b")])
        self.assertEqual([(i["id"], i["reason"]) for i in items], [("a", "fingerprint_changed"), ("b", "fingerprint_changed")])
        self.assertEqual([i["stored_fingerprint"] for i in items], [True, False])

    def test_a_probe_that_does_not_cover_the_tree_is_a_probe_mismatch(self):
        items = self.owed([record("a", fingerprint=OLD)], rows=[row("a", result="mismatch")])
        self.assertEqual([(i["id"], i["reason"]) for i in items], [("a", "probe_mismatch")])

    def test_a_covering_probe_makes_the_record_current_and_not_owed(self):
        self.assertEqual(self.owed([record("a", fingerprint=OLD)], rows=[row("a")]), [])

    def test_a_route_without_a_record_is_owed_with_the_groups_that_measure_it(self):
        routes = [route_row("cap", "measured", status.CALIBRATED), route_row("cap", "new", NOT_MEASURED, groups=["f: t"])]
        (item,) = self.owed([], routes=routes)
        self.assertEqual(
            (item["id"], item["reason"], item["groups"], item["unwired"]),
            ("route:cap/new", "no_record", ["f: t"], False),
        )

    def test_an_allocated_record_that_does_not_exist_is_an_owed_slot_once(self):
        routes = [
            route_row("cap", "x", NOT_MEASURED, slots=["cov.s1", "cov.s2"]),
            route_row("cap", "y", NOT_MEASURED, slots=["cov.s1"]),
        ]
        items = self.owed([], routes=routes)
        self.assertEqual([(i["kind"], i["id"]) for i in items], [("slot", "cov.s1"), ("slot", "cov.s2")])

    def test_an_unmapped_route_is_owed_but_unwired(self):
        (item,) = self.owed([], routes=[route_row("cap", "lost", status.UNMAPPED)])
        self.assertEqual((item["id"], item["unwired"]), ("route:cap/lost", True))

    def test_a_route_that_publishes_no_interval_is_never_owed(self):
        self.assertEqual(self.owed([], routes=[route_row("cap", "refuses", status.NO_INTERVAL)]), [])

    def test_the_counts_cover_every_reason(self):
        items = self.owed([record("a", fingerprint=OLD)], routes=[route_row("c", "n", NOT_MEASURED, groups=["f"])])
        self.assertEqual(status.owed_counts(items), {"no_record": 1, "fingerprint_changed": 1, "probe_mismatch": 0})


class RouteMapTests(unittest.TestCase):
    CAPABILITY = [
        {
            "id": "cap",
            "inference_claim": "interval",
            "coverage_records": ["cov.allocated"],
            "routes": [
                {"name": "analytic", "status": "supported"},
                {"name": "bootstrap", "status": "supported"},
                {"name": "refuses", "status": "supported"},
                {"name": "lost", "status": "supported"},
            ],
        }
    ]

    def rows(self, records, route_map):
        return {r["route"]: r for r in status.interval_routes(self.CAPABILITY, records, route_map)}

    def test_a_mapped_route_is_measured_by_the_records_of_its_groups(self):
        mapped = {("cap", "bootstrap"): {"groups": [("boot_calibration", "")], "no_interval": ""}}
        rows = self.rows([{"id": "cov.allocated", "test": "crates/c/tests/other.rs::t"}], mapped)
        self.assertEqual(rows["bootstrap"]["state"], status.NOT_MEASURED)
        self.assertEqual(rows["bootstrap"]["mapped_groups"], ["boot_calibration"])
        records = [{"id": "cov.allocated", "test": "crates/c/tests/other.rs::t"}, {"id": "cov.boot", "test": "crates/c/tests/boot_calibration.rs::t"}]
        self.assertEqual(self.rows(records, mapped)["bootstrap"]["state"], status.CALIBRATED)
        below = [records[0], records[1] | {"boundary": True}]
        self.assertEqual(self.rows(below, mapped)["bootstrap"]["state"], status.MEASURED_BELOW_NOMINAL)

    def test_an_unmapped_route_keeps_the_allocation_of_its_capability_record(self):
        rows = self.rows([{"id": "cov.allocated"}], {})
        self.assertEqual(rows["analytic"]["state"], status.CALIBRATED)
        rows = self.rows([], {})
        self.assertEqual(rows["analytic"]["state"], status.NOT_MEASURED)
        self.assertEqual(rows["analytic"]["missing_slots"], ["cov.allocated"])

    def test_a_route_with_no_allocation_no_record_and_no_group_is_unmapped(self):
        capability = [{"id": "c", "inference_claim": "interval", "routes": [{"name": "r", "status": "supported"}]}]
        (row_,) = status.interval_routes(capability, [], {})
        self.assertEqual(row_["state"], status.UNMAPPED)

    def test_a_route_declared_to_publish_no_interval_is_no_interval(self):
        rows = self.rows([], {("cap", "refuses"): {"groups": [], "no_interval": "refuses the request"}})
        self.assertEqual(rows["refuses"]["state"], status.NO_INTERVAL)

    def test_the_header_counts_every_state_including_not_measured_and_unmapped(self):
        capability = [{"id": "c", "inference_claim": "interval", "routes": [{"name": "r", "status": "supported"}]}]
        out = io.StringIO()
        with (
            patch.object(status, "read_text", return_value=""),
            patch.object(status.tomllib, "loads", return_value={"record": capability}),
            patch.object(status, "load_records", return_value=[]),
            patch.object(status, "load_route_map", return_value={}),
            redirect_stdout(out),
        ):
            self.assertEqual(status.main(["intervals"]), 0)
        self.assertIn("0 not_measured, 1 unmapped", out.getvalue())

    def test_the_route_file_loads_groups_and_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes.toml"
            path.write_text(
                '[[route]]\nrecord="cap"\nroutes=["a","b"]\ngroups=["f: t","g"]\n'
                '[[route]]\nrecord="cap"\nroutes=["c"]\nno_interval="refuses"\n',
                encoding="utf-8",
            )
            loaded = status.load_route_map(path)
        self.assertEqual(loaded[("cap", "a")]["groups"], [("f", "t"), ("g", "")])
        self.assertIs(loaded[("cap", "a")], loaded[("cap", "b")])
        self.assertEqual(loaded[("cap", "c")]["no_interval"], "refuses")

    def test_the_route_file_shape_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parity").mkdir()
            (root / "crates" / "c" / "tests").mkdir(parents=True)
            (root / "crates" / "c" / "tests" / "real_calibration.rs").write_text("", encoding="utf-8")
            (root / "parity" / "calibration_routes.toml").write_text(
                '[[route]]\nrecord="cap"\nroutes=["a"]\ngroups=["real_calibration: t"]\n'
                '[[route]]\nrecord="cap"\nroutes=["a"]\ngroups=["ghost_calibration"]\n'
                '[[route]]\nrecord="cap"\nroutes=["b"]\n'
                '[[route]]\nrecord="cap"\nroutes=["c"]\ngroups=["real_calibration"]\nno_interval="x"\n',
                encoding="utf-8",
            )
            problems = status.route_map_problems(root)
        text = "\n".join(problems)
        self.assertIn("mapped twice", text)
        self.assertIn("ghost_calibration", text)
        self.assertEqual(len([p for p in problems if "either `groups` or `no_interval`" in p]), 2)

    def test_the_shipped_route_map_is_well_formed_and_names_real_routes(self):
        self.assertEqual(status.route_map_problems(), [])
        capability = status.tomllib.loads(status.read_text(status.CAPABILITY_RECORDS))["record"]
        names = {(str(r["id"]), str(x.get("name"))) for r in capability for x in r.get("routes") or []}
        self.assertEqual(sorted(set(status.load_route_map()) - names), [])


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
