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
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_probe as probe  # noqa: E402
import calibration_judgments as judgments  # noqa: E402
import calibration_status as status  # noqa: E402
from evidence_earning import measurement_binding, measurement_digest, probe_binding  # noqa: E402
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
    if "fingerprint" in extra:
        # A stored manifest; the tree "now" is `manifest:<NOW>`.
        extra["dependency_manifest"] = extra.pop("fingerprint")
    return rec | extra


def row(rid: str, **extra) -> dict:
    return {
        "record_id": rid,
        "calibration_sha": MEASURED,
        "sha": PROBED,
        "dependency_manifest": NOW,
        "k": 100,
        "bound": 3 / 100,
        "result": "pass",
    } | {("dependency_manifest" if k == "fingerprint" else k): v for k, v in extra.items()}


def with_digests(records, rows):
    """Bind each probe row to its record's measurement, as a passing probe does."""
    by_id = {r["id"]: r for r in records}
    return [r if "measurement_digest" in r or r["record_id"] not in by_id
            else {**r, "measurement_digest": measurement_digest(by_id[r["record_id"]])} for r in rows]


class ClassifyTests(unittest.TestCase):
    def classify(self, records, rows=(), now=NOW):
        now = "manifest:" + now if now else None
        return status.classify(records, with_digests(records, list(rows)), {str(r["id"]): now for r in records})

    def test_a_matching_manifest_is_current_by_measurement(self):
        (s,) = self.classify([record("a", fingerprint=NOW)]).values()
        self.assertEqual((s.status, s.via), (status.CURRENT, "measurement"))

    def test_a_passing_probe_with_a_matching_manifest_is_current_by_probe(self):
        (s,) = self.classify([record("a", fingerprint=OLD)], [row("a")]).values()
        self.assertEqual((s.status, s.via), (status.CURRENT, "probe"))

    def test_a_probe_of_another_tree_does_not_count(self):
        (s,) = self.classify([record("a", fingerprint=OLD)], [row("a", fingerprint=OLD)]).values()
        self.assertEqual(s.status, status.PROBE_OWED)

    def test_failed_and_fingerprintless_probes_do_not_count(self):
        legacy = row("a")
        del legacy["dependency_manifest"]
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

    def test_a_record_without_a_manifest_is_never_current_by_measurement(self):
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


class JudgmentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.root.joinpath("parity").mkdir()
        self.record = record("a") | {"fingerprint": OLD, "dependency_manifest": "pass:calibration:a"}
        self.probe = with_digests([self.record], [row("a") | {
            "dependency_manifest": "pass:probe:a", "fingerprint": NOW,
        }])[0]
        self.measurement_address = "c3" * 32
        self.probe_address = "d4" * 32
        self.reviewed = "e5" * 32
        self.checker = Mock()
        self.checker.check.return_value = {"current": False}
        self.checker.pass_index = {
            "pass:calibration:a": self.measurement_address, "pass:probe:a": self.probe_address,
        }
        self.manifests = {
            self.measurement_address: {
                "record": measurement_binding(self.record), "nix_build": {"execution": "exact-binary"},
            },
            self.probe_address: {
                "record": probe_binding(self.probe), "nix_build": {"execution": "exact-binary"},
            },
        }
        self.checker.load.side_effect = self.manifests.__getitem__
        self.checker.prior_receipt.side_effect = self.prior_receipt
        self.checker.judgment_fingerprint.return_value = self.reviewed

    def prior_receipt(self, binding, address):
        manifest = self.manifests[address]
        if manifest["record"] != binding:
            raise ValueError("prior receipt does not bind this record")
        return manifest

    def carry(self, attestations):
        anchored = judgments.new_chain(self.checker, self.record, attestations)
        start, reviewed = judgments.review(self.checker, self.record, anchored, None, attestations)
        anchored["chain"].append(dict(zip(judgments.CHAIN_FIELDS,
            [start, reviewed, "Reviewed the revised implementation.", "reviewer"], strict=True)))
        return anchored

    def currency(self, attestation):
        with (
            patch("evidence_manifests.Checker", return_value=self.checker),
            patch.object(status, "tree_fingerprinter", return_value=object()),
            patch.object(status, "_same_platform", return_value=True),
            patch.object(status, "load_attestations", return_value=attestation),
        ):
            now = status.tree_fingerprints([self.record], root=self.root)
        return status.classify([self.record], attestation, now)["a"]

    def test_judgment_binds_the_measured_receipt_and_remains_rooted(self):
        with patch("evidence_fingerprint.coverage_fingerprint", return_value=NOW):
            chain = self.carry([])
            judgments.write(self.root, {judgments.key(chain): chain})
            self.assertEqual(self.currency([]), status.Status(status.CURRENT, "judgment", 1))
        self.assertEqual(judgments.references(self.root), {self.measurement_address})
        self.checker.judgment_fingerprint.assert_called_with(
            measurement_binding(self.record), self.measurement_address, context=[self.measurement_address, NOW],
        )
        self.checker.judgment_fingerprint.return_value = OLD
        with patch("evidence_fingerprint.coverage_fingerprint", return_value=NOW):
            self.assertEqual(self.currency([]).status, status.PROBE_OWED)

    def test_bounded_probe_is_a_connected_previous_carry(self):
        with (
            patch("evidence_fingerprint.coverage_fingerprint", return_value=NOW),
            patch("evidence_platforms.receipt_platform", return_value="platform"),
        ):
            chain = self.carry([self.probe])
            self.assertEqual(chain["probe_receipt"], self.probe_address)
            self.assertEqual(chain["chain"][0]["from_fingerprint"], self.probe["fingerprint"])
            judgments.write(self.root, {judgments.key(chain): chain})
            self.assertEqual(self.currency([self.probe]), status.Status(status.CURRENT, "judgment", 2))
            mismatch = self.probe | {"result": "mismatch"}
            self.assertEqual(self.currency([mismatch]).status, status.PROBE_OWED)
            chain["chain"][0]["from_fingerprint"] = OLD
            with self.assertRaisesRegex(ValueError, "does not begin"):
                judgments.review(self.checker, self.record, chain, None, [])
        self.assertEqual(judgments.references(self.root), {self.measurement_address, self.probe_address})

    def test_disconnected_or_unexecuted_provenance_is_refused(self):
        self.manifests[self.measurement_address]["nix_build"] = {}
        with self.assertRaisesRegex(ValueError, "exact binary"):
            judgments.new_chain(self.checker, self.record, [])
        self.manifests[self.measurement_address]["nix_build"] = {"execution": "exact-binary"}
        chain = judgments.new_chain(self.checker, self.record, []) | {"probe_receipt": self.probe_address}
        self.manifests[self.probe_address]["record"]["measurement_digest"] = OLD
        with self.assertRaisesRegex(ValueError, "exact measurement"):
            judgments.source_binding(self.checker, self.record, chain)

    def test_justify_extends_a_saved_chain_and_reports_its_length(self):
        output = io.StringIO()
        command = ["justify", "--record", "a", "--reason", "Reviewed.", "--author", "Reviewer"]
        with (
            patch.object(status, "ROOT", self.root),
            patch.object(status, "load_records", return_value=[self.record]),
            patch.object(status, "load_attestations", return_value=[]),
            patch.object(status, "current_ids", return_value=[]),
            patch.object(status, "tree_fingerprinter", return_value=object()),
            patch("evidence_manifests.Checker", return_value=self.checker),
            patch("evidence_fingerprint.coverage_fingerprint", return_value=NOW),
            redirect_stdout(output),
        ):
            self.assertEqual(status.main(command), 0)
            revised = "f6" * 32
            self.checker.judgment_fingerprint.return_value = revised
            self.assertEqual(status.main(command), 0)
        saved, problems = judgments.load(self.root)
        self.assertEqual(problems, [])
        (saved_row,) = saved.values()
        chain = saved_row["chain"]
        self.assertEqual([item["from_fingerprint"] for item in chain], [OLD, self.reviewed])
        self.assertEqual([item["to_fingerprint"] for item in chain], [self.reviewed, revised])
        self.assertIn("(chain 2)", output.getvalue())


class CommandTests(unittest.TestCase):
    def run_status(self, *argv, routes=()):
        records = [record("a", fingerprint=NOW), record("b", fingerprint=OLD)]
        out = io.StringIO()
        with (
            patch.object(status, "load_records", return_value=records),
            patch.object(status, "interval_routes", return_value=list(routes)),
            patch.object(status, "load_attestations", return_value=[]),
            patch.object(status, "ledger_rows", return_value=[]),
            patch.object(status, "backlog_rows", return_value=[]),
            patch("calibration_groups.gate_groups", return_value=[]),
            patch.object(status, "tree_fingerprinter"),
            patch.object(status, "tree_fingerprints", return_value={"a": "manifest:" + NOW, "b": "manifest:" + NOW}),
            redirect_stdout(out),
        ):
            code = status.main(["status", *argv])
        return code, out.getvalue()

    def test_owed_records_never_fail_the_command(self):
        code, text = self.run_status()
        self.assertEqual(code, 0)
        self.assertIn("1 current", text)
        self.assertIn("inputs_changed: b", text)

    def test_owed_lists_only_the_ids(self):
        self.assertEqual(self.run_status("--owed")[1].split(), ["b"])

    def test_owed_lists_every_owed_record_and_route(self):
        routes = [
            route_row("r", "unlinked", NOT_MEASURED),
            route_row("r", "slot", NOT_MEASURED, slots=["cov.slot"]),
            route_row("r", "done", status.CALIBRATED),
        ]
        self.assertEqual(
            self.run_status("--owed", routes=routes)[1].split(),
            ["b", "route:r/unlinked", "cov.slot"],
        )

    def test_the_summary_counts_the_owed_items_by_reason(self):
        routes = [route_row("r", "unlinked", NOT_MEASURED)]
        text = self.run_status(routes=routes)[1]
        self.assertIn("flagged: 2 item(s): 1 no_record, 1 inputs_changed, 0 probe_mismatch", text)
        self.assertIn("(0 record(s) have no manifest)", text)

    def test_json_lists_the_owed_items_with_their_reasons(self):
        routes = [route_row("r", "unlinked", NOT_MEASURED)]
        data = json.loads(self.run_status("--json", routes=routes)[1])
        self.assertEqual(data["owed_by_reason"], {"no_record": 1, "inputs_changed": 1, "probe_mismatch": 0,
                                                    "missing_gate": 0, "stale_gate": 0})
        self.assertEqual(
            [(i["id"], i["reason"]) for i in data["owed"]],
            [("b", "inputs_changed"), ("route:r/unlinked", "no_record")],
        )
        self.assertTrue(data["owed"][1]["unwired"])

    def test_json_carries_counts_and_per_record_status(self):
        data = json.loads(self.run_status("--json")[1])
        self.assertEqual(data["records"], {"a": status.CURRENT, "b": status.PROBE_OWED})
        self.assertEqual(data["counts"][status.CURRENT], 1)


NOT_MEASURED = status.NOT_MEASURED


def route_row(record_id, route, state, slots=()):
    return {
        "record": record_id,
        "route": route,
        "allocated": len(slots),
        "measured": 0,
        "missing_slots": list(slots),
        "state": state,
    }


class OwedItemTests(unittest.TestCase):
    def owed(self, records, rows=(), routes=(), fingerprints=None):
        fingerprints = fingerprints or {str(r["id"]): "manifest:" + NOW for r in records}
        return status.owed_items(records, with_digests(records, list(rows)), fingerprints, list(routes))

    def test_a_current_record_is_not_owed(self):
        self.assertEqual(self.owed([record("a", fingerprint=NOW)]), [])

    def test_changed_inputs_or_no_manifest_is_inputs_changed(self):
        items = self.owed([record("a", fingerprint=OLD), record("b")])
        self.assertEqual([(i["id"], i["reason"]) for i in items], [("a", "inputs_changed"), ("b", "inputs_changed")])
        self.assertEqual([i["has_manifest"] for i in items], [True, False])

    def test_a_probe_that_does_not_cover_the_tree_is_a_probe_mismatch(self):
        items = self.owed([record("a", fingerprint=OLD)], rows=[row("a", result="mismatch")])
        self.assertEqual([(i["id"], i["reason"]) for i in items], [("a", "probe_mismatch")])

    def test_a_covering_probe_makes_the_record_current_and_not_owed(self):
        self.assertEqual(self.owed([record("a", fingerprint=OLD)], rows=[row("a")]), [])

    def test_a_route_without_a_record_is_owed_and_unwired(self):
        routes = [route_row("cap", "measured", status.CALIBRATED), route_row("cap", "new", NOT_MEASURED)]
        (item,) = self.owed([], routes=routes)
        self.assertEqual(
            (item["id"], item["reason"], item["groups"], item["unwired"]),
            ("route:cap/new", "no_record", [], True),
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
        items = self.owed([record("a", fingerprint=OLD)], routes=[route_row("c", "n", NOT_MEASURED)])
        self.assertEqual(status.owed_counts(items), {"no_record": 1, "inputs_changed": 1, "probe_mismatch": 0,
                                                    "missing_gate": 0, "stale_gate": 0})


def unit(linked=(), calibration="not_measured"):
    return {"calibration": list(linked), "state": {"support": "supported", "calibration": calibration}}


class UnitIndexRouteTests(unittest.TestCase):
    CAPABILITY = [
        {
            "id": "cap",
            "inference_claim": "interval",
            "coverage_records": ["cov.allocated"],
            "routes": [
                {"name": "analytic", "status": "supported"},
                {"name": "bootstrap", "status": "supported"},
            ],
        }
    ]

    def rows(self, records, index):
        return {r["route"]: r for r in status.interval_routes(self.CAPABILITY, records, index)}

    def test_a_route_is_measured_by_the_records_linked_to_its_unit(self):
        index = {("capability", "cap"): unit(["cov.boot"])}
        rows = self.rows([{"id": "cov.allocated"}], index)
        self.assertEqual(rows["bootstrap"]["state"], status.NOT_MEASURED)
        self.assertEqual(rows["bootstrap"]["missing_slots"], ["cov.boot"])
        records = [{"id": "cov.allocated"}, {"id": "cov.boot"}]
        self.assertEqual(self.rows(records, index)["bootstrap"]["state"], status.CALIBRATED)
        below = [records[0], records[1] | {"boundary": True}]
        self.assertEqual(self.rows(below, index)["bootstrap"]["state"], status.MEASURED_BELOW_NOMINAL)

    def test_a_record_outside_the_index_keeps_its_allocation(self):
        for index in (None, {}):
            with self.subTest(index=index):
                self.assertEqual(self.rows([{"id": "cov.allocated"}], index)["analytic"]["state"], status.CALIBRATED)
                rows = self.rows([], index)
                self.assertEqual(rows["analytic"]["state"], status.NOT_MEASURED)
                self.assertEqual(rows["analytic"]["missing_slots"], ["cov.allocated"])

    def test_a_unit_with_no_linked_record_reads_its_stated_calibration(self):
        self.assertEqual(self.rows([], {("capability", "cap"): unit()})["analytic"]["state"], status.NOT_MEASURED)
        index = {("capability", "cap"): unit(calibration="estimator_grid_not_measured")}
        self.assertEqual(self.rows([], index)["analytic"]["state"], status.NOT_MEASURED)
        self.assertEqual(self.rows([], {("capability", "cap"): unit(calibration="")})["analytic"]["state"], status.UNMAPPED)
        index = {("capability", "cap"): unit(calibration=status.NO_INTERVAL_REPORTED)}
        self.assertEqual(self.rows([], index)["analytic"]["state"], status.NO_INTERVAL)

    def test_a_route_with_no_allocation_and_no_index_is_unmapped(self):
        capability = [{"id": "c", "inference_claim": "interval", "routes": [{"name": "r", "status": "supported"}]}]
        (row_,) = status.interval_routes(capability, [])
        self.assertEqual(row_["state"], status.UNMAPPED)

    def test_a_refusal_does_not_inherit_its_capability_measurement(self):
        capability = [{
            **self.CAPABILITY[0],
            "routes": [
                *self.CAPABILITY[0]["routes"],
                {"name": "refusal", "status": "supported", "no_interval_reason": "Refuses intervals."},
            ],
        }]
        records = [{"id": "cov.linked", "boundary": True}]
        index = {("capability", "cap"): unit(["cov.linked"])}
        rows = {r["route"]: r for r in status.interval_routes(capability, records, index, set())}
        self.assertEqual(rows["bootstrap"]["state"], status.MEASUREMENT_OWED)
        self.assertEqual(rows["bootstrap"]["allocated"], 1)
        self.assertEqual(rows["refusal"]["state"], status.NO_INTERVAL)
        self.assertEqual(rows["refusal"]["allocated"], 0)

    def test_a_current_measurement_with_a_flagged_link_is_not_calibrated(self):
        records = [{"id": "cov.linked"}]
        index = {("capability", "cap"): unit(["cov.linked"]) | {"calibration_current": []}}
        rows = status.interval_routes(self.CAPABILITY, records, index, {"cov.linked"})
        self.assertEqual({row["state"] for row in rows}, {status.MEASUREMENT_OWED})

    def test_the_header_counts_every_state_including_not_measured_and_unmapped(self):
        capability = [{"id": "c", "inference_claim": "interval", "routes": [{"name": "r", "status": "supported"}]}]
        out = io.StringIO()
        with (
            patch.object(status, "read_text", return_value=""),
            patch.object(status.tomllib, "loads", return_value={"record": capability}),
            patch.object(status, "load_records", return_value=[]),
            patch.object(status, "load_unit_index", return_value=None),
            patch.object(status, "current_ids", return_value=[]),
            redirect_stdout(out),
        ):
            self.assertEqual(status.main(["intervals"]), 0)
        self.assertIn("0 not_measured, 1 unmapped", out.getvalue())


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

    def test_a_stale_boundary_does_not_govern_a_current_record(self):
        records = {"a": {"boundary": True}, "b": {}}
        self.assertEqual(status.interval_state(["a", "b"], records, {"b"}), status.OUTSIDE_MEASURED_SCOPE)

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
            patch.object(status, "current_owed", return_value=([], {}, {status.CURRENT: 0})),
            redirect_stdout(out),
        ):
            self.assertEqual(status.main(["--check"]), 1)
        self.assertIn("bad", out.getvalue())

    def test_snapshot_receives_only_the_current_ids_from_the_same_check(self):
        states = {"b": status.Status(status.CURRENT, "probe"), "a": status.Status(status.PROBE_OWED, "")}
        with (
            patch.object(status, "shape_problems", return_value=[]),
            patch.object(status, "current_owed", return_value=([], states, {status.CURRENT: 1})) as currency,
            patch("generate_support_matrix_docs.check", return_value=1) as documents,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(status.main(["--check", "--check-support-docs"]), 1)
        currency.assert_called_once()
        documents.assert_called_once_with(attesting_ids=["b"])

    def test_informational_subcommands_exit_zero(self):
        for command in ("intervals", "backlog", "plan"):
            with self.subTest(command=command), redirect_stdout(io.StringIO()):
                self.assertEqual(status.main([command]), 0)


if __name__ == "__main__":
    unittest.main()
