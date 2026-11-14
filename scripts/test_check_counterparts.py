"""Counterpart inventory checker and generator: unit tests."""

from __future__ import annotations

import unittest

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import check_counterparts as check  # noqa: E402
import generate_counterparts as gen  # noqa: E402
from method_evidence import Evidence  # noqa: E402


def rows_for(**overrides: dict) -> list[dict]:
    base = {
        "kind": "method",
        "method": "M",
        "route": "r",
        "uncertainty": "analytic",
        "closure": "delivered",
        "status": "supported",
        "reason": "",
        "next_attempt": "",
        "sampling_design": "x",
        "reuse_key": "k",
        "decision_port": "u",
        "artifact": "u",
        "scope_status": "mandatory",
        "gap": "none",
    }
    rows = []
    for quadrant in gen.QUADRANTS:
        design, inference = quadrant.split(".")
        row = base | {"id": f"M.{quadrant}", "design": design, "inference": inference}
        rows.append(row | overrides.get(row["id"], {}))
    return rows


class CounterpartChecks(unittest.TestCase):
    def test_a_complete_method_passes(self) -> None:
        self.assertEqual(check.problems(rows_for()), [])

    def test_a_missing_quadrant_is_found(self) -> None:
        rows = [r for r in rows_for() if r["id"] != "M.temporal.bayesian"]
        self.assertTrue(any("missing_quadrant" in e for e in check.problems(rows)))

    def test_closed_rows_are_narrowed_promises(self) -> None:
        rows = rows_for(
            **{
                "M.static.bayesian": {
                    "status": "closed_with_next_attempt",
                    "reason": "r",
                    "next_attempt": "n",
                }
            }
        )
        self.assertTrue(any("closure_kind" in e for e in check.problems(rows)))

    def test_vocabulary_is_closed(self) -> None:
        rows = rows_for(**{"M.static.bayesian": {"status": "partial"}})
        self.assertTrue(any("vocabulary" in e for e in check.problems(rows)))

    def test_report_names_the_missing_side_and_next_attempt(self) -> None:
        rows = rows_for(
            **{
                "M.temporal.bayesian": {
                    "status": "closed_with_next_attempt",
                    "closure": "narrowed_promise",
                    "reason": "r",
                    "next_attempt": "derive the posterior",
                }
            }
        )
        text = check.report(rows)
        self.assertIn("M: temporal.bayesian missing; next attempt: derive the posterior", text)


class GeneratorChecks(unittest.TestCase):
    def test_child_id_splits_into_method_design_inference(self) -> None:
        self.assertEqual(
            gen.split_id("P07.curve.temporal.bayesian"), ("P07.curve", "temporal", "bayesian")
        )
        self.assertEqual(gen.split_id("P04.static.frequentist"), ("P04", "static", "frequentist"))

    def test_gap_kinds_map_to_statuses(self) -> None:
        child = {"gap": "theory", "missing": "no identification", "module": "none"}
        self.assertEqual(
            gen.child_status(child, [], False), ("closed_with_next_attempt", "narrowed_promise")
        )
        child = {"gap": "target", "missing": "not a static estimand", "module": "none"}
        self.assertEqual(
            gen.child_status(child, [], False), ("not_applicable_with_theory_reason", "delivered")
        )
        child = {"gap": "none", "missing": "none", "module": "m"}
        self.assertEqual(
            gen.child_status(child, [{"status": "supported"}], True), ("supported", "delivered")
        )
        self.assertEqual(
            gen.child_status(child, [{"status": "execution_only"}], True),
            ("not_supported", "delivered"),
        )

    def test_a_child_executes_only_on_a_method_record_or_a_verified_module(self) -> None:
        child = {"gap": "none", "missing": "none", "module": "crates/c/src/m.rs"}
        self.assertEqual(gen.child_status(child, [], True), ("not_supported", "delivered"))
        self.assertEqual(
            gen.child_status(child, [], False), ("closed_with_next_attempt", "narrowed_promise")
        )
        narrowed = {
            "gap": "validation",
            "missing": "interval not_measured",
            "module": "crates/c/src/m.rs",
        }
        self.assertEqual(
            gen.child_status(narrowed, [{"status": "supported"}], True),
            ("supported", "narrowed_promise"),
        )

    def test_uncertainty_comes_from_the_best_method_record(self) -> None:
        manifest = {
            "method": [
                {
                    "id": "a",
                    "route": "r",
                    "child": "",
                    "estimator": "e1",
                    "inference": "frequentist",
                    "uncertainty": "none",
                    "uncertainty_basis": "point_only_cell|x",
                    "status": "supported",
                },
                {
                    "id": "b",
                    "route": "r",
                    "child": "",
                    "estimator": "e2",
                    "inference": "frequentist",
                    "uncertainty": "analytic",
                    "uncertainty_basis": "construction_fixture|y",
                    "status": "supported",
                },
                {
                    "id": "c",
                    "route": "r",
                    "child": "",
                    "estimator": "e3",
                    "inference": "bayesian",
                    "uncertainty": "posterior",
                    "uncertainty_basis": "asserted_in_test|z",
                    "status": "supported",
                },
            ]
        }
        evidenced = gen.Evidenced(manifest, [], Evidence(gen.ROOT))
        self.assertEqual(evidenced.route_kind("r", "frequentist"), ("analytic", "construction_fixture"))
        self.assertEqual(evidenced.route_kind("r", "bayesian"), ("posterior", "asserted_in_test"))
        self.assertEqual(evidenced.route_kind("missing", "bayesian"), ("none", "no_method_record"))

    def test_a_point_only_group_is_asserted_only_when_every_record_is(self) -> None:
        def record(mid: str, basis: str) -> dict:
            return {
                "id": mid,
                "route": "r",
                "child": "",
                "estimator": mid,
                "inference": "frequentist",
                "uncertainty": "none",
                "uncertainty_basis": basis,
                "status": "supported",
            }

        asserted = record("a", "asserted_no_interval|test:crates/c/tests/t.rs::none")
        declared = record("b", "point_only_cell|cell:interval_none")
        both = gen.Evidenced({"method": [asserted, declared]}, [], Evidence(gen.ROOT))
        self.assertEqual(both.route_kind("r", "frequentist"), ("none", "point_only_cell"))
        only = gen.Evidenced({"method": [asserted]}, [], Evidence(gen.ROOT))
        self.assertEqual(only.route_kind("r", "frequentist"), ("none", "asserted_no_interval"))

    def test_a_point_only_route_rests_only_when_a_test_asserts_no_interval(self) -> None:
        route = {
            "id": "supported.Q.Dag",
            "target": "Q",
            "graph_class": "Dag",
            "design_time_model": "static",
            "inference": "Frequentist",
            "decision_ports": "unlisted",
            "status": "supported",
            "notes": "",
        }

        def row(basis: str) -> dict:
            method = {
                "id": "m",
                "route": "supported.Q.Dag",
                "child": "",
                "estimator": "e",
                "inference": "frequentist",
                "uncertainty": "none",
                "uncertainty_basis": basis,
                "status": "supported",
                "evidence": ["crates/c/tests/t.rs::runs"],
                "artifact_consumer": ["crates/c/src/reader.rs::read"],
            }
            evidenced = gen.Evidenced({"method": [method]}, [], Evidence(gen.ROOT))
            return gen.quadrant_row(route, "static", "frequentist", "Q", {}, evidenced)

        rests = row("asserted_no_interval|test:crates/c/tests/t.rs::none")
        self.assertEqual(
            (rests["closure"], rests["gap"], rests["reason"], rests["next_attempt"]),
            ("delivered", "none", "", ""),
        )
        for basis in ("point_only_cell|cell:interval_none", "declared_and_corroborated"):
            open_row = row(basis)
            self.assertEqual((open_row["closure"], open_row["gap"]), ("narrowed_promise", "validation"))
            self.assertTrue(open_row["next_attempt"])

    def test_unresolved_uncertainty_remains_unresolved(self) -> None:
        record = {"child": "", "route": "r", "estimator": "e", "inference": "frequentist", "uncertainty": "unresolved", "uncertainty_basis": "no_support|x"}
        evidenced = gen.Evidenced({"method": [record]}, [], Evidence(gen.ROOT))
        self.assertEqual(evidenced.route_kind("r", "frequentist"), ("unresolved", "no_support"))

    def test_an_executing_row_needs_a_target_a_design_a_fixture_and_a_consumer(self) -> None:
        ev = Evidence(gen.ROOT)
        row = rows_for()[0] | {
            "uncertainty_basis": "construction_fixture",
            "target": "AverageEffect",
            "sampling_design": "independent rows",
            "execution": "scripts/check_counterparts.py",
            "artifact": "scripts/generate_counterparts.py",
        }
        self.assertEqual(check.evidence_problems([row], ev), [])
        for field, value, needle in (
            ("target", "unlisted", "no recorded target"),
            ("sampling_design", "unlisted", "no recorded sampling design"),
            ("execution", "unlisted", "no execution fixture"),
            ("artifact", "unlisted", "no artifact consumer"),
            ("execution", "crates/none/tests/absent.rs", "does not exist"),
            ("uncertainty_basis", "no_method_record", "no method record"),
            ("uncertainty_basis", "guess", "outside the vocabulary"),
        ):
            found = check.evidence_problems([row | {field: value}], ev)
            self.assertTrue(any(needle in line for line in found), (field, found))
        closed = row | {
            "status": "closed_with_next_attempt",
            "target": "unlisted",
            "uncertainty_basis": "no_executing_route",
        }
        self.assertEqual(check.evidence_problems([closed], ev), [])

    def test_a_route_records_its_rows_under_a_declared_method(self) -> None:
        route = {"id": "x.part", "target": "Part"}
        self.assertEqual(gen.route_method(route), "Part")
        self.assertEqual(gen.route_method(route | {"counterpart_method": "Whole"}), "Whole")


def unit_route(rid: str, target: str, inference: str, **extra: str) -> dict:
    return {
        "id": rid,
        "target": target,
        "graph_class": "RandomizedTrial",
        "design_time_model": "declared design",
        "inference": inference,
        "decision_ports": "unlisted",
        "status": "graphless",
        "notes": "",
    } | extra


def unit_method(rid: str, inference: str, uncertainty: str, design: str = "static") -> dict:
    return {
        "id": f"method.{rid}",
        "route": rid,
        "child": "",
        "estimator": "unlisted",
        "inference": inference,
        "design": design,
        "uncertainty": uncertainty,
        "uncertainty_basis": "graphless_claim:x",
        "status": "graphless",
        "evidence": ["crates/c/tests/t.rs::runs"],
        "artifact_consumer": ["crates/c/src/reader.rs::read"],
    }


class FamilyRows(unittest.TestCase):
    """The quadrants of a graphless family come from its support rows and axis records."""

    ROUTES = [
        unit_route("graphless.fam.design.normal", "fam", "pointwise_95_normal_interval"),
        unit_route("graphless.fam.design.pseudo_posterior", "fam", "pointwise_95_posterior_quantiles"),
        unit_route("bayesian.fam_posterior", "FamTarget", "Bayesian"),
        unit_route("fam.over_time", "Other", "Frequentist"),
        unit_route("bayesian.fam.over_time", "Other", "Bayesian"),
    ]
    METHODS = [
        unit_method("graphless.fam.design.normal", "unlisted", "analytic"),
        unit_method("graphless.fam.design.pseudo_posterior", "unlisted", "posterior"),
        unit_method("bayesian.fam_posterior", "bayesian", "posterior"),
        unit_method("fam.over_time", "frequentist", "analytic"),
        unit_method("bayesian.fam.over_time", "bayesian", "posterior"),
    ]

    def graphless(self, temporal: dict) -> dict:
        return {
            "support": [{"family": "fam"}],
            "family_axis": [
                {
                    "family": "fam",
                    "axis": "bayesian",
                    "disposition": "counterpart",
                    "routes": ["bayesian.fam_posterior"],
                    "reason": "the posterior of the same contrast",
                },
                {"family": "fam", "axis": "temporal"} | temporal,
            ],
        }

    def rows(self, graphless: dict, routes: list[dict] | None = None, named=frozenset()) -> list[dict]:
        evidenced = gen.Evidenced({"method": self.METHODS}, [], Evidence(gen.ROOT))
        return gen.route_rows(routes or self.ROUTES, {}, evidenced, graphless, named)

    def quadrants(self, rows: list[dict]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(f"{row['design']}.{row['inference']}", []).append(row["route"])
        return out

    def test_every_quadrant_is_the_route_the_records_name(self) -> None:
        temporal = {
            "disposition": "counterpart",
            "routes": ["fam.over_time", "bayesian.fam.over_time"],
            "reason": "the same contrast over time",
        }
        rows = self.rows(self.graphless(temporal))
        self.assertEqual({row["method"] for row in rows}, {"fam"})
        self.assertEqual(
            self.quadrants(rows),
            {
                # A support route records its claim where others record the side: the posterior
                # claim is the Bayesian side, every other claim the frequentist one.
                "static.frequentist": ["graphless.fam.design.normal"],
                "static.bayesian": ["graphless.fam.design.pseudo_posterior", "bayesian.fam_posterior"],
                "temporal.frequentist": ["fam.over_time"],
                "temporal.bayesian": ["bayesian.fam.over_time"],
            },
        )
        self.assertEqual(check.problems(rows), [])
        ids = {row["route"]: row["id"] for row in rows}
        self.assertEqual(ids["graphless.fam.design.normal"], "graphless.fam.design.normal#static.frequentist")
        self.assertEqual(ids["fam.over_time"], "fam.over_time#temporal.frequentist>fam")

    def test_an_axis_the_records_decline_is_closed_with_their_reason(self) -> None:
        temporal = {"disposition": "not_routed", "reason": "no time-indexed form is routed"}
        rows = self.rows(self.graphless(temporal))
        absent = [row for row in rows if row["design"] == "temporal"]
        self.assertEqual(
            sorted(row["id"] for row in absent), ["fam#temporal.bayesian", "fam#temporal.frequentist"]
        )
        for row in absent:
            self.assertEqual(row["status"], "closed_with_next_attempt")
            self.assertEqual(row["closure"], "narrowed_promise")
            self.assertEqual(row["route"], "unlisted")
            self.assertEqual(row["reason"], "no time-indexed form is routed")
            self.assertTrue(row["next_attempt"])
        self.assertEqual(check.problems(rows), [])

    def test_a_side_no_record_names_is_closed_and_never_inferred_from_a_route_name(self) -> None:
        # The temporal axis names the frequentist route only. Its Bayesian twin exists in the
        # manifest, but no record says it is the family's temporal Bayesian side.
        temporal = {"disposition": "counterpart", "routes": ["fam.over_time"], "reason": "over time"}
        rows = self.rows(self.graphless(temporal))
        self.assertNotIn("bayesian.fam.over_time", {row["route"] for row in rows})
        absent = next(row for row in rows if row["id"] == "fam#temporal.bayesian")
        self.assertEqual(absent["status"], "closed_with_next_attempt")
        self.assertEqual(
            absent["reason"],
            "parity/support_graphless.toml records no temporal bayesian route for the fam family",
        )

    def test_a_family_recorded_only_over_time_has_the_static_boundary(self) -> None:
        temporal = {
            "disposition": "counterpart",
            "routes": ["fam.over_time", "bayesian.fam.over_time"],
            "reason": "the family's own structure",
        }
        graphless = self.graphless(temporal)
        graphless["family_axis"][0]["routes"] = ["bayesian.fam.over_time"]
        rows = self.rows(graphless, self.ROUTES[3:])
        static = [row for row in rows if row["design"] == "static"]
        self.assertEqual(
            {row["status"] for row in static}, {"not_applicable_with_theory_reason"}
        )
        self.assertEqual(check.problems(rows), [])

    def test_a_family_whose_routes_all_have_rows_adds_none(self) -> None:
        temporal = {
            "disposition": "counterpart",
            "routes": ["fam.over_time", "bayesian.fam.over_time"],
            "reason": "over time",
        }
        named = frozenset(route["id"] for route in self.ROUTES)
        self.assertEqual(self.rows(self.graphless(temporal), named=named), [])

    def test_a_declared_family_member_is_the_familys_own_side(self) -> None:
        temporal = {"disposition": "not_routed", "reason": "not routed"}
        routes = [
            unit_route("fam.bounds", "FamBounds", "Frequentist", counterpart_family="fam"),
            self.ROUTES[2],
        ]
        self.METHODS.append(unit_method("fam.bounds", "frequentist", "enclosure"))
        try:
            rows = self.rows(self.graphless(temporal), routes)
        finally:
            self.METHODS.pop()
        own = next(row for row in rows if row["route"] == "fam.bounds")
        self.assertEqual((own["id"], own["method"]), ("fam.bounds#static.frequentist>fam", "fam"))
        # The route's own target gets no group of its own.
        self.assertEqual({row["method"] for row in rows}, {"fam"})

    def test_a_counterpart_reason_does_not_narrow_a_family_row(self) -> None:
        temporal = {
            "disposition": "counterpart",
            "routes": ["fam.over_time", "bayesian.fam.over_time"],
            "reason": "over time",
        }
        routes = [
            route
            | (
                {"counterpart_reason": "reduced claim for another target", "counterpart_next_attempt": "n"}
                if route["id"] == "fam.over_time"
                else {}
            )
            for route in self.ROUTES
        ]
        row = next(r for r in self.rows(self.graphless(temporal), routes) if r["route"] == "fam.over_time")
        self.assertEqual((row["closure"], row["reason"], row["next_attempt"]), ("delivered", "", ""))


class ShippedInventory(unittest.TestCase):
    def test_every_manifest_family_route_has_a_row(self) -> None:
        rows = gen.build()
        rowed = {row["route"] for row in rows}
        graphless = gen.load(gen.GRAPHLESS)
        named = {rid for axis in graphless["family_axis"] for rid in axis.get("routes", [])}
        named |= {
            route["id"]
            for route in gen.load(gen.ROUTES)["route"]
            if route["id"].startswith("graphless.")
        }
        self.assertEqual(sorted(named - rowed), [])
        self.assertEqual(
            [e for e in check.problems(rows) if e.startswith("missing_quadrant")], []
        )

    def test_the_shipped_inventory_is_current(self) -> None:
        self.assertEqual(gen.OUT_TOML.read_text(encoding="utf-8"), gen.render_toml(gen.build()))


if __name__ == "__main__":
    unittest.main()
