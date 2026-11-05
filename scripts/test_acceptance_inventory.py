#!/usr/bin/env python3
"""Acceptance inventory: generation from the records and one negative case per rejection."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import check_acceptance_inventory as chk  # noqa: E402
import generate_acceptance_inventory as gen  # noqa: E402

SUITE = """
fn association_static() -> Rep {
    Rep {
        label: "association_static",
        family: RouteFamily::Associational,
    }
}

fn unused_rep() -> Rep {
    Rep {
        label: "unused",
    }
}

fn others() -> Vec<Other> {
    vec![
        Other { label: "chain_five", family: RouteFamily::PropagationChain, entry: Entry::Chain },
    ]
}

fn representatives() -> Vec<Rep> {
    vec![
        association_static(),
    ]
}
"""


def counterpart(**over) -> dict:
    row = {
        "id": "supported.AverageEffect.Dag#static.frequentist",
        "kind": "route",
        "method": "AverageEffect",
        "design": "static",
        "inference": "frequentist",
        "route": "supported.AverageEffect.Dag",
        "status": "supported",
        "uncertainty": "analytic",
        "uncertainty_basis": "measured_record",
        "target": "AverageEffect",
        "sampling_design": "static",
        "execution": "crates/a/tests/t.rs::runs",
        "reuse_key": "k",
        "decision_port": "unlisted",
        "artifact": "crates/a/src/consume.rs::consume",
        "closure": "delivered",
        "scope_status": "mandatory",
        "gap": "none",
        "reason": "",
        "next_attempt": "",
    }
    return row | over


def supported_obligation(**over) -> dict:
    row = {
        "id": "AverageEffect:Dag:explicit:Frequentist:none",
        "kind": "supported",
        "query": "AverageEffect",
        "graph_class": "Dag",
        "structure": "explicit",
        "inference": "Frequentist",
        "validation": "none",
        "identification_evidence": "internal_known_truth|crates/a/tests/t.rs|runs|fixture",
        "uncertainty": "coverage_records",
    }
    return row | over


def association_row(route: str, design: str, inference: str) -> dict:
    return counterpart(
        id=f"{route}#{design}.{inference}",
        route=route,
        status="not_supported",
        uncertainty="analytic",
        uncertainty_basis="declared_and_corroborated",
        design=design,
        inference=inference,
        scope_status="mandatory",
    )


class Fixture:
    def __init__(self, root: Path):
        for rel, body in (
            ("crates/a/tests/t.rs", "#[test]\nfn runs() {}\n"),
            ("crates/a/src/consume.rs", "pub fn consume() {}\n"),
        ):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(body, encoding="utf-8")
        self.sources = {
            "root": root,
            "counterparts": [counterpart()],
            "manifest": {
                "route": [
                    {
                        "id": "supported.AverageEffect.Dag",
                        "target": "AverageEffect",
                        "inference": "Frequentist",
                        "status": "supported",
                        "decision_ports": "unlisted",
                    }
                ],
                "method": [],
            },
            "families": [
                {
                    "id": "static_effect",
                    "status": "covered",
                    "representatives": ["association_static"],
                    "routes": ["supported.AverageEffect.Dag"],
                }
            ],
            "scope": [
                {
                    "id": "supported.AverageEffect.Dag",
                    "status": "mandatory",
                    "list": "mandatory",
                    "source": "route_manifest",
                }
            ],
            "children": [],
            "obligations": [supported_obligation()],
            "representatives": {"association_static"},
            "child_findings": {},
        }

    def rows(self) -> list[dict]:
        return gen.build(self.sources)


class AcceptanceInventoryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = Fixture(Path(self._tmp.name))

    def problems(self, rows=None) -> list[str]:
        return chk.check(self.fx.rows() if rows is None else rows, self.fx.sources)

    def assertRejected(self, needle: str, rows=None):
        found = self.problems(rows)
        self.assertTrue(any(needle in p for p in found), (needle, found))

    # ---- generation ---------------------------------------------------------------------
    def test_a_fully_evidenced_route_is_complete_and_checks_clean(self):
        rows = self.fx.rows()
        self.assertEqual(rows[0]["acceptance"], "complete")
        self.assertEqual(
            rows[0]["support_obligations"], ["AverageEffect:Dag:explicit:Frequentist:none"]
        )
        self.assertEqual(self.problems(rows), [])

    def test_render_round_trips_through_toml(self):
        import tomllib

        text = gen.render(self.fx.rows())
        self.assertEqual(
            tomllib.loads(text)["row"][0]["id"], "supported.AverageEffect.Dag#static.frequentist"
        )

    def test_representative_labels_come_from_the_registered_list_only(self):
        self.assertEqual(
            gen.registered_representatives(SUITE), {"association_static", "chain_five"}
        )

    # ---- incomplete ---------------------------------------------------------------------
    def test_executing_route_without_its_claimed_uncertainty_is_incomplete(self):
        self.fx.sources["counterparts"] = [
            counterpart(
                status="not_supported",
                uncertainty="none",
                uncertainty_basis="declared_and_corroborated",
            )
        ]
        self.fx.sources["obligations"] = []
        row = self.fx.rows()[0]
        self.assertEqual(row["acceptance"], "incomplete")
        self.assertTrue(any("claimed uncertainty" in r for r in row["incomplete_reasons"]), row)

    def test_executing_route_without_a_resolvable_consumer_is_incomplete(self):
        for artifact, needle in (
            ("unlisted", "no artifact consumer"),
            ("crates/a/src/gone.rs::consume", "does not exist"),
            ("crates/a/src/consume.rs::absent", "does not mention"),
        ):
            self.fx.sources["counterparts"] = [counterpart(artifact=artifact)]
            row = self.fx.rows()[0]
            self.assertEqual(row["acceptance"], "incomplete", artifact)
            self.assertTrue(any(needle in r for r in row["incomplete_reasons"]), row)

    def test_a_narrowed_promise_is_incomplete_with_its_reason(self):
        self.fx.sources["counterparts"] = [
            counterpart(
                closure="narrowed_promise",
                gap="validation",
                reason="point only",
                next_attempt="derive the interval",
            )
        ]
        row = self.fx.rows()[0]
        self.assertEqual(row["acceptance"], "incomplete")
        self.assertTrue(any("derive the interval" in r for r in row["incomplete_reasons"]))
        self.assertEqual(self.problems(), [])

    def test_unregistered_association_counterparts_are_incomplete_until_registered(self):
        routes = [
            ("association.descriptive.static", "static", "frequentist"),
            ("association.descriptive.temporal", "temporal", "frequentist"),
            ("bayesian.association.descriptive.static", "static", "bayesian"),
            ("bayesian.association.descriptive.temporal", "temporal", "bayesian"),
        ]
        self.fx.sources["counterparts"] = [association_row(*r) for r in routes]
        self.fx.sources["manifest"] = {
            "route": [
                {
                    "id": r,
                    "target": "Association",
                    "inference": "x",
                    "status": "execution_only",
                    "decision_ports": "unlisted",
                }
                for r, _, _ in routes
            ],
            "method": [],
        }
        self.fx.sources["families"] = [
            {
                "id": "associational",
                "status": "covered",
                "representatives": ["association_static"],
                "routes": [r for r, _, _ in routes],
            }
        ]
        self.fx.sources["scope"] = []
        self.fx.sources["obligations"] = []
        by_route = {r["route"]: r for r in self.fx.rows()}
        self.assertEqual(by_route["association.descriptive.static"]["acceptance"], "complete")
        for route in routes[1:]:
            row = by_route[route[0]]
            self.assertEqual(row["acceptance"], "incomplete", route)
            self.assertTrue(
                any("no conformance representative" in r for r in row["incomplete_reasons"]), row
            )
        self.fx.sources["representatives"] = {
            "association_static",
            "association_temporal",
            "association_static_bayesian",
            "association_temporal_bayesian",
        }
        self.assertTrue(all(r["acceptance"] == "complete" for r in self.fx.rows()))

    def test_a_manifest_route_no_counterpart_names_is_a_row_and_incomplete(self):
        self.fx.sources["manifest"]["route"].append(
            {
                "id": "graphless.x",
                "target": "x",
                "inference": "i",
                "status": "graphless",
                "decision_ports": "unlisted",
            }
        )
        self.fx.sources["families"][0]["routes"].append("graphless.x")
        rows = self.fx.rows()
        extra = [r for r in rows if r["kind"] == "manifest_route"]
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0]["acceptance"], "incomplete")
        self.assertTrue(any("no counterpart row" in r for r in extra[0]["incomplete_reasons"]))

    # ---- rejections ---------------------------------------------------------------------
    def test_rejects_a_counterpart_row_missing_from_the_inventory(self):
        self.assertRejected(
            "missing: no inventory row for supported.AverageEffect.Dag#static.frequentist", rows=[]
        )

    def test_rejects_a_route_obligation_no_row_carries(self):
        self.fx.sources["obligations"].append(
            supported_obligation(id="AverageEffect:Dag:explicit:Bayesian:none", inference="Bayesian")
        )
        rows = self.fx.rows()
        rows[0]["obligations"] = ["AverageEffect:Dag:explicit:Frequentist:none"]
        self.assertRejected(
            "route obligation AverageEffect:Dag:explicit:Bayesian:none is carried by no inventory row",
            rows,
        )

    def test_rejects_a_method_child_and_scope_entry_without_a_row(self):
        self.fx.sources["children"] = [
            {"id": "P99.x.static.frequentist", "route": "unlisted", "gap": "none"}
        ]
        self.fx.sources["scope"].append(
            {
                "id": "P99.x.static.frequentist",
                "status": "mandatory",
                "list": "mandatory",
                "source": "method_children",
            }
        )
        self.assertRejected("missing: method child P99.x.static.frequentist", None)
        self.assertRejected("missing: release-scope child P99.x.static.frequentist", None)

    def test_rejects_a_manifest_route_without_a_row(self):
        rows = self.fx.rows()
        rows[0]["route"] = "supported.Other.Dag"
        self.assertRejected("missing: manifest route supported.AverageEffect.Dag", rows)

    def test_rejects_a_status_that_conflicts_with_the_counterpart_record(self):
        rows = self.fx.rows()
        rows[0]["status"] = "not_supported"
        self.assertRejected("conflict: supported.AverageEffect.Dag#static.frequentist: status", rows)

    def test_rejects_unsupported_status_where_an_explicit_support_exists(self):
        self.fx.sources["counterparts"] = [counterpart(status="not_supported")]
        self.assertRejected("not supported although an explicit support obligation exists", None)

    def test_rejects_a_closure_kind_that_conflicts_with_the_inventory(self):
        rows = self.fx.rows()
        rows[0]["closure_kind"] = "narrowed_promise"
        self.assertRejected(
            "conflict: supported.AverageEffect.Dag#static.frequentist: closure_kind", rows
        )

    def test_rejects_a_scope_list_that_conflicts_with_the_row(self):
        self.fx.sources["scope"][0]["list"] = "research"
        self.assertRejected("scope list 'research' conflicts with counterpart scope_status", None)

    def test_rejects_a_child_gap_that_conflicts_with_the_row(self):
        row = counterpart(
            id="P99.x.static.frequentist",
            kind="method",
            method="P99.x",
            route="unlisted",
            status="not_supported",
            uncertainty_basis="declared_and_corroborated",
            gap="none",
        )
        self.fx.sources["counterparts"] = [row]
        self.fx.sources["obligations"] = []
        self.fx.sources["scope"] = []
        self.fx.sources["children"] = [
            {"id": "P99.x.static.frequentist", "route": "unlisted", "gap": "statistical"}
        ]
        self.assertRejected("child gap 'statistical' conflicts with counterpart gap 'none'", None)

    def test_rejects_a_route_in_no_conformance_family(self):
        self.fx.sources["families"][0]["routes"] = []
        self.assertRejected("is in no conformance family", None)

    def test_rejects_a_supported_row_without_an_explicit_support(self):
        self.fx.sources["obligations"] = []
        self.assertRejected(
            "unsupported: supported.AverageEffect.Dag#static.frequentist: supported row has no explicit support obligation",
            None,
        )

    def test_rejects_an_obligation_without_identification_evidence(self):
        self.fx.sources["obligations"] = [supported_obligation(identification_evidence="|||")]
        self.assertRejected("has no explicit identification evidence", None)

    def test_rejects_a_stale_row_and_a_row_naming_nothing(self):
        rows = self.fx.rows()
        rows[0]["compiler_execution"] = "crates/a/tests/t.rs::other"
        self.assertRejected(
            "stale: supported.AverageEffect.Dag#static.frequentist differs from the records in compiler_execution",
            rows,
        )
        ghost = copy.deepcopy(self.fx.rows()[0])
        ghost["id"] = "ghost"
        self.assertRejected("stale: row ghost names no counterpart row", self.fx.rows() + [ghost])

    def test_rejects_a_row_marked_complete_with_open_reasons(self):
        self.fx.sources["counterparts"] = [counterpart(artifact="unlisted")]
        rows = self.fx.rows()
        rows[0]["acceptance"] = "complete"
        self.assertRejected("marked complete with open reasons", rows)

    def test_rejects_a_counterpart_row_recorded_twice(self):
        self.fx.sources["counterparts"] = [counterpart(), counterpart()]
        self.assertRejected(
            "counterpart row supported.AverageEffect.Dag#static.frequentist appears 2 times", None
        )

    def test_rejects_a_duplicate_inventory_row(self):
        rows = self.fx.rows()
        self.assertRejected("appears twice", rows + [copy.deepcopy(rows[0])])


if __name__ == "__main__":
    unittest.main()
