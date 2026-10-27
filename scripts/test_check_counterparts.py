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
        "status": "licensed",
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

    def test_weaker_temporal_uncertainty_needs_a_reason(self) -> None:
        rows = rows_for(**{"M.temporal.bayesian": {"uncertainty": "none"}})
        self.assertTrue(any("weaker_temporal" in e for e in check.problems(rows)))
        recorded = rows_for(
            **{
                "M.temporal.bayesian": {
                    "uncertainty": "none",
                    "reason": "serial dependence has no likelihood",
                    "closure": "narrowed_promise",
                    "next_attempt": "derive the dependence-aware posterior",
                }
            }
        )
        self.assertEqual(check.problems(recorded), [])

    def test_not_implemented_is_not_a_reason(self) -> None:
        rows = rows_for(
            **{
                "M.static.bayesian": {
                    "reason": "this is not yet implemented",
                    "closure": "narrowed_promise",
                    "next_attempt": "n",
                }
            }
        )
        self.assertTrue(any("not_implemented" in e for e in check.problems(rows)))

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
            gen.child_status(child, [{"status": "licensed"}], True), ("licensed", "delivered")
        )
        self.assertEqual(
            gen.child_status(child, [{"status": "execution_only"}], True),
            ("executing_unlicensed", "delivered"),
        )

    def test_a_child_executes_only_on_a_method_record_or_a_verified_module(self) -> None:
        child = {"gap": "none", "missing": "none", "module": "crates/c/src/m.rs"}
        self.assertEqual(gen.child_status(child, [], True), ("executing_unlicensed", "delivered"))
        self.assertEqual(
            gen.child_status(child, [], False), ("closed_with_next_attempt", "narrowed_promise")
        )
        narrowed = {
            "gap": "validation",
            "missing": "interval unmeasured",
            "module": "crates/c/src/m.rs",
        }
        self.assertEqual(
            gen.child_status(narrowed, [{"status": "licensed"}], True),
            ("licensed", "narrowed_promise"),
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
                    "status": "licensed",
                },
                {
                    "id": "b",
                    "route": "r",
                    "child": "",
                    "estimator": "e2",
                    "inference": "frequentist",
                    "uncertainty": "analytic",
                    "uncertainty_basis": "measured_record|y",
                    "status": "licensed",
                },
                {
                    "id": "c",
                    "route": "r",
                    "child": "",
                    "estimator": "e3",
                    "inference": "bayesian",
                    "uncertainty": "posterior",
                    "uncertainty_basis": "asserted_in_test|z",
                    "status": "licensed",
                },
            ]
        }
        evidenced = gen.Evidenced(manifest, [], Evidence(gen.ROOT))
        self.assertEqual(evidenced.route_kind("r", "frequentist"), ("analytic", "measured_record"))
        self.assertEqual(evidenced.route_kind("r", "bayesian"), ("posterior", "asserted_in_test"))
        self.assertEqual(evidenced.route_kind("missing", "bayesian"), ("none", "no_method_record"))

    def test_an_executing_row_needs_a_target_a_design_a_fixture_and_a_consumer(self) -> None:
        ev = Evidence(gen.ROOT)
        row = rows_for()[0] | {
            "uncertainty_basis": "measured_record",
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

    def test_the_shipped_inventory_is_current(self) -> None:
        self.assertEqual(gen.OUT_TOML.read_text(encoding="utf-8"), gen.render_toml(gen.build()))


if __name__ == "__main__":
    unittest.main()
