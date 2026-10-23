"""Counterpart inventory checker and generator: unit tests."""
from __future__ import annotations

import unittest

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import check_counterparts as check  # noqa: E402
import generate_counterparts as gen  # noqa: E402


def rows_for(**overrides: dict) -> list[dict]:
    base = {
        "kind": "method", "method": "M", "route": "r", "uncertainty": "analytic", "closure": "delivered",
        "status": "licensed", "reason": "", "next_attempt": "", "sampling_design": "x", "reuse_key": "k",
        "decision_port": "u", "artifact": "u", "scope_status": "mandatory", "gap": "none",
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
        recorded = rows_for(**{"M.temporal.bayesian": {
            "uncertainty": "none", "reason": "serial dependence has no likelihood",
            "closure": "narrowed_promise", "next_attempt": "derive the dependence-aware posterior"}})
        self.assertEqual(check.problems(recorded), [])

    def test_not_implemented_is_not_a_reason(self) -> None:
        rows = rows_for(**{"M.static.bayesian": {
            "reason": "this is not yet implemented", "closure": "narrowed_promise", "next_attempt": "n"}})
        self.assertTrue(any("not_implemented" in e for e in check.problems(rows)))

    def test_closed_rows_are_narrowed_promises(self) -> None:
        rows = rows_for(**{"M.static.bayesian": {
            "status": "closed_with_next_attempt", "reason": "r", "next_attempt": "n"}})
        self.assertTrue(any("closure_kind" in e for e in check.problems(rows)))

    def test_vocabulary_is_closed(self) -> None:
        rows = rows_for(**{"M.static.bayesian": {"status": "partial"}})
        self.assertTrue(any("vocabulary" in e for e in check.problems(rows)))

    def test_report_names_the_missing_side_and_next_attempt(self) -> None:
        rows = rows_for(**{"M.temporal.bayesian": {
            "status": "closed_with_next_attempt", "closure": "narrowed_promise",
            "reason": "r", "next_attempt": "derive the posterior"}})
        text = check.report(rows)
        self.assertIn("M: temporal.bayesian missing; next attempt: derive the posterior", text)


class GeneratorChecks(unittest.TestCase):
    def test_child_id_splits_into_method_design_inference(self) -> None:
        self.assertEqual(gen.split_id("P07.curve.temporal.bayesian"), ("P07.curve", "temporal", "bayesian"))
        self.assertEqual(gen.split_id("P04.static.frequentist"), ("P04", "static", "frequentist"))

    def test_gap_kinds_map_to_statuses(self) -> None:
        child = {"gap": "theory", "missing": "no identification", "module": "none"}
        self.assertEqual(gen.child_status(child, None), ("closed_with_next_attempt", "narrowed_promise"))
        child = {"gap": "target", "missing": "not a static estimand", "module": "none"}
        self.assertEqual(gen.child_status(child, None), ("not_applicable_with_theory_reason", "delivered"))
        child = {"gap": "none", "missing": "none", "module": "m"}
        self.assertEqual(gen.child_status(child, {"status": "licensed"}), ("licensed", "delivered"))
        self.assertEqual(gen.child_status(child, None), ("executing_unlicensed", "delivered"))

    def test_the_shipped_inventory_is_current(self) -> None:
        self.assertEqual(gen.OUT_TOML.read_text(), gen.render_toml(gen.build()))


if __name__ == "__main__":
    unittest.main()
