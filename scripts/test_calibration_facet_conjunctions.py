"""Conjunctive implementation facets preserve unrelated numerical evidence."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import calibration_facets as facets


class ConjunctiveFacetTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name in ("core.rs", "response.rs", "curve.rs"):
            (self.root / name).write_text("// fixture\n")
        self.root_patch = patch.object(facets, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.old = "core core.rs\nestimator.response response.rs\nkey estimator.response estimator response.*\n"
        self.rule = (
            "estimator.levels curve.rs\n"
            "key_all estimator.levels estimator=response.kennedy_dr,response.intervention_gcomp "
            "query=ResponseCurve,InterventionResponse inference=Frequentist\n"
        )
        self.record = {"id": "curve", "estimator": "response.kennedy_dr", "query": "ResponseCurve",
                       "inference": "Frequentist", "calibration_sha": "a" * 40}

    def load(self, extra: str = ""):
        return facets.load_surface(text=self.old + extra)

    def test_each_clause_is_required_and_patterns_inside_clause_are_alternatives(self) -> None:
        surface = self.load(self.rule)
        self.assertEqual(surface.errors, [])
        self.assertIn("estimator.levels", facets.record_facets(self.record, surface))
        intervention = self.record | {"estimator": "response.intervention_gcomp", "query": "InterventionResponse"}
        self.assertIn("estimator.levels", facets.record_facets(intervention, surface))
        for changed in ({"estimator": "response.bayesian"}, {"query": "PointDerivative"}, {"inference": "Bayesian"}):
            self.assertNotIn("estimator.levels", facets.record_facets(self.record | changed, surface))

    def test_old_surface_blob_and_existing_key_semantics_remain_unchanged(self) -> None:
        historical = self.load()
        derivative = self.record | {"query": "PointDerivative"}
        self.assertEqual(facets.record_facets(derivative, historical), ["core", "estimator.response"])
        union = self.load("key estimator.response query OtherQuery\n")
        self.assertIn("estimator.response", facets.record_facets({"query": "OtherQuery"}, union))
        self.assertIn("estimator.response", facets.record_facets(derivative, union))

    def test_changed_level_path_owes_actual_users_and_widening_detectably_adds_debt(self) -> None:
        records = [self.record, self.record | {"id": "derivative", "query": "PointDerivative"},
                   self.record | {"id": "bayes", "estimator": "response.bayesian", "inference": "Bayesian"}]
        def owed(surface):
            return {r["id"] for assessment in facets.assess(surface, records, {"a" * 40: ["curve.rs"]})
                    for r in assessment.stale}
        self.assertEqual(owed(self.load(self.rule)), {"curve"})
        widened = self.rule.replace("query=ResponseCurve,InterventionResponse", "query=ResponseCurve,InterventionResponse,PointDerivative")
        self.assertEqual(owed(self.load(widened)), {"curve", "derivative"})

    def test_widening_guard_detects_directory_shadows_and_new_conjuncts(self) -> None:
        old = facets.Surface(entries=[("core", "core.rs"), ("estimator.response", "response/")],
                             keys=[("estimator.response", "estimator", ["response.*"])])
        shadow = facets.Surface(entries=old.entries + [("estimator.levels", "response/curve.rs")], keys=old.keys)
        self.assertTrue(any("response/curve.rs moved" in error for error in
                            facets.surface_narrowing_problems(old, shadow, ["response/curve.rs"])))
        narrowed = facets.Surface(entries=old.entries,
                                  all_keys=[("estimator.response", [("estimator", ["response.*"]), ("query", ["ResponseCurve"])])])
        self.assertTrue(any("key dependency" in error for error in facets.surface_narrowing_problems(old, narrowed, [])))
        broader = facets.Surface(entries=old.entries, keys=[("estimator.response", "estimator", ["*"])])
        self.assertEqual(facets.surface_narrowing_problems(old, broader, []), [])
        # A glob-looking string matching another glob does not prove language inclusion.
        tricky_old = facets.Surface(entries=old.entries, keys=[("estimator.response", "estimator", ["a*"])])
        tricky_new = facets.Surface(entries=old.entries, keys=[("estimator.response", "estimator", ["a?"])])
        self.assertTrue(facets.surface_narrowing_problems(tricky_old, tricky_new, []))

    def test_malformed_or_unowned_conjunctions_fail_closed(self) -> None:
        for clause in ("query=ResponseCurve query=Other", "invented=foo query=ResponseCurve",
                       "query=ResponseCurve, inference=Frequentist", "query=ResponseCurve"):
            with self.subTest(clause=clause):
                self.assertTrue(self.load("estimator.levels curve.rs\nkey_all estimator.levels " + clause).errors)
        self.assertTrue(self.load("key_all unknown query=A inference=B").errors)
        self.assertTrue(self.load("key_all core query=A inference=B").errors)


if __name__ == "__main__":
    unittest.main()
