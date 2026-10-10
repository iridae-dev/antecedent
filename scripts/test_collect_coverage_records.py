"""Focused guardrails for incremental coverage-record collection."""

from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_coverage_records as collector  # noqa: E402


class DelegatedIntervalCarrierTest(unittest.TestCase):
    def test_rust_private_builder_is_not_public_but_unrestricted_wrapper_is(self) -> None:
        import check_interval_coordinates as intervals

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.rs"
            source.write_text("pub(crate) fn candidate_interval_replay() {}\n"
                              "pub fn candidate_interval_internal() {}\n")
            with patch.object(intervals, "ROOT", root), patch.object(intervals.cai, "ROOT", root), \
                    patch.object(intervals, "RELEASE", "2.3"):
                self.assertEqual(intervals.surface_interval_findings("source.rs"),
                                 [(2, "candidate_interval_internal")])

    def test_exact_projection_delegates_only_to_owned_method_with_factory_proof(self) -> None:
        import check_interval_coordinates as intervals

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "carrier.py"
            source.write_text("class Scalar:\n    @property\n    def interval(self):\n"
                              "        return float(self._body['lower']), float(self._body['upper'])\n")
            field = {"name": "interval", "source": "carrier.py", "class": "Scalar",
                     "method_records": ["method"], "authority_fixture": "factory-proof",
                     "why": "Original authorized endpoint projection"}
            record = {"id": "carrier", "prerequisite_records": ["method"],
                      "fixtures": [{"id": "factory-proof", "role": "negative", "evidence_assertion": "reject_forgery"}],
                      "delegated_interval_fields": [field]}
            owners = {"method": {"coverage_records": ["actual-method"],
                                  "inference_outputs": [{"allocated_coverage_records": ["actual-method"]}]}}
            findings = [("carrier.py", 3, "interval")]
            with patch.object(intervals, "ROOT", root):
                errors = []
                self.assertEqual(intervals.delegated_interval_findings(record, findings, owners, errors), [])
                self.assertEqual(errors, [])
                for altered_field in ({**field, "method_records": ["foreign"]},
                                      {**field, "authority_fixture": "absent"},
                                      {**field, "name": "new_interval"},
                                      {**field, "method_records": []}):
                    errors = []
                    altered = {**record, "delegated_interval_fields": [altered_field]}
                    self.assertEqual(intervals.delegated_interval_findings(altered, findings, owners, errors), findings)
                    self.assertTrue(errors)
                errors = []
                self.assertEqual(intervals.delegated_interval_findings(record, findings, {"method": {}}, errors), findings)
                self.assertTrue(errors)
                source.write_text("class Scalar:\n    @property\n    def interval(self):\n"
                                  "        return float(self._body['lower']) - 1, float(self._body['upper']) + 1\n")
                errors = []
                self.assertEqual(intervals.delegated_interval_findings(record, findings, owners, errors), findings)
                self.assertTrue(errors, "Numerically recomputed interval cannot borrow delegated evidence")


class ArchivedSupersededMeasurementTest(unittest.TestCase):
    def test_retired_emitter_binds_immutable_history_and_selected_replacement(self) -> None:
        import check_interval_coordinates as intervals

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "crates/fixture/tests/calibration.rs"
            source.parent.mkdir(parents=True)
            old = "cov.fixture.dag.frequentist.l95.old_method"
            replacement = "cov.fixture.dag.frequentist.l95.corrected_method"
            source.write_text(f'#[test] #[ignore] fn old_method() {{ emit("{old}"); }}')
            subprocess.run(["git", "init", "-q", directory], check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                            "commit", "-q", "-m", "fixture"], cwd=root, check=True)
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            source.unlink()  # Retirement removes the live emitter; immutable history remains.
            entry = {"id": old, "harness": "crates/fixture/tests/calibration.rs", "test": "old_method",
                     "status": "failed", "measurement_commit": sha, "failure": "below unchanged floor",
                     "disposition": "retired_superseded", "replacement_coverage_id": replacement}
            record = {"id": "fixture", "historical_coverage_records": [entry], "inference_outputs": []}
            with patch.object(intervals, "ROOT", root):
                errors = []
                self.assertEqual(intervals.historical_allocations(record, [replacement], set(), errors), {old})
                self.assertEqual(errors, [])
                for field, value, expected in (
                    ("test", "invented", "no original ignored harness emitter"),
                    ("measurement_commit", "0" * 40, "no original ignored harness emitter"),
                    ("replacement_coverage_id", old, "selected replacement"),
                    ("status", "passed", "needs failed status"),
                ):
                    altered = {**record, "historical_coverage_records": [{**entry, field: value}]}
                    errors = []
                    intervals.historical_allocations(altered, [replacement], set(), errors)
                    self.assertTrue(any(expected in error for error in errors), errors)
                errors = []
                intervals.historical_allocations(record, [replacement], {old}, errors)
                self.assertTrue(any("passing licensed record" in error for error in errors))
                allocated = {**record, "inference_outputs": [{"allocated_coverage_records": [old]}]}
                errors = []
                intervals.historical_allocations(allocated, [replacement], set(), errors)
                self.assertTrue(any("active inference output" in error for error in errors))


class ConditionalDgpReferenceTest(unittest.TestCase):
    def test_conditional_record_labels_resolve_both_actual_branches_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "suite.rs"
            source.write_text(
                'fn measure(latent: bool) { let key = RecordKey { '
                'dgp: if latent { "latent_rows" } else { "markov_rows" }, }; }'
            )
            self.assertEqual(collector.conditional_record_dgp_labels(source),
                             {"latent_rows", "markov_rows"})

    def test_comments_unrelated_literals_and_computed_branches_do_not_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "suite.rs"
            source.write_text(
                '// dgp: if latent { "forged" } else { "forged2" }\n'
                'fn measure() { let unrelated = "forged"; '
                'let key = RecordKey { dgp: if latent { label() } else { "computed" }, }; }'
            )
            self.assertEqual(collector.conditional_record_dgp_labels(source), set())


class AppendAttestedRecordsTest(unittest.TestCase):
    def test_append_preserves_old_rows_and_original_sha(self) -> None:
        old = {"old": {"id": "old", "calibration_sha": "old-sha", "observed": 0.8}}
        new = {"new": {"id": "new", "calibration_sha": "new-sha", "observed": 0.9}}
        merged = collector.append_attested_records(old, new)
        self.assertEqual(set(merged), {"old", "new"})
        self.assertEqual(merged["old"], old["old"])
        self.assertEqual(merged["new"], new["new"])
        self.assertEqual(set(old), {"old"})

    def test_append_refuses_collision(self) -> None:
        with self.assertRaisesRegex(SystemExit, "refuses existing record ids"):
            collector.append_attested_records({"same": {}}, {"same": {}})

    def test_selected_cell_sync_leaves_other_cells_byte_identical(self) -> None:
        source = (
            '[[cell]]\nquery = "Elasticity"\ngraph_class = "Dag"\n'
            'structure = "explicit"\ninference = "Frequentist"\nvalidation = "none"\n'
            'estimators = ["elasticity.plugin"]\n'
            'calibration_reason = "estimator_grid_not_measured"\n\n'
            '[[cell]]\nquery = "Other"\ngraph_class = "Dag"\n'
            'structure = "explicit"\ninference = "Frequentist"\nvalidation = "none"\n'
            'calibration = ["old"]\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "licensed.toml"
            path.write_text(source)
            original = collector.LICENSED
            collector.LICENSED = path
            try:
                changed = collector.sync_licensed_cells(
                    {
                        "new": {
                            "query": "Elasticity",
                            "graph_class": "Dag",
                            "inference": "Frequentist",
                            "structure": "fixed",
                            "nominal": 0.95,
                            "boundary": False,
                            "estimator": "elasticity.plugin",
                        }
                    },
                    {("Elasticity", "Dag", "explicit", "Frequentist", "none"): ["new"]},
                )
            finally:
                collector.LICENSED = original
            self.assertEqual(changed, 1)
            self.assertIn('calibration = ["new"]', path.read_text())
            self.assertEqual(
                path.read_text().split('[[cell]]')[2],
                source.split('[[cell]]')[2],
            )



class LicensedCellSyncTest(unittest.TestCase):
    @staticmethod
    def source(inference: str = "Bayesian", bound: bool = True) -> str:
        return (
            '[[cell]]\nquery="AverageEffect"\ngraph_class="Admg"\n'
            f'structure="explicit"\ninference="{inference}"\nvalidation="none"\n'
            'estimators=["functional.effect"]\n'
            + ('calibration = ["frontdoor"]\ncalibration_reason = "boundary_record"\n'
               if bound else 'calibration_reason = "estimator_grid_not_measured"\n')
        )

    @staticmethod
    def record(estimator: str, inference: str = "Bayesian", boundary: bool = False) -> dict:
        return dict(query="AverageEffect", graph_class="Admg", structure="fixed",
                    inference=inference, estimator=estimator, nominal=0.95, boundary=boundary)

    def test_new_passing_fisher_and_bayesian_do_not_rebind_frontdoor_boundary(self) -> None:
        for inference, estimator in (("Bayesian", "nested_markov_bayesian"),
                                     ("Frequentist", "nested_markov_fisher")):
            with self.subTest(inference=inference), tempfile.TemporaryDirectory() as directory:
                source = self.source(inference)
                path = Path(directory) / "licensed.toml"
                path.write_text(source)
                with patch.object(collector, "LICENSED", path):
                    collector.sync_licensed_cells({
                        "frontdoor": self.record("functional.effect", inference, True),
                        "new_passing": self.record(estimator, inference),
                    })
                parsed = collector.tomllib.loads(path.read_text())["cell"][0]
                self.assertEqual(parsed["calibration"], ["frontdoor"])
                self.assertEqual(parsed["calibration_reason"], "boundary_record")

    def test_default_sync_leaves_unmeasured_cell_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(bound=False)
            path = Path(directory) / "licensed.toml"
            path.write_text(source)
            with patch.object(collector, "LICENSED", path):
                self.assertEqual(collector.sync_licensed_cells({
                    "new": self.record("nested_markov_bayesian")}), 0)
            self.assertEqual(path.read_text(), source)

    def test_missing_bound_record_refuses_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self.source()
            path = Path(directory) / "licensed.toml"
            path.write_text(source)
            with patch.object(collector, "LICENSED", path), self.assertRaisesRegex(
                SystemExit, "bound calibration records absent"
            ):
                collector.sync_licensed_cells({"new": self.record("nested_markov_bayesian")})
            self.assertEqual(path.read_text(), source)

    def test_explicit_same_axes_foreign_estimator_refuses_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self.source()
            path = Path(directory) / "licensed.toml"
            path.write_text(source)
            with patch.object(collector, "LICENSED", path), self.assertRaisesRegex(
                SystemExit, "not owned by the licensed cell"
            ):
                collector.sync_licensed_cells({"new": self.record("nested_markov_bayesian")},
                    {("AverageEffect", "Admg", "explicit", "Bayesian", "none"): ["new"]})
            self.assertEqual(path.read_text(), source)

    def test_refresh_uses_new_measurement_of_original_bound_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "licensed.toml"
            path.write_text(self.source())
            with patch.object(collector, "LICENSED", path):
                collector.sync_licensed_cells({"frontdoor": self.record("functional.effect")})
            parsed = collector.tomllib.loads(path.read_text())["cell"][0]
            self.assertEqual(parsed["calibration"], ["frontdoor"])
            self.assertNotIn("calibration_reason", parsed)

    def test_schema_ownership_does_not_require_new_foreign_posteriors(self) -> None:
        for inference, foreign in (("Bayesian", "nested_markov_bayesian"),
                                   ("Frequentist", "nested_markov_fisher")):
            cell = collector.tomllib.loads(self.source(inference))["cell"][0]
            records = {"frontdoor": self.record("functional.effect", inference, True),
                       "new_posterior": self.record(foreign, inference)}
            self.assertEqual(collector.licensed_cell_calibration_problems(cell, records), [])
            cell["calibration"].append("new_posterior")
            self.assertTrue(any("foreign estimator" in error for error in
                                collector.licensed_cell_calibration_problems(cell, records)))

    def test_schema_requires_every_actual_owned_grid_and_rejects_missing_record(self) -> None:
        cell = collector.tomllib.loads(self.source())["cell"][0]
        records = {"frontdoor": self.record("functional.effect", boundary=True),
                   "new_owned_grid": self.record("functional.effect")}
        errors = collector.licensed_cell_calibration_problems(cell, records)
        self.assertTrue(any("missing required owned records" in error for error in errors))
        cell["calibration"].append("new_owned_grid")
        self.assertEqual(collector.licensed_cell_calibration_problems(cell, records), [])
        del records["frontdoor"]
        self.assertTrue(any("unknown bound record" in error for error in
                            collector.licensed_cell_calibration_problems(cell, records)))

    def test_schema_rejects_other_structure_even_with_same_estimator(self) -> None:
        cell = collector.tomllib.loads(self.source())["cell"][0]
        record = self.record("functional.effect")
        record["structure"] = "graph_posterior"
        self.assertTrue(any("scientific coordinate" in error for error in
                            collector.licensed_cell_calibration_problems(cell, {"frontdoor": record})))

    def test_default_refresh_refuses_foreign_existing_binding_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "licensed.toml"
            source = self.source()
            path.write_text(source)
            with patch.object(collector, "LICENSED", path), self.assertRaisesRegex(
                SystemExit, "not owned by the licensed cell"
            ):
                collector.sync_licensed_cells({"frontdoor": self.record("nested_markov_bayesian")})
            self.assertEqual(path.read_text(), source)



    @staticmethod
    def scientific_owner_evidence_errors(cell: dict, root: Path = collector.ROOT) -> list[str]:
        # Static resolver roots are redirected only for isolated synthetic schema fixtures.
        with patch("test_evidence.ROOT", root):
            return collector.calibration_estimator_evidence_problems(cell, root)

    @staticmethod
    def scientific_owner(root: Path, *, ignored: bool = False, comment_only: bool = False) -> dict:
        source = root / "crates/owner/src/lib.rs"
        proof = root / "crates/owner/tests/owner.rs"
        source.parent.mkdir(parents=True)
        proof.parent.mkdir(parents=True)
        (root / "crates/owner/Cargo.toml").write_text(
            '[package]\nname = "owner"\nversion = "0.1.0"\nedition = "2021"\n'
        )
        source.write_text("// pub struct NativeOwner;" if comment_only else "pub struct NativeOwner;")
        proof.write_text("#[test]\n" + ("#[ignore]\n" if ignored else "")
                         + "fn scientific_owner_proof() { assert_eq!(1, 1); }")
        return {"estimator": "actual.scientific", "source": "crates/owner/src/lib.rs",
                "source_symbol": "NativeOwner", "evidence_test": "crates/owner/tests/owner.rs",
                "evidence_assertion": "scientific_owner_proof"}

    def test_separate_scientific_owner_does_not_change_public_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = collector.tomllib.loads(self.source())["cell"][0]
            selectors = list(cell["estimators"])
            cell["calibration_estimators"] = [self.scientific_owner(root)]
            self.assertEqual(self.scientific_owner_evidence_errors(cell, root), [])
            self.assertTrue(collector.licensed_record_matches(cell, self.record("actual.scientific")))
            self.assertEqual(cell["estimators"], selectors)
            self.assertFalse(collector.licensed_record_matches(cell, self.record("nested_markov_bayesian")))

    def test_scientific_owner_requires_actual_native_code_not_comment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = {"estimators": [], "calibration_estimators": [
                self.scientific_owner(root, comment_only=True)]}
            self.assertTrue(any("outside comments/tests" in error for error in
                                self.scientific_owner_evidence_errors(cell, root)))

    def test_scientific_owner_requires_nonignored_asserting_native_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.scientific_owner(root, ignored=True)
            cell = {"estimators": [], "calibration_estimators": [entry]}
            self.assertTrue(any("ignore" in error for error in
                                self.scientific_owner_evidence_errors(cell, root)))
            proof = root / entry["evidence_test"]
            proof.write_text("#[test] fn scientific_owner_proof() {}")
            self.assertTrue(self.scientific_owner_evidence_errors(cell, root))

    def test_scientific_owner_malformed_or_duplicate_declarations_refuse(self) -> None:
        for entries in ["actual.scientific", ["actual.scientific"], [{"estimator": "forged"}]]:
            self.assertTrue(self.scientific_owner_evidence_errors(
                {"estimators": [], "calibration_estimators": entries}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.scientific_owner(root)
            self.assertTrue(any("duplicate" in error for error in
                self.scientific_owner_evidence_errors(
                    {"estimators": [], "calibration_estimators": [entry, entry]}, root)))


def _point(k: int, observed: float, role: str, **extra) -> dict:
    """One grid point's `calibration-record` payload, as the harness emits it."""
    payload = {
        "id": "cov.classical_transport.admg.frequentist.percentile_bootstrap.l95.t",
        "query": "ClassicalTransport",
        "nominal": 0.95,
        "n_min": 200 * 2**k,
        "n_max": 200 * 2**k,
        "replicates_min": 199,
        "posterior_draws_min": 0,
        "unidentified_mass_max": 0.0,
        "observed": observed,
        "mcse": 0.011,
        "replicates": 400,
        "bound_replicates": 400,
        "boundary": False,
        "role": role,
        "grid_point": k,
    }
    payload.update(extra)
    return payload


UPPER = {"side": "upper", "target": "true upper extremal bound U"}



class OneSidedRoleTest(unittest.TestCase):
    def test_a_one_sided_record_keeps_its_side_and_target(self) -> None:
        points = {k: _point(k, obs, "one_sided", one_sided=UPPER) for k, obs in enumerate((0.99, 0.985, 1.0))}
        record = collector.merge_grid("rid", points)
        self.assertEqual(record["role"], "one_sided")
        self.assertEqual(record["one_sided"], UPPER)
        self.assertFalse(record["boundary"])
        self.assertEqual({p["role"] for p in record["grid"]}, {"one_sided"})

    def test_a_one_sided_record_must_state_its_side_and_target(self) -> None:
        points = {k: _point(k, 0.99, "one_sided") for k in range(3)}
        with self.assertRaisesRegex(SystemExit, "states one_sided"):
            collector.merge_grid("rid", points)
        bad = {k: _point(k, 0.99, "one_sided", one_sided={"side": "both", "target": "t"}) for k in range(3)}
        with self.assertRaisesRegex(SystemExit, "side = lower|upper"):
            collector.merge_grid("rid", bad)

    def test_only_a_one_sided_record_states_a_side(self) -> None:
        points = {k: _point(k, 0.95, "gated", one_sided=UPPER) for k in range(3)}
        with self.assertRaisesRegex(SystemExit, "states one_sided"):
            collector.merge_grid("rid", points)

    def test_one_sided_does_not_mix_with_another_role(self) -> None:
        points = {k: _point(k, 0.99, "one_sided", one_sided=UPPER) for k in range(2)}
        points[2] = _point(2, 0.95, "gated", one_sided=UPPER)
        with self.assertRaisesRegex(SystemExit, "another construction|incompatible roles|states one_sided"):
            collector.merge_grid("rid", points)

    def test_an_unknown_role_is_refused(self) -> None:
        points = {k: _point(k, 0.95, "two_sided_ish") for k in range(3)}
        with self.assertRaisesRegex(SystemExit, "unknown coverage role"):
            collector.merge_grid("rid", points)

    def test_the_registry_writes_one_sided_after_role(self) -> None:
        import tomllib

        record = collector.merge_grid(
            "rid", {k: _point(k, 0.99, "one_sided", one_sided=UPPER) for k in range(3)}
        )
        record.update({key: "x" for key in collector.FIELDS if key not in record})
        record.update(facets=["core"], calibration_sha="1" * 40, surface_list_blob="1" * 40)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "records.toml"
            original = collector.surface_list_blob
            collector.surface_list_blob = lambda sha: "1" * 40
            try:
                collector.write_registry({"rid": record}, out, retag=False)
            finally:
                collector.surface_list_blob = original
            text = out.read_text()
            parsed = tomllib.loads(text)["record"][0]
        self.assertEqual(parsed["one_sided"], UPPER)
        self.assertLess(text.index('role = "one_sided"'), text.index("one_sided = {"))
        self.assertLess(text.index("one_sided = {"), text.index("grid = ["))


if __name__ == "__main__":
    unittest.main()
