"""Focused guardrails for incremental coverage-record collection."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _repo import ensure_scripts_on_path, write_text

ensure_scripts_on_path()
import collect_coverage_records as collector  # noqa: E402


class DelegatedIntervalCarrierTest(unittest.TestCase):
    def test_rust_private_builder_is_not_public_but_unrestricted_wrapper_is(self) -> None:
        import check_interval_coordinates as intervals

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.rs"
            source.write_text("pub(crate) fn candidate_interval_replay() {}\n"
                              "pub fn candidate_interval_internal() {}\n", encoding="utf-8", newline="\n")
            with patch.object(intervals, "ROOT", root), patch.object(intervals.cai, "ROOT", root):
                self.assertEqual(intervals.surface_interval_findings("source.rs"),
                                 [(2, "candidate_interval_internal")])

    def test_exact_projection_delegates_only_to_owned_method_with_factory_proof(self) -> None:
        import check_interval_coordinates as intervals

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "carrier.py"
            source.write_text("class Scalar:\n    @property\n    def interval(self):\n"
                              "        return float(self._body['lower']), float(self._body['upper'])\n", encoding="utf-8", newline="\n")
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
                                  "        return float(self._body['lower']) - 1, float(self._body['upper']) + 1\n", encoding="utf-8", newline="\n")
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
            source.write_text(f'#[test] #[ignore] fn old_method() {{ emit("{old}"); }}', encoding="utf-8", newline="\n")
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
            , encoding="utf-8", newline="\n")
            self.assertEqual(collector.conditional_record_dgp_labels(source),
                             {"latent_rows", "markov_rows"})

    def test_comments_unrelated_literals_and_computed_branches_do_not_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "suite.rs"
            source.write_text(
                '// dgp: if latent { "forged" } else { "forged2" }\n'
                'fn measure() { let unrelated = "forged"; '
                'let key = RecordKey { dgp: if latent { label() } else { "computed" }, }; }'
            , encoding="utf-8", newline="\n")
            self.assertEqual(collector.conditional_record_dgp_labels(source), set())


class PassingGroupHarvestTest(unittest.TestCase):
    def test_failed_recheck_and_incomplete_grid_skip_whole_groups(self) -> None:
        sha = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def log(name: str, failed: bool = False, measured: str = sha) -> Path:
                path = root / name
                result = "FAILED. 0 passed; 1 failed;" if failed else "ok. 1 passed; 0 failed;"
                path.write_text(f"test result: {result} finished in 0.1s\ncalibration-measured-at {measured}\n")
                return path
            complete = [log(f"complete.p{p}.log") for p in range(3)]
            failed = [log(f"failed.p{p}.log") for p in range(3)] + [log("failed.p1.recheck.log", True)]
            incomplete = [log("partial.p0.log"), log("partial.p2.log")]
            wrong_sha = [log("wrong.log", measured="b" * 40)]
            self.assertEqual(collector.passing_group_logs(complete + failed + incomplete + wrong_sha, sha), complete)

    def test_malformed_whole_file_records_skip_the_entire_group(self) -> None:
        sha = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            logs = []
            for point in range(3):
                log = Path(directory) / f"whole_file.p{point}.log"
                log.write_text(f"test result: ok. 2 passed; 0 failed;\ncalibration-measured-at {sha}\n")
                logs.append(log)
            with patch.object(collector, "merged_records", side_effect=SystemExit("duplicate record id")):
                self.assertEqual(collector.passing_group_logs(logs, sha), [])


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

    def test_replace_record_requires_explicit_collision_allowlist_and_fresh_evidence(self) -> None:
        old = {"changed": {"id": "changed", "calibration_sha": "old"},
               "unrelated": {"id": "unrelated", "calibration_sha": "old", "observed": 0.95}}
        new = {"changed": {"id": "changed", "calibration_sha": "new", "observed": 0.951}}
        merged = collector.append_attested_records(old, new, mode="--keep-existing", replace_ids=["changed"])
        self.assertEqual(merged["changed"], new["changed"])
        self.assertIs(merged["unrelated"], old["unrelated"])
        with self.assertRaisesRegex(SystemExit, "refuses existing record ids"):
            collector.append_attested_records(old, new, mode="--keep-existing")
        for invalid in ("unrelated", "not-existing"):
            with self.assertRaisesRegex(SystemExit, "both existing and new evidence"):
                collector.append_attested_records(old, new, replace_ids=[invalid])

    def test_keep_existing_cli_preserves_stale_rows_and_previous_gate_evidence(self) -> None:
        old = {"id": "old", "calibration_sha": "old-sha", "facets": ["core"]}
        new = {"id": "new", "calibration_sha": "new-sha", "facets": ["core"]}
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "gates.toml"
            ledger.write_text('[[gate]]\ngroup="old-gate"\npassed=1\ncalibration_sha="old-sha"\n')
            argv = ["collector", "--keep-existing", "--no-cells", "--sha", "a" * 40]
            with patch.object(sys, "argv", argv), patch.object(collector, "ROOT", Path(directory)), patch.object(collector, "OUT", Path(directory) / "records.toml"), patch.object(collector, "GATES", ledger), patch.object(collector, "load_records", return_value={"new": new}), patch.object(collector.facets, "load_records", return_value=[old]), patch.object(collector, "write_registry") as write, patch.object(collector, "write_gate_ledger") as gates, patch.object(collector, "gate_ledger", return_value={}), patch.object(collector.subprocess, "run"):
                self.assertEqual(collector.main(), 0)
                self.assertEqual(write.call_args.args[0], {"old": old, "new": new})
                self.assertEqual(write.call_args.kwargs, {"retag": False})
                self.assertEqual(gates.call_args.args[1]["old-gate"]["calibration_sha"], "old-sha")

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
            path.write_text(source, encoding="utf-8", newline="\n")
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
            self.assertIn('calibration = ["new"]', path.read_text(encoding="utf-8"))
            self.assertEqual(
                path.read_text(encoding="utf-8").split('[[cell]]')[2],
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
                path.write_text(source, encoding="utf-8", newline="\n")
                with patch.object(collector, "LICENSED", path):
                    collector.sync_licensed_cells({
                        "frontdoor": self.record("functional.effect", inference, True),
                        "new_passing": self.record(estimator, inference),
                    })
                parsed = collector.tomllib.loads(path.read_text(encoding="utf-8"))["cell"][0]
                self.assertEqual(parsed["calibration"], ["frontdoor"])
                self.assertEqual(parsed["calibration_reason"], "boundary_record")

    def test_default_sync_leaves_unmeasured_cell_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(bound=False)
            path = Path(directory) / "licensed.toml"
            path.write_text(source, encoding="utf-8", newline="\n")
            with patch.object(collector, "LICENSED", path):
                self.assertEqual(collector.sync_licensed_cells({
                    "new": self.record("nested_markov_bayesian")}), 0)
            self.assertEqual(path.read_text(encoding="utf-8"), source)

    def test_missing_bound_record_refuses_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self.source()
            path = Path(directory) / "licensed.toml"
            path.write_text(source, encoding="utf-8", newline="\n")
            with patch.object(collector, "LICENSED", path), self.assertRaisesRegex(
                SystemExit, "bound calibration records absent"
            ):
                collector.sync_licensed_cells({"new": self.record("nested_markov_bayesian")})
            self.assertEqual(path.read_text(encoding="utf-8"), source)

    def test_explicit_same_axes_foreign_estimator_refuses_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self.source()
            path = Path(directory) / "licensed.toml"
            path.write_text(source, encoding="utf-8", newline="\n")
            with patch.object(collector, "LICENSED", path), self.assertRaisesRegex(
                SystemExit, "not owned by the licensed cell"
            ):
                collector.sync_licensed_cells({"new": self.record("nested_markov_bayesian")},
                    {("AverageEffect", "Admg", "explicit", "Bayesian", "none"): ["new"]})
            self.assertEqual(path.read_text(encoding="utf-8"), source)

    def test_refresh_uses_new_measurement_of_original_bound_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "licensed.toml"
            path.write_text(self.source(), encoding="utf-8", newline="\n")
            with patch.object(collector, "LICENSED", path):
                collector.sync_licensed_cells({"frontdoor": self.record("functional.effect")})
            parsed = collector.tomllib.loads(path.read_text(encoding="utf-8"))["cell"][0]
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
            path.write_text(source, encoding="utf-8", newline="\n")
            with patch.object(collector, "LICENSED", path), self.assertRaisesRegex(
                SystemExit, "not owned by the licensed cell"
            ):
                collector.sync_licensed_cells({"frontdoor": self.record("nested_markov_bayesian")})
            self.assertEqual(path.read_text(encoding="utf-8"), source)



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
        , encoding="utf-8", newline="\n")
        source.write_text("// pub struct NativeOwner;" if comment_only else "pub struct NativeOwner;", encoding="utf-8", newline="\n")
        proof.write_text("#[test]\n" + ("#[ignore]\n" if ignored else "")
                         + "fn scientific_owner_proof() { assert_eq!(1, 1); }", encoding="utf-8", newline="\n")
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
            proof.write_text("#[test] fn scientific_owner_proof() {}", encoding="utf-8", newline="\n")
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


class FacetTaggingTest(unittest.TestCase):
    def test_empty_measurement_needs_no_repository_reference_scan(self):
        with patch.object(collector.facets, "load_surface") as surface, patch.object(collector.facets, "references") as refs:
            collector.tag_facets({})
        surface.assert_not_called()
        refs.assert_not_called()





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
            text = out.read_text(encoding="utf-8")
            parsed = tomllib.loads(text)["record"][0]
        self.assertEqual(parsed["one_sided"], UPPER)
        self.assertLess(text.index('role = "one_sided"'), text.index("one_sided = {"))
        self.assertLess(text.index("one_sided = {"), text.index("grid = ["))


class ConfirmedPassingRangeTest(unittest.TestCase):
    """A boundary record recovers a passing range only from an independent confirmation."""

    @staticmethod
    def boundary_record(failing: tuple[int, ...], role: str = "gated") -> dict:
        points = {
            k: _point(k, 0.90 if k in failing else 0.95, role, boundary=k in failing)
            for k in range(3)
        }
        return collector.merge_grid("rid", points)

    @staticmethod
    def confirmed(record: dict, passing: tuple[int, ...], **extra) -> dict:
        collector.attach_confirmation(
            "rid", record, {k: _point(k, 0.951, record["role"], **extra) for k in passing}
        )
        return record

    def test_the_upper_passing_points_recover_their_range_and_the_failure_stays(self) -> None:
        record = self.confirmed(self.boundary_record((0,)), (1, 2))
        self.assertEqual(record["confirmed_passing_range"], [400, 800])
        self.assertTrue(record["boundary"])
        self.assertEqual(record["observed"], 0.90)
        self.assertEqual([p["point"] for p in record["confirmation"]], [1, 2])
        self.assertTrue(record["grid"][0]["boundary"])

    def test_the_lower_passing_points_recover_their_range(self) -> None:
        record = self.confirmed(self.boundary_record((2,)), (0, 1))
        self.assertEqual(record["confirmed_passing_range"], [200, 400])

    def test_a_point_the_confirmation_did_not_pass_is_not_recovered(self) -> None:
        self.assertNotIn("confirmed_passing_range", self.confirmed(self.boundary_record((0,)), (2,)))
        self.assertNotIn("confirmed_passing_range", self.confirmed(self.boundary_record((0,)), ()))

    def test_a_middle_failure_and_a_lone_point_are_not_subranges(self) -> None:
        self.assertNotIn("confirmed_passing_range", self.confirmed(self.boundary_record((1,)), (0, 2)))
        self.assertNotIn("confirmed_passing_range", self.confirmed(self.boundary_record((0, 1)), (2,)))

    def test_a_named_boundary_and_a_passing_record_recover_nothing(self) -> None:
        named = self.confirmed(self.boundary_record((0,), role="named_boundary"), (1, 2))
        self.assertNotIn("confirmation", named)
        passing = collector.merge_grid("rid", {k: _point(k, 0.95, "gated") for k in range(3)})
        collector.attach_confirmation("rid", passing, {1: _point(1, 0.95, "gated")})
        self.assertNotIn("confirmation", passing)

    def test_a_confirmation_of_another_design_is_refused(self) -> None:
        record = self.boundary_record((0,))
        with self.assertRaisesRegex(SystemExit, "another construction"):
            self.confirmed(record, (1,), estimator="foreign")
        with self.assertRaisesRegex(SystemExit, "re-measures the same design"):
            self.confirmed(record, (1,), n_min=999, n_max=999)

    def test_confirmation_lines_are_collected_apart_from_the_measurement(self) -> None:
        def line(k: int, confirmation: bool, failing: bool = False) -> str:
            payload = _point(
                k, 0.90 if failing else 0.951, "gated", boundary=failing, measured_at="a" * 40
            )
            if confirmation:
                payload["confirmation"] = True
            return "calibration-record " + json.dumps(payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = []
            for k in range(3):
                measured = root / f"g.p{k}.log"
                write_text(measured, line(k, False, failing=k == 0) + "\n")
                logs.append(measured)
            for k in (1, 2):
                confirm = root / f"g.p{k}.confirm.log"
                write_text(confirm, line(k, True) + "\n")
                logs.append(confirm)
            records = collector.merged_records(logs)
            self.assertEqual(len(records), 1)
            self.assertEqual(next(iter(records.values()))["confirmed_passing_range"], [400, 800])
            # A confirmation line in a measurement log is refused, not merged in.
            write_text(logs[0], line(0, True) + "\n")
            with self.assertRaisesRegex(SystemExit, "is a confirmation line"):
                collector.merged_records(logs)

    def test_the_registry_writes_the_confirmation_after_role(self) -> None:
        import tomllib

        record = self.confirmed(self.boundary_record((0,)), (1, 2))
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
            text = out.read_text(encoding="utf-8")
            parsed = tomllib.loads(text)["record"][0]
        self.assertEqual(parsed["confirmed_passing_range"], [400, 800])
        self.assertEqual([p["point"] for p in parsed["confirmation"]], [1, 2])
        self.assertLess(text.index('role = "gated"'), text.index("confirmation = ["))
        self.assertLess(text.index("confirmed_passing_range = "), text.index("grid = ["))


class ConfirmationMatchingTest(unittest.TestCase):
    """A confirmation combines with its measurement only when every matching axis agrees."""

    CONSTRUCTION = {
        "estimator": "aipw",
        "interval_method": "bootstrap_se",
        "se_kind": "",
        "dependence": "iid",
        "functional": "all_observed.mean",
        "identification": "point",
    }

    def measured(self) -> dict:
        points = {
            k: _point(k, 0.90 if k == 0 else 0.95, "gated", boundary=k == 0, **self.CONSTRUCTION)
            for k in range(3)
        }
        return collector.merge_grid("rid", points)

    def refused(self, **changed) -> str:
        """The refusal of a confirmation whose construction differs by `changed`."""
        record = self.measured()
        confirmation = {
            k: _point(k, 0.951, "gated", **{**self.CONSTRUCTION, **changed}) for k in (1, 2)
        }
        with self.assertRaises(SystemExit) as raised:
            collector.attach_confirmation("rid", record, confirmation)
        self.assertNotIn("confirmation", record, "nothing of a refused confirmation is combined")
        self.assertNotIn("confirmed_passing_range", record)
        self.assertTrue(record["boundary"] and record["grid"][0]["boundary"])
        return str(raised.exception)

    def test_an_identical_construction_combines(self) -> None:
        record = self.measured()
        confirmation = {k: _point(k, 0.951, "gated", **self.CONSTRUCTION) for k in (1, 2)}
        collector.attach_confirmation("rid", record, confirmation)
        self.assertEqual(record["confirmed_passing_range"], [400, 800])

    def test_calibration_subrange_construction_mismatch_refused(self) -> None:
        for field, other in (
            ("estimator", "linear_regression"),
            ("interval_method", "percentile_bootstrap"),
            ("se_kind", "hc3"),
        ):
            message = self.refused(**{field: other})
            self.assertIn("another construction", message)
            self.assertIn(field, message)

    def test_calibration_subrange_estimand_mismatch_refused(self) -> None:
        message = self.refused(functional="treated.mean_difference")
        self.assertIn("another construction", message)
        self.assertIn("functional", message)

    def test_calibration_subrange_nominal_level_mismatch_refused(self) -> None:
        message = self.refused(nominal=0.90)
        self.assertIn("another construction", message)
        self.assertIn("nominal", message)

    def test_calibration_subrange_dependence_mismatch_refused(self) -> None:
        message = self.refused(dependence="panel_cluster")
        self.assertIn("another construction", message)
        self.assertIn("dependence", message)

    def test_calibration_subrange_identification_mismatch_refused(self) -> None:
        message = self.refused(identification="partial")
        self.assertIn("another construction", message)
        self.assertIn("identification", message)

    def test_calibration_subrange_attestation_mismatch_refused(self) -> None:
        def line(k: int, measured_at: str, confirmation: bool = False) -> str:
            failing = k == 0 and not confirmation
            payload = _point(
                k,
                0.90 if failing else 0.951,
                "gated",
                boundary=failing,
                measured_at=measured_at,
                **self.CONSTRUCTION,
            )
            if confirmation:
                payload["confirmation"] = True
            return "calibration-record " + json.dumps(payload)

        attested, other = "a" * 40, "b" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = []
            for k in range(3):
                measured = root / f"g.p{k}.log"
                write_text(measured, line(k, attested) + "\n")
                logs.append(measured)
            for k in (1, 2):
                confirm = root / f"g.p{k}.confirm.log"
                write_text(confirm, line(k, other, confirmation=True) + "\n")
                logs.append(confirm)
            # A confirmation measured at another commit is not combined with the measurement.
            with self.assertRaisesRegex(
                SystemExit, f"says it was measured at {other}, not at {attested}"
            ):
                collector.merged_records(logs, sha=attested)
            # The same commit combines.
            for k in (1, 2):
                write_text(root / f"g.p{k}.confirm.log", line(k, attested, confirmation=True) + "\n")
            records = collector.merged_records(logs, sha=attested)
            self.assertEqual(next(iter(records.values()))["confirmed_passing_range"], [400, 800])


class FourPointRangeTest(unittest.TestCase):
    """The derivation of a passing range does not depend on a grid of three points."""

    SIZES = (400, 800, 1600, 3200)

    def grid(self, failing: tuple[int, ...]) -> list[dict]:
        return [
            {"point": k, "n_min": n, "n_max": n, "boundary": k in failing}
            for k, n in enumerate(self.SIZES)
        ]

    def confirmed(self, points: tuple[int, ...]) -> list[dict]:
        return [{"point": k, "n_min": self.SIZES[k], "n_max": self.SIZES[k], "boundary": False} for k in points]

    def test_the_upper_three_points_recover_800_to_3200(self) -> None:
        self.assertEqual(
            collector.confirmed_passing_range(self.grid((0,)), self.confirmed((1, 2, 3))),
            [800, 3200],
        )

    def test_without_confirmation_or_with_a_gap_in_it_nothing_is_recovered(self) -> None:
        self.assertIsNone(collector.confirmed_passing_range(self.grid((0,)), []))
        self.assertIsNone(collector.confirmed_passing_range(self.grid((0,)), self.confirmed((1, 3))))

    def test_a_failed_interior_point_is_never_inside_a_range(self) -> None:
        # 400 passed, 800 failed, 1600 and 3200 passed: only the upper run is a subrange.
        grid = self.grid((1,))
        self.assertEqual(
            collector.confirmed_passing_range(grid, self.confirmed((0, 2, 3))), [1600, 3200]
        )
        # With three points the passing ends are lone points: no range at all.
        three = self.grid((1,))[:3]
        self.assertIsNone(collector.confirmed_passing_range(three, self.confirmed((0, 2))))


class GateProtocolTest(unittest.TestCase):
    def test_marker_is_observed_in_log_never_inferred_and_survives_ledger_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old.log", root / "new.log"
            passed = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
            old.write_text(passed)
            new.write_text("CALIBRATION_GATE_PROTOCOL population_linear\n" + passed)
            rows = collector.gate_ledger([old, new], "a" * 40)
            self.assertNotIn("protocol", rows["old"])
            self.assertEqual(rows["new"]["protocol"], "population_linear")
            ledger = root / "gates.toml"
            with patch.object(collector, "GATES", ledger):
                collector.write_gate_ledger({"old": rows["old"]}, {"new": rows["new"]})
            parsed = {row["group"]: row for row in collector.tomllib.loads(ledger.read_text())["gate"]}
            self.assertEqual(parsed, rows)
            new.write_text("CALIBRATION_GATE_PROTOCOL population_linear\nCALIBRATION_GATE_PROTOCOL other\n" + passed)
            with self.assertRaisesRegex(SystemExit, "conflicting"):
                collector.gate_ledger([new], "a" * 40)


if __name__ == "__main__":
    unittest.main()
