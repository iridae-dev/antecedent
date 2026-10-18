"""Release-closure receipts: each passes on a complete fixture and fails on every defect."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import check_release_closure as closure  # noqa: E402

WHEEL_STEP = """
  (cd python && uv run maturin build --release --out "$wheel_dir")
  uv pip install --python "$venv_dir/venv/bin/python" "${wheels[0]}"
  cd "$run_dir"
  "$venv_dir/venv/bin/python" "$ROOT/scripts/gate_release_candidate_2.py" "$venv_dir"
  "$venv_dir/venv/bin/python" "$ROOT/scripts/run_doc_examples.py"
  "$venv_dir/venv/bin/python" -m pytest -q -p no:cacheprovider "$ROOT/python/tests"
"""
CI = "cargo nextest run -P ci --workspace"
CALIBRATION_GATE = "run_ps practitioner_design\n"


def case(cid: str, kind: str, test: str, **extra: Any) -> dict[str, Any]:
    targets = {"runtime_seconds": 1, "memory_bytes": 1}
    if kind == "calibrated_interval":
        targets |= {"nominal_coverage": 0.95, "coverage_floor": 0.93, "min_replicates": 2000}
    return {"id": cid, "area": extra.pop("area", "ingest"), "configuration": extra.pop("configuration", {"rows": 100}),
            "expected": {"kind": kind, "outputs": extra.pop("outputs", ["o"])},
            "targets": targets, "evidence": {"status": "bound", "test": test}}


def record(**extra: Any) -> dict[str, Any]:
    point = {"point": 0, "replicates": 2000, "mcse": 0.005, "observed": 0.95, "role": "gated", "boundary": False}
    base = {"id": "rec", "test": "crates/c/tests/t.rs::design", "nominal": 0.95, "n_min": 10, "n_max": 1000,
            "role": "gated", "boundary": False, "grid": [point], "calibration_sha": "a", "surface_list_blob": "b",
            "facets": ["core"], "estimator": "e", "interval_method": "m"}
    return {**base, **extra}


class ReleaseClosureTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "crates/c/tests").mkdir(parents=True)
        (self.root / "python/tests").mkdir(parents=True)

    def put(self, rel: str, text: str) -> None:
        (self.root / rel).write_text(text, encoding="utf-8", newline="\n")

    def test_the_installed_wheel_step_runs_every_example_and_the_suite_outside_the_source_tree(self) -> None:
        self.assertEqual(closure.installed_wheel_problems(WHEEL_STEP), [])
        for needle, label in (("run_doc_examples.py", "documented snippets"), ("-m pytest", "Python suite"),
                              ("gate_release_candidate_2.py", "wheel is the imported package")):
            broken = "\n".join(line for line in WHEEL_STEP.splitlines() if needle not in line)
            self.assertTrue(any(label in p for p in closure.installed_wheel_problems(broken)), needle)
        system_python = WHEEL_STEP.replace('"$venv_dir/venv/bin/python" "$ROOT/scripts/run_doc_examples.py"',
                                           'python3 "$ROOT/scripts/run_doc_examples.py"')
        self.assertTrue(closure.installed_wheel_problems(system_python))
        in_tree = WHEEL_STEP.replace('  cd "$run_dir"\n', "")
        self.assertTrue(any("leave the source tree" in p for p in closure.installed_wheel_problems(in_tree)))
        self.assertTrue(closure.example_problems(self.root))
        self.put("python/tests/test_examples_run.py", "import runpy\nEXAMPLES = 'examples/python'\nrunpy.run_path\n")
        self.assertEqual(closure.example_problems(self.root), [])

    def test_a_practitioner_scale_case_needs_its_gate_registration_and_a_full_record(self) -> None:
        data = {"case": [case("scale", "calibrated_interval", "crates/c/tests/t.rs::practitioner_design",
                              area="practitioner_scale", configuration={"participants": 100})]}
        rec = record(test="crates/c/tests/t.rs::practitioner_design")
        self.assertEqual(closure.practitioner_scale_problems(data, [rec], CALIBRATION_GATE), [])
        self.assertTrue(any("does not run" in p for p in closure.practitioner_scale_problems(data, [rec], "")))
        self.assertTrue(any("no coverage record" in p for p in closure.practitioner_scale_problems(data, [], CALIBRATION_GATE)))
        unbound = copy.deepcopy(data)
        unbound["case"][0]["evidence"] = {"status": "planned"}
        self.assertTrue(any("not bound" in p for p in closure.practitioner_scale_problems(unbound, [rec], CALIBRATION_GATE)))

    def test_a_refusal_case_must_be_bound_to_a_live_asserting_test(self) -> None:
        self.put("crates/c/tests/t.rs", '#[test]\nfn refuses() {\n    assert!(run().is_err());\n}\n'
                 '#[test]\n#[ignore]\nfn off() {\n    assert!(run().is_err());\n}\n')
        live = {"case": [case("r", "refusal", "crates/c/tests/t.rs::refuses")]}
        self.assertEqual(closure.refusal_boundary_problems(live, self.root, CI), [])
        self.assertTrue(any("no release gate executes" in p for p in closure.refusal_boundary_problems(live, self.root, "")))
        off = {"case": [case("r", "refusal", "crates/c/tests/t.rs::off")]}
        self.assertTrue(any("ignored" in p for p in closure.refusal_boundary_problems(off, self.root, CI)))
        planned = copy.deepcopy(live)
        planned["case"][0]["evidence"] = {"status": "planned"}
        self.assertTrue(any("no bound test" in p for p in closure.refusal_boundary_problems(planned, self.root, CI)))
        elsewhere = {"case": [case("r", "refusal", "docs/t.rs::refuses")]}
        self.assertTrue(closure.refusal_boundary_problems(elsewhere, self.root, CI))

    def test_artifact_cases_must_compare_the_replay_with_the_original(self) -> None:
        self.put("crates/c/tests/t.rs", '#[test]\nfn replays() {\n    assert_eq!(saved, loaded);\n}\n'
                 '#[test]\nfn runs() {\n    assert!(loaded.is_ok());\n}\n')
        outputs = ["replayed draws bit-identical to the original draws"]
        good = {"case": [case("a", "value", "crates/c/tests/t.rs::replays", outputs=outputs)]}
        self.assertEqual(closure.replay_identity_problems(good, self.root), [])
        weak = {"case": [case("a", "value", "crates/c/tests/t.rs::runs", outputs=outputs)]}
        self.assertTrue(any("never compares" in p for p in closure.replay_identity_problems(weak, self.root)))
        unrelated = {"case": [case("a", "value", "crates/c/tests/t.rs::runs")]}
        self.assertEqual(closure.replay_identity_problems(unrelated, self.root), [])

    def calibrated(self) -> dict[str, Any]:
        return {"case": [case("cal", "calibrated_interval", "crates/c/tests/t.rs::design")]}

    def test_split_enrollment_binds_the_actual_analysis_grid_and_rejects_false_sizes(self) -> None:
        data = self.calibrated()
        target = data["case"][0]
        target["configuration"] = {"participants": 2000, "analysis_rows": 500,
                                   "discovery_fraction": 0.75}
        rec = record(grid=[
            {**record()["grid"][0], "point": 0, "n_min": 250, "n_max": 250,
             "replicates": 400},
            {**record()["grid"][0], "point": 1, "n_min": 500, "n_max": 500},
            {**record()["grid"][0], "point": 2, "n_min": 1000, "n_max": 1000,
             "replicates": 400}])
        self.assertEqual(closure.calibrated_record_problems(data, [rec]), [])
        rec["grid"][1]["replicates"] = 1999
        self.assertTrue(any("point 1" in p and "below 2000" in p
                            for p in closure.calibrated_record_problems(data, [rec])))
        target["configuration"]["analysis_rows"] = 1000
        self.assertEqual(closure.records_of(target, [rec]), [])
        self.assertTrue(any("analysis_rows must equal" in p
                            for p in closure.calibrated_record_problems(data, [rec])))
        target["configuration"] = {"participants": 3000, "analysis_rows": 1500,
                                   "discovery_fraction": 0.5}
        self.assertEqual(closure.case_size(target), 1500)
        self.assertEqual(closure.split_size_problems(target), [])
        target["configuration"]["record_size_basis"] = "procedure_rows"
        self.assertEqual(closure.case_size(target), 3000)
        self.assertEqual(closure.split_size_problems(target), [])
        target["configuration"]["record_size_basis"] = "ambiguous"
        self.assertTrue(closure.split_size_problems(target))

    def test_parent_split_case_cannot_borrow_a_legacy_half_split_record(self) -> None:
        target = case("parent", "calibrated_interval",
                      "crates/antecedent/tests/sample_split_parent_set_effect_calibration.rs::design",
                      configuration={"participants": 2000, "analysis_rows": 500,
                                     "discovery_fraction": 0.75})
        legacy = record(test=target["evidence"]["test"], functional="total_effect.variables_15")
        current = {**legacy, "id": "v2",
                   "functional": "total_effect.variables_15.discovery_fraction_0_75.topology_layered"}
        self.assertEqual(closure.records_of(target, [legacy, current]), [current])

    def test_weighted_split_new_allocation_keeps_procedure_scope_and_version_binding(self) -> None:
        target = case("weighted", "calibrated_interval",
                      "crates/antecedent/tests/sample_split_weighted_effect_calibration.rs::design",
                      configuration={"participants": 3000, "analysis_rows": 750,
                                     "discovery_fraction": 0.75, "record_size_basis": "procedure_rows"})
        legacy = record(test=target["evidence"]["test"], n_min=3000, n_max=3000,
                        functional="weighted_population_total_effect")
        current = {**legacy, "id": "v2", "functional":
                   "weighted_population_total_effect.discovery_fraction_0_75.conflict_resolution_treatment_invariant"}
        self.assertEqual(closure.case_size(target), 3000)
        self.assertEqual(closure.records_of(target, [legacy, current]), [current])

        target["configuration"].update(interval_variance="hc3_sampling_weighted",
                                       record_functional_suffix="variance_hc3")
        hc3 = {**current, "id": "v3", "se_kind": "hc3_sampling_weighted",
               "functional": current["functional"] + ".variance_hc3"}
        self.assertEqual(closure.records_of(target, [legacy, current, hc3]), [hc3])
        wrong_variance = {**hc3, "se_kind": "hc1_sampling_weighted"}
        wrong_suffix = {**hc3, "functional": current["functional"] + ".variance_hc3_extra"}
        for wrong in (wrong_variance, wrong_suffix):
            self.assertEqual(closure.records_of(target, [wrong]), [])

    def test_calibrated_records_need_two_thousand_replicates_and_uncertainty(self) -> None:
        data = self.calibrated()
        self.assertEqual(closure.calibrated_record_problems(data, [record()]), [])
        few = record(grid=[{**record()["grid"][0], "replicates": 400}])
        self.assertTrue(any("below 2000" in p for p in closure.calibrated_record_problems(data, [few])))
        for mcse in (None, 0.0, float("nan")):
            blind = record(grid=[{**record()["grid"][0], "mcse": mcse}])
            self.assertTrue(any("Monte Carlo uncertainty" in p for p in closure.calibrated_record_problems(data, [blind])), mcse)
        low = record(grid=[{**record()["grid"][0], "observed": 0.90}])
        self.assertTrue(any("failed measurement" in p for p in closure.calibrated_record_problems(data, [low])))
        self.assertTrue(any("no coverage record" in p for p in closure.calibrated_record_problems(data, [])))
        other_level = record(nominal=0.90)
        self.assertTrue(any("no coverage record" in p for p in closure.calibrated_record_problems(data, [other_level])))
        too_small = record(n_min=500, n_max=1000)
        self.assertTrue(any("no coverage record" in p for p in closure.calibrated_record_problems(data, [too_small])))

    def test_a_boundary_record_needs_a_confirmation_covering_the_case(self) -> None:
        data = self.calibrated()
        self.assertEqual(closure.confirmation_problems(data, [record()]), [])
        boundary = record(boundary=True)
        self.assertTrue(any("no independent confirmation" in p for p in closure.confirmation_problems(data, [boundary])))
        confirmed = record(boundary=True, confirmed_passing_range=[50, 1000],
                           confirmation=[{"point": 1, "replicates": 2000}])
        self.assertEqual(closure.confirmation_problems(data, [confirmed]), [])
        misses = record(boundary=True, confirmed_passing_range=[500, 1000], confirmation=[{"replicates": 2000}])
        self.assertTrue(closure.confirmation_problems(data, [misses]))
        thin = record(boundary=True, confirmed_passing_range=[50, 1000], confirmation=[{"replicates": 400}])
        self.assertTrue(any("replicates" in p for p in closure.confirmation_problems(data, [thin])))
        bare = record(confirmed_passing_range=[50, 1000])
        self.assertTrue(any("no confirmation" in p for p in closure.confirmation_problems(data, [bare])))

    def test_every_interval_case_binds_a_record_for_its_public_test_and_the_attested_gate(self) -> None:
        data = self.calibrated()
        candidate, release = "bash scripts/gate_release.sh", (
            "gate_calibration_attestation.sh\ncheck_acceptance_manifest.py --require-bound\n")
        gate = "cargo test design\n"
        self.assertEqual(closure.construction_problems(data, [record()], candidate, release, gate), [])
        unattested = closure.construction_problems(data, [record()], candidate, "", gate)
        self.assertTrue(any("calibration attestation" in p for p in unattested))
        self.assertTrue(any("--require-bound" in p or "every manifest case" in p for p in unattested))
        self.assertTrue(any("does not run the candidate" in p or "gate_release.sh" in p
                            for p in closure.construction_problems(data, [record()], "", release, gate)))
        self.assertTrue(any("does not run design" in p
                            for p in closure.construction_problems(data, [record()], candidate, release, "")))
        unbound = record(surface_list_blob="")
        self.assertTrue(any("lacks surface_list_blob" in p
                            for p in closure.construction_problems(data, [unbound], candidate, release, gate)))
        stale = record(facets=[])
        self.assertTrue(any("lacks facets" in p for p in closure.construction_problems(data, [stale], candidate, release, gate)))

    def test_declared_calibration_point_must_match_the_actual_size_binding(self) -> None:
        target = case("declared", "calibrated_interval", "crates/c/tests/t.rs::design",
                      configuration={"participants": 100, "calibration_grid_point": 2})
        rec = record(grid=[{"point": 2, "n_min": 100, "n_max": 100, "replicates": 2000,
                            "mcse": .005, "observed": .95, "role": "gated"}])
        self.assertEqual(closure.calibrated_record_problems({"case":[target]}, [rec]), [])
        target["configuration"]["calibration_grid_point"] = 1
        self.assertTrue(any("differs from the configured" in p for p in
                            closure.calibrated_record_problems({"case":[target]}, [rec])))

    def test_donor_counts_are_explicit_case_sizes(self) -> None:
        self.assertEqual(closure.case_size(case("knn", "value", "x::y", configuration={"donors":100000})), 100000)


if __name__ == "__main__":
    unittest.main()
