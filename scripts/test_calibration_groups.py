"""Calibration selection/prebuild ownership checks; never run a measurement."""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import calibration_groups as groups  # noqa: E402


class SimulationBudgetTest(unittest.TestCase):
    def test_non_divisible_job_count_does_not_round_workers_up(self):
        with patch.object(groups, "CORES", 12):
            self.assertEqual(groups.threads_per_job(5), 2)
            self.assertEqual(groups.threads_per_job(20), 1)

    def test_widened_job_waits_for_enough_workers_and_returns_them(self):
        budget = groups.SimulationBudget(4)
        started, acquired = threading.Event(), threading.Event()

        def other():
            started.set()
            with budget.claim(2):
                acquired.set()
                self.assertEqual(budget.available, 2)

        with ThreadPoolExecutor(max_workers=1) as pool:
            with budget.claim(3):
                future = pool.submit(other)
                self.assertTrue(started.wait(5))
                self.assertFalse(acquired.wait(0.05))
                self.assertEqual(budget.available, 1)
            future.result(timeout=5)
        self.assertEqual(budget.available, 4)
        with self.assertRaisesRegex(RuntimeError, "failed job"):
            with budget.claim(4):
                raise RuntimeError("failed job")
        self.assertEqual(budget.available, 4)
        with self.assertRaises(ValueError):
            with budget.claim(5):
                self.fail("over-budget job was admitted")

    def test_tail_widening_never_reduces_base_workers(self):
        group = groups.Group(1, "suite: long", False, False)
        with (
            patch.object(groups, "CORES", 12),
            patch.object(groups, "last_durations", return_value={group.label: 100}),
        ):
            self.assertEqual(groups.tail_threads([(group, None)], 2, 6), {group.label: 6})


class WorkspaceSuiteOwnershipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.environment = patch.dict("os.environ", {"CARGO_TARGET_DIR": str(self.root / "target")})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        (self.root / "Cargo.toml").write_text(
            '[workspace]\nmembers=["crates/*"]\n', encoding="utf-8", newline="\n"
        )
        self.member("antecedent", "antecedent", ["facade_current", "facade_new"])
        # Cargo package names, rather than directory spelling, own the command.
        self.member("io-folder", "antecedent-io", ["nested_bayesian"])
        self.member("estimate-folder", "antecedent-estimate", ["nested_fisher", "recovery"])
        self.member("antecedent-prob", "antecedent-prob", [])
        self.addCleanup(groups._workspace_packages.cache_clear)
        self.addCleanup(groups._suite_owner.cache_clear)
        self.root_patch = patch.object(groups, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def member(self, folder: str, name: str, suites: list[str], extra: str = "") -> None:
        path = self.root / "crates" / folder
        (path / "tests").mkdir(parents=True)
        (path / "Cargo.toml").write_text(
            f'[package]\nname="{name}"\nversion="1.0.0"\n' + extra, encoding="utf-8", newline="\n"
        )
        for suite in suites:
            (path / "tests" / f"{suite}.rs").write_text(
                "// record-emitting fixture\n", encoding="utf-8", newline="\n"
            )

    def candidates(self) -> list[groups.Group]:
        return [
            groups.Group(i, label, False, True)
            for i, label in enumerate(
                [
                    "facade_current: existing",
                    "facade_new: missing",
                    "nested_bayesian: missing",
                    "nested_fisher: missing",
                    "recovery: missing",
                    "antecedent-prob: sbc",
                    "external_gate",
                ],
                1,
            )
        ]

    def test_missing_io_estimate_and_facade_records_selected_when_all_existing_current(
        self,
    ) -> None:
        records = [{"id": "current", "test": "crates/antecedent/tests/facade_current.rs::existing"}]
        facets = SimpleNamespace(
            load_records=lambda records=records: records,
            load_surface=lambda: {},
            assess=lambda surface, rows: [SimpleNamespace(stale=[], resolved=True, records=rows)],
        )
        with patch.dict(sys.modules, {"calibration_facets": facets}):
            selection = groups.select(self.candidates(), False)
        self.assertEqual(selection.owed_records, 0)
        self.assertEqual([g.index for g in selection.groups], [2, 3, 4, 5, 6, 7])
        for index in (2, 3, 4, 5):
            self.assertEqual(selection.reasons[index], "no record in the registry")
        self.assertEqual(selection.reasons[6], "pass/fail gate, no record")
        # If the missing cells disappear, current registry rows do not trigger gates.
        with patch.dict(sys.modules, {"calibration_facets": facets}):
            self.assertEqual(
                groups.select([self.candidates()[0], self.candidates()[5]], False).groups, []
            )

    def test_missing_only_preserves_stale_records_and_skips_drift_and_recordless_gates(
        self,
    ) -> None:
        records = [{"id": "stale", "test": "crates/antecedent/tests/facade_current.rs::existing"}]
        facets = SimpleNamespace(load_records=lambda records=records: records)
        ledger = self.root / "gates.toml"
        ledger.write_text(
            '[[gate]]\ngroup="antecedent-prob__sbc.p0"\npassed=1\n[[gate]]\ngroup="antecedent-prob__sbc.p1"\npassed=1\n[[gate]]\ngroup="antecedent-prob__sbc.p2"\npassed=1\n[[gate]]\ngroup="external_gate.p0"\npassed=1\n[[gate]]\ngroup="external_gate.p1"\npassed=1\n[[gate]]\ngroup="external_gate.p2"\npassed=1\n',
            encoding="utf-8",
        )
        with (
            patch.dict(sys.modules, {"calibration_facets": facets}),
            patch.object(groups, "GATE_REGISTRY", ledger),
        ):
            selection = groups.select(self.candidates(), False, missing_only=True)
        self.assertEqual([g.index for g in selection.groups], [2, 3, 4, 5])
        narrowed = groups.filter_selection(selection, ["nested_*: missing"])
        self.assertEqual([g.index for g in narrowed.groups], [3, 4])
        with self.assertRaisesRegex(SystemExit, "matches no selected group"):
            groups.filter_selection(selection, ["facade_current: existing"])

    def test_package_label_matches_sampled_transport_integration_records(self) -> None:
        record = {
            "test": "crates/antecedent-estimate/tests/counterfactual_transport_sampled.rs::sampled_mediation_covariate_shift"
        }
        self.assertTrue(
            groups.measures("antecedent-estimate", "sampled_mediation_covariate_shift", record)
        )
        self.assertFalse(groups.measures("antecedent-estimate", "sampled_latent_district", record))

    def test_legacy_parent_record_does_not_attest_default_version_two_measurement(self) -> None:
        group = groups.Group(
            1, "sample_split_parent_set_effect_calibration: parent_case", False, True
        )
        legacy = {
            "id": "legacy",
            "test": "crates/antecedent/tests/sample_split_parent_set_effect_calibration.rs::parent_case",
            "functional": "total_effect.variables_10.max_degree_4",
        }
        current = {
            **legacy,
            "id": "current",
            "functional": legacy["functional"] + ".discovery_fraction_0_75",
        }
        self.assertTrue(groups.measures(group.head, group.test_filter, legacy))
        self.assertFalse(groups.measures_current_protocol(group, legacy))
        self.assertTrue(groups.measures_current_protocol(group, current))
        ledger = self.root / "empty-gates.toml"
        ledger.write_text(
            "".join(
                f'[[gate]]\ngroup="{group.safe}.p{point}"\npassed=1\ncalibration_sha="legacy-sha"\n'
                for point in range(3)
            ),
            encoding="utf-8",
        )
        for record, expected in ((legacy, [group]), (current, [])):
            facets = SimpleNamespace(load_records=lambda record=record: [record])
            with (
                patch.dict(sys.modules, {"calibration_facets": facets}),
                patch.object(groups, "GATE_REGISTRY", ledger),
            ):
                self.assertEqual(groups.select([group], False, missing_only=True).groups, expected)

    def test_pag_trial_shift_requires_its_distinct_functional_and_record(self) -> None:
        group = groups.Group(
            684,
            "discovered_transport_calibration: discovered_pag_transport_contrast_complete_procedure_coverage",
            False,
            True,
        )
        legacy = {
            "id": "cov.legacy_v1",
            "test": "crates/antecedent/tests/discovered_transport_calibration.rs::discovered_pag_transport_contrast_complete_procedure_coverage",
            "functional": "target_interventional_contrast.v1",
            "interval_method": "percentile_bootstrap",
        }
        marker = "declared_selection_joint_pretreatment_standardization"
        current = {
            **legacy,
            "functional": "target_interventional_contrast." + marker,
            "id": "cov.new_" + marker,
        }
        self.assertFalse(groups.measures_current_protocol(group, legacy))
        self.assertTrue(groups.measures_current_protocol(group, current))
        self.assertFalse(groups.measures_current_protocol(group, {**current, "id": legacy["id"]}))

    def test_upgraded_weighted_protocols_require_new_records_despite_old_gate_success(self) -> None:
        for name, marker in (
            ("weighted_matching_interval", "matching=population_linear"),
            ("weighted_distance_matching_interval", "matching=population_linear"),
            ("weighted_stratification_interval", "stratification=regression_linear_hc3"),
            ("weighted_ipw_hajek_interval", "weighting=balanced_linear_hc3"),
            ("weighted_ipw_hajek_cluster_interval", "weighting=balanced_linear_hc3"),
        ):
            group = groups.Group(1, f"weighted_interval_calibration: {name}", False, True)
            old = {
                "id": "old",
                "test": f"crates/antecedent/tests/weighted_interval_calibration.rs::{name}",
                "functional": "all_observed.mean",
                "interval_method": "analytic_se",
            }
            new = {
                **old,
                "id": "new." + marker.partition("=")[2],
                "functional": old["functional"] + "+" + marker,
                "interval_method": (
                    "balanced_population_linear_score"
                    if name == "weighted_ipw_hajek_interval"
                    else "balanced_population_score"
                )
                if "weighting=" in marker
                else "analytic_se",
            }
            if name == "weighted_ipw_hajek_interval":
                new["functional"] += (
                    "+interval_variance=homoskedastic_gaussian_arm_linear_population"
                )
                new["se_kind"] = "population_linear_homoskedastic"
            ledger = self.root / "weighted-gates.toml"
            ledger.write_text(
                "".join(
                    f'[[gate]]\ngroup="{group.safe}.p{point}"\npassed=1\n' for point in range(3)
                ),
                encoding="utf-8",
            )
            self.assertTrue(groups.measures(group.head, group.test_filter, old))
            for records, expected in (([old], [group]), ([old, new], [])):
                with (
                    self.subTest(name=name, current=len(records) == 2),
                    patch.dict(
                        sys.modules,
                        {
                            "calibration_facets": SimpleNamespace(
                                load_records=lambda records=records: records
                            )
                        },
                    ),
                    patch.object(groups, "GATE_REGISTRY", ledger),
                ):
                    self.assertEqual(
                        groups.select([group], False, missing_only=True).groups, expected
                    )
            if "weighting=" in marker:
                self.assertFalse(
                    groups.measures_current_protocol(
                        group, {**new, "interval_method": "analytic_se"}
                    )
                )
                if name == "weighted_ipw_hajek_interval":
                    self.assertFalse(
                        groups.measures_current_protocol(
                            group, {**new, "interval_method": "balanced_population_score"}
                        )
                    )
                    self.assertFalse(
                        groups.measures_current_protocol(group, {**new, "se_kind": "homoskedastic"})
                    )
                    self.assertFalse(
                        groups.measures_current_protocol(
                            group, {**new, "functional": old["functional"] + "+" + marker}
                        )
                    )
            else:
                self.assertFalse(groups.measures_current_protocol(group, {**new, "id": old["id"]}))

    def test_population_aipw_protocol_requires_actual_model_and_corrected_point_identity(
        self,
    ) -> None:
        for name in (
            "weighted_aipw_population_linear_interval",
            "weighted_dml_aipw_population_linear_interval",
            "weighted_dr_learner_population_linear_interval",
        ):
            group = groups.Group(1, f"weighted_interval_calibration: {name}", False, True)
            record = {
                "id": "new",
                "test": f"crates/antecedent/tests/weighted_interval_calibration.rs::{name}",
                "interval_method": "population_linear_aipw_score",
                "se_kind": "population_linear_homoskedastic",
                "functional": "all_observed.mean+interval_variance=homoskedastic_gaussian_arm_linear_population+point_correction=arm_linear_projection",
            }
            self.assertTrue(groups.measures_current_protocol(group, record))
            for field, value in (
                ("interval_method", "analytic_se"),
                ("se_kind", "homoskedastic"),
                ("functional", "all_observed.mean+point_correction=arm_linear_projection"),
                (
                    "functional",
                    "all_observed.mean+interval_variance=homoskedastic_gaussian_arm_linear_population",
                ),
            ):
                self.assertFalse(groups.measures_current_protocol(group, {**record, field: value}))

    def test_weighted_split_new_method_requires_current_fraction_and_conflict_rule(self) -> None:
        name = "weighted_split_effect_interval_calibrated"
        group = groups.Group(1, "sample_split_weighted_effect_calibration: " + name, False, True)
        marker = "discovery_fraction_0_75.conflict_resolution_treatment_invariant.variance_hc3"
        old = {
            "id": "old",
            "test": "crates/antecedent/tests/sample_split_weighted_effect_calibration.rs::" + name,
            "functional": "weighted_population_total_effect",
        }
        new = {
            **old,
            "id": "new." + marker.replace(".", "_"),
            "functional": old["functional"] + "." + marker,
            "interval_method": "student_t_sandwich_effective_n",
            "se_kind": "hc3_sampling_weighted",
        }
        intermediate = {
            **new,
            "functional": new["functional"].removesuffix(".variance_hc3"),
            "se_kind": "hc1_sampling_weighted",
        }
        self.assertFalse(groups.measures_current_protocol(group, intermediate))
        self.assertFalse(
            groups.measures_current_protocol(group, {**new, "se_kind": "hc1_sampling_weighted"})
        )
        self.assertFalse(groups.measures_current_protocol(group, old))
        self.assertTrue(groups.measures_current_protocol(group, new))
        self.assertFalse(groups.measures_current_protocol(group, {**new, "id": "old"}))
        ledger = self.root / "old-weighted-split-gates.toml"
        ledger.write_text(
            "".join(f'[[gate]]\ngroup="{group.safe}.p{point}"\npassed=1\n' for point in range(3)),
            encoding="utf-8",
        )
        with (
            patch.dict(
                sys.modules, {"calibration_facets": SimpleNamespace(load_records=lambda: [old])}
            ),
            patch.object(groups, "GATE_REGISTRY", ledger),
        ):
            self.assertEqual(groups.select([group], False, missing_only=True).groups, [group])
        with patch.dict(
            sys.modules, {"calibration_facets": SimpleNamespace(load_records=lambda: [new])}
        ):
            self.assertEqual(groups.select([group], False, missing_only=True).groups, [])

    def test_actual_cluster_reference_is_a_new_protocol(self) -> None:
        for name in (
            "weighted_aipw_cluster_dml_interval",
            "weighted_dml_cluster_interval",
            "weighted_dr_learner_cluster_interval",
        ):
            group = groups.Group(1, f"weighted_interval_calibration: {name}", False, True)
            record = {
                "test": f"crates/antecedent/tests/weighted_interval_calibration.rs::{name}",
                "interval_method": "student_analytic",
                "functional": "all_observed.mean+reference=cluster_df",
            }
            self.assertTrue(groups.measures_current_protocol(group, record))
            self.assertFalse(
                groups.measures_current_protocol(
                    group, {**record, "interval_method": "analytic_se"}
                )
            )
            self.assertFalse(
                groups.measures_current_protocol(
                    group, {**record, "functional": "all_observed.mean"}
                )
            )

    def test_atc_model_protocol_requires_heteroskedastic_control_target(self) -> None:
        group = groups.Group(
            1, "weighted_interval_calibration: weighted_aipw_atc_interval", False, True
        )
        record = {
            "test": "crates/antecedent/tests/weighted_interval_calibration.rs::weighted_aipw_atc_interval",
            "interval_method": "population_linear_atc_hc3",
            "se_kind": "population_linear_heteroskedastic_hc3",
            "functional": "untreated.mean+interval_variance=heteroskedastic_arm_linear_training_hc3",
        }
        self.assertTrue(groups.measures_current_protocol(group, record))
        for field, value in (
            ("interval_method", "analytic_se"),
            ("se_kind", "homoskedastic"),
            (
                "functional",
                "all_observed.mean+interval_variance=heteroskedastic_arm_linear_training_hc3",
            ),
            ("functional", "untreated.mean"),
        ):
            self.assertFalse(groups.measures_current_protocol(group, {**record, field: value}))

    def test_gaussian_study_current_protocol_keeps_fixed_effect_and_requires_distinct_role(
        self,
    ) -> None:
        for name in (
            "pooled_random_effects_five_studies",
            "pooled_random_effects_strong_heterogeneity_five_studies",
            "pooled_random_effects_homogeneous_five_studies",
        ):
            group = groups.Group(1, f"evidence_synthesis_calibration: {name}", False, True)
            old = {
                "id": "old",
                "test": f"crates/antecedent/tests/evidence_synthesis_calibration.rs::{name}",
                "interval_method": "hksj_t",
                "role": "point",
            }
            new = {
                **old,
                "id": "new",
                "interval_method": "independent_gaussian_study_t",
                "role": "conservative_point",
            }
            self.assertFalse(groups.measures_current_protocol(group, old))
            self.assertTrue(groups.measures_current_protocol(group, new))
            self.assertFalse(groups.measures_current_protocol(group, {**new, "role": "point"}))
            self.assertFalse(
                groups.measures_current_protocol(group, {**new, "interval_method": "hksj_t"})
            )
        fixed = groups.Group(
            1, "evidence_synthesis_calibration: pooled_fixed_effect_five_studies", False, True
        )
        record = {
            "test": "crates/antecedent/tests/evidence_synthesis_calibration.rs::pooled_fixed_effect_five_studies",
            "interval_method": "analytic_se",
        }
        self.assertTrue(groups.measures_current_protocol(fixed, record))

    def test_recordless_knn_requires_protocol_marker_at_every_grid_point(self) -> None:
        group = groups.Group(
            1, "knn_matching_calibration: knn_matching_interval_ci_coverage", False, True
        )
        ledger = self.root / "knn-gates.toml"
        for markers, expected in ((0, [group]), (2, [group]), (3, [])):
            ledger.write_text(
                "".join(
                    f'[[gate]]\ngroup="{group.safe}.p{point}"\npassed=1\n'
                    + ('protocol="population_linear"\n' if point < markers else "")
                    for point in range(3)
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    sys.modules, {"calibration_facets": SimpleNamespace(load_records=lambda: [])}
                ),
                patch.object(groups, "GATE_REGISTRY", ledger),
            ):
                self.assertEqual(groups.select([group], False, missing_only=True).groups, expected)

    def test_prebuild_groups_actual_package_targets_and_retains_lib_commands(self) -> None:
        candidates = self.candidates() + [groups.Group(8, "nested_bayesian: another", False, True)]
        with patch.object(
            groups.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ) as run:
            self.assertEqual(groups.prebuild(candidates), 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            commands,
            [
                ["cargo", "test", "--release", "-p", "antecedent-prob", "--lib", "--no-run"],
                [
                    "cargo",
                    "test",
                    "--release",
                    "-p",
                    "antecedent",
                    "--test",
                    "facade_current",
                    "--test",
                    "facade_new",
                    "--no-run",
                ],
                [
                    "cargo",
                    "test",
                    "--release",
                    "-p",
                    "antecedent-estimate",
                    "--test",
                    "nested_fisher",
                    "--test",
                    "recovery",
                    "--no-run",
                ],
                [
                    "cargo",
                    "test",
                    "--release",
                    "-p",
                    "antecedent-io",
                    "--test",
                    "nested_bayesian",
                    "--no-run",
                ],
            ],
        )
        self.assertTrue(all(call.kwargs == {"cwd": self.root} for call in run.call_args_list))

    def test_explicit_test_target_and_ambiguous_owner(self) -> None:
        self.member(
            "custom",
            "custom-package",
            [],
            '\nautotests=false\n[[test]]\nname="custom_suite"\npath="oracle.rs"\n',
        )
        (self.root / "crates/custom/oracle.rs").write_text(
            "// explicit integration target\n", encoding="utf-8", newline="\n"
        )
        groups._workspace_packages.cache_clear()
        self.assertEqual(groups._suite_owner(self.root, "custom_suite"), "custom-package")
        self.member("duplicate", "duplicate-package", ["nested_bayesian"])
        groups._workspace_packages.cache_clear()
        with self.assertRaisesRegex(SystemExit, "ambiguous calibration test target"):
            groups._suite_owner(self.root, "nested_bayesian")

    def test_prebuild_failure_stops_before_later_packages(self) -> None:
        with patch.object(
            groups.subprocess, "run", return_value=SimpleNamespace(returncode=1)
        ) as run:
            self.assertEqual(groups.prebuild(self.candidates()), 1)
        self.assertEqual(run.call_count, 1)


class EvidenceArchiveTest(unittest.TestCase):
    def test_two_runs_keep_evidence_and_console_in_unique_sha_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs, consoles, archive = (root / name for name in ("records", "console", "archive"))
            sha = "a" * 40
            with (
                patch.object(groups, "LOG_DIR", logs),
                patch.object(groups, "CONSOLE_DIR", consoles),
                patch.object(groups, "ARCHIVE_DIR", archive),
            ):
                for index in range(2):
                    logs.mkdir()
                    consoles.mkdir()
                    (logs / "design.log").write_text(
                        f"result {index}\ncalibration-measured-at {sha}\n", encoding="utf-8"
                    )
                    (logs / "design.tallies.jsonl").write_text(f"tally {index}\n", encoding="utf-8")
                    (consoles / "design.txt").write_text(f"console {index}\n", encoding="utf-8")
                    groups.archive_logs()
                runs = list((archive / sha).iterdir())
                self.assertEqual(len(runs), 2)
                self.assertEqual(
                    {
                        (p / "records/design.tallies.jsonl").read_text(encoding="utf-8")
                        for p in runs
                    },
                    {"tally 0\n", "tally 1\n"},
                )
                self.assertEqual(
                    {(p / "console/design.txt").read_text(encoding="utf-8") for p in runs},
                    {"console 0\n", "console 1\n"},
                )
                self.assertFalse(logs.exists())

    def test_explicit_replicate_budget_reaches_only_selected_measurement(self) -> None:
        group = groups.Group(1, "suite: selected", False, False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(groups, "measurement_source_sha", return_value="a" * 40),
                patch.object(groups, "formal_decision_environment", return_value={}),
                patch.dict(
                    groups.os.environ,
                    {
                        "ANTECEDENT_PARENT_PROTOCOL_VERSION": "1",
                        "ANTECEDENT_WEIGHTED_SPLIT_PROTOCOL": "1",
                    },
                ),
                patch.object(groups, "FIRST_NSIM", 20000),
                patch.object(groups, "RECHECK_NSIM", 20001),
                patch.object(groups, "last_durations", return_value={}),
                patch.object(groups, "archive_logs"),
                patch.object(groups, "prebuild", return_value=0),
                patch.object(groups.subprocess, "check_output", return_value="a" * 40),
                patch.object(
                    groups.subprocess, "run", return_value=SimpleNamespace(returncode=0)
                ) as run,
                patch.object(groups, "LOG_DIR", root / "records"),
                patch.object(groups, "CONSOLE_DIR", root / "console"),
                patch.object(groups, "TIMINGS", root / "times.tsv"),
            ):
                self.assertEqual(groups.run(groups.Selection([group], 0, {1: "selected"}), 2, 1), 0)
                self.assertEqual(run.call_count, 1)
                env = run.call_args.kwargs["env"]
                self.assertEqual(env["ANTECEDENT_CALIBRATION_NSIM"], "20000")
                self.assertEqual(env["ANTECEDENT_CALIBRATION_RECHECK_NSIM"], "20001")
                self.assertEqual(env["ANTECEDENT_CALIBRATION_SHARD"], "0/2")
                self.assertEqual(env["RUST_TEST_THREADS"], "1")
                self.assertNotIn("ANTECEDENT_PARENT_PROTOCOL_VERSION", env)
                self.assertNotIn("ANTECEDENT_WEIGHTED_SPLIT_PROTOCOL", env)

    def test_run_preserves_historical_longest_first_without_current_pilot(self) -> None:
        short = groups.Group(1, "suite: short", False, False)
        long = groups.Group(2, "suite: long", False, False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(groups, "measurement_source_sha", return_value="a" * 40),
                patch.object(groups, "formal_decision_environment", return_value={}),
                patch.object(
                    groups, "last_durations", return_value={short.label: 1, long.label: 100}
                ),
                patch.object(groups, "pilot_timings", return_value={}),
                patch.object(groups, "archive_logs"),
                patch.object(groups, "prebuild", return_value=0),
                patch.object(groups.subprocess, "check_output", return_value="a" * 40),
                patch.object(groups, "LOG_DIR", root / "records"),
                patch.object(groups, "CONSOLE_DIR", root / "console"),
                patch.object(groups, "ThreadPoolExecutor") as executor,
            ):
                executor.return_value.__enter__.return_value.map.return_value = []
                selection = groups.Selection([short, long], 0, {1: "missing", 2: "missing"})
                self.assertEqual(groups.run(selection, 2, 2), 0)
                scheduled = executor.return_value.__enter__.return_value.map.call_args.args[1]
                self.assertEqual(scheduled, [(long, None), (short, None)])


class FormalDecisionEnvironmentTest(unittest.TestCase):
    def test_followup_foreign_group_stops_before_archiving_build_or_draws(self):
        import json

        import calibration_decision as decision
        import calibration_family as family

        roster = json.loads(
            family.read_source(
                groups.ROOT, family.MANIFEST_PATH, family.allocations.ORIGINAL_SOURCE
            )
        )["groups"]
        config = decision.config_for_manifest(
            {"groups": roster, "validation_domain": decision.PAG_CONFIDENCE_SET_DOMAIN}
        )
        foreign = groups.Group(1, roster[0]["label"], False, True)
        with (
            patch.object(groups, "measurement_source_sha", return_value="a" * 40),
            patch.object(family, "current_config", return_value=config),
            patch.object(groups, "archive_logs") as archive,
            patch.object(groups, "prebuild") as build,
            self.assertRaisesRegex(ValueError, "allocated group"),
        ):
            groups.run(groups.Selection([foreign], 0, {1: "selected"}), len(roster), 1)
        archive.assert_not_called()
        build.assert_not_called()

    def test_direct_run_source_binding_rejects_dirty_source(self):
        with (
            patch.object(groups.subprocess, "check_output", return_value=" M numerical.rs\n"),
            self.assertRaisesRegex(SystemExit, "clean committed source"),
        ):
            groups.measurement_source_sha()
        with patch.object(groups.subprocess, "check_output", side_effect=["", "a" * 40]):
            self.assertEqual(groups.measurement_source_sha(), "a" * 40)

    def test_complete_family_overrides_foreign_selected_config(self):
        import json

        import calibration_decision as decision
        import calibration_family as family

        config = decision.config_for_manifest({"groups": [{"label": "one"}, {"label": "two"}]})
        with (
            patch.object(family, "current_config", return_value=config) as validate,
            patch.dict(groups.os.environ, {decision.ENV: "foreign small family"}),
            patch.object(groups, "FIRST_NSIM", 400),
        ):
            result = groups.formal_decision_environment()
        validate.assert_called_once_with(groups.ROOT)
        self.assertEqual(json.loads(result[decision.ENV]), config)
        self.assertEqual(config["family_cells"], 2 * 3 * 256)

    def test_confirmation_and_unregistered_budget_fail_closed(self):
        with (
            patch.dict(groups.os.environ, {"ANTECEDENT_CALIBRATION_CONFIRMATION": "1"}),
            self.assertRaisesRegex(SystemExit, "confirmation"),
        ):
            groups.formal_decision_environment()
        with (
            patch.object(groups, "FIRST_NSIM", 401),
            self.assertRaisesRegex(SystemExit, "registered"),
        ):
            groups.formal_decision_environment()


if __name__ == "__main__":
    unittest.main()
