"""The calibration runner: the group table, first-pass replicates, test arguments and one
(group, grid point) measurement with its recheck."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_runner as cr
from _repo import REPO_ROOT


def group(tests=("one",), grid=True, ignored=True, label="suite: one", target="suite"):
    return cr.GroupSpec(1, label, "antecedent", target, (), tests, ignored, grid)


class TableTests(unittest.TestCase):
    def test_the_table_lists_every_gate_group_in_order(self):
        groups = cr.load_groups(REPO_ROOT)
        self.assertEqual([g.index for g in groups], list(range(1, len(groups) + 1)))
        self.assertEqual(len({g.label for g in groups}), len(groups))
        whole = [g for g in groups if g.tests is None]
        self.assertTrue(whole and all(g.ignored and g.target != "lib" for g in whole))


class FirstPassTests(unittest.TestCase):
    acceptance = cr.Acceptance(frozenset({"bound", "grid", "point"}), frozenset({"grid"}),
                               frozenset({("point", 2)}))

    def nsim(self, test, point, **kw):
        return cr.first_pass_nsim(group(label=f"suite: {test}"), point, self.acceptance, first=400,
                                  acceptance_nsim=2000, smoke=kw.get("smoke", False),
                                  parent_protocol=kw.get("parent", "2"))

    def test_bound_tests_run_the_acceptance_count_at_their_bound_points(self):
        self.assertEqual(self.nsim("plain", 1), 400)
        self.assertEqual(self.nsim("bound", 1), 2000)
        self.assertEqual(self.nsim("bound", 0), 400)  # only the base point is bound
        self.assertEqual(self.nsim("grid", 0), 2000)  # bound at all three sizes
        self.assertEqual(self.nsim("point", 2), 2000)  # an explicit non-base binding
        self.assertEqual(self.nsim("point", 0), 400)
        self.assertEqual(self.nsim("bound", 1, smoke=True), 400)

    def test_a_fixed_budget_target_runs_its_budget_at_every_point(self):
        fixed = group(label="adaptive_whole_procedure_calibration: case",
                      target="adaptive_whole_procedure_calibration")
        nsim = lambda point, smoke=False: cr.first_pass_nsim(  # noqa: E731
            fixed, point, self.acceptance, first=400, acceptance_nsim=2000, smoke=smoke)
        self.assertEqual([nsim(p) for p in (0, 1, 2)], [2000, 2000, 2000])
        self.assertEqual(nsim(0, smoke=True), 400)


class ArgumentTests(unittest.TestCase):
    def test_arguments_are_those_cargo_passed_to_libtest(self):
        self.assertEqual(cr.libtest_args(group()), ["one", "--exact", "--ignored", "--nocapture", "--test-threads", "1"])
        self.assertEqual(cr.libtest_args(group(tests=None)), ["--ignored", "--nocapture", "--test-threads", "1"])
        self.assertEqual(cr.libtest_args(group(tests=None), ["a", "b"])[:3], ["a", "b", "--exact"])
        self.assertNotIn("--ignored", cr.libtest_args(group(ignored=False)))


class MeasureFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, g, **kw):
        from dataclasses import asdict
        return {"group": asdict(g), "point": 1, "stem": str(self.dir / "suite_one.p1"), "binary": "/bin/test",
                "cwd": str(self.dir), "env": {"PATH": "/bin"}, "tools": {}, "first_nsim": 400,
                "recheck_nsim": 2000, "smoke": False, "sha": "a" * 40, "decision_context": "{}", **kw}


class MeasureTests(MeasureFixture):
    def test_a_recheck_extends_the_first_pass_and_both_logs_are_stamped(self):
        calls = []
        outputs = iter([
            "calibration-recheck please\ncalibration-recheck-test t2\ntest result: ok. 2 passed; 0 failed\n",
            "test result: ok. 1 passed; 0 failed\n",
        ])

        def fake(binary, args, *, cwd, env, tools, merge_output):
            calls.append((args, env))
            return 0, next(outputs), {"src.rs": {(len(calls), 1)}}, None

        with patch("evidence_capture.instrumented_run", fake):
            result = cr.measure(self.spec(group(tests=None)))
        self.assertEqual(result["status"], 0)
        self.assertTrue(result["rechecked"])
        self.assertEqual(result["coverage"], {"src.rs": {(1, 1), (2, 1)}})
        first, recheck = calls
        self.assertEqual(first[1]["ANTECEDENT_CALIBRATION_NSIM"], "400")
        self.assertEqual(first[1]["ANTECEDENT_CALIBRATION_GRID_POINT"], "1")
        self.assertEqual(recheck[1]["ANTECEDENT_CALIBRATION_NSIM"], "2000")
        self.assertEqual(recheck[1]["ANTECEDENT_CALIBRATION_REPLICATE_START"], "400")
        self.assertEqual(recheck[1]["ANTECEDENT_CALIBRATION_PRIOR_TALLIES"],
                         str(self.dir / "suite_one.p1.tallies.jsonl"))
        self.assertEqual(recheck[0][:2], ["t2", "--exact"])  # only the test that asked
        log = (self.dir / "suite_one.p1.log").read_text(encoding="utf-8")
        self.assertTrue(log.endswith(f"calibration-measured-at {'a' * 40}\ncalibration-decision-context {{}}\n"))
        again = (self.dir / "suite_one.p1.recheck.log").read_text(encoding="utf-8")
        self.assertIn("calibration-rechecked-test t2\n", again)

    def test_a_run_with_no_passed_test_fails_and_is_not_rechecked(self):
        with patch("evidence_capture.instrumented_run",
                   return_value=(0, "calibration-recheck x\nrunning 0 tests\n", {}, None)):
            result = cr.measure(self.spec(group()))
        self.assertEqual(result["status"], 1)
        self.assertFalse(result["rechecked"])
        self.assertIn("no test ran", result["problems"][0])

    def test_a_smoke_run_is_never_rechecked(self):
        with patch("evidence_capture.instrumented_run",
                   return_value=(0, "calibration-recheck x\ntest result: ok. 1 passed\n", {}, None)) as run:
            result = cr.measure(self.spec(group(), smoke=True))
        self.assertEqual((result["status"], result["rechecked"], run.call_count), (0, False, 1))


class ReleaseMeasureTests(MeasureFixture):
    def test_replicates_run_on_the_release_binary_and_a_slice_gives_the_coverage(self):
        runs = []

        class Done:
            def __init__(self, stdout):
                self.returncode, self.stdout = 0, stdout

        def plain(argv, **kw):
            runs.append((argv[0], kw["env"]))
            return Done("calibration-recheck x\ntest result: ok. 1 passed; 0 failed\n" if len(runs) == 1
                        else "test result: ok. 1 passed; 0 failed\n")

        def instrumented(binary, args, *, cwd, env, tools, merge_output, failed_runs_count):
            self.assertTrue(failed_runs_count)
            runs.append((binary, env))
            return 0, "test result: ok. 1 passed; 0 failed\n", {"src.rs": {(7, 1)}}, None

        with patch("subprocess.run", plain), patch("evidence_capture.instrumented_run", instrumented):
            result = cr.measure(self.spec(group(), release_binary="/bin/release",
                                          env={"PATH": "/bin", "ANTECEDENT_CALIBRATION_THREADS": "4"}))
        self.assertEqual((result["status"], result["rechecked"]), (0, True))
        (first, a), (recheck, b), (sliced, c) = runs
        self.assertEqual((first, recheck, sliced), ("/bin/release", "/bin/release", "/bin/test"))
        self.assertEqual((a["ANTECEDENT_CALIBRATION_NSIM"], b["ANTECEDENT_CALIBRATION_NSIM"]), ("400", "2000"))
        self.assertEqual(a["ANTECEDENT_CALIBRATION_THREADS"], "4")
        # The slice is a smoke run of a few replicates on one thread, writing no tallies.
        self.assertEqual((c["ANTECEDENT_CALIBRATION_NSIM"], c["ANTECEDENT_CALIBRATION_SMOKE"],
                          c["ANTECEDENT_CALIBRATION_THREADS"]), (str(cr.SLICE_NSIM), "1", "1"))
        self.assertNotIn("ANTECEDENT_CALIBRATION_TALLY_OUT", c)
        self.assertEqual(c["ANTECEDENT_CALIBRATION_GRID_POINT"], "1")
        self.assertEqual(result["coverage"], {"src.rs": {(7, 1)}})

    def test_an_unreadable_slice_fails_the_measurement(self):
        class Done:
            returncode, stdout = 0, "test result: ok. 1 passed; 0 failed\n"
        with patch("subprocess.run", return_value=Done()), \
                patch("evidence_capture.instrumented_run", return_value=(0, "", None, "no profile")):
            result = cr.measure(self.spec(group(), release_binary="/bin/release"))
        self.assertEqual(result["status"], 1)
        self.assertIn("coverage slice", result["problems"][0])


    def test_a_slice_whose_assertion_fails_still_gives_the_coverage(self):
        class Done:
            returncode, stdout = 0, "test result: ok. 1 passed; 0 failed\n"
        sliced = (101, "test t ... its own output\nFAILED\ntest result: FAILED. 0 passed; 1 failed\n",
                  {"src.rs": {(3, 1)}}, None)
        with patch("subprocess.run", return_value=Done()), \
                patch("evidence_capture.instrumented_run", return_value=sliced):
            result = cr.measure(self.spec(group(), release_binary="/bin/release"))
        self.assertEqual((result["status"], result["problems"]), (0, []))
        self.assertEqual(result["coverage"], {"src.rs": {(3, 1)}})


class LogLineTests(unittest.TestCase):
    def test_a_calibration_line_joined_to_the_progress_prefix_starts_its_own_line(self):
        joined = "test t ... calibration-decision {}\nok\ntest u ... ok\n"
        self.assertEqual(cr.log_lines(joined), "test t ... \ncalibration-decision {}\nok\ntest u ... ok\n")


class RecordBindingTests(unittest.TestCase):
    def test_records_map_to_the_stems_that_emitted_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "g.p0.log").write_text('calibration-record {"id": "r1"}\n', encoding="utf-8")
            (d / "g.p0.recheck.log").write_text('calibration-record {"id": "r1"}\n', encoding="utf-8")
            (d / "g.p1.log").write_text('calibration-record {"id": "r1"}\ncalibration-record {"id": "r2"}\n', encoding="utf-8")
            self.assertEqual(cr.record_runs(d), {"r1": {"g.p0", "g.p1"}, "r2": {"g.p1"}})

    def test_executed_regions_are_reported_at_checkout_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, root, store = Path(tmp) / "logs", Path(tmp).resolve() / "root", Path(tmp).resolve() / "store"
            for base in (root, store):
                (base / "crates/a/src").mkdir(parents=True)
                (base / "crates/a/src/lib.rs").write_text("fn f() {}\n", encoding="utf-8")
            d.mkdir()
            group = cr.GroupSpec(index=1, label="l", package="a", target="lib", features=(), tests=None,
                                 ignored=True, grid=True)
            (d / f"{group.safe}.p0.coverage.json").write_text('{"/antecedent-source/crates/a/src/lib.rs": [[1, 1]]}', encoding="utf-8")
            (d / cr.BUILDS).write_text(json.dumps({"a|lib|": {"source": {"store_path": str(store)}}}), encoding="utf-8")
            with patch.object(cr, "load_groups", return_value=[group]):
                built, regions = cr.run_evidence(root, d, {f"{group.safe}.p0"})
            self.assertEqual(regions, {str(root / "crates/a/src/lib.rs"): {(1, 1)}})
            self.assertEqual(built["source"]["store_path"], str(store))


if __name__ == "__main__":
    unittest.main()
