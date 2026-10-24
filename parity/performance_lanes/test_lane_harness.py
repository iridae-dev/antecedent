#!/usr/bin/env python3
"""Tests for the lane harness: command handling, OS measurement, verdicts and records.

Run with `python3 parity/performance_lanes/test_lane_harness.py`. No cargo, no antecedent
import. The measurement tests run a real child process that allocates real memory.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import lane_verdicts as v
import lanes_common as common
import measure
import run_lanes
import stage_frozen
import verify_entry_points

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
import bench_regression  # noqa: E402

MIB = 1 << 20


def cell(ns, rss, *, digest="d", accounting=None, outcome="completed"):
    repeats = [
        {
            "outcome": "completed",
            "section_ns": n,
            "wall_ns": n + 5,
            "cpu_user_ns": n,
            "cpu_system_ns": 1,
            "peak_rss_bytes": r,
            "data_digest": digest,
            "accounting": accounting or {"replicates": 399},
        }
        for n, r in zip(ns, rss, strict=True)
    ]
    return {"outcome": outcome, "repeats": repeats if outcome == "completed" else []}


class Commands(unittest.TestCase):
    def test_env_assignments_and_kinds(self) -> None:
        command = common.parse_command(
            "USER_PATH_N=10000 cargo bench -p antecedent --bench user_path -- user_path_dag"
        )
        self.assertEqual(command.environment, {"USER_PATH_N": "10000"})
        self.assertEqual(common.command_kind(command), "cargo_bench")
        self.assertEqual(
            common.prebuild_argv(command),
            (
                "cargo",
                "bench",
                "-p",
                "antecedent",
                "--bench",
                "user_path",
                "--no-run",
                "--",
                "user_path_dag",
            ),
        )
        toolchain = common.parse_command(
            "cargo +1.85 bench -p antecedent-discovery --bench pcmci"
        )
        self.assertEqual(common.prebuild_argv(toolchain)[-1], "--no-run")
        run = common.parse_command("cargo run --release -q -p a --example e")
        self.assertEqual(common.prebuild_argv(run)[1], "build")
        self.assertEqual(
            common.command_kind(common.parse_command("python examples/python/x.py")),
            "example",
        )
        entry = common.parse_command(
            f"python {common.ENTRY_DIR}survival.py --role control"
        )
        self.assertEqual(common.command_kind(entry), "entry")
        self.assertIsNone(common.prebuild_argv(entry))

    def test_rows_are_set_by_the_mechanism_the_command_has(self) -> None:
        entry = common.parse_command(
            f"python {common.ENTRY_DIR}survival.py --role control"
        )
        self.assertEqual(common.with_rows(entry, "1e5").argv[-2:], ("--rows", "100000"))
        bench = common.parse_command(
            "USER_PATH_N=10000 cargo bench -p antecedent --bench user_path"
        )
        self.assertEqual(
            common.with_rows(bench, "1e6").environment["USER_PATH_N"], "1000000"
        )
        fixed = common.parse_command(
            "cargo bench -p antecedent-prob --bench laplace_glm"
        )
        with self.assertRaises(ValueError):
            common.with_rows(fixed, "1e5")
        self.assertIs(common.with_rows(fixed, common.DEFAULT_TIER), fixed)

    def test_every_registered_command_parses_and_the_checker_passes(self) -> None:
        lanes, workloads = common.load_tables()
        for rows in common.entries_by_control(lanes).values():
            for entry in rows:
                common.parse_command(entry["frozen_command"])
                common.parse_command(entry["candidate_command"])
        layout = stage_frozen.Layout.under(
            Path(tempfile.gettempdir()) / "no_such_lane_work"
        )
        self.assertEqual(verify_entry_points.check_static(layout, lanes, workloads), [])


class OperatingSystemMeasurement(unittest.TestCase):
    """The peak resident set comes from the kernel for a real child that touches memory."""

    def child(self, megabytes: int) -> dict:
        code = (
            "import sys\n"
            f"buf = bytearray({megabytes} * 1024 * 1024)\n"
            "for i in range(0, len(buf), 4096):\n"
            "    buf[i] = 1\n"
            "print(sum(buf[::4096]))\n"
        )
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            return measure.run_measured(
                [sys.executable, "-c", code],
                env={"PATH": "/usr/bin:/bin"},
                cwd=root,
                stdout_path=root / "out",
                stderr_path=root / "err",
                timeout=120,
            )

    def test_peak_resident_set_follows_what_the_child_allocated(self) -> None:
        small, large = self.child(8), self.child(160)
        for run in (small, large):
            self.assertEqual(run["outcome"], "completed", run["stderr_tail"])
        self.assertGreaterEqual(large["peak_rss_bytes"], 160 * MIB)
        self.assertGreaterEqual(
            large["peak_rss_bytes"], small["peak_rss_bytes"] + 120 * MIB
        )
        self.assertGreater(large["minor_faults"], small["minor_faults"])
        self.assertGreater(large["wall_ns"], 0)

    def test_failures_timeouts_and_memory_errors_are_classified_not_dropped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            args = {
                "env": {"PATH": "/usr/bin:/bin"},
                "cwd": root,
                "stdout_path": root / "o",
                "stderr_path": root / "e",
            }
            failed = measure.run_measured(
                [sys.executable, "-c", "raise SystemExit(3)"], **args
            )
            self.assertEqual(
                (failed["returncode"], failed["outcome"]), (3, "unsuccessful")
            )
            memory = measure.run_measured(
                [sys.executable, "-c", "raise MemoryError('x')"], **args
            )
            self.assertEqual(memory["outcome"], "oom")
            slow = measure.run_measured(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout=0.3,
                **args,
            )
            self.assertTrue(slow["timed_out"])
            self.assertEqual(slow["outcome"], "unsuccessful")


class Verdicts(unittest.TestCase):
    def test_median_is_floor_midpoint(self) -> None:
        self.assertEqual(v.median([10, 31, 20, 40]), 25)
        self.assertEqual(v.median([5, 1, 3]), 3)

    def test_first_answer_goal_uses_the_median_and_requires_the_same_work(self) -> None:
        frozen = cell([100, 110, 90, 105, 95], [10 * MIB] * 5)
        fast = cell([30, 40, 35, 33, 31], [9 * MIB] * 5)
        verdict = v.first_answer_verdict(frozen, fast, 2.0)
        self.assertTrue(verdict["target_met"])
        self.assertAlmostEqual(verdict["speedup"], 100 / 33)
        slow = cell([80, 90, 85, 70, 75], [9 * MIB] * 5)
        self.assertFalse(v.first_answer_verdict(frozen, slow, 2.0)["target_met"])
        omitted = cell([10] * 5, [9 * MIB] * 5, accounting={"replicates": 39})
        rejected = v.first_answer_verdict(frozen, omitted, 2.0)
        self.assertFalse(rejected["target_met"])
        self.assertIn("counts or statuses", rejected["reason"])
        changed = cell([10] * 5, [9 * MIB] * 5, digest="other")
        self.assertIn(
            "different data", v.first_answer_verdict(frozen, changed, 2.0)["reason"]
        )
        failed = cell([], [], outcome="oom")
        self.assertFalse(v.first_answer_verdict(frozen, failed, 2.0)["target_met"])

    def test_reuse_needs_both_baselines(self) -> None:
        solo = cell([1000] * 5, [MIB] * 5)
        prepared = cell([400] * 5, [MIB] * 5)
        candidate = cell([30] * 5, [MIB] * 5)
        self.assertTrue(v.reuse_verdict(solo, prepared, candidate, 10.0)["target_met"])
        slower = cell([60] * 5, [MIB] * 5)
        verdict = v.reuse_verdict(solo, prepared, slower, 10.0)
        self.assertFalse(verdict["target_met"])
        self.assertTrue(verdict["against"]["solo"]["target_met"])
        self.assertFalse(v.reuse_verdict(None, prepared, candidate, 10.0)["target_met"])

    def test_relative_change_needs_five_repeats_and_reports_the_lower_bound(
        self,
    ) -> None:
        self.assertIsNone(v.relative_change([10, 11, 12], [5, 5, 5]))
        change = v.relative_change([100, 101, 99, 100, 102], [50, 51, 49, 50, 52])
        self.assertAlmostEqual(change["point_estimate"], -0.5, places=2)
        self.assertLess(change["lower_bound"], change["point_estimate"] + 1e-9)
        self.assertLess(change["lower_bound"], 0.0)

    def test_capacity_is_a_tier_above_the_frozen_build_in_its_resources(self) -> None:
        tiers = ("1e4", "1e5", "1e6", "1e7")
        frozen = {
            "1e4": cell([10] * 5, [100 * MIB] * 5),
            "1e5": cell([100] * 5, [400 * MIB] * 5),
            "1e6": cell([], [], outcome="oom"),
        }
        candidate = {
            "1e4": cell([8] * 5, [90 * MIB] * 5),
            "1e5": cell([60] * 5, [300 * MIB] * 5),
            "1e6": cell([90] * 5, [380 * MIB] * 5),
        }
        met = v.capacity_verdict(frozen, candidate, tiers, 1)
        self.assertTrue(met["target_met"], met["reason"])
        self.assertEqual((met["frozen_tier"], met["candidate_tier"]), ("1e5", "1e6"))
        too_big = dict(candidate, **{"1e6": cell([90] * 5, [500 * MIB] * 5)})
        self.assertFalse(
            v.capacity_verdict(frozen, too_big, tiers, 1)["within_peak_rss"]
        )
        too_slow = dict(candidate, **{"1e6": cell([500] * 5, [380 * MIB] * 5)})
        self.assertFalse(v.capacity_verdict(frozen, too_slow, tiers, 1)["within_time"])
        two = v.capacity_verdict(frozen, candidate, tiers, 2)
        self.assertFalse(two["target_met"])
        everywhere = {t: cell([10] * 5, [MIB] * 5) for t in tiers}
        self.assertIn(
            "not registered",
            v.capacity_verdict(everywhere, everywhere, tiers, 1)["reason"],
        )


class Planning(unittest.TestCase):
    def config(
        self, run_dir: Path, lanes=common.LANE_IDS, threads=(1, 4)
    ) -> run_lanes.Config:
        return run_lanes.Config(
            layout=stage_frozen.Layout.under(run_dir / "work"),
            run_dir=run_dir,
            lanes=tuple(lanes),
            tiers=common.TIERS,
            threads=threads,
            repeats=5,
            timeout=None,
            hardware_class="apple_silicon",
            build="portable",
            time_factor=1.0,
        )

    def test_the_plan_covers_every_lane_control_for_both_builds(self) -> None:
        lanes, workloads = common.load_tables()
        with tempfile.TemporaryDirectory() as scratch:
            config = self.config(Path(scratch))
            membership = run_lanes.plan_cells(config, lanes, workloads)
        for lane, row in common.lane_rows(lanes).items():
            planned = {(c.control, c.arm) for c in membership[lane]}
            for control in row["controls"]:
                self.assertIn((control, "frozen"), planned, (lane, control))
                self.assertIn((control, "candidate"), planned, (lane, control))
        reuse = {
            c.arm
            for c in membership["reuse_throughput"]
            if c.control == "prepared_query_families"
        }
        self.assertIn("frozen_solo", reuse)
        survival = {
            c.tier for c in membership["row_capacity"] if c.control == "survival"
        }
        self.assertEqual(survival, set(common.TIERS))
        fixed = {c.tier for c in membership["row_capacity"] if c.control == "glm"}
        self.assertEqual(fixed, {common.DEFAULT_TIER})

    def test_invocations_pair_one_script_with_two_interpreters(self) -> None:
        lanes, workloads = common.load_tables()
        entries = common.primary_entries(lanes, workloads)
        with tempfile.TemporaryDirectory() as scratch:
            config = self.config(Path(scratch))
            frozen = run_lanes.Cell("survival", "python", "frozen", "1e5", 4)
            candidate = run_lanes.Cell("survival", "python", "candidate", "1e5", 4)
            f_argv, f_env, f_cwd, kind = run_lanes.invocation(config, entries, frozen)
            c_argv, c_env, c_cwd, _ = run_lanes.invocation(config, entries, candidate)
        self.assertEqual(kind, "entry")
        self.assertEqual(f_argv[1], c_argv[1])
        self.assertNotEqual(f_argv[0], c_argv[0])
        self.assertIn("control", f_argv)
        self.assertIn("candidate", c_argv)
        self.assertEqual(f_argv[f_argv.index("--rows") + 1], "100000")
        self.assertEqual(
            (f_env["RAYON_NUM_THREADS"], c_env["RAYON_NUM_THREADS"]), ("4", "4")
        )
        self.assertEqual(f_cwd, config.layout.frozen_root)
        self.assertEqual(c_cwd, config.layout.repo)
        solo = run_lanes.Cell(
            "prepared_query_families", "python", "frozen_solo", common.DEFAULT_TIER, 1
        )
        self.assertEqual(
            run_lanes.invocation(config, entries, solo)[0][-2:], ["--mode", "solo"]
        )

    def test_staging_lists_what_the_tag_lacks(self) -> None:
        lanes, _ = common.load_tables()
        with tempfile.TemporaryDirectory() as scratch:
            layout = stage_frozen.Layout.under(Path(scratch) / "work")
            (layout.frozen_root / "crates" / "antecedent-discovery").mkdir(parents=True)
            (
                layout.frozen_root / "crates" / "antecedent-discovery" / "Cargo.toml"
            ).write_text('[package]\nname = "antecedent-discovery"\n', encoding="utf-8")
            actions = stage_frozen.missing_in_frozen(layout, lanes)
            registered = [a for a in actions if a["action"] == "register_bench"]
            self.assertIn("fci_search", {a["bench"] for a in registered})
            for action in registered:
                if action["crate"] == "antecedent-discovery":
                    stage_frozen.apply_action(layout, action)
            text = (
                layout.frozen_root / "crates/antecedent-discovery/Cargo.toml"
            ).read_text(encoding="utf-8")
            self.assertIn('name = "fci_search"', text)
            again = stage_frozen.missing_in_frozen(layout, lanes)
            self.assertNotIn(
                "fci_search",
                {a.get("bench") for a in again if a["action"] == "register_bench"},
            )


class Records(unittest.TestCase):
    def test_a_written_record_is_read_back_by_the_regression_report(self) -> None:
        lanes, workloads = common.load_tables()
        with tempfile.TemporaryDirectory() as scratch:
            config = PlanningConfig.make(Path(scratch))
            membership = {
                "first_answer": [
                    run_lanes.Cell(
                        "survival", "python", "candidate", common.DEFAULT_TIER, 1
                    )
                ]
            }
            cells = {
                run_lanes.Cell(
                    "survival", "python", a, common.DEFAULT_TIER, 1
                ).key: cell(ns, [r] * 5)
                for a, ns, r in (
                    ("frozen", [300, 310, 290, 305, 295], 200 * MIB),
                    ("candidate", [100, 110, 90, 105, 95], 150 * MIB),
                )
            }
            summary = run_lanes.write_records(
                config, lanes, workloads, membership, cells
            )
            self.assertTrue(summary[0]["target_met"], summary)
            paths = sorted((Path(scratch) / "resource").rglob("resource.json"))
            self.assertEqual(len(paths), 1)
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            result = bench_regression.resource_result(payload)
        self.assertEqual(result["lane"], "first_answer")
        self.assertEqual(result["row_tier"], "control_default")
        self.assertEqual(result["measurement"]["peak_rss_bytes"], 150 * MIB)
        self.assertLess(result["relative"]["point_estimate"], -0.5)
        self.assertIn("peak_rss_bytes", result["measurement"]["variability"])


class PlanningConfig:
    @staticmethod
    def make(run_dir: Path) -> run_lanes.Config:
        return Planning().config(run_dir)


if __name__ == "__main__":
    unittest.main()
