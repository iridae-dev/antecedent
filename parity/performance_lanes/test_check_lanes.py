#!/usr/bin/env python3
"""Unit tests for check_lanes.py: a lane control needs a frozen control and a candidate.

Run with `python3 parity/performance_lanes/test_check_lanes.py`. Each case copies the two
tables into a scratch tree, breaks one thing, and expects the checker to fail on it.
"""

from __future__ import annotations

import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path

import check_lanes

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LINKED = ("crates", "examples", "python")


def entry_block(text: str, control: str, surface: str) -> str:
    pattern = re.compile(
        r'\[\[entry_point\]\]\ncontrol = "%s"\nsurface = "%s"\n.*?(?=\n\[\[|\Z)' % (control, surface),
        re.S,
    )
    match = pattern.search(text)
    assert match, (control, surface)
    return match.group(0)


class CheckLanes(unittest.TestCase):
    def run_checker(self, lanes=lambda t: t, workloads=lambda t: t) -> str:
        """Run the checker on edited copies of the tables; return its exit message or ''."""
        with tempfile.TemporaryDirectory() as scratch:
            tree = Path(scratch)
            for name in LINKED:
                (tree / name).symlink_to(REPO / name)
            lanes_dir = tree / "parity" / "performance_lanes"
            controls_dir = tree / "parity" / "controls"
            lanes_dir.mkdir(parents=True)
            controls_dir.mkdir(parents=True)
            (lanes_dir / "entry_points").symlink_to(HERE / "entry_points")
            (lanes_dir / "lanes.toml").write_text(
                lanes((HERE / "lanes.toml").read_text(encoding="utf-8")), encoding="utf-8"
            )
            source = (REPO / "parity" / "controls" / "workloads.toml").read_text(encoding="utf-8")
            (controls_dir / "workloads.toml").write_text(workloads(source), encoding="utf-8")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    check_lanes.main(lanes_dir)
            except SystemExit as exit_:
                return str(exit_.code)
            return ""

    def test_the_registered_tables_pass(self) -> None:
        self.assertEqual(self.run_checker(), "")

    def test_a_control_without_any_entry_point_fails(self) -> None:
        def drop(text: str) -> str:
            for surface in ("python",):
                text = text.replace(entry_block(text, "survival", surface), "")
            return text

        self.assertIn("no registered entry point pair", self.run_checker(lanes=drop))

    def test_an_entry_point_without_a_candidate_command_fails(self) -> None:
        def strip(text: str) -> str:
            block = entry_block(text, "policy", "python")
            lines = [ln for ln in block.splitlines() if not ln.startswith("candidate_command")]
            return text.replace(block, "\n".join(lines))

        self.assertIn("keys must be", self.run_checker(lanes=strip))

    def test_an_entry_point_without_a_frozen_command_fails(self) -> None:
        def strip(text: str) -> str:
            block = entry_block(text, "fci", "python")
            lines = [ln for ln in block.splitlines() if not ln.startswith("frozen_command")]
            return text.replace(block, "\n".join(lines))

        self.assertIn("keys must be", self.run_checker(lanes=strip))

    def test_an_empty_candidate_command_fails(self) -> None:
        def blank(text: str) -> str:
            block = entry_block(text, "nonlinear_bayes", "python")
            return text.replace(
                block, re.sub(r'candidate_command = ".*"', 'candidate_command = ""', block)
            )

        self.assertIn("candidate_command", self.run_checker(lanes=blank))

    def test_a_missing_entry_point_file_fails(self) -> None:
        def move(text: str) -> str:
            return text.replace(
                "entry_points/survival.py\"]", "entry_points/absent_survival.py\"]"
            )

        self.assertIn("does not exist", self.run_checker(lanes=move))

    def test_a_control_still_marked_missing_in_the_workloads_table_fails(self) -> None:
        def flip(text: str) -> str:
            block = re.search(r'\[\[control\]\]\nid = "policy"\n.*?\n\n', text, re.S).group(0)
            fixed = block.replace('status = "defined"', 'status = "missing_entry_point"')
            fixed = re.sub(r'command = ".*"\n', 'command = ""\n', fixed, count=1)
            fixed = re.sub(r"candidate_command = .*\n", "", fixed)
            return text.replace(block, fixed + 'nearest = "examples/python/policy_value_evaluation.py"\n\n')

        message = self.run_checker(workloads=flip)
        self.assertNotEqual(message, "")

    def test_commands_that_differ_without_a_role_fail(self) -> None:
        def diverge(text: str) -> str:
            block = entry_block(text, "refresh", "python")
            return text.replace(
                block,
                block.replace(
                    'candidate_command = "python examples/python/analysis_workflow.py"',
                    'candidate_command = "python examples/python/propensity_weighting.py"',
                ),
            )

        self.assertIn("differ without a --role switch", self.run_checker(lanes=diverge))

    def test_roles_must_pair_control_with_candidate(self) -> None:
        def swap(text: str) -> str:
            block = entry_block(text, "survival", "python")
            return text.replace(block, block.replace("--role candidate", "--role control"))

        self.assertIn("--role control", self.run_checker(lanes=swap))

    def test_an_unregistered_bench_fails(self) -> None:
        def rename(text: str) -> str:
            block = entry_block(text, "fci", "rust")
            return text.replace(block, block.replace("--bench fci_search", "--bench fci_absent"))

        self.assertIn("not registered", self.run_checker(lanes=rename))

    def test_a_missing_scalar_direct_baseline_fails(self) -> None:
        def drop(text: str) -> str:
            return re.sub(r"\[scalar_direct\].*?(?=\n\[\[lane\]\])", "", text, flags=re.S)

        self.assertIn("scalar_direct table is missing", self.run_checker(lanes=drop))

    def test_a_switch_the_library_does_not_read_fails(self) -> None:
        def rename(text: str) -> str:
            return text.replace("ANTECEDENT_BASELINE_SCALAR_KERNELS", "ANTECEDENT_NO_SUCH_SWITCH")

        self.assertIn("does not read", self.run_checker(lanes=rename))

    def test_a_control_outside_every_lane_fails(self) -> None:
        def orphan(text: str) -> str:
            block = entry_block(text, "adjustment", "python")
            return text + "\n" + block.replace('control = "adjustment"', 'control = "neural_crossfit"') + "\n"

        self.assertIn("outside every lane", self.run_checker(lanes=orphan))


if __name__ == "__main__":
    unittest.main()
