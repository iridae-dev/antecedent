"""Fault controls for the child-process test selector check."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import check_child_test_selectors as selectors


def tree(source: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix="child-selectors-"))
    (root / "crates/p/tests").mkdir(parents=True)
    (root / "crates/p/tests/t.rs").write_text(source, encoding="utf-8")
    return root


class ChildTestSelectors(unittest.TestCase):
    def test_a_module_path_name_with_a_run_assertion_passes(self):
        self.assertEqual(selectors.check(tree(selectors.GOOD)), [])

    def test_a_bare_name_is_refused(self):
        bare = selectors.GOOD.replace("module_path!()", "other()")
        problems = selectors.check(tree(bare))
        self.assertEqual(len(problems), 1)
        self.assertIn("module_path!()", problems[0])

    def test_a_hard_coded_module_path_is_refused(self):
        hard = selectors.GOOD.replace("module_path!()", '"suite::m"')
        self.assertEqual(len(selectors.check(tree(hard))), 1)

    def test_a_missing_run_assertion_is_refused(self):
        problems = selectors.check(tree(selectors.GOOD.replace("running 1 test", "ok")))
        self.assertEqual(len(problems), 1)
        self.assertIn("running 1 test", problems[0])

    def test_files_without_a_child_spawn_are_ignored(self):
        self.assertEqual(selectors.check(tree('#[ignore = "run with --exact"]\nfn f() {}')), [])

    def test_self_test_passes_and_the_repository_is_clean(self):
        self.assertEqual(selectors.self_test(), 0)
        self.assertEqual(selectors.check(selectors.REPO_ROOT), [])


if __name__ == "__main__":
    unittest.main()
