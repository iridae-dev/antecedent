"""Fault controls for the ignored-test classification and the slow lane."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import check_ignored_tests as lanes

SOURCE = """
#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "calibration: final measurement only"]
    fn measured() { assert!(true); }

    #[test]
    #[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
    fn write_golden() { assert!(true); }

    #[test]
    #[ignore = "timing harness"]
    fn timing_probe() { assert!(true); }

    #[test]
    #[ignore = "exhaustive sweep; minutes in a debug build"]
    fn sweep() { assert_eq!(1, 1); }

    #[test]
    fn ordinary() { assert_eq!(1, 1); }
}
"""
REGISTRY = """
[[slow]]
path = "crates/p/src/lib.rs"
name = "sweep"
reason = "exhaustive"
expected_seconds = 60
"""


def tree(source: str = SOURCE, registry: str = REGISTRY) -> Path:
    root = Path(tempfile.mkdtemp(prefix="ignored-lanes-"))
    crate = root / "crates/p"
    (crate / "src").mkdir(parents=True)
    (root / "parity").mkdir()
    (crate / "Cargo.toml").write_text('[package]\nname="p"\nversion="0.1.0"\n', encoding="utf-8")
    (crate / "src/lib.rs").write_text(source, encoding="utf-8")
    (root / lanes.REGISTRY).write_text(registry, encoding="utf-8")
    return root


class IgnoredTestLanes(unittest.TestCase):
    def test_every_ignored_test_lands_in_exactly_one_class(self):
        ignored, problems = lanes.classify(tree())
        self.assertEqual(problems, [])
        self.assertEqual(
            {t["name"]: t["class"] for t in ignored},
            {"measured": "calibration", "write_golden": "writer", "timing_probe": "timing", "sweep": "slow"},
        )

    def test_a_new_ignored_test_in_no_class_fails(self):
        extra = SOURCE.replace("fn ordinary()", '#[ignore = "slow for now"]\n    fn newcomer() {}\n    #[test]\n    fn ordinary()')
        _, problems = lanes.classify(tree(extra))
        self.assertEqual(len(problems), 1)
        self.assertIn("newcomer", problems[0])
        self.assertIn("in no class", problems[0])

    def test_an_ignore_without_a_reason_fails(self):
        bare = SOURCE.replace('#[ignore = "timing harness"]', "#[ignore]")
        _, problems = lanes.classify(tree(bare))
        self.assertEqual(len(problems), 1)
        self.assertIn("needs a written reason", problems[0])

    def test_a_registered_test_that_is_no_longer_ignored_fails(self):
        live = SOURCE.replace('#[ignore = "exhaustive sweep; minutes in a debug build"]', "")
        _, problems = lanes.classify(tree(live))
        self.assertEqual(len(problems), 1)
        self.assertIn("no longer #[ignore]d", problems[0])

    def test_a_registered_test_that_no_longer_exists_fails(self):
        gone = SOURCE.replace("fn sweep()", "fn renamed_sweep()")
        _, problems = lanes.classify(tree(gone))
        self.assertTrue(any("sweep" in p and "no such test" in p for p in problems), problems)
        # the renamed test is also unclassified, so the rename cannot hide
        self.assertTrue(any("renamed_sweep" in p and "in no class" in p for p in problems), problems)

    def test_a_registered_test_in_a_missing_file_fails(self):
        registry = REGISTRY.replace("src/lib.rs", "src/missing.rs")
        _, problems = lanes.classify(tree(registry=registry))
        self.assertTrue(any("file does not exist" in p for p in problems), problems)

    def test_a_registry_row_cannot_shadow_a_rule_or_repeat_or_omit_a_field(self):
        shadow = REGISTRY + '\n[[other]]\npath = "crates/p/src/lib.rs"\nname = "timing_probe"\nreason = "x"\n'
        self.assertTrue(any("already classifies it" in p for p in lanes.classify(tree(registry=shadow))[1]))
        twice = REGISTRY + REGISTRY
        self.assertTrue(any("listed twice" in p for p in lanes.classify(tree(registry=twice))[1]))
        for broken in (REGISTRY.replace("expected_seconds = 60", ""), REGISTRY.replace("expected_seconds = 60", "expected_seconds = 0"),
                       REGISTRY.replace('reason = "exhaustive"', 'reason = ""')):
            self.assertTrue(lanes.classify(tree(registry=broken))[1], broken)

    def test_slow_selection_is_the_exact_libtest_identity(self):
        root = tree()
        _, problems = lanes.classify(root)
        self.assertEqual(problems, [])
        rows = lanes.slow_rows(root)
        self.assertEqual([(r["crate"], r["cargo_target"], r["libtest_name"]) for r in rows], [("p", "lib", "tests::sweep")])
        self.assertEqual(lanes.nextest_filter(rows), "(package(=p) and kind(lib) and test(=tests::sweep))")

    def test_listing_must_show_registered_tests_ignored_and_no_unseen_ignored_tests(self):
        root = tree()
        ignored, _ = lanes.classify(root)
        case = {"ignored": True}
        good = {"rust-suites": {"p": {"testcases": {n: case for n in (
            "tests::measured", "tests::write_golden", "tests::timing_probe", "tests::sweep")}}}}
        self.assertEqual(lanes.listing_problems(root, good, ignored), [])
        missing = {"rust-suites": {"p": {"testcases": {"tests::sweep": {"ignored": False}}}}}
        self.assertTrue(any("not a selected ignored test" in p for p in lanes.listing_problems(root, missing, ignored)))
        deselected = {"rust-suites": {"p": {"testcases": {**good["rust-suites"]["p"]["testcases"],
                      "tests::sweep": {"ignored": True, "filter-match": {"status": "mismatch"}}}}}}
        self.assertTrue(any("not a selected" in p for p in lanes.listing_problems(root, deselected, ignored)))
        unseen = {"rust-suites": {"p": {"testcases": {**good["rust-suites"]["p"]["testcases"], "tests::macro_made": case}}}}
        self.assertTrue(any("macro_made" in p for p in lanes.listing_problems(root, unseen, ignored)))

    def test_the_committed_registry_classifies_every_ignored_test(self):
        ignored, problems = lanes.classify(lanes.REPO_ROOT)
        self.assertEqual(problems, [])
        self.assertGreater(len(lanes.slow_rows(lanes.REPO_ROOT)), 0)
        self.assertTrue(all(t["class"] for t in ignored))


if __name__ == "__main__":
    unittest.main()
