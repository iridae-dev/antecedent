"""Parity evidence cannot hide duplicate rows or empty exit artifacts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import parity_rows as gate
import gate_release as release


class ParityRowsIntegrity(unittest.TestCase):
    def test_duplicate_pending_and_done_rows_fail_both_lookup_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rows.toml").write_text('[[capabilities]]\nid="x"\nstatus="pending"\n[[capabilities]]\nid="x"\nstatus="done"\n', encoding="utf-8")
            (root / "evidence.txt").write_text("evidence", encoding="utf-8")
            with patch.object(gate, "ROOT", root):
                self.assertTrue(any("duplicate" in p for p in gate.evidence_map_problems({"x": "evidence.txt"}, ["rows.toml"])))
                self.assertTrue(any("duplicate" in p for p in gate.require_done("rows.toml", ["x"], "gate")))

    def test_exit_artifacts_are_nonempty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty").write_text("", encoding="utf-8")
            (root / "directory").mkdir()
            (root / "present").write_text("ok", encoding="utf-8")
            with patch.object(gate, "ROOT", root):
                self.assertEqual(len(gate.exit_artifact_problems(["empty", "directory", "present"])), 2)


class ParityLiveTestEvidence(unittest.TestCase):
    def check(self, suffix, content):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample" / "tests" / ("test_case" + suffix)
            path.parent.mkdir(parents=True)
            path.write_text(content, encoding="utf-8")
            with patch.object(gate, "ROOT", root):
                return gate._evidence_problem("case", str(path.relative_to(root)))

    def test_rust_comment_string_and_commented_assertion_are_not_evidence(self):
        for content in (
            '// #[test] fn fake() { assert!(true); }\n',
            'const S: &str = "#[test] fn fake() { assert!(true); }";\n',
            '#[test] fn fake() { /* assert!(false); */ let x = 1; }\n',
            '#[test] #[ignore] fn ignored() { assert_eq!(1, 2); }\n',
        ):
            with self.subTest(content=content):
                self.assertIsNotNone(self.check(".rs", content))
        self.assertIsNone(self.check(".rs", '#[test] fn checks() { assert_eq!(1, 2); }\n'))
        self.assertIsNone(self.check(".rs", 'fn check() { assert_eq!(1, 2); } #[test] fn checks() { check(); }\n'))

    def test_python_comments_docstrings_skips_and_commented_assertion_fail(self):
        for content in (
            '# def test_fake(): assert False\n',
            '"""def test_fake(): assert False"""\n',
            'def test_fake():\n    # assert False\n    pass\n',
            'import pytest\n@pytest.mark.skip\ndef test_fake():\n    assert 1 == 2\n',
            'def test_broken(:\n    assert False\n',
        ):
            with self.subTest(content=content):
                self.assertIsNotNone(self.check(".py", content))
        self.assertIsNone(self.check(".py", 'def test_checks():\n    assert 1 == 2\n'))
        self.assertIsNone(self.check(".py", 'def check():\n    assert 1 == 2\ndef test_checks():\n    check()\n'))


class ReleaseUnsafeInventory(unittest.TestCase):
    def test_all_three_added_crates_are_enforced_and_comments_do_not_count(self):
        crates = ("crates/antecedent-learn", "crates/antecedent-learn-burn",
                  "crates/antecedent-test-support")
        self.assertTrue(set(crates).issubset(release.FORBID_CRATES))
        with tempfile.TemporaryDirectory() as tmp, patch.object(release, "FORBID_CRATES", crates):
            root = Path(tmp)
            for crate in crates:
                lib = root / crate / "src/lib.rs"
                lib.parent.mkdir(parents=True)
                lib.write_text("#![forbid(unsafe_code)]\n", encoding="utf-8")
            self.assertEqual(release.forbid_unsafe_problems(root), [])
            for crate in crates:
                lib = root / crate / "src/lib.rs"
                lib.write_text("// #![forbid(unsafe_code)]\n", encoding="utf-8")
                problems = release.forbid_unsafe_problems(root)
                self.assertEqual(len(problems), 1)
                self.assertIn(crate, problems[0])
                lib.write_text("#![forbid(unsafe_code)]\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
