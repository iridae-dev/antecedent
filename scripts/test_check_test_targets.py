"""Test-target checker: unit tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import check_test_targets as check  # noqa: E402


def crate_with(root: Path, files: dict[str, str], manifest: str) -> Path:
    crate = root / "crates" / "demo"
    (crate / "tests").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(manifest, encoding="utf-8")
    for name, text in files.items():
        (crate / "tests" / name).write_text(text, encoding="utf-8")
    return crate


MANIFEST = '[package]\nname = "demo"\nautotests = false\n\n[[test]]\nname = "suite_01"\npath = "tests/suite_01.rs"\n'


class TestTargetChecks(unittest.TestCase):
    def test_comments_literals_and_disabled_modules_do_not_register_orphans(self) -> None:
        for declaration in (
            '// #[path = "orphan.rs"] mod orphan;\n',
            'const DECOY: &str = r#"#[path = "orphan.rs"] mod orphan;"#;\n',
            '#[cfg(any())]\n#[path = "orphan.rs"] mod orphan;\n',
        ):
            with self.subTest(declaration=declaration), tempfile.TemporaryDirectory() as tmp:
                crate = crate_with(Path(tmp), {"suite_01.rs": declaration, "orphan.rs": ""}, MANIFEST)
                self.assertEqual(check.orphans_of(crate), ["demo/tests/orphan.rs"])

    def test_an_unregistered_suite_and_its_includes_are_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crate = crate_with(
                Path(tmp),
                {
                    "suite_01.rs": "",
                    "suite_02.rs": '#[path = "hidden.rs"]\nmod hidden;\n',
                    "hidden.rs": "",
                },
                MANIFEST,
            )
            self.assertEqual(
                check.orphans_of(crate), ["demo/tests/hidden.rs", "demo/tests/suite_02.rs"]
            )

    def test_a_file_included_through_another_included_file_is_compiled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crate = crate_with(
                Path(tmp),
                {
                    "suite_01.rs": '#[path = "outer.rs"]\nmod outer;\n',
                    "outer.rs": '#[path = "inner.rs"]\nmod inner;\n',
                    "inner.rs": "",
                },
                MANIFEST,
            )
            self.assertEqual(check.orphans_of(crate), [])

    def test_a_file_no_target_includes_is_an_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crate = crate_with(
                Path(tmp),
                {
                    "suite_01.rs": '#[path = "outer.rs"]\nmod outer;\n',
                    "outer.rs": "",
                    "lonely.rs": "",
                },
                MANIFEST,
            )
            self.assertEqual(check.orphans_of(crate), ["demo/tests/lonely.rs"])

    def test_an_include_that_leaves_the_tests_directory_resolves_from_the_includer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crate = crate_with(
                Path(tmp),
                {
                    "suite_01.rs": '#[path = "outer.rs"]\nmod outer;\n',
                    "outer.rs": '#[path = "../shared/fixture.rs"]\nmod f;\n',
                },
                MANIFEST,
            )
            (crate / "shared").mkdir()
            (crate / "shared" / "fixture.rs").write_text("", encoding="utf-8")
            self.assertEqual(check.orphans_of(crate), [])

    def test_a_crate_with_automatic_tests_is_not_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crate = crate_with(Path(tmp), {"lonely.rs": ""}, '[package]\nname = "demo"\n')
            self.assertEqual(check.orphans_of(crate), [])

    def test_the_repository_has_no_orphans(self) -> None:
        self.assertEqual(check.main(), 0)


if __name__ == "__main__":
    unittest.main()
