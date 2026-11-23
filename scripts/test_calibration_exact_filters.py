"""Calibration groups name exact, source-backed libtest identities."""
import unittest

from _repo import REPO_ROOT, ensure_scripts_on_path

ensure_scripts_on_path()
import calibration_runner as runner  # noqa: E402 -- local helper path is installed immediately above
from test_evidence import (  # noqa: E402
    libtest_name,
    rust_items,
    target_files,
)


class CalibrationExactFilters(unittest.TestCase):
    groups = runner.load_groups(REPO_ROOT)

    def test_library_groups_name_the_exact_registered_test(self):
        library = [g for g in self.groups if g.target == "lib"]
        self.assertTrue(library)
        cache = {}
        for group in library:
            with self.subTest(group=group.label):
                lib = (REPO_ROOT / "crates" / group.package / "src/lib.rs").resolve()
                if group.package not in cache:
                    names = set()
                    for path in target_files(lib, lib.parent.parent):
                        for item in rust_items(path)[1]:
                            if item.kind == "fn" and any("test" in attr for attr in item.attrs):
                                names.add(libtest_name(path, item.name, lib))
                    cache[group.package] = names
                for exact in group.tests:
                    self.assertIn(exact, cache[group.package], f"missing exact libtest identity: {group.package}::{exact}")

    def test_block_length_group_has_two_exact_identities(self):
        group = next(g for g in self.groups
                     if g.label == "antecedent: analysis::execute::block_length_tests::shared_block_length_sensitivity")
        self.assertEqual(group.tests, tuple(
            "analysis::execute::block_length_tests::shared_block_length_sensitivity_" + s
            for s in ("iid_n160", "ar1_rho05_n160")))

    def test_only_whole_file_groups_name_no_exact_test(self):
        whole = {g.target for g in self.groups if g.tests is None}
        self.assertEqual(whole, {"temporal_response_calibration", "panel_calibration"})


if __name__ == "__main__":
    unittest.main()
