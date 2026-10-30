"""Named calibration groups dispatch exact, source-backed libtest identities."""
import re
import unittest
from pathlib import Path

from _repo import REPO_ROOT, ensure_scripts_on_path

ensure_scripts_on_path()
from test_evidence import libtest_name, rust_items, target_files


class CalibrationExactFilters(unittest.TestCase):
    def test_library_groups_name_the_exact_registered_test(self):
        text = (REPO_ROOT / "scripts/gate_calibration.sh").read_text().replace("\\\n", " ")
        helper = re.search(r"(?ms)^run_ignored\(\) \{(.*?)^\}", text).group(1)
        self.assertRegex(helper, r'--lib "\$exact" -- --ignored --exact')
        calls = re.findall(r"(?m)^run_ignored\s+(antecedent(?:-[\w]+)?)\s+([\w:]+)\s+([\w:]+)\s*$", text)
        self.assertTrue(calls)
        cache = {}
        for package, label, exact in calls:
            with self.subTest(package=package, label=label):
                lib = (REPO_ROOT / "crates" / package / "src/lib.rs").resolve()
                if package not in cache:
                    names = set()
                    for path in target_files(lib, lib.parent.parent):
                        for item in rust_items(path)[1]:
                            if item.kind == "fn" and any("test" in attr for attr in item.attrs):
                                names.add(libtest_name(path, item.name, lib))
                    cache[package] = names
                self.assertTrue(exact in cache[package], f"missing exact libtest identity: {package}::{exact}")
                self.assertTrue(exact == label or exact.endswith("::" + label))

    def test_block_length_group_has_two_exact_identities(self):
        text = (REPO_ROOT / "scripts/gate_calibration.sh").read_text().replace("\\\n", " ")
        group = next(line for line in text.splitlines() if line.startswith('check "antecedent: analysis::execute::block_length_tests::shared_block_length_sensitivity"'))
        self.assertIn("-- --ignored --exact", group)
        for suffix in ("iid_n160", "ar1_rho05_n160"):
            self.assertIn("analysis::execute::block_length_tests::shared_block_length_sensitivity_" + suffix, group)

    def test_integration_helpers_use_exact_filters(self):
        text = (REPO_ROOT / "scripts/gate_calibration.sh").read_text().replace("\\\n", " ")
        for name in ("run_static_mixture_test", "run_ignored_bayes_temporal"):
            body = re.search(r"(?ms)^" + name + r"\(\) \{(.*?)^\}", text).group(1)
            self.assertRegex(body, r'"\$filter"\s+--\s+--ignored\s+--exact')


if __name__ == "__main__":
    unittest.main()
