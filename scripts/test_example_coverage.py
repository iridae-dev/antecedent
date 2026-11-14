"""Every example-catalog entry is executed, compiled only, or skipped for a stated reason."""

from __future__ import annotations

import unittest

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import example_coverage as cov  # noqa: E402

SUITE = """
#[path = "../../../examples/rust/called.rs"]
mod called;

#[allow(dead_code)]
#[path = "../../../examples/rust/included_only.rs"]
mod included_only;

#[test]
fn example_called_runs() {
    called::run().unwrap();
}
"""
MANIFEST = """
[[test]]
name = "not_an_example"
path = "../../examples/rust/a_test.rs"

[[example]]
name = "called"
path = "../../examples/rust/called.rs"

[[example]]
name = "built"
required-features = ["x"]
path = "../../examples/rust/built.rs"
"""


class SelectionParsingTest(unittest.TestCase):
    def test_rust_suite_runs_only_the_examples_whose_run_it_calls(self) -> None:
        self.assertEqual(cov.rust_suite_runs(SUITE), {"examples/rust/called.rs"})

    def test_cargo_example_targets_are_the_example_tables_only(self) -> None:
        self.assertEqual(
            cov.cargo_example_targets(MANIFEST),
            {"examples/rust/called.rs", "examples/rust/built.rs"},
        )


class AccountingTest(unittest.TestCase):
    def rows(self, entry: dict, claims: dict) -> dict[str, dict]:
        return {row["kind"]: row for row in cov.account([{"id": "e", **entry}], claims)}

    def test_the_strongest_runner_decides_the_disposition(self) -> None:
        claims = {
            "a.py": [(cov.CONDITIONAL, "suite", "needs scipy"), (cov.EXECUTED, "docs", "")],
            "a.rs": [(cov.COMPILED, "Cargo.toml", "")],
            "a.ipynb": [(cov.CONDITIONAL, "suite", "needs pandas")],
        }
        rows = self.rows({"python": "a.py", "rust": "a.rs", "notebook": "a.ipynb"}, claims)
        self.assertEqual(rows["python"]["disposition"], cov.EXECUTED)
        self.assertEqual(rows["rust"]["disposition"], cov.COMPILED)
        self.assertEqual(rows["notebook"]["disposition"], cov.CONDITIONAL)
        self.assertEqual(cov.problems(list(rows.values())), [])
        self.assertIn("e [rust] compiled only: Cargo.toml", cov.render(list(rows.values())))

    def test_an_artifact_no_runner_selects_fails_unless_the_catalog_says_why(self) -> None:
        rows = self.rows({"python": "a.py"}, {})
        self.assertEqual(rows["python"]["disposition"], cov.NONE)
        self.assertEqual(len(cov.problems(list(rows.values()))), 1)
        stated = self.rows({"python": "a.py", "unexecuted": {"python": "needs a GPU"}}, {})
        self.assertEqual(stated["python"]["disposition"], "skipped")
        self.assertEqual(cov.problems(list(stated.values())), [])
        self.assertIn("skipped: needs a GPU", cov.render(list(stated.values())))

    def test_a_reason_left_on_an_artifact_that_now_runs_is_stale(self) -> None:
        rows = self.rows(
            {"python": "a.py", "unexecuted": {"python": "was slow"}},
            {"a.py": [(cov.EXECUTED, "suite", "")]},
        )
        self.assertTrue(any("stale" in problem for problem in cov.problems(list(rows.values()))))


class CatalogTest(unittest.TestCase):
    def test_every_catalog_artifact_has_a_disposition(self) -> None:
        rows = cov.current()
        self.assertEqual(cov.problems(rows), [])
        # The accounting covers the whole catalog, one row per declared artifact.
        entries = cov.load_toml(cov.ROOT / "examples/catalog.toml")["example"]
        declared = [entry[kind] for entry in entries for kind in cov.KINDS if kind in entry]
        self.assertEqual([row["path"] for row in rows], declared)
        self.assertTrue(all(row["runners"] or row["reason"] for row in rows))


if __name__ == "__main__":
    unittest.main()
