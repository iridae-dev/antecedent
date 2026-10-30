"""Retained-route citations name actual non-ignored Rust test declarations."""
import tempfile
import unittest
from pathlib import Path

from gate_retained_routes import _test_citation_problems


class RetainedCitations(unittest.TestCase):
    def test_comment_helper_ignored_missing_and_escape_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "fixture.rs"
            path.write_text("// fn commented() {}\nfn helper() {}\n#[test] #[ignore] fn ignored() {}\n#[test] fn real_test() {}\n", encoding="utf-8")
            for citation in ("fixture.rs", "fixture.rs::commented", "fixture.rs::helper", "fixture.rs::ignored", "../outside.rs::real_test"):
                with self.subTest(citation=citation):
                    self.assertTrue(_test_citation_problems(root, {"id": "route", "retained_route_test": citation}))
            self.assertEqual(_test_citation_problems(root, {"id": "route", "retained_route_test": "fixture.rs::real_test"}), [])


if __name__ == "__main__":
    unittest.main()
