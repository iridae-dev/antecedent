#!/usr/bin/env python3
"""Tests for item-level review bindings: an edit to an unreviewed item never stales a review."""

import tempfile
import unittest
from pathlib import Path

from review_items import artifact_digest, artifact_problem

RUST = """
const TOL: f64 = 1e-9;

pub struct Kernel { width: f64 }

impl Kernel {
    pub fn new(width: f64) -> Self { Kernel { width } }
    pub fn eval(&self, x: f64) -> f64 { (x / self.width).exp() }
}

pub fn solve(x: f64) -> f64 {
    // note
    x * TOL
}

pub fn unrelated() -> u32 { 1 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn solves() { assert_eq!(solve(1.0), TOL); }
    #[test]
    fn other() { assert_eq!(unrelated(), 1); }
}
"""

PYTHON = """
LIMIT = 3

def reviewed(x):
    return x + LIMIT

def elsewhere(x):
    return x
"""


class ItemBinding(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def artifact(self, name: str, items, role="implementation") -> dict:
        artifact = {"role": role, "path": name, "sha256": ""}
        if items is not None:
            artifact["items"] = items
        artifact["sha256"] = artifact_digest(self.root, artifact)[0] or ""
        return artifact

    def test_rust_edit_outside_the_items_keeps_the_binding(self):
        self.write("a.rs", RUST)
        bound = self.artifact("a.rs", ["fn solve", "const TOL", "impl Kernel"])
        self.assertIsNone(artifact_problem(self.root, bound, "review"))
        self.write(
            "a.rs",
            RUST.replace("pub fn unrelated() -> u32 { 1 }", "pub fn unrelated() -> u32 { 2 }")
            .replace("assert_eq!(unrelated(), 1)", "assert_eq!(unrelated(), 2)"),
        )
        self.assertIsNone(artifact_problem(self.root, bound, "review"))

    def test_rust_formatting_and_comments_keep_the_binding(self):
        self.write("a.rs", RUST)
        bound = self.artifact("a.rs", ["fn solve"])
        self.write("a.rs", RUST.replace("// note", "// reworded note\n").replace("x * TOL", "x  *  TOL"))
        self.assertIsNone(artifact_problem(self.root, bound, "review"))

    def test_rust_edit_inside_an_item_stales_the_binding(self):
        self.write("a.rs", RUST)
        for items, old, new in (
            (["fn solve"], "x * TOL", "x + TOL"),
            (["const TOL"], "1e-9", "1e-8"),
            (["impl Kernel"], "x / self.width", "x * self.width"),
            (["module"], "pub fn unrelated() -> u32 { 1 }", "pub fn unrelated() -> u32 { 5 }"),
        ):
            self.write("a.rs", RUST)
            bound = self.artifact("a.rs", items)
            self.write("a.rs", RUST.replace(old, new))
            self.assertIn("stale artifact", artifact_problem(self.root, bound, "review") or "", items)

    def test_rust_test_functions_bind_only_the_named_tests(self):
        self.write("t.rs", RUST)
        bound = self.artifact("t.rs", ["fn solves"], role="assertion")
        self.write("t.rs", RUST.replace("assert_eq!(unrelated(), 1)", "assert_eq!(unrelated(), 9)"))
        self.assertIsNone(artifact_problem(self.root, bound, "review"))
        self.write("t.rs", RUST.replace("assert_eq!(solve(1.0), TOL)", "assert_eq!(solve(2.0), TOL)"))
        self.assertIn("stale artifact", artifact_problem(self.root, bound, "review") or "")

    def test_python_functions_bind_by_ast(self):
        self.write("p.py", PYTHON)
        bound = self.artifact("p.py", ["fn reviewed", "const LIMIT"], role="assertion")
        self.write("p.py", PYTHON.replace("return x\n", "return x * 2  # edited\n"))
        self.assertIsNone(artifact_problem(self.root, bound, "review"))
        self.write("p.py", PYTHON.replace("x + LIMIT", "x - LIMIT"))
        self.assertIn("stale artifact", artifact_problem(self.root, bound, "review") or "")

    def test_a_missing_item_is_a_finding(self):
        self.write("a.rs", RUST)
        bound = self.artifact("a.rs", ["fn solve"])
        bound["items"] = ["fn renamed_away"]
        self.assertIn("item not found", artifact_problem(self.root, bound, "review") or "")

    def test_whole_source_binding_is_only_for_reference_data(self):
        self.write("a.rs", RUST)
        self.write("gen.py", PYTHON)
        self.write("data.json", "{}")
        whole = {"role": "implementation", "path": "a.rs", "sha256": ""}
        self.assertIn("name the reviewed items", artifact_problem(self.root, whole, "review") or "")
        for name, role in (("gen.py", "generator"), ("data.json", "fixture")):
            artifact = self.artifact(name, None, role=role)
            self.assertIsNone(artifact_problem(self.root, artifact, "review"))
        data_with_items = {"role": "fixture", "path": "data.json", "items": ["fn x"], "sha256": ""}
        self.assertIn("data file binds whole", artifact_problem(self.root, data_with_items, "review") or "")


if __name__ == "__main__":
    unittest.main()
