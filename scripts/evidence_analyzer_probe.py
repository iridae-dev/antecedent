#!/usr/bin/env python3
"""Check item-level evidence precision with the real compiler driver on a disposable fixture.

Builds `tools/evidence-analyzer/driver.rs` with the pinned compiler (it needs the
`rustc-dev` component), compiles a small two-module workspace through it with
coverage instrumentation, runs one test and selects its items exactly as earning
does. It then applies each edit of a matrix to a copy and checks the selection
without rebuilding: an edit to code the test does not depend on must leave it
current, and an edit to code it does must owe it. Runs no Antecedent test and
writes no evidence.

    python3 scripts/evidence_analyzer_probe.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import evidence_items as ei
import evidence_manifests as em
from _repo import load_toml
from scientific_records import region_coverage

ROOT = Path(__file__).resolve().parents[1]
A = "crates/alpha/src/"
FILES = {
    "Cargo.toml": '[workspace]\nmembers = ["crates/alpha"]\nresolver = "2"\n',
    "crates/alpha/Cargo.toml": '[package]\nname = "alpha"\nversion = "0.1.0"\nedition = "2021"\n',
    A + "lib.rs": """//! Fixture.
#![allow(dead_code)]
pub mod est;
pub use est::*;

pub const SCALE: f64 = scale_of(2.0);
const fn scale_of(x: f64) -> f64 { x * 2.0 }

#[macro_export]
macro_rules! twice { ($e:expr) => { $e + $e } }

pub trait Estimator {
    const NAME: &'static str;
    type Out;
    fn fit(&self, data: &[f64]) -> Self::Out;
    fn label(&self) -> String { format!("{}", Self::NAME) }
}

pub fn run<E: Estimator>(e: &E, data: &[f64]) -> (E::Out, &'static str) { (e.fit(data), E::NAME) }

mod inner {
    pub fn helper(x: f64) -> f64 { x + 1.0 }
    pub fn called_not_run(x: f64) -> f64 { x }
    pub mod deeper { pub fn where_am_i() -> u32 { line!() } }
}
pub use inner::{called_not_run, helper};
""",
    A + "est/mod.rs": "mod mean;\nmod median;\npub use mean::Mean;\npub use median::Median;\n",
    A + "est/mean.rs": """use crate::Estimator;

#[derive(Debug, Clone, Default)]
pub struct Mean { pub bias: f64 }

#[derive(Debug, PartialEq)]
pub enum Kind { Plain, Weighted(f64) }

impl Mean {
    pub fn new() -> Self { Mean { bias: crate::SCALE } }
    fn unused(&self) -> f64 { self.bias * 3.0 }
}

impl Estimator for Mean {
    const NAME: &'static str = "mean";
    type Out = f64;
    fn fit(&self, data: &[f64]) -> f64 {
        let k = Kind::Plain;
        let s: f64 = data.iter().sum();
        let m = crate::twice!(s) / (2.0 * data.len() as f64);
        if k == Kind::Plain { m + self.bias + crate::helper(0.0) } else { crate::called_not_run(0.0) }
    }
}
""",
    A + "est/median.rs": """use crate::Estimator;
pub struct Median;
impl Estimator for Median {
    const NAME: &'static str = "median";
    type Out = f64;
    fn fit(&self, data: &[f64]) -> f64 { let mut v = data.to_vec(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
}
""",
    "crates/alpha/tests/accept.rs": """use alpha::{run, Mean, Estimator};
#[test]
fn mean_runs() {
    let m = Mean::new();
    let (v, name) = run(&m, &[1.0, 3.0]);
    assert_eq!(name, "mean");
    assert!(v > 0.0);
    assert_eq!(m.label(), "mean");
}
""",
}
NEW_ESTIMATOR = ("use crate::Estimator;\npub struct Mode;\nimpl Estimator for Mode { const NAME: &'static str = "
                 "\"mode\"; type Out = f64; fn fit(&self, d: &[f64]) -> f64 { d[0] } }\n")
# (expected, label, [(file, old, new)]); a file with old None is created.
CASES = [
    ("current", "new estimator file, mod and use", [(A + "est/mod.rs", "mod median;", "mod median;\nmod mode;"),
     (A + "est/mod.rs", "pub use median::Median;", "pub use median::Median;\npub use mode::Mode;"),
     (A + "est/mode.rs", None, NEW_ESTIMATOR)]),
    ("current", "unexecuted method body", [(A + "est/mean.rs", "self.bias * 3.0", "self.bias * 4.0")]),
    ("current", "called function's body that never ran", [(A + "lib.rs", "-> f64 { x }", "-> f64 { x * 2.0 }")]),
    ("current", "new unrelated enum", [(A + "est/mean.rs", "impl Mean {", "pub enum Spare { A, B }\n\nimpl Mean {")]),
    ("current", "new unrelated use", [(A + "est/median.rs", "pub struct Median;", "use std::fmt::Write as _;\npub struct Median;")]),
    ("current", "comment and layout in the executed body", [(A + "est/mean.rs", "let s: f64 = data.iter().sum();",
                                                              "// sum it\n        let s: f64 =   data.iter().sum();")]),
    ("current", "unexecuted position read", [(A + "lib.rs", "line!()", "line!() + 1")]),
    ("current", "new unrelated inherent method", [(A + "est/mean.rs", "    fn unused(", "    pub fn extra(&self) -> f64 { 1.0 }\n    fn unused(")]),
    ("current", "new impl of the trait for another type", [(A + "est/median.rs", "impl Estimator for Median",
     "pub struct Other;\nimpl Estimator for Other { const NAME: &'static str = \"o\"; type Out = f64; "
     "fn fit(&self, _: &[f64]) -> f64 { 0.0 } }\nimpl Estimator for Median")]),
    ("owed", "executed body", [(A + "est/mean.rs", "(2.0 * data.len() as f64)", "(3.0 * data.len() as f64)")]),
    ("owed", "called function's signature", [(A + "lib.rs", "called_not_run(x: f64) -> f64", "called_not_run(x: f64) -> f32")]),
    ("owed", "used enum gains a variant", [(A + "est/mean.rs", "Weighted(f64) }", "Weighted(f64), Extra }")]),
    ("owed", "used constant", [(A + "lib.rs", "scale_of(2.0)", "scale_of(3.0)")]),
    ("owed", "const fn evaluated at compile time", [(A + "lib.rs", "x * 2.0 }", "x * 2.5 }")]),
    ("owed", "used macro", [(A + "lib.rs", "$e + $e", "$e * 2.0")]),
    ("owed", "inherent method shadowing a trait method", [(A + "est/mean.rs", "    fn unused(",
                                                           "    pub fn label(&self) -> String { String::new() }\n    fn unused(")]),
    ("owed", "derive removed from a used struct", [(A + "est/mean.rs", "#[derive(Debug, Clone, Default)]", "#[derive(Debug, Clone)]")]),
    ("owed", "root item shadowing a glob import", [(A + "lib.rs", "pub use inner::{", "pub struct Mean;\npub use inner::{")]),
    ("owed", "module declaration compiled out", [(A + "lib.rs", "pub mod est;", "#[cfg(any())]\npub mod est;")]),
    ("owed", "trait gains a default method", [(A + "lib.rs", "    fn label(&self)", "    fn extra(&self) -> u8 { 0 }\n    fn label(&self)")]),
    ("owed", "crate attribute", [(A + "lib.rs", "#![allow(dead_code)]", "#![allow(dead_code, unused)]")]),
    ("owed", "impl constant read through a generic", [(A + "est/mean.rs", '"mean";', '"Mean";')]),
]


def _rustc(*args: str, env: dict | None = None) -> str:
    return subprocess.run(["rustc", *args], env=env, check=True, capture_output=True, text=True).stdout


def earn(root: Path, driver: Path, tools: Path) -> list[dict]:
    """Build the fixture through the driver, run its test, and select its items."""
    env = {**os.environ, "RUSTFLAGS": "-C instrument-coverage", "CARGO_TARGET_DIR": str(root / "target"),
           "RUSTC_WORKSPACE_WRAPPER": str(driver), "ANTECEDENT_ANALYZER_OUTPUT": str(root / "inventory")}
    (root / "inventory").mkdir()
    built = subprocess.run(["cargo", "test", "--no-run", "--offline", "--message-format=json"], cwd=root, env=env,
                           check=True, capture_output=True, text=True).stdout
    binary = next(m["executable"] for m in map(json.loads, built.splitlines())
                  if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "accept")
    raw, data = root / "t.profraw", root / "t.profdata"
    subprocess.run([binary, "--exact", "mean_runs"], env={**os.environ, "LLVM_PROFILE_FILE": str(raw)},
                   check=True, capture_output=True)
    subprocess.run([str(tools / "llvm-profdata"), "merge", "-sparse", str(raw), "-o", str(data)], check=True)
    export = json.loads(subprocess.check_output([str(tools / "llvm-cov"), "export", f"-instr-profile={data}", binary]))
    selection = ei.Selection(root, ei.load_inventory(root / "inventory"))
    for path, positions in region_coverage(export).items():
        try:
            rel = Path(path).relative_to(root).as_posix()
        except ValueError:
            continue
        file = selection.file(rel)
        for line, column in positions:
            if file is not None:
                selection.execute(rel, file.offset(line, column))
    if selection.whole or selection.observes:
        raise RuntimeError(f"fixture fell back to whole files: {selection.whole}")
    entries, tables = selection.entries(em.Checker(root, {}), ["crates/alpha"], "crates/alpha")
    for address, data in tables.items():
        (root / ei.STORE).mkdir(parents=True, exist_ok=True)
        (root / ei.STORE / f"{address}.json").write_bytes(data)
    return entries


def owed(root: Path, entries: list[dict]) -> list[str]:
    em._FILE_DIGESTS.clear()
    ei._PARSED.clear()
    checker = em.Checker(root, {})
    names = next(e for e in entries if e["kind"] == "rust_names")
    changed = []
    for entry in entries:
        kind = entry["kind"]
        if kind == "rust_items":
            ok = checker.items.items_current(entry, names)
        elif kind == "rust_guards":
            ok = checker.items.guards_current(entry)
        elif kind == "rust_names":
            ok = True
        else:
            ok = checker.input_digest("file", entry["path"]) == entry["digest"]
        if not ok:
            changed.append(f"{kind}:{entry['path']}")
    return changed


def probe() -> int:
    pin = load_toml(ROOT / "tools/evidence-analyzer/pin.toml")
    identity = dict(line.split(": ", 1) for line in _rustc("-vV").splitlines() if ": " in line)
    if identity.get("release") != pin["release"] or identity.get("commit-hash") != pin["commit"]:
        raise RuntimeError("compiler identity differs from tools/evidence-analyzer/pin.toml")
    tools = Path(_rustc("--print", "sysroot").strip()) / "lib/rustlib" / identity["host"] / "bin"
    with tempfile.TemporaryDirectory(prefix="antecedent-analyzer-probe-") as tmp:
        scratch = Path(tmp).resolve()
        driver = scratch / "driver"
        _rustc(str(ROOT / "tools/evidence-analyzer/driver.rs"), "--edition=2021", "-C", "rpath=yes",
               "-o", str(driver), env={**os.environ, "RUSTC_BOOTSTRAP": "1"})
        base = scratch / "base"
        for rel, text in FILES.items():
            (base / rel).parent.mkdir(parents=True, exist_ok=True)
            (base / rel).write_text(text, encoding="utf-8")
        entries = earn(base, driver, tools)
        failures = 0
        for index, (expected, label, edits) in enumerate(CASES):
            root = scratch / f"case{index}"
            shutil.copytree(base, root, ignore=shutil.ignore_patterns("target", "*.profraw", "*.profdata"))
            for rel, old, new in edits:
                path = root / rel
                text = new if old is None else path.read_text(encoding="utf-8")
                if old is not None:
                    if old not in text:
                        raise RuntimeError(f"{label}: {old!r} not in {rel}")
                    text = text.replace(old, new, 1)
                path.write_text(text, encoding="utf-8")
            changed = owed(root, entries)
            got = "owed" if changed else "current"
            failures += got != expected
            print(f"{'ok  ' if got == expected else 'FAIL'} {expected:7} {label}: {', '.join(changed) or '-'}")
            shutil.rmtree(root)
        print(f"{len(CASES) - failures}/{len(CASES)} cases as expected")
        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(probe())
