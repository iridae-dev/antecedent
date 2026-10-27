#!/usr/bin/env python3
"""Self-test: the compiler-bypass check passes on the repo and fails on each reopened bypass.

python3 scripts/test_gate_compiler_bypass.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

from gate_checked_execution import ROOT, compiler_bypass_problems

COPIED = (
    "crates/antecedent/src/lib.rs",
    "crates/antecedent/src/prelude.rs",
    "crates/antecedent/src/design.rs",
    "crates/antecedent/src/identify_api.rs",
    "crates/antecedent/src/strategy_table/mod.rs",
    "crates/antecedent/src/strategy_table/dispatch.rs",
    "crates/antecedent/src/analysis/surface/estimate.rs",
    "parity/python_facade_surface.toml",
    "python/antecedent/estimation.py",
    "python/antecedent/identify.py",
    "python/src/bounds_api.rs",
)


def overlay(destination: Path) -> None:
    for rel in COPIED:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)


def mutated(destination: Path, rel: str, old: str, new: str) -> None:
    path = destination / rel
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"FAIL: self-test anchor {old!r} is missing from {rel}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    real = compiler_bypass_problems()
    if real:
        raise SystemExit("FAIL: the repository reopens a compiler bypass:\n" + "\n".join(real))
    cases = {
        "re-exported interference estimator": (
            "crates/antecedent/src/analysis/surface/estimate.rs",
            "pub use antecedent_estimate::{",
            "pub use antecedent_estimate::{estimate_interference, ",
        ),
        "re-exported raw static estimate": (
            "crates/antecedent/src/strategy_table/mod.rs",
            "pub use dispatch::{",
            "pub use dispatch::{identify_pag, ",
        ),
        "root re-export of identify": (
            "crates/antecedent/src/lib.rs",
            "pub use identify_api::Identification;",
            "pub use identify_api::{Identification, identify};",
        ),
        "public unbound ranking": (
            "crates/antecedent/src/design.rs",
            "pub(crate) fn rank_designs<A, O>(",
            "pub fn rank_designs<A, O>(",
        ),
        "public raw identification": (
            "crates/antecedent/src/strategy_table/dispatch.rs",
            "pub(crate) fn identify_pag(",
            "pub fn identify_pag(",
        ),
        "public structure identification": (
            "crates/antecedent/src/identify_api.rs",
            "pub(crate) fn identify_study(",
            "pub fn identify_study(",
        ),
        "registry row for a raw entry": (
            "parity/python_facade_surface.toml",
            '[[entry]]\nrust = "antecedent::design::preview_design_rank"',
            '[[entry]]\nrust = "antecedent::identify_api::identify_with"\nkind = "fn"\n'
            'status = "internal"\nreason = "x"\n\n[[entry]]\n'
            'rust = "antecedent::design::preview_design_rank"',
        ),
    }
    failures = []
    for label, (rel, old, new) in cases.items():
        with tempfile.TemporaryDirectory() as scratch:
            destination = Path(scratch)
            overlay(destination)
            mutated(destination, rel, old, new)
            if not compiler_bypass_problems(destination):
                failures.append(label)
    with tempfile.TemporaryDirectory() as scratch:
        destination = Path(scratch)
        overlay(destination)
        other = destination / "python/antecedent/elsewhere.py"
        other.write_text("from ._native import _identify_ate\n", encoding="utf-8")
        if not compiler_bypass_problems(destination):
            failures.append("python native imported outside its wrapper")
    with tempfile.TemporaryDirectory() as scratch:
        destination = Path(scratch)
        overlay(destination)
        wrapper = destination / "python/antecedent/identify.py"
        wrapper.write_text(
            wrapper.read_text(encoding="utf-8") + "\nfrom ._native import _binary_iv_ate_bounds\n",
            encoding="utf-8",
        )
        if not compiler_bypass_problems(destination):
            failures.append("retired binary-IV native named again in its wrapper")
    with tempfile.TemporaryDirectory() as scratch:
        destination = Path(scratch)
        overlay(destination)
        native = destination / "python/src/bounds_api.rs"
        native.write_text(
            native.read_text(encoding="utf-8")
            + '\n#[pyfunction(name = "_binary_iv_ate_bounds")]\nfn again() {}\n',
            encoding="utf-8",
        )
        if not compiler_bypass_problems(destination):
            failures.append("retired binary-IV native exported again by the extension")
    if failures:
        raise SystemExit("FAIL: the bypass check missed: " + ", ".join(failures))
    print(f"ok: compiler-bypass check passes on the repo and catches {len(cases) + 3} reopenings")


if __name__ == "__main__":
    sys.exit(main())
