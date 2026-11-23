#!/usr/bin/env python3
"""Self-test: the compiled-path gate passes on the repository and flags each planted bypass.

python3 scripts/test_gate_compiled_path.py [--self-test]

Each case plants synthetic snapshot lines, Python files or manifest rows in a temporary copy of
the files the gate reads, then asserts the gate names the planted item.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from gate_compiled_path import MANIFEST, REGISTRY_FILES, ROOT, compiled_path_problems
from _repo import load_toml

CTX = "&antecedent_core::execution::context::ExecutionContext"
EFFECT = "antecedent_estimate::effect_estimate::EffectEstimate"
RESULT = f"core::result::Result<{EFFECT}, antecedent_estimate::EstimationError>"


def overlay(destination: Path) -> None:
    """The files the gate reads, copied from the repository."""
    shutil.copytree(ROOT / "parity" / "public_api", destination / "parity" / "public_api")
    (destination / "parity").mkdir(exist_ok=True)
    shutil.copyfile(ROOT / MANIFEST, destination / MANIFEST)
    for rel in REGISTRY_FILES:
        if (ROOT / rel).is_file():
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, target)
    for source in ("python/antecedent", "python/src"):
        shutil.copytree(
            ROOT / source,
            destination / source,
            ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyc"),
        )
    # The Rust files a verdict row names, so that row is not stale in the copy.
    for row in load_toml(ROOT / MANIFEST).get("verdict", []):
        rel = row["path"].split("::")[0]
        if rel.endswith(".rs"):
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, target)


def append(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def manifest_row(path: str, disposition: str, **extra: str) -> str:
    fields = "".join(f'{key} = "{value}"\n' for key, value in extra.items())
    return f'\n[[row]]\npath = "{path}"\ndisposition = "{disposition}"\n{fields}'


def python_row(path: str, disposition: str, **extra: str) -> str:
    fields = "".join(f'{key} = "{value}"\n' for key, value in extra.items())
    return f'\n[[python]]\npath = "{path}"\ndisposition = "{disposition}"\n{fields}'


def compiled_route() -> str:
    """A route name present in the lowering registry text."""
    for rel in REGISTRY_FILES:
        if (ROOT / rel).is_file() and "ObservationalPolicy" in (ROOT / rel).read_text(encoding="utf-8"):
            return "ObservationalPolicy"
    raise SystemExit("FAIL: self-test needs a PlanRoute in the lowering registry")


def plant_deprecated_entry(root: Path, *, replacement: str | None, warn: bool | None) -> None:
    """A deprecated facade function, the native that calls it and the Python wrapper."""
    append(
        root,
        "parity/public_api/antecedent.txt",
        '#[deprecated(since = "1.0.0")] pub fn antecedent::planted::planted_old_fit('
        f"ctx: {CTX}) -> {RESULT}\n",
    )
    append(
        root,
        "python/src/planted_api.rs",
        "#[pyfunction]\nfn planted_native() {\n    antecedent::planted::planted_old_fit();\n}\n",
    )
    body = "    _deprecation.direct_path_deprecated('planted', 'fit')\n" if warn else ""
    append(
        root,
        "python/antecedent/planted_wrapper.py",
        "from . import _deprecation\n"
        "from ._native import planted_native as _planted_native\n\n\n"
        f"def planted_run():\n{body}    return _planted_native()\n",
    )
    manifest = manifest_row(
        "antecedent::planted::planted_new_fit", "compiled", route=compiled_route()
    )
    entry = {"replacement": replacement} if replacement else {}
    manifest += manifest_row("antecedent::planted::planted_old_fit", "deprecated", **entry)
    append(root, MANIFEST, manifest)
    # The compiled row's own candidate: a facade function that returns a result through the compiler.
    append(
        root,
        "parity/public_api/antecedent.txt",
        f"pub fn antecedent::planted::planted_new_fit(ctx: {CTX}) -> {RESULT}\n",
    )
    append(
        root,
        MANIFEST,
        python_row(
            "python/antecedent/planted_wrapper.py::planted_run",
            "deprecated",
            replacement="antecedent::planted::planted_new_fit",
        ),
    )


def case_with_context_reexport(root: Path) -> str:
    append(
        root,
        "parity/public_api/antecedent-estimate.txt",
        f"pub fn antecedent_estimate::planted::run_with_context(ctx: {CTX}) -> {RESULT}\n",
    )
    append(
        root,
        "parity/public_api/antecedent.txt",
        "pub use antecedent::estimate::run_with_context = "
        "antecedent_estimate::planted::run_with_context\n",
    )
    return "antecedent_estimate::planted::run_with_context: unclassified"


def case_method_on_reexported_type(root: Path) -> str:
    append(
        root,
        "parity/public_api/antecedent-estimate.txt",
        f"pub fn <antecedent_estimate::AipwAte>::fn planted_refit(&self, ctx: {CTX}) -> {RESULT}\n",
    )
    return "antecedent_estimate::AipwAte::planted_refit: unclassified"


def case_module_namespace_reexport(root: Path) -> str:
    append(
        root,
        "parity/public_api/antecedent-estimate.txt",
        "pub mod antecedent_estimate::planted_ns\n"
        f"pub fn antecedent_estimate::planted_ns::go(ctx: {CTX}) -> {RESULT}\n",
    )
    append(
        root,
        "parity/public_api/antecedent.txt",
        "pub use antecedent::estimate::planted_ns = antecedent_estimate::planted_ns\n",
    )
    return "antecedent_estimate::planted_ns::go: unclassified"


def case_deprecated_without_replacement(root: Path) -> str:
    plant_deprecated_entry(root, replacement=None, warn=True)
    return "disposition deprecated requires a non-empty 'replacement'"


def case_deprecated_replacement_not_compiled(root: Path) -> str:
    plant_deprecated_entry(root, replacement="antecedent::planted::missing", warn=True)
    return "does not name a compiled row"


def case_python_wrapper_missing_warning(root: Path) -> str:
    plant_deprecated_entry(root, replacement="antecedent::planted::planted_new_fit", warn=False)
    return "python/antecedent/planted_wrapper.py::planted_run: deprecated, but it does not call"


def case_python_literal_support_status(root: Path) -> str:
    append(
        root,
        "python/antecedent/planted_status.py",
        'class Planted:\n    support_status: str = "planted_literal"\n',
    )
    return "python/antecedent/planted_status.py::planted_literal"


def case_rust_capability_literal(root: Path) -> str:
    append(
        root,
        "crates/antecedent/src/planted_verdict.rs",
        'const PLANTED_CAPABILITY: &str = "planted.capability";\n',
    )
    return "crates/antecedent/src/planted_verdict.rs::PLANTED_CAPABILITY"


def case_stale_row(root: Path) -> str:
    append(root, MANIFEST, manifest_row("antecedent_estimate::nowhere::gone", "pending", owner="x"))
    return "stale row 'antecedent_estimate::nowhere::gone'"


def case_unclassified_python_wrapper(root: Path) -> str:
    plant_deprecated_entry(root, replacement="antecedent::planted::planted_new_fit", warn=True)
    path = root / MANIFEST
    text = path.read_text(encoding="utf-8")
    marker = python_row(
        "python/antecedent/planted_wrapper.py::planted_run",
        "deprecated",
        replacement="antecedent::planted::planted_new_fit",
    )
    path.write_text(text.replace(marker, ""), encoding="utf-8")
    return "python/antecedent/planted_wrapper.py::planted_run: unclassified Python wrapper"


CASES: list[tuple[str, Callable[[Path], str]]] = [
    ("a _with_context re-export", case_with_context_reexport),
    ("a method on a re-exported type", case_method_on_reexported_type),
    ("a module-namespace re-export", case_module_namespace_reexport),
    ("a deprecated row without a replacement", case_deprecated_without_replacement),
    ("a deprecated row whose replacement is not compiled", case_deprecated_replacement_not_compiled),
    ("a Python wrapper missing the deprecation warning", case_python_wrapper_missing_warning),
    ("an unclassified Python wrapper", case_unclassified_python_wrapper),
    ("a literal support_status", case_python_literal_support_status),
    ("a Rust *_CAPABILITY literal", case_rust_capability_literal),
    ("a stale row", case_stale_row),
]


def main() -> None:
    real = compiled_path_problems()
    if real:
        for problem in real[:20]:
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit("FAIL: the repository does not pass the compiled-path gate")
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="compiled-path-selftest-") as scratch:
        base = Path(scratch) / "base"
        overlay(base)
        clean = compiled_path_problems(base)
        if clean:
            raise SystemExit(f"FAIL: the unmodified copy is not clean: {clean[:3]}")
        # The positive control: a correct deprecated wrapper is accepted.
        control = Path(scratch) / "control"
        shutil.copytree(base, control)
        plant_deprecated_entry(
            control, replacement="antecedent::planted::planted_new_fit", warn=True
        )
        accepted = compiled_path_problems(control)
        if accepted:
            failures.append(f"a correct deprecated wrapper was refused: {accepted[:3]}")
        for index, (label, case) in enumerate(CASES):
            target = Path(scratch) / f"case{index}"
            shutil.copytree(base, target)
            expected = case(target)
            problems = compiled_path_problems(target)
            if not any(expected in problem for problem in problems):
                failures.append(f"{label}: expected {expected!r}, got {problems[:3]}")
            else:
                print(f"ok: {label} is flagged")
            shutil.rmtree(target)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("compiled-path gate self-test passed")


if __name__ == "__main__":
    main()
