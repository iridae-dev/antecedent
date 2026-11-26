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

from _repo import load_toml
from gate_compiled_path import (
    BOUND_REPLACEMENT_PREFIX,
    MANIFEST,
    REGISTRY_FILES,
    ROOT,
    bound_form_files,
    compiled_path_problems,
)

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
    # The sources that define the bound forms deprecated loaders name.
    for row in load_toml(ROOT / MANIFEST).get("row", []):
        replacement = str(row.get("replacement", ""))
        if replacement.startswith(BOUND_REPLACEMENT_PREFIX):
            for source in bound_form_files(ROOT, replacement[len(BOUND_REPLACEMENT_PREFIX) :]):
                target = destination / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
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


def plant_deprecated_entry(
    root: Path, *, replacement: str | None, warn: bool | str | None
) -> None:
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
    call = "_deprecation.direct_path_deprecated('planted', 'fit')"
    if warn == "conditional":
        body = f"    if True:\n        {call}\n"
    else:
        body = f"    {call}\n" if warn else ""
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


def plant_module_relay(root: Path) -> None:
    """A kernel module relayed by a second kernel crate, then reached by the facade through it."""
    append(
        root,
        "parity/public_api/antecedent-io-core.txt",
        "pub mod antecedent_io_core::planted_relay\n"
        "pub struct antecedent_io_core::planted_relay::PlantedFit\n"
        f"pub fn antecedent_io_core::planted_relay::relay_fit(ctx: {CTX}) -> {RESULT}\n"
        f"pub fn <antecedent_io_core::planted_relay::PlantedFit>::fn refit(&self, ctx: {CTX}) -> "
        f"{RESULT}\n",
    )
    append(
        root,
        "parity/public_api/antecedent-io.txt",
        "pub use antecedent_io::planted_relay = antecedent_io_core::planted_relay\n",
    )


def case_item_through_reexported_module(root: Path) -> str:
    plant_module_relay(root)
    append(
        root,
        "parity/public_api/antecedent.txt",
        "pub use antecedent::analysis::relay_fit = antecedent_io::planted_relay::relay_fit\n",
    )
    return "antecedent_io_core::planted_relay::relay_fit: unclassified"


def case_method_through_reexported_module(root: Path) -> str:
    plant_module_relay(root)
    append(
        root,
        "parity/public_api/antecedent.txt",
        "pub use antecedent::analysis::PlantedFit = antecedent_io::planted_relay::PlantedFit\n",
    )
    return "antecedent_io_core::planted_relay::PlantedFit::refit: unclassified"


def plant_super_relay(root: Path) -> None:
    """A kernel module the facade names through a relative `super::` path and a crate rename."""
    append(
        root,
        "parity/public_api/antecedent-design.txt",
        "pub mod antecedent_design::planted_super\n"
        "pub struct antecedent_design::planted_super::PlantedRun\n"
        f"pub fn antecedent_design::planted_super::run_super(ctx: {CTX}) -> {RESULT}\n"
        f"pub fn <antecedent_design::planted_super::PlantedRun>::fn refit(&self, ctx: {CTX}) -> "
        f"{RESULT}\n",
    )


def case_relative_super_item(root: Path) -> str:
    plant_super_relay(root)
    append(
        root,
        "parity/public_api/antecedent.txt",
        "pub use antecedent::analysis::planted_super::run_super = "
        "super::design_exports::design_crate::planted_super::run_super\n",
    )
    return "antecedent_design::planted_super::run_super: unclassified"


def case_relative_super_module_method(root: Path) -> str:
    plant_super_relay(root)
    append(
        root,
        "parity/public_api/antecedent.txt",
        "pub use antecedent::analysis::planted_super = "
        "super::design_exports::design_crate::planted_super\n"
        "pub use antecedent::analysis::planted_super::PlantedRun = "
        "super::design_exports::design_crate::planted_super::PlantedRun\n",
    )
    return "antecedent_design::planted_super::PlantedRun::refit: unclassified"


def case_deprecated_without_replacement(root: Path) -> str:
    plant_deprecated_entry(root, replacement=None, warn=True)
    return "disposition deprecated requires a non-empty 'replacement'"


def case_deprecated_replacement_not_compiled(root: Path) -> str:
    plant_deprecated_entry(root, replacement="antecedent::planted::missing", warn=True)
    return "does not name a compiled row"


def plant_kernel_method(root: Path, **extra: str) -> None:
    """A run method of a re-exported kernel type, classified `kernel`."""
    append(
        root,
        "parity/public_api/antecedent-estimate.txt",
        f"pub fn <antecedent_estimate::AipwAte>::fn planted_kernel_fit(&self, ctx: {CTX}) -> {RESULT}\n",
    )
    append(
        root,
        MANIFEST,
        manifest_row("antecedent_estimate::AipwAte::planted_kernel_fit", "kernel", **extra),
    )


def case_kernel_without_reason(root: Path) -> str:
    plant_kernel_method(root)
    return "disposition kernel requires a non-empty 'reason'"


def case_kernel_on_facade_item(root: Path) -> str:
    append(
        root,
        "parity/public_api/antecedent.txt",
        f"pub fn antecedent::planted::planted_facade_fit(ctx: {CTX}) -> {RESULT}\n",
    )
    append(
        root,
        MANIFEST,
        manifest_row("antecedent::planted::planted_facade_fit", "kernel", reason="planted"),
    )
    return "classified kernel, but it is not an item of a kernel crate"


PLANTED_LOADER = """
pub fn planted_load(bytes: &[u8]) -> Result<u8, IoError> {
    todo!()
}

pub fn planted_load_expecting(
    bytes: &[u8],
    retained: &RetainedArtifactIdentity,
) -> Result<u8, IoError> {
    BODY
}
"""


def plant_unbound_loader(root: Path, body: str) -> None:
    """A deprecated facade loader whose row names a bound form in a planted kernel file."""
    append(
        root,
        "crates/antecedent-io/src/planted_loader.rs",
        PLANTED_LOADER.replace("BODY", body),
    )
    append(
        root,
        "parity/public_api/antecedent.txt",
        '#[deprecated(since = "3.1.0")] pub fn antecedent::io::planted_load(bytes: &[u8]) -> '
        "core::result::Result<u8, antecedent_io_core::error::IoError>\n",
    )
    append(
        root,
        MANIFEST,
        manifest_row(
            "antecedent::io::planted_load",
            "deprecated",
            replacement=f"{BOUND_REPLACEMENT_PREFIX}antecedent_io::planted::planted_load_expecting",
        ),
    )


def case_bound_replacement_reads_unbound(root: Path) -> str:
    plant_unbound_loader(root, "planted_load(bytes)")
    return "does not name a bound form that binds the bytes"


def case_bound_replacement_missing(root: Path) -> str:
    append(
        root,
        "parity/public_api/antecedent.txt",
        '#[deprecated(since = "3.1.0")] pub fn antecedent::io::planted_missing(bytes: &[u8]) -> '
        "core::result::Result<u8, antecedent_io_core::error::IoError>\n",
    )
    append(
        root,
        MANIFEST,
        manifest_row(
            "antecedent::io::planted_missing",
            "deprecated",
            replacement=f"{BOUND_REPLACEMENT_PREFIX}antecedent_io::planted::planted_missing_expecting",
        ),
    )
    return "does not name a bound form that binds the bytes"


def case_deprecated_unknown_route(root: Path) -> str:
    plant_deprecated_entry(root, replacement="route:PlantedNoSuchRoute", warn=True)
    return "does not name a route in the lowering registry"


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


def plant_deprecated_facade(root: Path) -> None:
    """The deprecated facade function and its compiled replacement, with no native or wrapper."""
    append(
        root,
        "parity/public_api/antecedent.txt",
        '#[deprecated(since = "1.0.0")] pub fn antecedent::planted::planted_old_fit('
        f"ctx: {CTX}) -> {RESULT}\n"
        f"pub fn antecedent::planted::planted_new_fit(ctx: {CTX}) -> {RESULT}\n",
    )
    append(
        root,
        MANIFEST,
        manifest_row("antecedent::planted::planted_new_fit", "compiled", route=compiled_route())
        + manifest_row(
            "antecedent::planted::planted_old_fit",
            "deprecated",
            replacement="antecedent::planted::planted_new_fit",
        ),
    )


def case_renamed_native_wrapper(root: Path) -> str:
    plant_deprecated_facade(root)
    # A long signature puts the renaming attribute far above the `fn`.
    signature = ", ".join(f"argument_{index}=None" for index in range(60))
    append(
        root,
        "python/src/planted_renamed_api.rs",
        '#[pyfunction(name = "planted_renamed")]\n'
        f"#[pyo3(signature = ({signature}))]\n"
        "fn planted_renamed_py() {\n    antecedent::planted::planted_old_fit();\n}\n",
    )
    append(
        root,
        "python/antecedent/planted_renamed.py",
        "from ._native import planted_renamed as _planted_renamed\n\n\n"
        "def planted_renamed_run():\n    return _planted_renamed()\n",
    )
    return "python/antecedent/planted_renamed.py::planted_renamed_run: unclassified Python wrapper"


def case_renamed_native_method(root: Path) -> str:
    plant_deprecated_facade(root)
    append(
        root,
        "python/src/planted_stage_api.rs",
        '#[pyclass(name = "PlantedStage")]\nstruct PlantedStageNative;\n\n'
        "#[pymethods]\nimpl PlantedStageNative {\n"
        '    #[staticmethod]\n    #[pyo3(name = "run")]\n'
        "    fn run_py() {\n        antecedent::planted::planted_old_fit();\n    }\n}\n",
    )
    append(
        root,
        "python/antecedent/planted_stage.py",
        "from ._native import PlantedStage\n\n\n"
        "def planted_stage_run():\n    return PlantedStage.run()\n",
    )
    return "python/antecedent/planted_stage.py::planted_stage_run: unclassified Python wrapper"


CASES: list[tuple[str, Callable[[Path], str]]] = [
    ("a renamed native behind a long signature", case_renamed_native_wrapper),
    ("a renamed method of a native class", case_renamed_native_method),
    ("a _with_context re-export", case_with_context_reexport),
    ("a method on a re-exported type", case_method_on_reexported_type),
    ("a module-namespace re-export", case_module_namespace_reexport),
    ("an item re-exported through a re-exported module", case_item_through_reexported_module),
    ("a method of a type re-exported through a re-exported module", case_method_through_reexported_module),
    ("an item named through a relative super:: re-export", case_relative_super_item),
    ("a method of a type named through a relative super:: module re-export", case_relative_super_module_method),
    ("a deprecated row without a replacement", case_deprecated_without_replacement),
    ("a deprecated row whose replacement is not compiled", case_deprecated_replacement_not_compiled),
    ("a kernel row without a reason", case_kernel_without_reason),
    ("a kernel row on a facade item", case_kernel_on_facade_item),
    ("a deprecated row naming an unknown route", case_deprecated_unknown_route),
    ("a bound replacement that reads before binding", case_bound_replacement_reads_unbound),
    ("a bound replacement that does not exist", case_bound_replacement_missing),
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
        # A deprecated row may name a compiled route directly.
        by_route = Path(scratch) / "by-route"
        shutil.copytree(base, by_route)
        plant_deprecated_entry(by_route, replacement=f"route:{compiled_route()}", warn=True)
        accepted = compiled_path_problems(by_route)
        if accepted:
            failures.append(f"a deprecated row naming a route was refused: {accepted[:3]}")
        # A deprecation scoped by an `if` to the served cases still counts as warning first.
        scoped = Path(scratch) / "scoped"
        shutil.copytree(base, scoped)
        plant_deprecated_entry(
            scoped, replacement="antecedent::planted::planted_new_fit", warn="conditional"
        )
        accepted = compiled_path_problems(scoped)
        if accepted:
            failures.append(f"a deprecation scoped by an if was refused: {accepted[:3]}")
        # A deprecated unbound loader naming a bound form that binds first is accepted.
        bound = Path(scratch) / "bound"
        shutil.copytree(base, bound)
        plant_unbound_loader(bound, "load_expecting(bytes, retained, planted_load)")
        accepted = compiled_path_problems(bound)
        if accepted:
            failures.append(f"a deprecated loader naming its bound form was refused: {accepted[:3]}")
        # A kernel row with a reason is accepted.
        kernel = Path(scratch) / "kernel"
        shutil.copytree(base, kernel)
        plant_kernel_method(kernel, reason="kernel API; the type configures a compiled estimator")
        accepted = compiled_path_problems(kernel)
        if accepted:
            failures.append(f"a kernel row with a reason was refused: {accepted[:3]}")
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
