#!/usr/bin/env python3
"""Emit checked-execution citations, and fail when one does not run its retained program.

A licensed cell with no citation is not a silent pass and is not a license.
A citation whose test keeps or rebuilds a builder, or never runs the retained
program, fails this gate before any test binary starts. Gaussian compiler
integration is covered by cited public behavioral tests, which must be executed
by the acceptance runner; emitting a citation alone does not execute its test.

Checked-recipe obligations for those citations are the route and the test.
They must match the support matrix. Identification and calibration stay on the
license record. A closed cell stays closed. This does not weaken the retained
program check below.

Every licensed route dispatches a lowered program. The gate also fails, before
any test binary starts, when:

* a `[[route]]` in parity/licensed_routes.toml records no `lowering`, names a
  value that is not a registered route cell, or the record does not hold every
  licensed route;
* a partial lowering survives under analysis/lowering or planner/lowered
  (`LoweringDepth::Generic`, `WidthSource::SchemaBound`, a `generic` lowering);
* library code under analysis/{execute,prepared,lowering} or planner re-enters
  the public builder (`Study::tabular` and its peers, `StudyBuilder`, or a
  builder's `.build()`), outside `#[cfg(test)]` items and test files;
* any code outside antecedent-core writes an `ExecutionContext { .. }` literal,
  which would bypass the lowered-dispatch scope the context carries;
* a raw entry that runs without a lowered program or a contract is re-exported from the
  facade, is public where it must be crate-internal, loses its `#[doc(hidden)]`, keeps a
  row in the Python facade registry, or is imported by a Python module other than its
  staged wrapper (`compiler_bypass_problems`).

* the route set derived from the route manifest, the `route_lowerings!` registry and the
  licensed routes (`gate_retained_routes.retained_route_problems`) has a route with no
  lowered program entry, a registered cell no sealed operation serves (its prepared
  execution is only the catch-all), a `retained_operation` that is not a sealed operation
  serving the route's cell, a reintroduced `OrdinaryDispatch`, a dispatch entry that does not require the lowered program, a compile
  or run entry that lost the `public_builder_reentry` refusal, or a caller that swallows
  `route_not_lowered`.

The evidence rows add the compiler-path, lowered-dispatch and lowering-registry
assertions to the checked-execution citations.
"""

import json
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from _rust_source import rust_mask
from gate_recipe_obligations import matrix_alignment_problems
from gate_retained_routes import retained_route_problems
from test_evidence import checked_execution_body_problems, closure

ROOT = Path(__file__).resolve().parents[1]
LICENSED = ROOT / "parity" / "support_licensed.toml"
LICENSED_ROUTES = ROOT / "parity" / "licensed_routes.toml"
LICENSED_ROUTE_COUNT = 723
FACADE = ROOT / "crates" / "antecedent" / "src"
REGISTRY = FACADE / "analysis" / "lowering" / "registry.rs"
LOWERING_SOURCES = (
    FACADE / "analysis" / "lowering",
    FACADE / "planner" / "lowered.rs",
    FACADE / "planner" / "lowered",
)
REENTRY_SOURCES = (
    FACADE / "analysis" / "execute",
    FACADE / "analysis" / "prepared",
    FACADE / "analysis" / "lowering",
    FACADE / "planner",
)
_PARTIAL_LOWERING = re.compile(
    r"\bLoweringDepth::Generic\b|\bWidthSource::SchemaBound\b|\bfn\s+generic\b"
    r"|\bSchemaBound\b|^\s*Generic\s*[,(]",
    re.M,
)
_BUILDER_BUILD = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*builder[A-Za-z0-9_]*\s*\.\s*build\s*\(\s*\)", re.I
)
_CONTEXT_LITERAL = re.compile(r"\bExecutionContext\s*\{")
_CONTEXT_DECLARATION = re.compile(
    r"(?:->|\bfor|\bimpl(?:\s*<[^>]*>)?|\bstruct|\benum|\btrait)\s*"
    r"(?:[A-Za-z_][A-Za-z0-9_]*::)*ExecutionContext\s*\{$"
)
_CFG_TEST = re.compile(r"#\s*\[\s*cfg\s*\(\s*(?:test|all\s*\(\s*test\b[^\]]*)\)\s*\]")
AXIS = ("query", "graph_class", "structure", "inference", "validation")
_STUDY_BUILDER = re.compile(
    r"\bStudy::(?:tabular|series|series_multi|panel|events)\b|\bStudyBuilder\b"
)


def coordinate_of(cell: dict) -> str:
    return ":".join(str(cell.get(key)) for key in AXIS)


def execution_reentry_problems(body: str, assertion: str) -> list[str]:
    """The retained run must not be a public `Study` built after the builder is gone.

    Only the block that discards the builder is inspected, and only a `Study`
    constructor counts. A later refusal or a second coordinate in that same
    block is still a re-entry: the licensed execution was supposed to be over.
    """
    # Closures concatenate helpers as separate top-level blocks. Scan each.
    depth = 0
    start = None
    blocks: list[str] = []
    for index, character in enumerate(body):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(body[start : index + 1])
                start = None
    for block in blocks or [body]:
        discards = list(
            re.finditer(
                r"\b(?:drop\s*\(\s*(?:[A-Za-z_][A-Za-z0-9_]*)?builder[A-Za-z0-9_]*\s*\)"
                r"|del\s+(?:[A-Za-z_][A-Za-z0-9_]*)?builder[A-Za-z0-9_]*"
                r"|(?:[A-Za-z_][A-Za-z0-9_]*)?builder[A-Za-z0-9_]*\s*=\s*None)\b",
                block,
                re.I,
            )
        )
        if not discards:
            continue
        # A constructor before the retained run replaced that run. A constructor
        # after `estimate`/`execute` is a separate refusal or coordinate.
        executed = re.compile(
            r"\b(?:execute|estimate(?:_[A-Za-z0-9_]+)?|interval|refresh_series|run|evaluate_exact)\s*\("
        )
        for discard in discards:
            tail = block[discard.end() :]
            study = _STUDY_BUILDER.search(tail)
            if study is None:
                continue
            ran = executed.search(tail)
            if ran is not None and ran.start() < study.start():
                continue
            return [
                f"evidence assertion {assertion!r} re-enters the public builder "
                "instead of running its retained program"
            ]
    return []


def route_cell_names() -> set[str]:
    """Every registered route cell: the registry rows and the graph-posterior cells."""
    text = REGISTRY.read_text(encoding="utf-8")
    table = text[text.index("route_lowerings! {") :]
    table = table[: table.index("}")]
    names = set(
        re.findall(r"^\s*([A-Z][A-Za-z0-9]*)\s*=>\s*[a-z_]+\s+as\s+[A-Za-z]+\s*,", table, re.M)
    )
    names |= set(re.findall(r'=>\s*"(GraphPosterior[A-Za-z]+)"', text))
    return names


def lowering_record_problems() -> list[str]:
    """Every licensed route names the registered cell its lowered program was compiled for."""
    names = route_cell_names()
    if not names:
        return [f"{REGISTRY.relative_to(ROOT)}: no route cells found"]
    routes = tomllib.loads(LICENSED_ROUTES.read_text(encoding="utf-8")).get("route", [])
    problems = []
    if len(routes) != LICENSED_ROUTE_COUNT:
        problems.append(
            f"licensed_routes.toml holds {len(routes)} routes, expected {LICENSED_ROUTE_COUNT}"
        )
    for route in routes:
        coordinate = route.get("coordinate", "<no coordinate>")
        lowering = route.get("lowering")
        if not isinstance(lowering, str) or not lowering.strip():
            problems.append(f"{coordinate}: licensed route records no lowering")
        elif lowering not in names:
            problems.append(f"{coordinate}: lowering {lowering!r} is not a registered route cell")
    return problems


def rust_files(root: Path) -> list[Path]:
    return [root] if root.is_file() else sorted(root.rglob("*.rs"))


def is_test_file(path: Path) -> bool:
    return (
        path.name == "tests.rs"
        or path.name.endswith("_tests.rs")
        or "tests" in path.relative_to(ROOT).parts[:-1]
    )


def without_test_items(masked: str) -> str:
    """Blank every `#[cfg(test)]` item: an inline module or function, or a `mod x;`."""
    out = list(masked)
    for attribute in _CFG_TEST.finditer(masked):
        start = attribute.start()
        brace = masked.find("{", attribute.end())
        semicolon = masked.find(";", attribute.end())
        if semicolon != -1 and (brace == -1 or semicolon < brace):
            end = semicolon + 1
        elif brace == -1:
            continue
        else:
            depth = 0
            end = len(masked)
            for index in range(brace, len(masked)):
                if masked[index] == "{":
                    depth += 1
                elif masked[index] == "}":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
        for index in range(start, end):
            if out[index] != "\n":
                out[index] = " "
    return "".join(out)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def partial_lowering_problems() -> list[str]:
    problems = []
    for root in LOWERING_SOURCES:
        if not root.exists():
            continue
        for path in rust_files(root):
            masked = rust_mask(path.read_text(encoding="utf-8"))
            for match in _PARTIAL_LOWERING.finditer(masked):
                problems.append(
                    f"{path.relative_to(ROOT)}:{line_of(masked, match.start())}: partial lowering "
                    f"{match.group(0).strip()!r} survives; every route lowers completely"
                )
    return problems


def reentry_problems() -> list[str]:
    problems = []
    for root in REENTRY_SOURCES:
        for path in rust_files(root):
            if is_test_file(path):
                continue
            masked = without_test_items(rust_mask(path.read_text(encoding="utf-8")))
            for pattern in (_STUDY_BUILDER, _BUILDER_BUILD):
                for match in pattern.finditer(masked):
                    problems.append(
                        f"{path.relative_to(ROOT)}:{line_of(masked, match.start())}: library "
                        f"code re-enters the public builder ({match.group(0).strip()!r})"
                    )
    return problems


def context_literal_problems() -> list[str]:
    problems = []
    roots = [path for path in sorted((ROOT / "crates").iterdir()) if path.is_dir()]
    roots.append(ROOT / "python")
    for root in roots:
        if root.name == "antecedent-core" or not root.exists():
            continue
        for path in sorted(root.rglob("*.rs")):
            if "target" in path.relative_to(ROOT).parts:
                continue
            masked = rust_mask(path.read_text(encoding="utf-8"))
            for match in _CONTEXT_LITERAL.finditer(masked):
                line_start = masked.rfind("\n", 0, match.start()) + 1
                if _CONTEXT_DECLARATION.search(masked[line_start : match.end()]):
                    continue
                problems.append(
                    f"{path.relative_to(ROOT)}:{line_of(masked, match.start())}: "
                    "ExecutionContext struct literal outside antecedent-core"
                )
    return problems


# Raw entries that run an estimate, an identification or a ranking without a lowered
# program or a contract. The compiled path (`Study::compile`, `prepare`, `run`) is the entry.
# Re-exporting one from the facade, or making it public again, reopens the bypass.
BYPASS_CRATE_INTERNAL = {
    "crates/antecedent/src/strategy_table/dispatch.rs": (
        "estimate_static_effect",
        "identify_static",
        "identify_pag",
        "identify_admg",
    ),
    "crates/antecedent/src/design.rs": ("rank_designs",),
    # Structure identification is reached through `StudyBuilder::identify_structure`, which
    # assembles a compiled study over a table with no rows; the raw entries stay crate-internal.
    "crates/antecedent/src/identify_api.rs": (
        "identify_study",
        "identify",
        "identify_with",
        "identify_dag",
    ),
}
# Public only as hidden path items: absent from the crate root, the prelude and the docs.
BYPASS_DOC_HIDDEN: dict[str, tuple[str, ...]] = {}
BYPASS_FACADE_SURFACES = (
    "crates/antecedent/src/lib.rs",
    "crates/antecedent/src/prelude.rs",
    "crates/antecedent/src/strategy_table/mod.rs",
    "crates/antecedent/src/design.rs",
    "crates/antecedent/src/analysis/surface",
)
BYPASS_REEXPORTS = (
    "estimate_static_effect",
    "identify_static",
    "identify_pag",
    "identify_admg",
    "estimate_interference",
    "trial_to_target_effect",
    "evaluate_exact_transport",
    "identify",
    "identify_with",
    "identify_dag",
    "rank_designs",
)
# Python natives with no compiled twin. Only the staged wrapper that owns each may import it.
BYPASS_PYTHON_NATIVES = (
    "_identify_ate_admg",
    "_identify_ate",
)
# Natives that had no compiled twin and were retired when the compiled study replaced them. No
# Python module may name one again, the staged wrapper included: the wrapper goes through
# `analyze` (the compiled `IvBounds` study), and a native entry of the same name reopens the bypass.
RETIRED_PYTHON_NATIVES = ("_binary_iv_ate_bounds",)
RETIRED_NATIVE_SOURCES = ("python/src",)
BYPASS_PYTHON_WRAPPERS = ("estimation.py", "identify.py")
_PUB_USE = re.compile(r"\bpub\s+use\b[^;]*;")


def _reexport_named(name: str) -> re.Pattern[str]:
    # The last path segment or a list item; `pub use crate::identify::{..}` is a module.
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w|\s*::)")


def compiler_bypass_problems(root: Path = ROOT) -> list[str]:
    """Raw estimate, identification and ranking entries stay behind the compiled path."""
    problems: list[str] = []
    for surface in BYPASS_FACADE_SURFACES:
        target = root / surface
        files = (
            [target]
            if target.is_file()
            else sorted(target.rglob("*.rs"))
            if target.is_dir()
            else []
        )
        for path in files:
            masked = rust_mask(path.read_text(encoding="utf-8"))
            for statement in _PUB_USE.finditer(masked):
                for name in BYPASS_REEXPORTS:
                    if _reexport_named(name).search(statement.group(0)):
                        problems.append(
                            f"{path.relative_to(root)}:{line_of(masked, statement.start())}: "
                            f"the facade re-exports the raw entry {name!r}, which runs without a "
                            "lowered program or a contract"
                        )
    for rel, names in BYPASS_CRATE_INTERNAL.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: file not found; update the bypass gate with the move")
            continue
        masked = rust_mask(path.read_text(encoding="utf-8"))
        for name in names:
            for match in re.finditer(rf"(?m)^\s*pub\s+(?:const\s+)?fn\s+{name}\b", masked):
                problems.append(
                    f"{rel}:{line_of(masked, match.start())}: {name} is public; it must stay "
                    "pub(crate) so only compiled programs reach it"
                )
    for rel, names in BYPASS_DOC_HIDDEN.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: file not found; update the bypass gate with the move")
            continue
        masked = rust_mask(path.read_text(encoding="utf-8"))
        for name in names:
            for match in re.finditer(rf"(?m)^pub\s+fn\s+{name}\b", masked):
                before = masked[: match.start()]
                item_end = max(before.rfind("}"), before.rfind(";"))
                if "#[doc(hidden)]" not in before[item_end + 1 :]:
                    problems.append(
                        f"{rel}:{line_of(masked, match.start())}: {name} lost #[doc(hidden)]; "
                        "an analysis goes through Study::compile"
                    )
    registry = root / "parity" / "python_facade_surface.toml"
    if registry.is_file():
        raw_rows = {
            "antecedent::" + path
            for path in (
                "design::rank_designs",
                "estimate::estimate_static_effect",
                "estimate::identify_admg",
                "estimate::identify_pag",
                "estimate::identify_static",
                "estimate::estimate_interference",
                "estimate::trial_to_target_effect",
                "estimate::evaluate_exact_transport",
                "identify",
                "identify_api::identify",
                "identify_api::identify_with",
                "identify_api::identify_dag",
            )
        }
        for row in tomllib.loads(registry.read_text(encoding="utf-8")).get("entry", []):
            if row.get("rust") in raw_rows:
                problems.append(
                    f"parity/python_facade_surface.toml: row {row['rust']!r} names a raw entry "
                    "that the facade no longer exports"
                )
    package = root / "python" / "antecedent"
    pattern = re.compile(rf"(?<!\w)(?:{'|'.join(BYPASS_PYTHON_NATIVES)})\b")
    if package.is_dir():
        for path in sorted(package.rglob("*.py")):
            if path.name in BYPASS_PYTHON_WRAPPERS:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line) and not line.lstrip().startswith("#"):
                    problems.append(
                        f"{path.relative_to(root)}:{number}: names a native with no compiled twin "
                        f"({pattern.search(line).group(0)}); only {', '.join(BYPASS_PYTHON_WRAPPERS)} "
                        "own the staged wrappers"
                    )
    retired = re.compile(rf"(?<!\w)(?:{'|'.join(RETIRED_PYTHON_NATIVES)})\b")
    for source in (*RETIRED_NATIVE_SOURCES, "python/antecedent"):
        base = root / source
        if not base.is_dir():
            continue
        for path in sorted([*base.rglob("*.py"), *base.rglob("*.rs"), *base.rglob("*.pyi")]):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                comment = "//" if path.suffix == ".rs" else "#"
                if retired.search(line) and not line.lstrip().startswith(comment):
                    problems.append(
                        f"{path.relative_to(root)}:{number}: names the retired native "
                        f"{retired.search(line).group(0)!r}; the compiled IvBounds study "
                        "(analyze with quasi.IvBounds) is the entry"
                    )
    return problems


def compiler_problems() -> list[str]:
    return (
        lowering_record_problems()
        + partial_lowering_problems()
        + reentry_problems()
        + context_literal_problems()
        + compiler_bypass_problems()
        + retained_route_problems()
    )


def citation_problems(test_rel: str, assertion: str, cache: dict) -> list[str]:
    key = (test_rel, assertion)
    if key not in cache:
        path = ROOT / test_rel
        if not path.is_file():
            cache[key] = [f"test {test_rel!r} does not exist"]
        else:
            body = closure(path, assertion)
            cache[key] = checked_execution_body_problems(
                body, assertion
            ) + execution_reentry_problems(body, assertion)
    return cache[key]


def cell_problems(cell: dict, cache: dict) -> list[str]:
    label = coordinate_of(cell)
    estimators = cell.get("estimators")
    entries = cell.get("checked_execution")
    if not isinstance(estimators, list) or not estimators:
        return [
            f"{label}: licensed cell has no estimators, so it is not a checked-execution license"
        ]
    if not isinstance(entries, list) or not entries:
        return [
            f"{label}: no retained-execution citation; an unlanded route is not a silent pass "
            "and is not a license"
        ]
    problems: list[str] = []
    named: list[str] = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            problems.append(f"{label}: checked_execution entry #{index} must be an inline table")
            continue
        estimator, test_rel, assertion = (
            entry.get("estimator"),
            entry.get("test"),
            entry.get("assertion"),
        )
        if not all(
            isinstance(value, str) and value.strip() for value in (estimator, test_rel, assertion)
        ):
            problems.append(
                f"{label}: checked_execution entry #{index} needs estimator, test and assertion"
            )
            continue
        named.append(estimator)
        for problem in citation_problems(test_rel, assertion, cache):
            problems.append(f"{label}::estimator={estimator}: {problem}")
    if len(named) != len(set(named)):
        problems.append(f"{label}: checked_execution names an estimator twice")
    if set(named) != set(estimators):
        problems.append(
            f"{label}: checked_execution estimators {sorted(set(named))} != "
            f"licensed estimators {sorted(set(estimators))}"
        )
    return problems


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} EVIDENCE_OUT SHARD")
    cells = tomllib.loads(LICENSED.read_text(encoding="utf-8")).get("cell", [])
    cache: dict = {}
    failures: list[str] = []
    rows = []
    for cell in cells:
        failures.extend(cell_problems(cell, cache))
        coordinate = coordinate_of(cell)
        for entry in cell.get("checked_execution") or []:
            if isinstance(entry, dict):
                rows.append(
                    (
                        f"{coordinate}::estimator={entry.get('estimator')}",
                        str(entry.get("test", "")),
                        str(entry.get("assertion", "")),
                    )
                )

    def cached_cite(test: str, assertion: str) -> list[str]:
        # cell_problems already reported a cached citation. A miss is still checked.
        if (test, assertion) in cache:
            return []
        return citation_problems(test, assertion, cache)

    alignment = matrix_alignment_problems(ROOT, cite=cached_cite)
    compiler = compiler_problems()
    if failures or alignment or compiler:
        if failures:
            print(
                f"FAIL: checked execution: {len(failures)} licensed route(s) bypass the retained plan",
                flush=True,
            )
            for failure in failures:
                print(f"FAIL: {failure}", flush=True)
        if alignment:
            print(
                f"FAIL: recipe obligations: {len(alignment)} cross-check failure(s)",
                flush=True,
            )
            for failure in alignment:
                print(f"FAIL: {failure}", flush=True)
        if compiler:
            print(
                f"FAIL: lowered dispatch: {len(compiler)} route(s) or source site(s) bypass the "
                "lowered program",
                flush=True,
            )
            for failure in compiler:
                print(f"FAIL: {failure}", flush=True)
        raise SystemExit(1)
    if not rows:
        raise SystemExit("FAIL: support_licensed.toml has no checked-execution citations")
    # Public behavioral assertions replace function-name checks of a parallel demonstration.
    gaussian_tests = "crates/antecedent/src/analysis/execute/prepared_gaussian_tests.rs"
    rows.extend(
        [
            (
                "gaussian_compiled_execution",
                gaussian_tests,
                "prepared_gaussian_retains_the_compiled_program_after_builder_drop",
            ),
            (
                "gaussian_compiled_resource_refusal",
                gaussian_tests,
                "prepared_gaussian_refuses_cancelled_and_insufficient_resource_execution",
            ),
            (
                "gaussian_compiled_retention",
                gaussian_tests,
                "prepared_gaussian_reservations_follow_builder_cache_and_result_lifetimes",
            ),
            (
                "gaussian_compiled_design_width",
                gaussian_tests,
                "prepared_gaussian_quotes_selected_columns_and_resampling_work",
            ),
            (
                "gaussian_compiled_materialization",
                gaussian_tests,
                "prepared_gaussian_materializes_the_actual_categorical_design",
            ),
            (
                "gaussian_compiled_refresh",
                gaussian_tests,
                "prepared_gaussian_refresh_replans_changed_rows_without_reidentifying",
            ),
            (
                "dose_grid_compiled_reuse",
                "crates/antecedent/src/analysis/bayesian_posterior_row.rs",
                "public_preparation_inspects_without_fitting_and_retargets_the_frozen_posterior",
            ),
            (
                "dose_grid_compiled_invalidation",
                "crates/antecedent/src/analysis/bayesian_posterior_row.rs",
                "changed_inputs_refit_and_execution_policy_refuses_before_fitting",
            ),
            (
                "cyclic_compiled_reuse_replay",
                "crates/antecedent/src/analysis/cyclic.rs",
                "common_prepared_handle_retains_fit_retargets_and_independently_replays",
            ),
            (
                "gaussian_compiled_compile_cancellation",
                gaussian_tests,
                "compile_observes_cancellation_and_a_refused_compile_holds_no_lease",
            ),
        ]
    )
    # The finite-discrete cells plan a supplied table under the pool, observe cancellation
    # between plans and survive an independent consumer of the exported joint.
    discrete = "crates/antecedent/src/analysis/finite_discrete.rs"
    rows.extend(
        [
            (
                "discrete_supplied_table",
                discrete,
                "a_supplied_table_and_its_posterior_draws_plan_to_the_hand_expectation",
            ),
            (
                "discrete_independent_consumer",
                discrete,
                "the_supplied_table_survives_an_independent_consumer",
            ),
            (
                "discrete_cancellation",
                discrete,
                "cancellation_stops_planning_before_and_between_plans",
            ),
            (
                "discrete_resource_refusal",
                discrete,
                "a_table_the_pool_cannot_hold_is_refused_before_it_is_built",
            ),
        ]
    )
    # Layer five declares a projection and a cancellable scan for every route that reads a table.
    rows.append(
        (
            "lowered_scan_operators",
            "crates/antecedent/src/analysis/lowering/scan.rs",
            "the_scan_declares_a_projection_and_a_cancellable_scan",
        )
    )
    # An association counterpart retains its own operation, not the program-only wrapper.
    rows.append(
        (
            "association_retained_operations",
            "crates/antecedent/tests/association_route.rs",
            "every_association_counterpart_retains_its_own_operation_and_executes_its_program",
        )
    )
    # Every licensed route dispatches the lowered program compile attached.
    lowered_dispatch = "crates/antecedent/tests/lowered_dispatch.rs"
    rows.extend(
        [
            (
                "compiler_path_licensed_routes",
                "crates/antecedent/tests/licensed_compiler.rs",
                "every_licensed_cell_completes_the_compiler_path",
            ),
            (
                "lowered_dispatch_refuses_unlowered_plan",
                lowered_dispatch,
                "dispatch_refuses_a_plan_without_a_lowered_program",
            ),
            (
                "lowered_dispatch_refuses_reentry",
                lowered_dispatch,
                "lowered_dispatch_refuses_public_builder_reentry",
            ),
            (
                "lowered_dispatch_identifies_once",
                lowered_dispatch,
                "one_shot_run_identifies_once_per_family",
            ),
            (
                "lowered_dispatch_retained_identification",
                lowered_dispatch,
                "compiled_execution_takes_the_identification_the_program_retained",
            ),
            (
                "lowering_registry_coverage",
                "crates/antecedent/src/analysis/lowering/tests.rs",
                "every_registered_cell_lowers_completely_and_the_licensed_routes_name_registered_cells",
            ),
        ]
    )
    shard = sys.argv[2]
    if shard != "all":
        # Keep all citations for one test target together, so one assertion is
        # executed by exactly one shard. Balance by distinct cited assertions.
        tests_by_path = defaultdict(set)
        for _, path, assertion in rows:
            if not path.startswith("crates/"):
                raise SystemExit(f"FAIL: checked shard needs Rust evidence; found {path}")
            tests_by_path[path].add(assertion)
        loads = [0] * 4
        assignment = {}
        for path, tests in sorted(tests_by_path.items(), key=lambda item: (-len(item[1]), item[0])):
            index = min(range(4), key=lambda i: (loads[i], i))
            assignment[path] = index
            loads[index] += len(tests) + 1
        index = int(shard.split("/")[0])
        total = len(rows)
        rows = [row for row in rows if assignment[row[1]] == index]
        if not rows:
            raise SystemExit(f"FAIL: checked shard {shard} has no citations")
        print(f"checked shard {shard}: {len(rows)}/{total} estimator citations", flush=True)
    with Path(sys.argv[1]).open("w", encoding="utf-8", newline="\n") as out:
        for label, path, assertion in rows:
            out.write("[[fixture_evidence]]\n")
            out.write(f"id = {json.dumps(label)}\n")
            out.write(f"evidence_test = {json.dumps(path)}\n")
            out.write(f"evidence_assertion = {json.dumps(assertion)}\n\n")
    print(
        f"checked execution: {len(cells)} licensed cells, {len(rows)} estimator citations",
        flush=True,
    )


if __name__ == "__main__":
    main()
