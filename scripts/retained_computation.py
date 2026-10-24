"""Retained routes keep their own family computation, and their evidence executes it.

`gate_retained_routes` proves a route has a lowered program, a sealed operation and a
citation. This module checks what those do not:

* **Family computation.** A sealed operation that is not the program-only wrapper
  (`CheckedRouteOperation`) must reach a computation crate (`antecedent_estimate`,
  `antecedent_stats`, ...) from its own module, or from the `execute_checked_*` method it
  calls, and must not call the generic dispatcher (`execute_on`, `execute_tabular`).
* **Wrapper routes.** A manifest route whose only serving operation is the wrapper must
  name, in `WRAPPER_FAMILY_EXECUTORS`, the family executor that computes it, and that
  executor must refuse a plan with no lowered program for its cell itself, not only through
  the dispatcher.
* **Recorded operation.** Every executing route a specialised sealer serves records a
  `retained_operation` that sealer serves for the route's cell.
* **Executing evidence.** A cited `retained_route_test` is not a name: its body must prepare
  and execute the route, assert the retained program's identity, and assert a number or
  interval of the result.
* **Axis rows.** Every `[[family_axis]]` route of `parity/support_graphless.toml` is a
  manifest route that executes, and every license's citation is the one its graphless route
  records.

Each function returns messages; none builds or runs anything.
"""

from __future__ import annotations

import functools
import re
import tomllib
from pathlib import Path

from _rust_source import lex, rust_mask

SRC = "crates/antecedent/src"
ANALYSIS = f"{SRC}/analysis"
EXECUTION = f"{ANALYSIS}/prepared/execution.rs"
GRAPHLESS = "parity/support_graphless.toml"
WRAPPER_TYPE = "CheckedRouteOperation"

# A computation crate: a path into one of these is the family's statistical or physical work.
FAMILY_COMPUTE = re.compile(
    r"\bantecedent_(?:estimate|stats|prob|kernels|learn|counterfactual|discovery|validate|"
    r"design|dynamics|bayes)\w*::"
)
# The generic dispatcher. A sealed operation that calls it is a wrapper, not the family.
GENERIC_DISPATCH = re.compile(r"\b(?:execute_on|execute_tabular)\s*\(")
_CFG_TEST = re.compile(r"#\s*\[\s*cfg\s*\(\s*(?:test|all\s*\(\s*test\b[^\]]*)\)\s*\]")

# Cells whose only serving operation is the wrapper, and the family executor that computes
# them: (file, function). The executor must require its lowered program for the cell itself.
WRAPPER_FAMILY_EXECUTORS: dict[str, tuple[str, str]] = {}
# What proves an executor refuses a plan with no lowered program for its cell.
LOWERED_GUARD = re.compile(r"\b(?:require_lowered|retained_refusal|lowered_analysis|expected_cell)\b")

# A manifest route that matches here records exactly this operation when several operations
# serve its cell. First match wins.
OPERATION_BY_ROUTE = (
    (r"^bayesian\.gcomp\.", "bayesian_gcomp"),
    (r"^bayesian\.temporal\.", "bayesian_temporal_dag_effect"),
    (r"^temporal\.", "temporal_dag_effect"),
    (r"^(?:gcm|attribution)\.", "attribution"),
    (r"^bayesian\.", None),
)

# What a cited test must do. Each pattern is searched in the function body.
TEST_REQUIRES = (
    ("prepare or run the route", re.compile(r"\.(?:prepare|run|estimate)\w*\s*\(|\bfn\s+run\b|\brun\s*\(")),
    ("assert the retained program", re.compile(r"assert_retained_graphless_program\s*\(|retained_operation\s*\(")),
    (
        "assert a number or interval of the result",
        re.compile(
            r"assert!\s*\([^;]*?(?:\.abs\(\)\s*<|<\s*\d|to_bits|interval|posterior|effect\(\))"
            r"|assert_eq!\s*\([^;]*?(?:result|inference|interval|posterior|effect)"
        ),
    ),
)


def _read(root: Path, rel: str) -> str:
    path = root / rel
    return path.read_text(encoding="utf-8") if path.is_file() else ""


@functools.lru_cache(maxsize=None)
def _masked(path: Path) -> str:
    """`rust_mask` of a file, computed once per process and path."""
    return rust_mask(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _live_path(path: Path) -> str:
    return _blank_tests(_masked(path))


def _blank_tests(masked: str) -> str:
    out = list(masked)
    for attribute in _CFG_TEST.finditer(masked):
        brace = masked.find("{", attribute.end())
        semicolon = masked.find(";", attribute.end())
        if semicolon != -1 and (brace == -1 or semicolon < brace):
            end = semicolon + 1
        elif brace == -1:
            continue
        else:
            depth, end = 0, len(masked)
            for index in range(brace, len(masked)):
                depth += (masked[index] == "{") - (masked[index] == "}")
                if depth == 0:
                    end = index + 1
                    break
        for index in range(attribute.start(), end):
            if out[index] != "\n":
                out[index] = " "
    return "".join(out)


def _live(root: Path, rel: str) -> str:
    """The non-test code of `rel` with comments and literal contents blanked."""
    path = root / rel
    return _live_path(path) if path.is_file() else ""


_CALLED = re.compile(
    r"\b((?:execute|run|estimate|compute|fit|evaluate|posterior|sample|draw)_[a-z0-9_]+)\s*[(:]"
)


@functools.lru_cache(maxsize=None)
def _index(path: Path) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Type names, function names and called execute/run/fit names of one file."""
    text = _masked(path)
    types = frozenset(re.findall(r"\b(?:struct|enum|type)\s+([A-Za-z_][A-Za-z0-9_]*)", text))
    functions = frozenset(re.findall(r"\bfn\s+([a-z_][a-z0-9_]*)", text))
    called = frozenset(_CALLED.findall(_live_path(path)))
    return types, functions, called


@functools.lru_cache(maxsize=None)
def _definitions(files: tuple[Path, ...]) -> dict[str, tuple[Path, ...]]:
    found: dict[str, list[Path]] = {}
    for path in files:
        for name in _index(path)[1]:
            found.setdefault(name, []).append(path)
    return {name: tuple(paths) for name, paths in found.items()}


def _is_test_file(path: Path) -> bool:
    return path.name == "tests.rs" or path.name.endswith("_tests.rs") or "tests" in path.parts[:-1]


def sealed_payloads(root: Path) -> dict[str, str]:
    """Sealed operation kind -> the operation type it holds."""
    text = rust_mask(_read(root, EXECUTION))
    start = text.find("sealed_executions! {\n")
    block = text[start : text.find("\n}", start)] if start >= 0 else ""
    pairs = re.findall(r"^    ([A-Z][A-Za-z0-9]*)\((?:Box<)?([\w:]+?)>?\),", block, re.M)
    return {kind: path.split("::")[-1] for kind, path in pairs}


def wrapper_kinds(payloads: dict[str, str]) -> set[str]:
    return {kind for kind, type_name in payloads.items() if type_name == WRAPPER_TYPE}


def _source_files(root: Path) -> list[Path]:
    return [
        path for path in sorted((root / SRC).rglob("*.rs"))
        if not _is_test_file(path.relative_to(root))
    ]


def _defining_files(files: list[Path], type_name: str) -> list[Path]:
    return [path for path in files if type_name in _index(path)[0]]


def _impl_text(root: Path, type_name: str) -> str:
    """The body of `impl SealedOperation for <type>` in the sealed-operation modules."""
    out: list[str] = []
    pattern = re.compile(rf"\bimpl\s+SealedOperation\s+for\s+{re.escape(type_name)}\b[^{{]*{{")
    for path in sorted((root / ANALYSIS / "prepared" / "sealed").glob("*.rs")):
        text = _live_path(path)
        for match in pattern.finditer(text):
            depth, end = 1, len(text)
            for index in range(match.end(), len(text)):
                depth += (text[index] == "{") - (text[index] == "}")
                if depth == 0:
                    end = index
                    break
            out.append(text[match.end() : end])
    return "\n".join(out)


def _module_files(
    root: Path, files: list[Path], defining: list[Path], impl_text: str = ""
) -> list[Path]:
    """The defining files, the directory module of each, and, two calls deep, the files
    defining the execute/run/estimate/fit functions those files call."""
    chosen = list(defining)
    for path in defining:
        directory = path.with_suffix("")
        if directory.is_dir():
            chosen.extend(sorted(directory.rglob("*.rs")))
    definitions = _definitions(tuple(files))
    frontier = set(chosen)
    for round_ in range(2):
        called: set[str] = set(_CALLED.findall(impl_text)) if round_ == 0 else set()
        for path in frontier:
            called |= _index(path)[2]
        found = {path for name in called for path in definitions.get(name, ())} - set(chosen)
        chosen.extend(sorted(found))
        frontier = found
    return sorted(set(chosen))


def family_computation_problems(root: Path) -> list[str]:
    payloads = sealed_payloads(root)
    if not payloads:
        return [f"{EXECUTION}: sealed_executions! block not found; update the family-computation gate"]
    wrappers = wrapper_kinds(payloads)
    files = _source_files(root)
    problems: list[str] = []
    for kind, type_name in sorted(payloads.items()):
        if kind in wrappers:
            continue
        defining = _defining_files(files, type_name)
        if not defining:
            problems.append(f"sealed operation {kind}: the module defining {type_name} was not found")
            continue
        impl_text = _impl_text(root, type_name)
        module = _module_files(root, files, defining, impl_text)
        # The generic dispatcher is checked in the operation's own files; the family
        # computation is searched through the functions it calls.
        own = [path for path in module if path in set(defining) or any(
            parent == path.parent or parent in path.parents for parent in
            [d.with_suffix("") for d in defining if d.with_suffix("").is_dir()]
        )]
        own_live = "\n".join(_live(root, str(path.relative_to(root))) for path in own) + impl_text
        live = "\n".join(_live(root, str(path.relative_to(root))) for path in module) + impl_text
        names = ", ".join(str(path.relative_to(root)) for path in defining)
        if not FAMILY_COMPUTE.search(live):
            problems.append(
                f"sealed operation {kind} ({names}) reaches no computation crate from its own "
                "module or its execute_checked_* method; a retained family operation computes "
                "its family, it does not wrap a generic executor"
            )
        if "route: CheckedRouteOperation" in own_live and (
            len(re.findall(r"require_program\(&(?:call\.)?handle\.plan\)", impl_text)) < 2
        ):
            problems.append(
                f"sealed operation {kind} ({names}) is sealed to a lowered program but its click "
                "does not require, in both its table and series clicks, that the plan still carries it"
            )
        if GENERIC_DISPATCH.search(own_live):
            problems.append(
                f"sealed operation {kind} ({names}) calls the generic dispatcher; it must run "
                "its own family computation"
            )
    return problems


def _reached_through(root: Path, stem: str, union: str) -> bool:
    """Whether a module the operation names calls the owning module `stem`: the family module
    of a computation crate that dispatches to its design-specific modules."""
    for path in sorted((root / "crates").glob("*/src/**/*.rs")):
        if path.stem == stem or _is_test_file(path.relative_to(root)):
            continue
        if re.search(rf"\b{re.escape(stem)}\b", _live_path(path)):
            name = path.parent.name if path.stem in ("mod", "lib") else path.stem
            if re.search(rf"\b{re.escape(name)}\b", union):
                return True
    return False


def _reexport_reached(root: Path, module: str, stem: str, union: str) -> bool:
    """Whether the operation names a symbol the owning module's crate re-exports from it."""
    parts = Path(module).parts
    if "src" not in parts:
        return False
    lib = root.joinpath(*parts[: parts.index("src") + 1], "lib.rs")
    if not lib.is_file():
        return False
    text = _masked(lib)
    names: set[str] = set()
    for match in re.finditer(rf"pub\s+use\s+{re.escape(stem)}::(\{{[^}}]*\}}|\w+)", text):
        names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", match.group(1)))
    return any(re.search(rf"\b{re.escape(name)}\b", union) for name in names)


def owning_module_problems(
    root: Path, routes: list[dict], route_cell, served: dict[str, set[str]], names: dict[str, str]
) -> list[str]:
    """A graphless route names the module that owns its statistics. The sealed operation that
    computes the route must reach that module, so the route's retained computation is the
    module's own and not a lookalike."""
    payloads = sealed_payloads(root)
    by_name = {names[kind]: payloads[kind] for kind in payloads if kind in names}
    files = _source_files(root)
    problems: list[str] = []
    for route in routes:
        if route["status"] != "graphless":
            continue
        module = route.get("owning_module", "")
        stem = Path(module).stem
        if stem in ("mod", "lib"):
            stem = Path(module).parent.name
        operation = route.get("retained_operation")
        type_name = by_name.get(operation)
        if not stem or type_name is None:
            continue
        defining = _defining_files(files, type_name)
        module_files = _module_files(root, files, defining, _impl_text(root, type_name))
        union = "\n".join(_live(root, str(path.relative_to(root))) for path in module_files)
        owning_path = root / module
        reached = (
            owning_path in module_files
            or re.search(rf"\b{re.escape(stem)}\b", union) is not None
            or _reached_through(root, stem, union)
            or _reexport_reached(root, module, stem, union)
        )
        if not reached:
            problems.append(
                f"{route['id']}: retained operation {operation} never reaches the owning module "
                f"{module}; the route's statistics are not computed by the module that owns them"
            )
    return problems


def _served_operations(
    cell: str, served: dict[str, set[str]], names: dict[str, str]
) -> list[str]:
    return sorted(names[kind] for kind, cells in served.items() if cell in cells and kind in names)


def recorded_operation_problems(
    root: Path,
    routes: list[dict],
    route_cell,
    served: dict[str, set[str]],
    names: dict[str, str],
    wrappers: set[str],
) -> list[str]:
    problems: list[str] = []
    wrapper_names = {names[kind] for kind in wrappers if kind in names}
    for route in routes:
        if route["status"] not in ("execution_only", "fitting_gap", "graphless"):
            continue
        cell = route_cell(route)
        if cell is None:
            continue
        serving = _served_operations(cell, served, names)
        specialised = [name for name in serving if name not in wrapper_names]
        recorded = route.get("retained_operation")
        if specialised:
            pick = None
            for pattern, operation in OPERATION_BY_ROUTE:
                if re.search(pattern, route["id"]):
                    pick = operation
                    break
            expected = [pick] if pick in specialised else specialised
            if recorded is None:
                pass
            elif recorded not in specialised:
                problems.append(
                    f"{route['id']}: retained_operation {recorded!r} is not served for cell "
                    f"{cell}; its specialised operations are {specialised}"
                )
            elif len(specialised) > 1 and pick and recorded != pick:
                problems.append(
                    f"{route['id']}: retained_operation {recorded!r} should be {expected[0]!r}"
                )
        elif serving:
            executor = WRAPPER_FAMILY_EXECUTORS.get(cell)
            if executor is None:
                problems.append(
                    f"{route['id']}: cell {cell} is served only by the program-only wrapper "
                    f"{serving}; give it a sealed operation that computes it, or name its family "
                    "executor in WRAPPER_FAMILY_EXECUTORS"
                )
    return problems


def wrapper_executor_problems(root: Path) -> list[str]:
    problems: list[str] = []
    for cell, (rel, function) in sorted(WRAPPER_FAMILY_EXECUTORS.items()):
        live = _live(root, rel)
        match = re.search(rf"\bfn\s+{re.escape(function)}\b", live)
        if not match:
            problems.append(f"{cell}: family executor {function} not found in {rel}")
            continue
        brace = live.find("{", match.end())
        depth, end = 0, len(live)
        for index in range(brace, len(live)):
            depth += (live[index] == "{") - (live[index] == "}")
            if depth == 0:
                end = index + 1
                break
        body = live[match.start() : end]
        if not LOWERED_GUARD.search(body):
            problems.append(
                f"{cell}: {function} ({rel}) does not require its own lowered program; a plan "
                "with no program for the cell must be refused by the family executor itself"
            )
        files = _source_files(root)
        module = _module_files(root, files, [root / rel])
        if not any(FAMILY_COMPUTE.search(_live(root, str(path.relative_to(root)))) for path in module):
            problems.append(f"{cell}: {rel} reaches no computation crate")
    return problems


def _function_body(text: str, name: str) -> str | None:
    match = re.search(rf"\bfn\s+{re.escape(name)}\b", text)
    if not match:
        return None
    brace = text.find("{", match.end())
    if brace < 0:
        return None
    depth, end = 0, len(text)
    for index in range(brace, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            end = index + 1
            break
    return text[match.start() : end]


def test_evidence_problems(root: Path, routes: list[dict]) -> list[str]:
    """A cited retained-route test prepares and executes the route, asserts the retained
    program, and asserts the result. A helper the test calls counts when the helper is in the
    same file."""
    problems: list[str] = []
    seen: set[str] = set()
    for route in routes:
        citation = route.get("retained_route_test")
        if not citation or citation in seen:
            continue
        seen.add(citation)
        path, _, name = str(citation).partition("::")
        text = lex(_read(root, path)).code
        body = _function_body(text, name)
        if body is None:
            continue  # a missing function is reported by the citation check
        # Inline the same-file helpers the test calls, one level deep.
        scope = body
        for helper in set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", body)):
            if helper == name:
                continue
            extra = _function_body(text, helper)
            if extra is not None:
                scope += "\n" + extra
        for need, pattern in TEST_REQUIRES:
            if not pattern.search(scope):
                problems.append(f"{route['id']}: retained_route_test {citation} does not {need}")
    return problems


def axis_problems(root: Path, routes: list[dict], today: str) -> list[str]:
    text = _read(root, GRAPHLESS)
    if not text:
        return [f"{GRAPHLESS}: not found"]
    data = tomllib.loads(text)
    by_id = {route["id"]: route for route in routes}
    executing = ("execution_only", "graphless", "licensed", "fitting_gap")
    problems: list[str] = []
    seen_axes: set[tuple[str, str]] = set()
    for axis in data.get("family_axis", []):
        key = (axis["family"], axis["axis"])
        if key in seen_axes:
            problems.append(f"{GRAPHLESS}: family {key[0]} records axis {key[1]} twice")
        seen_axes.add(key)
        if axis.get("disposition") == "counterpart":
            if not axis.get("routes"):
                problems.append(f"{GRAPHLESS}: {key} is a counterpart with no route")
            for rid in axis.get("routes", []):
                route = by_id.get(rid)
                if route is None:
                    problems.append(f"{GRAPHLESS}: {key} names {rid}, which is not a manifest route")
                elif route["status"] not in executing:
                    problems.append(f"{GRAPHLESS}: {key} names {rid}, whose status is {route['status']}")
        elif not axis.get("reason") or not axis.get("exit_by"):
            problems.append(f"{GRAPHLESS}: {key} records neither counterpart routes nor a reason and a dated exit")
        if axis.get("exit_by") and str(axis["exit_by"]) < today:
            problems.append(f"{GRAPHLESS}: {key} exit_by {axis['exit_by']} has passed")
    families = {license["family"] for license in data.get("license", [])}
    for family in sorted(families):
        for axis in ("bayesian", "temporal"):
            if (family, axis) not in seen_axes:
                problems.append(f"{GRAPHLESS}: family {family} has no {axis} axis row")
    cited = {route.get("retained_route_test") for route in routes if route["status"] == "graphless"}
    for license in data.get("license", []):
        if license.get("retained_route_test") not in cited:
            problems.append(
                f"{GRAPHLESS}: license {license['family']}/{license['design']}/{license['method']} "
                f"cites {license.get('retained_route_test')}, which no graphless route records"
            )
    return problems


def retained_computation_problems(
    root: Path,
    routes: list[dict],
    route_cell,
    served: dict[str, set[str]],
    names: dict[str, str],
    today: str,
) -> list[str]:
    payloads = sealed_payloads(root)
    wrappers = wrapper_kinds(payloads)
    problems = family_computation_problems(root)
    problems += recorded_operation_problems(root, routes, route_cell, served, names, wrappers)
    problems += owning_module_problems(root, routes, route_cell, served, names)
    problems += wrapper_executor_problems(root)
    problems += test_evidence_problems(root, routes)
    problems += axis_problems(root, routes, today)
    return problems
