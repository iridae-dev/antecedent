#!/usr/bin/env python3
"""Reconcile Python capabilities with executing journey evidence.

Python parity here is SEMANTIC capability parity, not API parity. Python is not Rust: mirroring
the Rust API item by item in Python is an explicit non-goal. Every capability Antecedent supports
must be reachable from Python in Python's own idiom; an individual Rust method, type, function
or re-export need not have a Python counterpart, and its absence is never a gap.

The capability universe is the supported units of `scripts/gate_support.py` (the one support
count), collapsed to what a Python user recognizes as "a thing I can do":

  query:<Q>        a matrix question (average effect, pulse effect, mediation, ...). The matrix
                   has one cell per (query, graph class, structure, inference, validation), but a
                   Python user states the question by constructing one query object and passes
                   the graph, the structure, the inference paradigm and the validation as
                   arguments of the same call. The cells of one query differ only by those
                   arguments, so the query is the unit; its cells stay with gate_support.
  design:<family>  a graphless design family (randomized effect, policy value, survival, ...).
                   The design and the method inside a family are arguments of one Python entry,
                   so the family is the unit.
  capability:<id>  a supported capability record outside the matrix (transport stages, adaptive
                   procedures, decisions, bridges): a record is one question a consumer asks
                   (`consumer_question`), which is already the Python user's granularity.

A unit is exercised when a row of `parity/python_capabilities/capabilities.toml` (status covered,
plumbing or deferred) cites a Python test whose reachable code exercises it: the test names the
query or family, is the Python test the capability record itself cites, or calls one of the
record's Python-named routes. A row may also state `supports = [unit ids]` when no static link
can show it. Rows may keep `rust` / `rust_types` as optional traceability pointers; if present
they must resolve, but their absence is never a finding.

A GAP is a supported unit that no such row exercises and that carries no recorded
`rust_only_reason` (a row with status "rust_only", a reason and the `supports` it applies to: a
capability intentionally not offered in Python). --require-complete fails on any gap, on any
integrity finding, and on absent or stale execution evidence. Static assertion scopes establish
reviewable semantics, not execution.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("parity/python_capabilities/capabilities.toml")
SURFACE = Path("parity/python_capabilities/rust_surface.json")
#: Statuses whose cited journeys must execute. A deferred row cites the journeys that do not
#: depend on the measurement it waits for (the typed refusal while not_measured, for example).
EXECUTING = {"covered", "plumbing", "deferred"}

#: Rows deferred in the last `check` call: (id, deferred_to, deferred dimensions). They are
#: reported beside the unresolved gaps and never counted as closed capabilities.
LAST_DEFERRED: list[tuple[str, str, list[str]]] = []

DIMENSIONS = {
    "success",
    "inspection",
    "controls",
    "refusal",
    "branch",
    "lifecycle",
    "artifact",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(root: Path) -> str:
    """Hash actual source inputs; no timestamp or installed-wheel inference."""
    h = hashlib.sha256()
    paths = [root / "Cargo.toml", root / "Cargo.lock"]
    for prefix in ("crates", "python", "macros"):
        directory = root / prefix
        if directory.exists():
            paths.extend(
                p
                for p in directory.rglob("*")
                if p.is_file()
                and p.suffix in {".rs", ".py", ".pyi", ".toml"}
                and not set(p.relative_to(root).parts) & {"target", "__pycache__", ".venv"}
            )
    for path in sorted(set(paths)):
        if path.is_file():
            h.update(path.relative_to(root).as_posix().encode() + b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def rust_surface(snapshot: str) -> dict[str, dict[str, str]]:
    """Keep exact method owners and export edges, including aliases and reexports."""
    items = {}
    for raw in snapshot.splitlines():
        if "#[doc(hidden)]" in raw or raw.startswith("# "):
            continue
        line = re.sub(r"^(#\[[^\]]*\]\s*)+", "", raw)
        method = re.match(
            r"^pub (?:const )?fn (<antecedent::.+?>)::(?:const )?fn (\w+)(?:[<(])", line
        )
        free = re.match(r"^pub (?:const )?fn (antecedent::[\w:]+)(?:[<(])", line)
        export = re.match(r"^pub use (antecedent::[\w:]+) = (.+)$", line)
        typ = re.match(r"^pub (?:struct|enum|trait|type) (antecedent::[\w:]+)", line)
        if method:
            key, kind = f"{method[1]}::{method[2]}", "method"
        elif free:
            key, kind = free[1], "function"
        elif export:
            key, kind = f"export:{export[1]}", "reexport"
        elif typ:
            key, kind = typ[1], "type"
        else:
            continue
        # Overloads remain separately auditable instead of silently replacing an entry.
        if key in items and items[key]["declaration"] != line:
            key += "#" + hashlib.sha256(line.encode()).hexdigest()[:12]
        items[key] = {"kind": kind, "declaration": line}
        if export:
            items[key]["target"] = export[2]
    return items


def current_surface(root: Path) -> dict[str, dict[str, str]]:
    items = rust_surface((root / "parity/public_api/antecedent.txt").read_text(encoding="utf-8"))
    # Preserve feature-conditional source entries not represented by normal-feature rustdoc.
    from check_python_surface import source_entries

    for symbol, kind in source_entries(root).items():
        if symbol not in items:
            parts = symbol.split("::")
            module = parts[2]
            source = root / f"crates/antecedent/src/analysis/{module}.rs"
            if not source.is_file():
                source = root / f"crates/antecedent/src/analysis/{module}/mod.rs"
            name = re.escape(parts[-1])
            text = source.read_text(encoding="utf-8")
            declaration = re.search(
                rf"^pub (?:const )?(?:fn|struct) {name}[^\n]*", text, re.MULTILINE
            )
            items[symbol] = {
                "kind": "function" if kind == "fn" else kind,
                "declaration": declaration[0] if declaration else symbol,
                "source": source.relative_to(root).as_posix(),
                "availability": "source-visible; reconcile conditional feature in executing build",
            }
    return items


def write_surface(root: Path) -> None:
    """Regenerate the inventory from the facade snapshot and conditional source."""
    inventory = {
        "schema": 1,
        "source": "parity/public_api/antecedent.txt plus conditional public analysis source",
        "items": current_surface(root),
    }
    (root / SURFACE).write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def local_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path outside repository: {name}")
    return path


def functions(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in result:
                raise ValueError(f"ambiguous duplicate function scope {node.name}")
            result[node.name] = node
    return result


def called_names(node: ast.AST) -> set[str]:
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def reachable_scopes(scope: str, defs: dict) -> set[str]:
    pending, seen = [scope], set()
    while pending:
        name = pending.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        pending.extend(called_names(defs[name]) - seen)
    return seen


def evidence_problems(root: Path, evidence: dict) -> list[str]:
    problems = []
    test = evidence.get("test", "")
    if "::" not in test:
        return [f"invalid test selector {test!r}"]
    filename, name = test.split("::", 1)
    if "[" in name or "::" in name:
        return [f"use a concrete module-level test scope (all its parameters execute): {test}"]
    try:
        tree = ast.parse(local_path(root, filename).read_text(encoding="utf-8"))
        defs = functions(tree)
    except (OSError, ValueError, SyntaxError) as error:
        return [f"{test}: {error}"]
    if name not in defs or not name.startswith("test_"):
        return [f"{test}: missing test function"]
    scopes = reachable_scopes(name, defs)
    if not evidence.get("assertion_scopes"):
        problems.append(f"{test}: no assertion scopes")
    for scope in evidence.get("assertion_scopes", []):
        if scope not in scopes:
            problems.append(f"{test}: assertion scope {scope} is not reached by this test")
            continue
        # An explicit exception assertion is valid evidence; arbitrary bare calls are not.
        nodes = list(ast.walk(defs[scope]))
        if not any(
            isinstance(n, ast.Assert)
            or (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in {"raises", "assertEqual", "assertTrue"}
            )
            for n in nodes
        ):
            problems.append(f"{test}: {scope} contains no assertion")
    for scope in scopes:
        node = defs[scope]
        decorators = " ".join(ast.unparse(d) for d in node.decorator_list)
        if re.search(r"\b(?:skip|skipif|xfail)\b", decorators):
            problems.append(f"{test}: skipped/xfail evidence scope {scope}")
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            if isinstance(call.func, ast.Attribute) and call.func.attr in {
                "skip",
                "xfail",
            }:
                problems.append(f"{test}: dynamic skip/xfail in {scope}")
    dimensions = set(evidence.get("dimensions", []))
    if not dimensions or dimensions - DIMENSIONS:
        problems.append(f"{test}: invalid semantic dimensions")
    if not evidence.get("assertion_contract", "").strip():
        problems.append(f"{test}: no reviewed semantic assertion contract")
    return problems


def execution_problems(root: Path, manifest: dict, execution: dict) -> list[str]:
    problems = []
    if execution.get("schema") != 1:
        problems.append("unsupported executing evidence schema")
    if execution.get("source_digest") != source_digest(root):
        problems.append("execution source digest is stale")
    if execution.get("manifest_sha256") != digest(root / MANIFEST):
        problems.append("execution capability manifest is stale")
    try:
        extension = Path(execution.get("native_path", ""))
        if not extension.is_file() or digest(extension) != execution.get("native_sha256"):
            problems.append("executed extension is missing or changed")
    except OSError:
        problems.append("executed extension cannot be read")
    for key, name in [
        ("checker_sha256", "scripts/check_python_capabilities.py"),
        ("runner_sha256", "scripts/run_python_capabilities.py"),
    ]:
        path = root / name
        if path.is_file() and execution.get(key) != digest(path):
            problems.append(f"execution {key} is stale")
    if not execution.get("build_receipt_verified"):
        problems.append("execution has no verified source-to-extension build receipt")
    required_selectors = {
        evidence["test"]
        for capability in manifest.get("capability", [])
        if capability.get("status") in EXECUTING
        for evidence in capability.get("evidence", [])
    }
    if set(execution.get("selected", [])) != required_selectors:
        problems.append("executing selection differs from required manifest journeys")
    observed = execution.get("tests", [])
    collected = set(execution.get("collected", []))
    if not collected or collected != {test.get("nodeid") for test in observed}:
        problems.append("collected parameters and observed executing cases differ")
    if not observed:
        problems.append("execution selected no tests")
    if execution.get("exit_code") != 0:
        problems.append("executing journey run failed")
    for capability in manifest.get("capability", []):
        if capability.get("status") not in EXECUTING:
            continue
        for evidence in capability.get("evidence", []):
            matches = [test for test in observed if test.get("selector") == evidence["test"]]
            if not matches:
                problems.append(
                    f"{capability['id']}: evidence test did not execute: {evidence['test']}"
                )
            elif any(test.get("outcome") != "passed" for test in matches):
                problems.append(f"{capability['id']}: evidence contains failed/skipped test")
    return problems


def type_members(items: dict[str, dict[str, str]], type_name: str) -> set[str]:
    """The inventory entries of one declared type: itself, its duplicates and its inherent methods."""
    members = set()
    for key, value in items.items():
        if key.startswith("export:"):
            continue
        if (
            key == type_name
            or key.startswith(type_name + "#")
            or value["kind"] == "method"
            and (key.startswith(f"<{type_name}>::") or key.startswith(f"<{type_name}<"))
        ):
            members.add(key)
    return members


def test_tokens(root: Path, test: str) -> set[str]:
    """Names and string words in a cited test's reachable code and in its module-level setup."""
    filename, name = test.split("::", 1)
    try:
        tree = ast.parse(local_path(root, filename).read_text(encoding="utf-8"))
        defs = functions(tree)
    except (OSError, ValueError, SyntaxError):
        return set()
    nodes: list[ast.AST] = [defs[scope] for scope in reachable_scopes(name, defs)]
    nodes += [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    tokens: set[str] = set()
    for node in nodes:
        for item in ast.walk(node):
            if isinstance(item, ast.Name):
                tokens.add(item.id)
            elif isinstance(item, ast.Attribute):
                tokens.add(item.attr)
            elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                tokens.update(re.findall(r"[A-Za-z0-9_]+", item.value))
    return tokens


def route_exercised(route: str, tokens: set[str]) -> bool:
    """A Python-named route `antecedent.<path>` whose callable (and class, if any) the test names."""
    parts = route.split(".")[1:]
    if not parts or parts[-1] not in tokens:
        return False
    return parts[-2] in tokens if len(parts) > 1 and parts[-2][:1].isupper() else True


def unit_exercised(unit: dict, row: dict, tokens: set[str], tests: set[str]) -> bool:
    if unit["id"] in row.get("supports", []):
        return True
    if tokens & set(unit.get("names", [])):
        return True
    files = {t.split("::")[0] for t in tests}
    for cited in unit.get("tests", []):
        if cited in tests or ("::" not in cited and cited in files):
            return True
    python = row.get("python")
    return any(
        route == python or route_exercised(route, tokens) for route in unit.get("routes", [])
    )


def module_tokens(path: Path) -> frozenset[str]:
    """Every name a test module mentions (cached; used only to explain a gap)."""
    key = str(path)
    if key not in _MODULE_TOKENS:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            tree = ast.parse("")
        _MODULE_TOKENS[key] = frozenset(
            item.id if isinstance(item, ast.Name) else item.attr
            for item in ast.walk(tree)
            if isinstance(item, (ast.Name, ast.Attribute))
        )
    return _MODULE_TOKENS[key]


_MODULE_TOKENS: dict[str, frozenset[str]] = {}


def gap_reason(root: Path, unit: dict) -> str:
    """Why a unit is a gap: Python tests that mention it but no row cites, or none at all."""
    near = []
    for path in sorted((root / "python/tests").glob("test_*.py")):
        tokens = set(module_tokens(path))
        if tokens & set(unit.get("names", [])) or any(
            route_exercised(route, tokens) for route in unit.get("routes", [])
        ):
            near.append(path.relative_to(root).as_posix())
    if near:
        return (
            "no capabilities.toml row cites a Python test of it (uncited tests touching it: "
            + ", ".join(near[:3])
            + ")"
        )
    missing = unit.get("unresolved_routes", [])
    if missing:
        return (
            "no Python test exercises it and its declared Python route(s) do not resolve to a "
            f"Python definition ({', '.join(missing[:2])}); either reach it in Python's own idiom "
            "or record a rust_only_reason"
        )
    return "no Python test found that exercises it, and no recorded rust_only_reason"


def semantic_gaps(root: Path, manifest: dict, universe: list[dict]) -> list[str]:
    """Supported capability units that no executing row exercises and none records as Rust-only."""
    rows = []
    for cap in manifest.get("capability", []):
        if cap.get("status") not in EXECUTING:
            continue
        tests = {e["test"] for e in cap.get("evidence", []) if "::" in e.get("test", "")}
        tokens: set[str] = set()
        for test in tests:
            tokens |= test_tokens(root, test)
        rows.append((cap, tokens, tests))
    rust_only = {
        unit
        for cap in manifest.get("capability", [])
        if cap.get("status") == "rust_only" and str(cap.get("rust_only_reason", "")).strip()
        for unit in cap.get("supports", [])
    }
    gaps = []
    for unit in universe:
        if unit["id"] in rust_only:
            continue
        if not any(unit_exercised(unit, row[0], row[1], row[2]) for row in rows):
            gaps.append(f"{unit['id']}: {gap_reason(root, unit)}")
    return gaps


def capability_universe() -> list[dict]:
    """The supported units of scripts/gate_support.py, collapsed to what a Python user chooses."""
    import gate_support

    result = gate_support.evaluate()
    records = {r["id"]: r for r in gate_support._load(gate_support.RECORDS).get("record") or []}
    units: dict[str, dict] = {}
    for unit in result["matrix"]:
        if unit.supported:
            query = unit.ident.split("/")[0]
            units.setdefault(
                f"query:{query}",
                {"id": f"query:{query}", "names": [query, query.removesuffix("Effect")]},
            )
    for unit in result["graphless"]:
        if unit.supported:
            family = unit.ident.split(":")[0]
            units.setdefault(f"design:{family}", {"id": f"design:{family}", "names": [family]})
    for unit in result["capabilities"]:
        if not unit.supported:
            continue
        record = records[unit.ident]
        cited: set[str] = set()

        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, str) and value.startswith("python/tests/"):
                cited.add(value.split("::")[0] if value.endswith(".py") else value)
                cited.add(value) if "::" in value else None

        walk(record)
        routes = [
            r["name"]
            for r in record.get("routes") or []
            if r.get("status") == "supported" and r["name"].startswith("antecedent.")
        ]
        ident = unit.ident
        # A Bayesian twin (`<id>.bayesian`) is reached through the same Python call as its
        # frequentist base by choosing the inference paradigm, so it is one unit with the base.
        if ident.endswith(".bayesian") and ident.removesuffix(".bayesian") in records:
            ident = ident.removesuffix(".bayesian")
        merged = units.setdefault(
            f"capability:{ident}",
            {"id": f"capability:{ident}", "tests": [], "routes": [], "names": []},
        )
        merged["tests"] = sorted(set(merged["tests"]) | cited)
        merged["routes"] = sorted(set(merged["routes"]) | set(routes))
        merged["names"] = sorted(set(merged["names"]) | set(record.get("owned_exports") or []))
    from check_python_surface import PythonIndex

    index = PythonIndex(ROOT)
    for unit in units.values():
        unit["unresolved_routes"] = [
            r for r in unit.get("routes", []) if not index.resolves(r)
        ]
    return list(units.values())


def check(
    root: Path,
    manifest: dict,
    inventory: dict,
    *,
    complete: bool = False,
    execution: dict | None = None,
    universe: list[dict] | None = None,
) -> tuple[list[str], list[str]]:
    """Integrity problems and semantic gaps. `universe` is `capability_universe()`; None checks
    integrity only (no capability is demanded)."""
    problems, gaps = [], []
    LAST_DEFERRED.clear()
    if manifest.get("schema") != 1 or inventory.get("schema") != 1:
        problems.append("unsupported manifest/inventory schema")
    actual = current_surface(root)
    if actual != inventory.get("items"):
        problems.append("Rust surface inventory differs from snapshot; deliberately reconcile it")
    from check_python_surface import PythonIndex

    python_index = PythonIndex(root)
    registry = tomllib.loads(
        (root / "parity/python_facade_surface.toml").read_text(encoding="utf-8")
    )["entry"]
    internal = {row["rust"] for row in registry if row["status"] == "internal"}
    seen, old_internal = set(), set()
    unit_ids = {unit["id"] for unit in universe or []}
    for cap in manifest.get("capability", []):
        label = cap.get("id", "")
        if not label or label in seen:
            problems.append(f"duplicate or empty capability id: {label}")
        seen.add(label)
        status = cap.get("status")
        if status not in {"covered", "plumbing", "gap", "deferred", "rust_only"}:
            problems.append(f"{label}: invalid status")
        for legacy in cap.get("legacy_internal", []):
            if legacy not in cap.get("rust", []):
                problems.append(f"{label}: old internal row not tied to its exact Rust symbol")
        old_internal.update(cap.get("legacy_internal", []))
        if not cap.get("owner") or not cap.get("behavior"):
            problems.append(f"{label}: missing concrete owner/behavior")
        # `rust` / `rust_types` are optional traceability pointers; when present they resolve.
        for symbol in cap.get("rust", []):
            if symbol not in actual:
                problems.append(f"{label}: unknown Rust symbol {symbol}")
        for claimed in cap.get("rust_types", []):
            if not type_members(actual, claimed):
                problems.append(f"{label}: unknown Rust type {claimed}")
        if universe is not None:
            for unit in cap.get("supports", []):
                if unit not in unit_ids:
                    problems.append(f"{label}: supports unknown capability unit {unit}")
        if status == "rust_only":
            if not str(cap.get("rust_only_reason", "")).strip():
                problems.append(f"{label}: a rust_only row records why Python does not offer it")
            if not cap.get("supports"):
                problems.append(f"{label}: a rust_only row names the capability units it covers")
            continue
        required = set(cap.get("required_dimensions", []))
        if not required or required - DIMENSIONS:
            problems.append(f"{label}: invalid required dimensions")
        if status == "gap":
            if not cap.get("gap", "").strip():
                problems.append(f"{label}: gap has no concrete next action")
            gaps.append(f"{label}: {cap.get('gap', '')}")
        else:
            if not cap.get("python"):
                problems.append(f"{label}: no concrete Python consumer")
            elif not python_index.resolves(cap["python"]):
                problems.append(f"{label}: Python consumer does not resolve")
            if status == "plumbing" and not cap.get("composed_caller", "").strip():
                problems.append(f"{label}: plumbing has no exact composed caller chain")
            evidence = cap.get("evidence", [])
            if not evidence:
                problems.append(f"{label}: no executing evidence")
            delivered = set()
            for item in evidence:
                problems.extend(f"{label}: {p}" for p in evidence_problems(root, item))
                delivered.update(item.get("dimensions", []))
            if status == "deferred":
                waits = set(cap.get("deferred_dimensions", []))
                target = str(cap.get("deferred_to", "")).strip()
                if not target or not waits or waits - required or not (waits < required):
                    problems.append(
                        f"{label}: a deferred row names what it waits for (deferred_to) and a "
                        "strict subset of its required dimensions (deferred_dimensions)"
                    )
                else:
                    LAST_DEFERRED.append((label, target, sorted(waits)))
                    required = required - waits
            if required - delivered:
                problems.append(
                    f"{label}: missing semantic dimensions {sorted(required - delivered)}"
                )
    if old_internal != internal:
        problems.append(
            f"internal classifications not reconciled: missing={sorted(internal - old_internal)}, extra={sorted(old_internal - internal)}"
        )
    if universe is not None:
        gaps.extend(semantic_gaps(root, manifest, universe))
    if complete:
        problems.extend(gaps)
        if execution is None:
            problems.append("completion requires actual executing journey evidence")
        else:
            problems.extend(execution_problems(root, manifest, execution))
    return problems, gaps


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--list-gaps", action="store_true")
    parser.add_argument(
        "--write-surface",
        action="store_true",
        help="regenerate the Rust inventory before checking; does not close capability gaps",
    )
    args = parser.parse_args()
    if args.write_surface:
        write_surface(ROOT)
    manifest = tomllib.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    inventory = json.loads((ROOT / SURFACE).read_text(encoding="utf-8"))
    execution = json.loads(args.execution.read_text(encoding="utf-8")) if args.execution else None
    problems, gaps = check(
        ROOT,
        manifest,
        inventory,
        complete=args.require_complete,
        execution=execution,
        universe=capability_universe(),
    )
    for problem in problems:
        print(f"ERROR: {problem}")
    if args.list_gaps:
        for gap in gaps:
            print(f"GAP: {gap}")
    for label, target, waits in LAST_DEFERRED:
        print(f"DEFERRED: {label}: waits for {target} ({', '.join(waits)})")
    print(
        f"Integrity findings: {len(problems)}; semantic capability gaps: {len(gaps)}; "
        f"deferred to calibration or attestation: {len(LAST_DEFERRED)}"
    )
    print("Inventory integrity alone is not capability completion or runtime evidence.")
    return bool(problems)


if __name__ == "__main__":
    sys.exit(main())
