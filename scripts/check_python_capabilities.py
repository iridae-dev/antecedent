#!/usr/bin/env python3
"""Reconcile Python capabilities with Rust declarations and executing journey evidence.

The normal mode checks inventory integrity and reports open gaps. --require-complete
also refuses unresolved capabilities, unmapped methods/exports and absent or stale
execution evidence. Static assertion scopes establish reviewable semantics, not execution.
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
MANIFEST = Path("parity/python_api_pass/capabilities.toml")
SURFACE = Path("parity/python_api_pass/rust_surface.json")
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


def export_owners(items: dict[str, dict[str, str]]) -> dict[str, str]:
    """Resolve exact export edges; never guess an owner from a matching leaf name.

    External crate paths are terminal identities. Relative paths resolve only
    against an inventoried declaration/export, so flattened `super` exports
    without their original module remain open instead of gaining false parity.
    """
    declarations = {key for key in items if not key.startswith("export:")}
    exports = {
        key[7:]: value["target"] for key, value in items.items() if value["kind"] == "reexport"
    }

    def target_of(alias: str, target: str) -> str | None:
        if target.startswith("antecedent_"):
            return target
        if target.startswith("antecedent::"):
            return target
        if target.startswith("crate::"):
            return "antecedent::" + target[7:]
        namespace = alias.rsplit("::", 1)[0].split("::")
        parts = target.split("::")
        if parts[0] == "self":
            return "::".join(namespace + parts[1:])
        if parts[0] == "super":
            while parts and parts[0] == "super":
                if len(namespace) <= 1:
                    return None
                namespace.pop()
                parts.pop(0)
            return "::".join(namespace + parts)
        while namespace:
            candidate = "::".join(namespace + parts)
            if candidate in declarations or candidate in exports:
                return candidate
            namespace.pop()
        return None

    owners = {}
    for alias in exports:
        current, visited = alias, set()
        while current in exports and current not in visited:
            visited.add(current)
            target = target_of(current, exports[current])
            if target is None:
                break
            current = target
        else:
            if current not in visited and (
                current in declarations or current.startswith("antecedent_")
            ):
                owners["export:" + alias] = current
    return owners


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
        if capability.get("status") in {"covered", "plumbing"}
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
        if capability.get("status") not in {"covered", "plumbing"}:
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


def check(
    root: Path,
    manifest: dict,
    inventory: dict,
    *,
    complete: bool = False,
    execution: dict | None = None,
) -> tuple[list[str], list[str]]:
    problems, gaps = [], []
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
    seen, old_internal, mapped = set(), set(), set()
    for cap in manifest.get("capability", []):
        label = cap.get("id", "")
        if not label or label in seen:
            problems.append(f"duplicate or empty capability id: {label}")
        seen.add(label)
        status = cap.get("status")
        if status not in {"covered", "plumbing", "gap"}:
            problems.append(f"{label}: invalid status")
        for legacy in cap.get("legacy_internal", []):
            if legacy not in cap.get("rust", []):
                problems.append(f"{label}: old internal row not tied to its exact Rust symbol")
        old_internal.update(cap.get("legacy_internal", []))
        if not cap.get("owner") or not cap.get("behavior"):
            problems.append(f"{label}: missing concrete owner/behavior")
        rust = cap.get("rust", [])
        if not rust:
            problems.append(f"{label}: no exact Rust entry")
        for symbol in rust:
            if symbol not in actual:
                problems.append(f"{label}: unknown Rust symbol {symbol}")
            mapped.add(symbol)
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
            if required - delivered:
                problems.append(
                    f"{label}: missing semantic dimensions {sorted(required - delivered)}"
                )
    if old_internal != internal:
        problems.append(
            f"internal classifications not reconciled: missing={sorted(internal - old_internal)}, extra={sorted(old_internal - internal)}"
        )
    # An export spelling adds no scientific behavior. Its exact canonical owner
    # must still have a capability row; that row's gap/evidence checks remain.
    owners = export_owners(actual)
    mapped_owners = {owners.get(symbol, symbol) for symbol in mapped}
    mapped.update(alias for alias, owner in owners.items() if owner in mapped_owners)
    for symbol in sorted(actual.keys() - mapped):
        gaps.append(f"unmapped Rust {actual[symbol]['kind']}: {symbol}")
    if complete:
        problems.extend(gaps)
        if execution is None:
            problems.append("completion requires actual executing journey evidence")
        else:
            problems.extend(execution_problems(root, manifest, execution))
    return problems, gaps


def main() -> int:
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
        ROOT, manifest, inventory, complete=args.require_complete, execution=execution
    )
    for problem in problems:
        print(f"ERROR: {problem}")
    if args.list_gaps:
        for gap in gaps:
            print(f"GAP: {gap}")
    print(f"Integrity findings: {len(problems)}; unresolved capability/surface gaps: {len(gaps)}")
    print("Inventory integrity alone is not capability completion or runtime evidence.")
    return bool(problems)


if __name__ == "__main__":
    sys.exit(main())
