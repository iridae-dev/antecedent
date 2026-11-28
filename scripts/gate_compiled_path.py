#!/usr/bin/env python3
"""Every scientific computation the facade or the Python package reaches goes through the compiler.

Static: reads only committed files (the public-API snapshots under parity/public_api, the
lowering registry, python/, python/src) and the classification in parity/compiled_path.toml.

Candidates (each needs a row):

* every kernel function the facade re-exports (`pub use antecedent::P = <kernel>::Q`), and every
  executing method of a re-exported struct, enum or module (it takes an `ExecutionContext` or a
  data value, or returns a scientific result), so a namespace re-export or a method of a
  re-exported estimator cannot slip past a name list;
* every facade function or method whose return type, after unwrapping Result, Option, Vec, Box,
  Arc, references and tuples, is a scientific result type (the seed in `[result_types]`, closed
  over public fields and variants), and every deprecated facade function;
* every public Python function that reaches, through a native binding, a direct forwarder;
* every verdict literal that a generated table must own: a Rust `const *_CAPABILITY: &str`
  outside crates/antecedent/src/support/units_generated.rs and a Python
  `support_status: str = "<literal>"` outside python/antecedent/_direct_status.py.

Dispositions:

* `owner`: publishes the result. `unit` is the PlanRoute or the capability that carries the
  check. `route`, when present, is that PlanRoute in the lowering registry;
* `forwarder`: returns an owner's result and adds no claim. `unit` names the owner path or the
  route. Without `direct`, it is a kernel-crate item the unit runs. `direct = true` is a facade
  entry the snapshot marks `#[deprecated]` (every such entry is direct, or a consumer with `of`);
  its Python wrappers call `_deprecation.direct_path_deprecated(` first, directly or inside an
  `if` that scopes the warning to the cases the route serves;
* `consumer`: reads an artifact or result a unit already sealed and adds no claim; it takes no
  data value. `of` names the bound form when the consumer admits bytes (`*_expecting` /
  `*_retained` binds them to a `RetainedArtifactIdentity` before reading);
* `not_scientific`: computes no result (construction, serialization, identity, gating, data
  access); `reason` says what it does. It returns no result type and is not deprecated.

An unclassified candidate owes a claim. A stale row (a row no candidate matches) fails.

Usage:
  gate_compiled_path.py            check the repository
  gate_compiled_path.py --list     print the candidates by disposition
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from _repo import load_toml
from _rust_source import lex

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "parity/compiled_path.toml"
SNAPSHOTS = "parity/public_api"
FACADE = "antecedent"
REGISTRY_FILES = (
    "crates/antecedent/src/analysis/lowering/registry.rs",
    "crates/antecedent/src/analysis/lowering/program_families.rs",
)
GENERATED_VERDICTS_RS = "crates/antecedent/src/support/units_generated.rs"
GENERATED_VERDICTS_PY = "python/antecedent/_direct_status.py"
DEPRECATION_MODULE = "python/antecedent/_deprecation.py"
ROUTE_REPLACEMENT_PREFIX = "route:"
# A deprecated unbound artifact loader names its bound form: `bound:<canonical path>` of a
# `*_expecting` / `*_retained` function that takes a RetainedArtifactIdentity and binds the bytes
# (load_expecting / bind_retained_artifact, through bound_identity::bind) before reading them.
BOUND_REPLACEMENT_PREFIX = "bound:"

DISPOSITIONS = ("owner", "forwarder", "consumer", "not_scientific")
DISPOSITION_FIELD = {"owner": "unit", "forwarder": "unit", "not_scientific": "reason"}
CONTEXT_TYPE = "ExecutionContext"
DATA_TYPES = frozenset(
    {
        "TabularData",
        "TimeSeriesData",
        "MultiEnvironmentData",
        "PanelData",
        "TableFile",
        "TableView",
        "NetworkData",
    }
)

_ATTRS = re.compile(r"^((?:#\[[^\]]*\] )*)")
_FN = re.compile(
    r"^pub fn (?:<(?P<owner>.+?)>::(?:(?:const|async|unsafe) )*fn |(?P<path>[\w:]+::))(?P<name>\w+)(?P<rest>.*)$"
)
_TYPE = re.compile(r"^pub (struct|enum|trait|type|mod|union) ([\w:]+)")
_FIELD = re.compile(r"^pub field ([\w:]+)::\w+: (.*)$")
_VARIANT = re.compile(r"^pub variant ([\w:]+)::\w+(.*)$")
_USE = re.compile(r"^pub use (\S+) = (\S+)$")
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Fn:
    key: str  # canonical `path::name` or `Owner::name`
    owner: str | None
    name: str
    params: str
    ret: str
    hidden: bool
    deprecated: bool
    crate: str


def _balanced(text: str, start: int, open_: str, close: str) -> int:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == open_:
            depth += 1
        elif char == close and not (close == ">" and index and text[index - 1] == "-"):
            depth -= 1
            if depth == 0:
                return index
    return len(text) - 1


def _signature(rest: str) -> tuple[str, str]:
    """(parameters, return type) of what follows a function name in a snapshot line."""
    index = 0
    if rest.startswith("<"):
        index = _balanced(rest, 0, "<", ">") + 1
    if index >= len(rest) or rest[index] != "(":
        return "", ""
    end = _balanced(rest, index, "(", ")")
    params = rest[index + 1 : end]
    tail = rest[end + 1 :]
    ret = tail.split(" -> ", 1)[1] if " -> " in tail else ""
    return params, ret.split(" where ", 1)[0]


class Snapshots:
    """The committed public-API snapshots, parsed once."""

    def __init__(self, root: Path) -> None:
        self._merged: dict[str, str] | None = None
        self.fns: list[Fn] = []
        self.kinds: dict[str, str] = {}
        self.alias: dict[str, dict[str, str]] = defaultdict(dict)  # crate -> alias -> target
        self.fields: dict[str, list[str]] = defaultdict(list)  # simple type name -> field types
        directory = root / SNAPSHOTS
        files = sorted(directory.glob("*.txt")) if directory.is_dir() else []
        self.crates = [path.stem.replace("-", "_") for path in files]
        for path in files:
            crate = path.stem.replace("-", "_")
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("# ") or not line:
                    continue
                self._line(crate, line)
        self._absolutize()

    def _absolutize(self) -> None:
        """Rewrite each relative `pub use a::b::X = super::X` target to the path it names.

        The snapshot prints a relative re-export target as written in the source. `super::X`
        in module `m` names `X` in the parent of `m`, which re-exports it in turn (and the
        crate root, which has no parent, names the facade's analysis namespace); the chain is
        then followed by [`Self::resolve`]. A target that no item or re-export answers to is
        kept as written and names nothing.
        """
        known = {name for table in self.alias.values() for name in table}
        known |= set(self.kinds) | {fn.key for fn in self.fns}
        for table in self.alias.values():
            for alias, target in list(table.items()):
                if not target.startswith("super::"):
                    continue
                parts = alias.split("::")[:-1]
                rest = target
                while rest.startswith("super::"):
                    rest = rest[len("super::") :]
                    parts = parts[:-1]
                candidates = ["::".join([*parts, rest])] if parts else []
                # A module that renames a kernel crate (`use antecedent_design as
                # design_crate;`) names its items through the rename.
                segments = rest.split("::")
                for index, segment in enumerate(segments):
                    if segment.endswith("_crate"):
                        crate = f"antecedent_{segment[: -len('_crate')]}"
                        candidates.append("::".join([crate, *segments[index + 1 :]]))
                if len(parts) <= 1:
                    candidates.append(f"{FACADE}::analysis::{rest}")
                for candidate in candidates:
                    if candidate != alias and candidate in known:
                        table[alias] = candidate
                        break

    def _line(self, crate: str, line: str) -> None:
        use = _USE.match(line)
        if use:
            self.alias[crate][use.group(1)] = use.group(2)
            return
        attrs = _ATTRS.match(line).group(1)
        body = line[len(attrs) :]
        fn = _FN.match(body)
        if fn:
            owner = fn.group("owner")
            owner = re.sub(r"<.*$", "", owner) if owner else owner
            name = fn.group("name")
            params, ret = _signature(fn.group("rest"))
            key = f"{owner}::{name}" if owner else fn.group("path") + name
            self.fns.append(
                Fn(
                    key=key,
                    owner=owner,
                    name=name,
                    params=params,
                    ret=ret,
                    hidden="doc(hidden)" in attrs,
                    deprecated="#[deprecated" in attrs,
                    crate=crate,
                )
            )
            return
        typ = _TYPE.match(body)
        if typ:
            self.kinds[typ.group(2)] = typ.group(1)
            return
        field = _FIELD.match(body)
        if field:
            self.fields[field.group(1).split("::")[-1]].append(field.group(2))
            return
        variant = _VARIANT.match(body)
        if variant:
            self.fields[variant.group(1).split("::")[-1]].append(variant.group(2))

    def resolve(self, path: str) -> str:
        """The canonical path of `path`, following item and module re-exports.

        A path can name an item through a re-exported module (`pub use k::m = core::m` makes
        `k::m::Item` name `core::m::Item`), so the longest aliased prefix is rewritten too.
        """
        merged = self._merged
        if merged is None:
            merged = {}
            for table in self.alias.values():
                merged.update(table)
            self._merged = merged
        for _ in range(16):
            if path in merged:
                path = merged[path]
                continue
            parts = path.split("::")
            for cut in range(len(parts) - 1, 0, -1):
                prefix = "::".join(parts[:cut])
                if prefix in merged:
                    path = merged[prefix] + "::" + "::".join(parts[cut:])
                    break
            else:
                return path
        return path


def result_types(seed: list[str], snaps: Snapshots) -> set[str]:
    """The seed closed over public fields and variants (a type holding a result is a result)."""
    closed = set(seed)
    changed = True
    while changed:
        changed = False
        for owner, types in snaps.fields.items():
            if owner not in closed and any(set(_WORD.findall(t)) & closed for t in types):
                closed.add(owner)
                changed = True
    return closed


def _owner_path(owner: str) -> str:
    return re.sub(r"<.*$", "", owner)


def facade_reexports(snaps: Snapshots) -> dict[str, str]:
    """Facade alias -> canonical kernel path, for each `pub use antecedent::P = <kernel>::Q`."""
    out: dict[str, str] = {}
    kernel = {crate for crate in snaps.crates if crate != FACADE}
    for alias, target in snaps.alias.get(FACADE, {}).items():
        if not alias.startswith(FACADE + "::"):
            continue
        canonical = snaps.resolve(target)
        if canonical.split("::")[0] in kernel:
            out[alias] = canonical
    return out


def _returns_result(fn: Fn, results: set[str]) -> bool:
    return bool(set(_WORD.findall(fn.ret)) & results)


def _takes_data(fn: Fn) -> bool:
    return bool(set(_WORD.findall(fn.params)) & DATA_TYPES)


def _executing(fn: Fn, results: set[str]) -> bool:
    return (
        CONTEXT_TYPE in _WORD.findall(fn.params) or _takes_data(fn) or _returns_result(fn, results)
    )


def rust_candidates(snaps: Snapshots, results: set[str]) -> dict[str, Fn]:
    """Candidate key -> function, from the facade re-exports and the facade's own results."""
    found: dict[str, Fn] = {}
    reexports = facade_reexports(snaps)
    canonical = set(reexports.values())
    fn_keys = {fn.key: fn for fn in snaps.fns if fn.crate != FACADE}
    types = {c for c in canonical if snaps.kinds.get(c) in ("struct", "enum", "type", "union")}
    modules = {c for c in canonical if snaps.kinds.get(c) == "mod"}
    # A module re-export carries what the module defines and what it re-exports.
    for table in snaps.alias.values():
        for alias, target in table.items():
            target = snaps.resolve(target)
            for module in modules:
                if alias.startswith(module + "::"):
                    if snaps.kinds.get(target) in ("struct", "enum", "type", "union"):
                        types.add(target)
                    elif target in fn_keys:
                        canonical.add(target)
    for key in list(fn_keys):
        if fn_keys[key].owner is None and any(key.startswith(module + "::") for module in modules):
            canonical.add(key)
    for kind_path, kind in snaps.kinds.items():
        if kind in ("struct", "enum", "type", "union") and any(
            kind_path.startswith(module + "::") for module in modules
        ):
            types.add(kind_path)
    for key in canonical:
        if key in fn_keys and fn_keys[key].owner is None:
            found[key] = fn_keys[key]
    for fn in snaps.fns:
        if fn.crate == FACADE or fn.owner is None:
            continue
        if _owner_path(fn.owner) in types and _executing(fn, results):
            found[fn.key] = fn
    # The facade's own functions: results and deprecated wrappers.
    for fn in snaps.fns:
        if fn.crate != FACADE:
            continue
        if _returns_result(fn, results) or fn.deprecated:
            found[fn.key] = fn
    return found


def registry_text(root: Path) -> str:
    return "\n".join(
        (root / rel).read_text(encoding="utf-8") for rel in REGISTRY_FILES if (root / rel).is_file()
    )


_ROUTE_TABLE = re.compile(r"route_lowerings!\s*\{(.*?)^\}", re.S | re.M)
_ROUTE_ROW = re.compile(r"^\s*(\w+)\s*=>\s*\w+\s+as\s+\w+\s*,", re.M)
_FAMILY_ROW = re.compile(r"^\s*([A-Z]\w*)\s*=>\s*\w+\s*,\s*unit\s", re.M)


# A string-carrying plan route (`Self::CyclicFamily(_) => "CyclicFamily"`): the cyclic families,
# the standalone experiment plans and the propagation chains each lower under one stable name.
_STRING_ROUTE = re.compile(r'Self::\w+\(_\)\s*=>\s*"(\w+)"')


def route_names(registry: str) -> set[str]:
    """The route names of the `route_lowerings!` table, the program-family table and the
    string-carrying plan routes (`CyclicFamily`, `ExperimentPlanning`, `PropagationChain`)."""
    names: set[str] = set()
    for table in _ROUTE_TABLE.findall(registry):
        names |= set(_ROUTE_ROW.findall(table))
    names |= set(_FAMILY_ROW.findall(registry))
    names |= set(_STRING_ROUTE.findall(registry))
    return names


# -- Python ---------------------------------------------------------------------------------


_PYMETHODS = re.compile(r"#\[pymethods\]\s*impl(?:\s*<[^>{]*>)?\s+(\w+)")
_PYCLASS = re.compile(r"#\[pyclass\b([^\]]*)\]\s*(?:#\[[^\]]*\]\s*)*pub(?:\([^)]*\))?\s+(?:struct|enum)\s+(\w+)|#\[pyclass\b([^\]]*)\]\s*(?:#\[[^\]]*\]\s*)*(?:struct|enum)\s+(\w+)")
_PY_NAME = re.compile(r"\bname\s*=\s*\"(\w+)\"")
# How far before a `fn` its attributes are read; the head is cut at the previous item's end, so
# a long `#[pyo3(signature = ...)]` keeps the `#[pyfunction(name = ...)]` above it in view.
HEAD_WINDOW = 4000


def _python_name(raw_head: str) -> str | None:
    """The Python name an item's attributes give it (`#[pyfunction(name = ...)]`,
    `#[pyo3(name = ...)]`), when they rename it."""
    for attribute in re.findall(r"#\[(?:pyfunction|pyo3)\s*\(([^\]]*)\)\]", raw_head):
        found = _PY_NAME.search(attribute)
        if found:
            return found.group(1)
    return None


def _rust_functions(root: Path) -> tuple[dict[str, str], dict[str, str], Counter]:
    """Body text by function name over python/src, and the Rust function behind each name a
    Python caller can reach.

    A `#[pyfunction]` is reached by its Rust name and by the name `#[pyfunction(name = ...)]` or
    `#[pyo3(name = ...)]` gives it. A method of a `#[pymethods]` block is reached as
    `<Class>.<method>`, the Python names of both (`#[pyclass(name = ...)]`,
    `#[pyo3(name = ...)]`); its body is keyed `<RustType>::<rust fn>`.
    """
    bodies: dict[str, str] = defaultdict(str)
    exposed: dict[str, str] = {}
    definitions: Counter = Counter()
    base = root / "python" / "src"
    if not base.is_dir():
        return {}, {}, Counter()
    class_names: dict[str, str] = {}
    sources: list[tuple[str, str]] = []
    for path in sorted(base.rglob("*.rs")):
        raw = path.read_text(encoding="utf-8")
        sources.append((raw, lex(raw).skeleton))
        for match in _PYCLASS.finditer(raw):
            attributes = match.group(1) if match.group(2) else match.group(3)
            rust = match.group(2) or match.group(4)
            renamed = _PY_NAME.search(attributes or "")
            class_names[rust] = renamed.group(1) if renamed else rust
    for raw, skeleton in sources:
        blocks: list[tuple[int, int, str]] = []
        for match in _PYMETHODS.finditer(skeleton):
            brace = skeleton.find("{", match.end())
            if brace >= 0:
                blocks.append((brace, _balanced(skeleton, brace, "{", "}"), match.group(1)))
        for match in re.finditer(r"\bfn\s+(\w+)", skeleton):
            name = match.group(1)
            brace = skeleton.find("{", match.end())
            semi = skeleton.find(";", match.end())
            if brace < 0 or (0 <= semi < brace):
                continue
            end = _balanced(skeleton, brace, "{", "}")
            owner = next((cls for start, stop, cls in blocks if start < match.start() < stop), None)
            key = f"{owner}::{name}" if owner else name
            bodies[key] += " " + skeleton[match.end() : end]
            definitions[name] += 1
            head = skeleton[max(0, match.start() - HEAD_WINDOW) : match.start()]
            last_item = max(head.rfind("}"), head.rfind(";"))
            start = max(0, match.start() - HEAD_WINDOW) + (last_item + 1 if last_item >= 0 else 0)
            head = skeleton[start : match.start()]
            renamed = _python_name(raw[start : match.start()])
            if owner:
                python_class = class_names.get(owner, owner)
                exposed[f"{python_class}.{renamed or name}"] = key
            elif "#[pyfunction" in head:
                exposed[name] = key
                if renamed:
                    exposed[renamed] = key
    return dict(bodies), exposed, definitions


def _side_door_patterns(
    rows: list[dict], disposition: set[str]
) -> dict[str, tuple[str, re.Pattern[str]]]:
    patterns: dict[str, tuple[str, re.Pattern[str]]] = {}
    for row in rows:
        if row.get("disposition") not in disposition:
            continue
        path = row["path"]
        segments = path.split("::")
        leaf = segments[-1]
        owner = _owner_path(segments[-2]) if len(segments) > 1 else ""
        if owner[:1].isupper():
            patterns[path] = (
                leaf,
                re.compile(
                    rf"(?s)(?:\b{re.escape(owner)}\b.*(?:\.|::)\s*{re.escape(leaf)}\s*(?:::<[^>]*>)?\s*\()"
                ),
            )
        else:
            patterns[path] = (
                leaf,
                re.compile(rf"(?<![\w.]){re.escape(leaf)}\s*(?:::<[^>]*>)?\s*\("),
            )
    return patterns


def _direct(row: dict) -> bool:
    return row.get("direct") in (True, "true")


def python_reach(root: Path, rows: list[dict]) -> dict[str, set[str]]:
    """Native binding name -> the direct forwarders it reaches (transitively)."""
    bodies, exposed, definitions = _rust_functions(root)
    # A consumer of a bound form is reached by natives that serve both forms; the Python loader
    # warns through `_retained.admit` when it is called unbound.
    rows = [row for row in rows if _direct(row) and row.get("disposition") == "forwarder"]
    patterns = _side_door_patterns(rows, {"forwarder"})
    direct = {
        name: {k for k, (needle, p) in patterns.items() if needle in text and p.search(text)}
        for name, text in bodies.items()
    }
    # A helper is followed only when its name is defined once, so a common name such as `new`
    # does not join unrelated bindings.
    # A method of a #[pymethods] block is keyed `<Type>::<name>`; a call names it bare.
    keys: dict[str, list[str]] = defaultdict(list)
    for key in bodies:
        keys[key.rsplit("::", 1)[-1]].append(key)
    calls = {
        name: {
            keys[callee][0]
            for callee in re.findall(r"\b(\w+)\s*\(", text)
            if definitions[callee] == 1 and len(keys.get(callee, ())) == 1 and keys[callee][0] != name
        }
        for name, text in bodies.items()
    }
    reach = {name: set(found) for name, found in direct.items()}
    changed = True
    while changed:
        changed = False
        for name, callees in calls.items():
            for callee in callees:
                extra = reach.get(callee, set()) - reach[name]
                if extra:
                    reach[name] |= extra
                    changed = True
    return {
        python: reach[rust] for python, rust in exposed.items() if rust in reach and reach[rust]
    }


@dataclass
class PyFunction:
    key: str
    natives: set[str]
    first_call_deprecated: bool


def _native_aliases(tree: ast.AST) -> tuple[dict[str, str], set[str]]:
    aliases: dict[str, str] = {}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[-1] == "_native":
                for item in node.names:
                    aliases[item.asname or item.name] = item.name
            else:
                for item in node.names:
                    if item.name == "_native":
                        modules.add(item.asname or item.name)
        elif isinstance(node, ast.Import):
            for item in node.names:
                if item.name.split(".")[-1] == "_native":
                    modules.add(item.asname or item.name.split(".")[-1])
    return aliases, modules


def _is_deprecation_call(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return False
    func = node.value.func
    return (isinstance(func, ast.Attribute) and func.attr == "direct_path_deprecated") or (
        isinstance(func, ast.Name) and func.id == "direct_path_deprecated"
    )


def _warns_first(node: ast.AST) -> bool:
    """The first statement warns, or is an `if` whose body warns first.

    The second form is a deprecation scoped to the cases the compiled route serves: the other
    cases keep the direct path, undeprecated, until the route accepts them.
    """
    if _is_deprecation_call(node):
        return True
    return isinstance(node, ast.If) and bool(node.body) and _is_deprecation_call(node.body[0])


def _function_info(node: ast.AST, aliases: dict[str, str], modules: set[str]) -> tuple[set[str], set[str], bool]:
    natives: set[str] = set()
    called: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in aliases:
            natives.add(aliases[child.id])
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            if child.value.id in modules:
                natives.add(child.attr)
            elif child.value.id in aliases:
                # A method of a native class imported from `_native`: `Class.method`.
                natives.add(f"{aliases[child.value.id]}.{child.attr}")
            elif child.value.id == "self":
                called.add(child.attr)
        elif (
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Attribute)
            and isinstance(child.value.value, ast.Name)
            and child.value.value.id in modules
        ):
            # A method of a native class reached through the module: `_native.Class.method`.
            natives.add(f"{child.value.attr}.{child.attr}")
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            called.add(child.func.id)
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    return natives, called, bool(body) and _warns_first(body[0])


def python_functions(root: Path) -> list[PyFunction]:
    def natives_of(
        infos: dict[str, tuple[set[str], set[str], bool]],
        qualname: str,
        cls: str | None,
        seen: set[str],
    ) -> set[str]:
        if qualname in seen or qualname not in infos:
            return set()
        seen.add(qualname)
        natives, called, _ = infos[qualname]
        result = set(natives)
        for callee in called:
            private = callee.startswith("_")
            if cls and f"{cls}.{callee}" in infos and private:
                result |= natives_of(infos, f"{cls}.{callee}", cls, seen)
            elif callee in infos and private:
                result |= natives_of(infos, callee, None, seen)
        return result

    out: list[PyFunction] = []
    package = root / "python" / "antecedent"
    if not package.is_dir():
        return out
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if path.name.startswith("_") or any(part.startswith("_") for part in path.relative_to(package).parts[:-1]):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        aliases, modules = _native_aliases(tree)
        if not aliases and not modules:
            continue
        infos: dict[str, tuple[set[str], set[str], bool]] = {}
        scopes: list[tuple[str, str | None, ast.AST]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scopes.append((node.name, None, node))
            elif isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        scopes.append((member.name, node.name, member))
        for name, cls, node in scopes:
            infos[f"{cls}.{name}" if cls else name] = _function_info(node, aliases, modules)

        for name, cls, _node in scopes:
            if name.startswith("_") or (cls and cls.startswith("_")):
                continue
            qualname = f"{cls}.{name}" if cls else name
            natives = natives_of(infos, qualname, cls, set())
            if natives:
                out.append(PyFunction(f"{rel}::{qualname}", natives, infos[qualname][2]))
    return out


# -- verdict literals -----------------------------------------------------------------------

# A verdict literal is a string literal (the lexer blanks it, so its initializer holds no
# identifier); a constant that reads the generated unit table (`= units_generated::UNIT.as_str()`)
# names no verdict of its own.
_RUST_VERDICT = re.compile(r"\bconst\s+(\w*_CAPABILITY)\s*:\s*&(?:'static\s+)?str\s*=\s*(?!\s*[\w:]*[A-Za-z_])")
# Names of reason details (`external_response_binding.malformed_capability`), not support verdicts.
REASON_DETAILS_RS = "crates/antecedent-core/src/reason_details.rs"
_PY_VERDICT = re.compile(r"^\s*support_status\s*:\s*str\s*=\s*[\"']([^\"']+)[\"']", re.M)


def verdict_literals(root: Path) -> set[str]:
    found: set[str] = set()
    crates = root / "crates"
    if crates.is_dir():
        for path in sorted(crates.rglob("*.rs")):
            rel = path.relative_to(root).as_posix()
            if rel in (GENERATED_VERDICTS_RS, REASON_DETAILS_RS) or "/target/" in rel:
                continue
            raw = path.read_text(encoding="utf-8")
            if "_CAPABILITY" not in raw:
                continue
            for match in _RUST_VERDICT.finditer(lex(raw).skeleton):
                found.add(f"{rel}::{match.group(1)}")
    package = root / "python" / "antecedent"
    if package.is_dir():
        for path in sorted(package.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel == GENERATED_VERDICTS_PY:
                continue
            for match in _PY_VERDICT.finditer(path.read_text(encoding="utf-8")):
                found.add(f"{rel}::{match.group(1)}")
    return found


# -- the gate -------------------------------------------------------------------------------


_CRATE_SOURCES = {
    "antecedent": "crates/antecedent/src",
    "antecedent_io": "crates/antecedent-io/src",
    "antecedent_io_core": "crates/antecedent-io-core/src",
    "antecedent_design": "crates/antecedent-design/src",
}


def bound_form_files(root: Path, path: str) -> list[Path]:
    """Source files that define the bound form `path` names and bind before they read.

    Resolved by the leaf name inside the path's crate, since a canonical path may be a crate-root
    re-export of an item defined deeper.
    """
    import check_retained_artifact_loaders as loaders  # noqa: PLC0415 - sibling script

    segments = path.split("::")
    leaf = segments[-1]
    source = _CRATE_SOURCES.get(segments[0])
    if source is None or not (leaf.endswith("_expecting") or leaf.endswith("_retained")):
        return []
    loader = leaf[: -len("_expecting")] if leaf.endswith("_expecting") else leaf[: -len("_retained")]
    base = root / source
    if not base.is_dir():
        return []
    found: list[Path] = []
    for file in sorted(base.rglob("*.rs")):
        text = file.read_text(encoding="utf-8")
        if f"fn {leaf}" not in text:
            continue
        forms = loaders.bound_forms(loaders._strip_tests(text))
        if leaf in forms and loaders.form_problem(loader, forms[leaf], forms) is None:
            found.append(file)
    return found


def _row_problems(
    where: str,
    row: dict,
    line: str | None,
    registry: str,
    owner_paths: set[str],
    require_snapshot: bool,
    fn: Fn | None = None,
    results: set[str] | None = None,
    root: Path | None = None,
) -> list[str]:
    problems: list[str] = []
    disposition = row.get("disposition")
    if disposition not in DISPOSITIONS:
        return [f"{where}: disposition {disposition!r} is not one of {', '.join(DISPOSITIONS)}"]
    field = DISPOSITION_FIELD.get(disposition)
    if field and not str(row.get(field, "")).strip():
        problems.append(f"{where}: disposition {disposition} requires a non-empty {field!r}")
        return problems
    routes = route_names(registry)
    if disposition == "owner" and row.get("route") and row["route"] not in routes:
        problems.append(f"{where}: route {row['route']!r} is not in the lowering registry")
    if disposition == "forwarder":
        unit = str(row["unit"]).strip()
        if unit.startswith(ROUTE_REPLACEMENT_PREFIX):
            unit = unit[len(ROUTE_REPLACEMENT_PREFIX) :].strip()
        if unit not in routes and unit not in owner_paths:
            problems.append(f"{where}: unit {row['unit']!r} does not name an owner or a route")
        if _direct(row) and require_snapshot and (line is None or "#[deprecated" not in line):
            problems.append(f"{where}: direct forwarder, but the snapshot shows no #[deprecated]")
        if not _direct(row) and fn is not None and fn.crate == FACADE:
            problems.append(
                f"{where}: a facade item forwards only as a #[deprecated] direct entry "
                "(direct = true); otherwise it is the owner"
            )
    deprecated = require_snapshot and line is not None and "#[deprecated" in line
    if deprecated and not (
        (disposition == "forwarder" and _direct(row)) or (disposition == "consumer" and row.get("of"))
    ):
        problems.append(
            f"{where}: #[deprecated] in the snapshot; a deprecated entry is a direct forwarder "
            "(direct = true) or a consumer that names its bound form (of)"
        )
    if disposition == "consumer" and fn is not None and _takes_data(fn):
        problems.append(f"{where}: a consumer reads a sealed artifact, but this takes a data value")
    if disposition == "not_scientific" and fn is not None and _returns_result(fn, results or set()):
        problems.append(f"{where}: returns a result type, so it is an owner, a forwarder or a consumer")
    if disposition == "consumer" and row.get("of"):
        target = str(row["of"])
        if target.startswith(BOUND_REPLACEMENT_PREFIX):
            target = target[len(BOUND_REPLACEMENT_PREFIX) :].strip()
        if not bound_form_files(ROOT if root is None else root, target):
            problems.append(
                f"{where}: of {row['of']!r} does not name a bound form that binds "
                "the bytes to a RetainedArtifactIdentity before reading them"
            )
    return problems


def compiled_path_problems(root: Path = ROOT) -> list[str]:
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        return [f"{MANIFEST}: missing"]
    manifest = load_toml(manifest_path)
    rows = manifest.get("row", [])
    python_rows = manifest.get("python", [])
    problems: list[str] = []
    if "verdict" in manifest:
        problems.append(f"{MANIFEST}: [[verdict]] waivers are forbidden")

    snaps = Snapshots(root)
    results = result_types(manifest.get("result_types", {}).get("seed", []), snaps)
    candidates = rust_candidates(snaps, results)
    registry = registry_text(root)

    by_path: dict[str, dict] = {}
    for row in rows:
        path = row.get("path", "")
        if path in by_path:
            problems.append(f"{MANIFEST}: duplicate row {path!r}")
        by_path[path] = row
    owner_paths = {p for p, r in by_path.items() if r.get("disposition") == "owner"}

    for path, fn in sorted(candidates.items()):
        row = by_path.get(path)
        if row is None:
            problems.append(
                f"{path}: unclassified; claim is owed "
                f"(name an owner, a forwarder, or a consumer in {MANIFEST})"
            )
            continue
        # The snapshot line carries the attributes; rebuild just what the rows check.
        line = ("#[doc(hidden)] " if fn.hidden else "") + ("#[deprecated] " if fn.deprecated else "")
        problems += _row_problems(path, row, line, registry, owner_paths, True, fn, results, root)
    for path in sorted(set(by_path) - set(candidates)):
        problems.append(f"{MANIFEST}: stale row {path!r}; no candidate matches it")

    # Python.
    reach = python_reach(root, rows)
    py_by_key: dict[str, dict] = {}
    for row in python_rows:
        key = row.get("path", "")
        if key in py_by_key:
            problems.append(f"{MANIFEST}: duplicate python row {key!r}")
        py_by_key[key] = row
    py_candidates: dict[str, PyFunction] = {}
    for function in python_functions(root):
        reached: set[str] = set()
        for native in function.natives:
            reached |= reach.get(native, set())
        if reached:
            py_candidates[function.key] = function
            direct_reached = any(_direct(by_path[p]) for p in reached if p in by_path)
            row = py_by_key.get(function.key)
            if row is None:
                problems.append(
                    f"{function.key}: unclassified Python wrapper; claim is owed; its native reaches "
                    f"{sorted(reached)[0]!r} (name an owner, a forwarder, or a consumer)"
                )
                continue
            problems += _row_problems(function.key, row, None, registry, owner_paths, False)
            if _direct(row):
                if not function.first_call_deprecated:
                    problems.append(
                        f"{function.key}: deprecated, but it does not call "
                        "_deprecation.direct_path_deprecated( first"
                    )
            elif direct_reached:
                problems.append(
                    f"{function.key}: reaches a direct forwarder, so it must be a forwarder "
                    "and call _deprecation.direct_path_deprecated("
                )
    for key in sorted(set(py_by_key) - set(py_candidates)):
        problems.append(f"{MANIFEST}: stale python row {key!r}; no wrapper reaches it")
    if py_by_key and not (root / DEPRECATION_MODULE).is_file():
        problems.append(f"{DEPRECATION_MODULE}: missing")

    # Verdict literals.
    for key in sorted(verdict_literals(root)):
        problems.append(
            f"{key}: a verdict literal outside the generated tables (units_generated.rs / "
            "_direct_status.py); read the support model instead"
        )
    return problems


def candidate_counts(root: Path = ROOT) -> Counter:
    manifest = load_toml(root / MANIFEST)
    counts: Counter = Counter()
    for section in ("row", "python"):
        for row in manifest.get(section, []):
            counts[(section, row.get("disposition"))] += 1
    return counts


def main(argv: list[str]) -> int:
    if "--list" in argv:
        for (section, disposition), count in sorted(candidate_counts().items(), key=str):
            print(f"{section:8} {disposition:15} {count}")
        return 0
    problems = compiled_path_problems()
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        print(f"gate_compiled_path: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("gate_compiled_path: every candidate is classified")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
