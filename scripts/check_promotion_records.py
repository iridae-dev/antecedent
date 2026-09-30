"""Validate the 2.2 promotion records (parity/promotion_2_2.toml).

A record freezes a cell's scientific surface before implementation. A route is
licensed only when its record carries the evidence for it: a promoted record
cites executed positive, negative and artifact fixtures (and budget/calibration
evidence where its search or interval claims require them); an in_progress
record may license a non-uncertainty point_only/none route once those same
fixture roles cite executed evidence. Once a record is in_progress, its closed
routes cite executed runtime-refusal tests, its refusal details equal the
namespaced detail literals in non-test source, its declared public surface is
covered by its routes, and its search is metered by the shared SearchBudget.
Every route has a row in its owning registry that agrees with the record.

    python3 scripts/check_promotion_records.py [registry.toml] [--emit-evidence out.toml]

--emit-evidence writes every fixture and closed-route refusal that cites a test as
a [[fixture_evidence]] row, so scripts/gate_promotion.sh can execute it with
run_evidence_rows.py.

Environment overrides (gate self-tests only):
  PROMOTION_TRANSPORT_STAGES   owning registry for transport_stages routes
  PROMOTION_SUPPORT_LICENSED   owning registry for licensed support_licensed routes
  PROMOTION_SUPPORT_CLOSED     owning registry for closed support_licensed routes
  PROMOTION_EXTRA_SOURCES      os.pathsep-separated extra files scanned as non-test
                               source for refusal-detail literals
"""

import ast
import os
import re
import sys
from pathlib import Path

import tomllib
from test_evidence import (
    _is_cfg_test,
    closure,
    resolve_python_test,
    resolve_rust_test,
    rust_items,
    target_modules,
)

root = Path(__file__).resolve().parents[1]
args = sys.argv[1:]
emit_path = None
if "--emit-evidence" in args:
    at = args.index("--emit-evidence")
    emit_path = Path(args[at + 1])
    del args[at : at + 2]
registry_path = Path(args[0]) if args else root / "parity/promotion_2_2.toml"
registry = tomllib.loads(registry_path.read_text())

FROZEN = (
    "id", "workstream", "milestone", "work_package", "status", "consumer_question",
    "theorem", "reference", "guarantee", "graph_class", "population_semantics",
    "evidence_family", "provider", "estimand", "inference_claim", "owners",
    "identity_inputs", "wire_changes", "compatibility", "refusals", "routes", "fixtures",
)
STATUSES = {"frozen", "in_progress", "promoted", "carried_forward"}
# Statuses at which implementation exists, so refusals, details, surface and
# search metering are checked against code.
IMPLEMENTED = {"in_progress", "promoted"}
# "nominal" is deliberately absent: no 2.2 interval ships unmeasured.
CLAIMS = {"point_only", "calibrated", "structural_envelope", "assumption_range", "none"}
# The only route claims an in_progress record may license ahead of promotion.
EARLY_CLAIMS = {"point_only", "none"}
NOMINAL = "estimator_grid_not_measured"
STAGES = {"identify", "representation", "evaluate", "uncertainty", "prepare", "consume"}
REGISTRIES = {"transport_stages", "support_licensed"}
SEARCH_LIMITS = ("operation_limit", "depth_limit", "memory_limit")
# The one bounded-search contract (crates/antecedent-core/src/search.rs): it
# cannot be built without limits and observes cancellation and memory per charge.
SEARCH_CONTRACT = "antecedent_core::SearchBudget"
# Symbols whose appearance in a budget fixture shows it exercises that contract.
SEARCH_SYMBOLS = ("SearchBudget", "SearchReceipt", "SearchStop")
# Non-test source scanned for refusal-detail literals.
SOURCE_GLOBS = ("crates/*/src/**/*.rs", "python/src/**/*.rs", "python/antecedent/**/*.py")

codes = {
    row["id"]: row
    for row in tomllib.loads((root / "parity/reason_codes.toml").read_text())["code"]
}
runtime_codes = {cid for cid, row in codes.items() if "runtime_refusal" in row.get("applies_to", [])}
stages_path = Path(os.environ.get("PROMOTION_TRANSPORT_STAGES", root / "parity/transport_stages.toml"))
support_path = Path(os.environ.get("PROMOTION_SUPPORT_LICENSED", root / "parity/support_licensed.toml"))
support_closed_path = Path(os.environ.get("PROMOTION_SUPPORT_CLOSED", root / "parity/support_closed.toml"))
extra_sources = [Path(p) for p in os.environ.get("PROMOTION_EXTRA_SOURCES", "").split(os.pathsep) if p]
stage_routes = {
    row.get("route"): row for row in tomllib.loads(stages_path.read_text()).get("routes", [])
}
support_cells = tomllib.loads(support_path.read_text()).get("cell", [])
# The support axes cannot name a contrast, so a closed sub-cell contrast owned by
# a 2.2 record is a [[closed_contrast]] row (the support-matrix gates read only
# [[closed]]).
support_closed_contrasts = tomllib.loads(support_closed_path.read_text()).get("closed_contrast", [])
coverage_ids = {
    row.get("id")
    for row in tomllib.loads((root / "parity/coverage_records.toml").read_text()).get("record", [])
}

errors: list[str] = []
if registry.get("version") != 1 or registry.get("release") != "2.2":
    errors.append("promotion registry requires version 1 and release 2.2")


def resolve(path: str, assertion: str) -> list[str]:
    if path.endswith(".rs"):
        return resolve_rust_test(root / path, assertion)[1]
    if path.endswith(".py"):
        return resolve_python_test(root / path, assertion)
    return [f"{path}: evidence must be a collected Rust or Python test"]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def carries(value, needle: str) -> bool:
    """Whether any string anywhere in a TOML value contains `needle`."""
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(carries(v, needle) for v in value.values())
    if isinstance(value, list):
        return any(carries(v, needle) for v in value)
    return False


# ------------------------------------------------------------- non-test source


_crate_modules: dict[Path, dict[Path, bool]] = {}


def _crate_src(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name == "src":
            return parent
    return None


def _compiled_test_only(path: Path) -> bool | None:
    """True if every target compiling `path` does so only under cfg(test); None if
    no target of its crate compiles it."""
    src = _crate_src(path)
    if src is None:
        return None
    if src not in _crate_modules:
        merged: dict[Path, bool] = {}
        roots = [src / "lib.rs", src / "main.rs", *sorted((src / "bin").glob("*.rs"))]
        for top in roots:
            if top.is_file():
                for mod, test_only in target_modules(top.resolve()).items():
                    merged[mod] = merged.get(mod, True) and test_only
        _crate_modules[src] = merged
    return _crate_modules[src].get(path.resolve())


_nontest_cache: dict[Path, str] = {}


def nontest_rust(path: Path, *, in_crate: bool = True) -> str:
    """Comment-free source of `path` with #[cfg(test)] modules/functions and
    #[test] functions blanked; empty when the file is compiled only for tests or
    not compiled at all. String literals are kept."""
    key = path.resolve()
    if key in _nontest_cache:
        return _nontest_cache[key]
    text = ""
    if in_crate:
        test_only = _compiled_test_only(path)
        compiled = test_only is False
    else:
        compiled = True
    if compiled:
        masked, items = rust_items(key)
        if not re.search(r"#!\[cfg\(\s*test\s*\)\]", masked.code):
            code = list(masked.code)
            for item in items:
                if item.kind not in ("fn", "mod"):
                    continue
                if _is_cfg_test(item.attrs) or any(re.sub(r"\s+", "", a) == "#[test]" for a in item.attrs):
                    for k in range(item.start, item.body[1]):
                        if code[k] != "\n":
                            code[k] = " "
            text = "".join(code)
    _nontest_cache[key] = text
    return text


_RUST_STR = re.compile(r'(?<![\w#])"((?:\\.|[^"\\])*)"')


def rust_literals(path: Path, *, in_crate: bool = True) -> list[tuple[int, str]]:
    code = nontest_rust(path, in_crate=in_crate)
    return [(code.count("\n", 0, m.start()) + 1, m.group(1)) for m in _RUST_STR.finditer(code)]


def python_literals(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        return []
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def detail_literals(namespaces: set[str]) -> dict[str, dict[str, list[str]]]:
    """namespace -> detail -> ["file:line", ...] for every quoted literal in
    non-test source that is exactly `<ns>.<snake>` or starts `<ns>.<snake>:`
    (the "detail: message" convention)."""
    found: dict[str, dict[str, list[str]]] = {ns: {} for ns in namespaces}
    if not namespaces:
        return found
    shape = re.compile(r"^(" + "|".join(map(re.escape, sorted(namespaces))) + r")\.([a-z0-9_]+)(?::|$)")
    raw_hint = re.compile("|".join(re.escape(ns + ".") for ns in sorted(namespaces)))
    files = [(p, True) for pattern in SOURCE_GLOBS for p in sorted(root.glob(pattern))]
    files += [(p, False) for p in extra_sources]
    for path, in_crate in files:
        if not path.is_file() or not raw_hint.search(path.read_text(errors="ignore")):
            continue
        if path.suffix == ".py":
            literals = python_literals(path)
        else:
            literals = rust_literals(path, in_crate=in_crate)
        for line, value in literals:
            m = shape.match(value)
            if m:
                found[m.group(1)].setdefault(f"{m.group(1)}.{m.group(2)}", []).append(f"{rel(path)}:{line}")
    return found


def surface_symbols(path: Path) -> dict[str, str]:
    """Public top-level names mapped to their kind ("type" for a class/struct/enum,
    "function", or "other"): `__all__` if the Python file defines it, else defs and
    classes without a leading underscore; top-level pub fn/struct/enum/trait for a
    Rust facade."""
    if path.suffix == ".py":
        tree = ast.parse(path.read_text(errors="ignore"))
        kinds = {
            node.name: "type" if isinstance(node, ast.ClassDef) else "function"
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets) and isinstance(
                node.value, (ast.List, ast.Tuple)
            ):
                return {
                    e.value: kinds.get(e.value, "other")
                    for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                }
        return {name: kind for name, kind in kinds.items() if not name.startswith("_")}
    code = nontest_rust(path, in_crate=_crate_src(path) is not None)
    return {
        name: "function" if item == "fn" else "type" if item in ("struct", "enum") else "other"
        for item, name in re.findall(r"^pub\s+(fn|struct|enum|trait)\s+([A-Za-z_]\w*)", code, re.M)
    }


def camel(snake: str) -> str:
    return "".join(part.title() for part in snake.split("_"))


# ------------------------------------------------------------------- records


seen_records: set[str] = set()
seen_fixtures: set[str] = set()
seen_routes: set[str] = set()
evidence_rows: list[tuple[str, str, str]] = []
records = registry.get("record", [])
namespaces_in_use = {
    refusal.get("detail", "").split(".")[0]
    for rec in records
    if rec.get("status") in IMPLEMENTED
    for refusal in rec.get("refusals") or []
    if "." in refusal.get("detail", "")
}
code_details = detail_literals(namespaces_in_use)

for rec in records:
    rid = rec.get("id", "?")
    if rid in seen_records:
        errors.append(f"{rid}: duplicate record id")
    seen_records.add(rid)
    for key in FROZEN:
        value = rec.get(key)
        if value is None or (isinstance(value, (str, list)) and not value):
            errors.append(f"{rid}: missing frozen field {key}")
    status = rec.get("status")
    if status not in STATUSES:
        errors.append(f"{rid}: unknown status {status!r}")
    promoted = status == "promoted"
    implemented = status in IMPLEMENTED
    claim = rec.get("inference_claim")
    if claim not in CLAIMS:
        errors.append(f"{rid}: inference_claim {claim!r} is not a 2.2 claim ({', '.join(sorted(CLAIMS))})")
    cov = rec.get("coverage_records") or []
    if claim == "calibrated" and not cov:
        errors.append(f"{rid}: a calibrated interval must allocate its coverage record ids")
    if claim != "calibrated" and cov:
        errors.append(f"{rid}: coverage records allocated for a {claim} claim")
    if promoted:
        errors.extend(f"{rid}: unknown coverage record {cid}" for cid in cov if cid not in coverage_ids)

    # Bounded computation is mandatory on every new search.
    bounds = rec.get("bounds") or {}
    if bounds.get("cancellation") is not True:
        errors.append(f"{rid}: bounds.cancellation must be true")
    if rec.get("search") is True:
        for key in SEARCH_LIMITS:
            if not bounds.get(key):
                errors.append(f"{rid}: a bounded search must declare bounds.{key}")
        if bounds.get("contract") != SEARCH_CONTRACT:
            errors.append(f"{rid}: a bounded search must run under {SEARCH_CONTRACT}")
    elif rec.get("search") is not False:
        errors.append(f"{rid}: search must be declared true or false")

    # The search is metered by the shared contract in code, not only pre-flighted:
    # every declared implementation file names SearchBudget and together they charge it.
    search_impl = rec.get("search_impl")
    if search_impl is not None and (
        not isinstance(search_impl, list) or not all(isinstance(p, str) and p for p in search_impl)
    ):
        errors.append(f"{rid}: search_impl must be a list of source paths")
        search_impl = []
    if rec.get("search") is True and implemented and not search_impl:
        errors.append(
            f"{rid}: a search at status {status} must declare search_impl = [<.rs files>] whose non-test "
            f"source builds a SearchBudget and calls .charge( on it"
        )
    charged = False
    for src in search_impl or []:
        path = root / src
        if not path.is_file() or path.suffix != ".rs":
            errors.append(f"{rid}: search_impl {src} is not a Rust source file")
            continue
        code = nontest_rust(path, in_crate=_crate_src(path) is not None)
        if "SearchBudget" not in code:
            errors.append(
                f"{rid}: search_impl {src} non-test source never names SearchBudget; "
                f"every declared implementation file runs under {SEARCH_CONTRACT}"
            )
        charged = charged or ".charge(" in code
    if search_impl and not charged:
        errors.append(
            f"{rid}: search_impl files lack .charge(; the search must charge {SEARCH_CONTRACT} "
            f"per step, not only pre-flight it"
        )

    # A registered top-level code plus a unique, namespaced detail code: callers
    # switch on the pair, so neither may be prose or collide inside a record.
    details: set[str] = set()
    for refusal in rec.get("refusals") or []:
        code, detail = refusal.get("code"), refusal.get("detail", "")
        if code not in runtime_codes:
            errors.append(f"{rid}: refusal code {code!r} is not a registered runtime_refusal code")
        if not refusal.get("when"):
            errors.append(f"{rid}: refusal {code!r} needs its condition")
        parts = detail.split(".")
        if len(parts) != 2 or not all(part.replace("_", "").isalnum() and part.islower() for part in parts):
            errors.append(f"{rid}: refusal detail {detail!r} must be <namespace>.<snake_case>")
        if detail in details:
            errors.append(f"{rid}: duplicate refusal detail {detail}")
        details.add(detail)
    namespaces = {d.split(".")[0] for d in details}
    if len(namespaces) > 1:
        errors.append(f"{rid}: refusal details must share one namespace")

    # Refusal boundary == code: once implemented, each declared detail is a
    # literal in non-test source and each namespaced literal there is declared.
    if implemented and len(namespaces) == 1:
        ns = next(iter(namespaces))
        in_code = code_details.get(ns, {})
        for detail in sorted(details - set(in_code)):
            errors.append(
                f"{rid}: refusal detail {detail} is not emitted by non-test source "
                f"(no \"{detail}\" or \"{detail}: ...\" literal in {', '.join(SOURCE_GLOBS)})"
            )
        for detail in sorted(set(in_code) - details):
            errors.append(
                f"{rid}: non-test source emits undeclared refusal detail {detail} "
                f"at {', '.join(in_code[detail])}; add it to the record's refusals or remove it"
            )

    # Fixtures first: a route's license depends on which roles cite executed evidence.
    workstream = str(rec.get("workstream", "")).lower()
    roles: set[str] = set()
    evidenced: set[str] = set()
    for fixture in rec.get("fixtures") or []:
        fid, role = fixture.get("id", ""), fixture.get("role")
        if fid in seen_fixtures:
            errors.append(f"{rid}: duplicate fixture id {fid}")
        seen_fixtures.add(fid)
        if not fid.startswith(f"{workstream}.") or not fid.endswith(f".{role}"):
            errors.append(f"{rid}: fixture {fid} must be <workstream>.<name>.<role>")
        if not fixture.get("intent"):
            errors.append(f"{rid}: fixture {fid} needs an intent")
        roles.add(role)
        path, assertion = fixture.get("evidence_test", ""), fixture.get("evidence_assertion", "")
        if bool(path) != bool(assertion):
            errors.append(f"{rid}: fixture {fid} needs both evidence_test and evidence_assertion")
        elif path:
            problems = resolve(path, assertion)
            errors.extend(f"{rid}: {fid}: {p}" for p in problems)
            evidence_rows.append((fid, path, assertion))
            ok = not problems
            if role == "budget" and rec.get("search") is True and ok:
                body = closure(root / path, assertion)
                if not any(symbol in body for symbol in SEARCH_SYMBOLS):
                    errors.append(f"{rid}: budget fixture {fid} does not exercise {SEARCH_CONTRACT}")
                    ok = False
            if ok:
                evidenced.add(role)
        elif promoted:
            errors.append(f"{rid}: promoted record lacks executed evidence for fixture {fid}")
    required = {"positive", "negative", "artifact"}
    if rec.get("search") is True:
        required.add("budget")
    if claim == "calibrated":
        required.add("calibration")
    missing = sorted(required - roles)
    if missing:
        errors.append(f"{rid}: missing fixture roles {', '.join(missing)}")
    # Roles an in_progress record must evidence before licensing any route.
    early_roles = {"positive", "negative", "artifact"} | ({"budget"} if rec.get("search") is True else set())
    unevidenced = sorted(early_roles - evidenced)

    route_names: list[str] = []
    for route in rec.get("routes") or []:
        name = route.get("name")
        if not name or name in seen_routes:
            errors.append(f"{rid}: duplicate or missing route name {name!r}")
        seen_routes.add(name)
        route_names.append(name or "")
        stage = route.get("stage")
        if stage not in STAGES:
            errors.append(f"{rid}: {name}: invalid stage {stage!r}")
        owner = route.get("registry", "transport_stages")
        if owner not in REGISTRIES:
            errors.append(f"{rid}: {name}: unknown owning registry {owner!r}")
        permanent = route.get("permanent_in_release") is True
        route_claim = route.get("claim")
        if route_claim is not None and route_claim not in CLAIMS:
            errors.append(f"{rid}: {name}: claim {route_claim!r} is not a 2.2 claim")
        route_status = route.get("status")
        licensed = route_status == "licensed"
        if route_status == "closed":
            reason = route.get("reason_code")
            if reason not in runtime_codes:
                errors.append(f"{rid}: {name}: closed route needs a registered runtime reason_code")
            # A closed route of implemented work must be shown to refuse at runtime.
            rpath, rassert = route.get("refusal_test", ""), route.get("refusal_assertion", "")
            if bool(rpath) != bool(rassert):
                errors.append(f"{rid}: {name}: closed route needs both refusal_test and refusal_assertion")
            elif rpath:
                problems = resolve(rpath, rassert)
                errors.extend(f"{rid}: {name}: refusal evidence: {p}" for p in problems)
                evidence_rows.append((f"{name}.refusal", rpath, rassert))
                if not problems and reason:
                    body = closure(root / rpath, rassert)
                    if reason not in body and camel(reason) not in body:
                        errors.append(
                            f"{rid}: {name}: refusal test {rpath}::{rassert} never names the "
                            f"route's reason_code {reason} (or {camel(reason)})"
                        )
            elif implemented:
                errors.append(
                    f"{rid}: {name}: closed route of a record at status {status} needs refusal_test + "
                    f"refusal_assertion citing a test that calls it and observes {reason}"
                )
        elif licensed:
            if permanent:
                errors.append(f"{rid}: {name}: licensed but marked permanent_in_release (closed)")
            elif promoted:
                pass
            elif status == "in_progress":
                if stage == "uncertainty":
                    errors.append(f"{rid}: {name}: uncertainty route licensed before its record is promoted")
                if route_claim not in EARLY_CLAIMS:
                    errors.append(
                        f"{rid}: {name}: licensed before promotion requires claim = \"point_only\" "
                        f"or \"none\" on the route (got {route_claim!r})"
                    )
                if unevidenced:
                    errors.append(
                        f"{rid}: {name}: licensed before promotion but the record cites no executed "
                        f"evidence for fixture role(s) {', '.join(unevidenced)}"
                    )
            else:
                errors.append(f"{rid}: {name}: licensed while its record is {status}")
        else:
            errors.append(f"{rid}: {name}: status must be closed or licensed")

        # Owning registry must hold a row that agrees with the record.
        owning_rows: list[dict] = []
        if owner == "transport_stages":
            row = stage_routes.get(name)
            if row is None:
                errors.append(
                    f"{rid}: {name} has no row in transport_stages.toml; add [[routes]] route = \"{name}\", "
                    f"stage = \"{stage}\", status = \"{'licensed' if licensed else 'closed'}\""
                    + ("" if licensed else f", reason_code = \"{route.get('reason_code')}\"")
                )
            else:
                owning_rows.append(row)
                there = row.get("status")
                if row.get("stage") != stage:
                    errors.append(f"{rid}: {name}: transport_stages.toml stage {row.get('stage')!r} != record stage {stage!r}")
                if there == "licensed" and not licensed:
                    errors.append(f"{rid}: {name} is licensed in transport_stages.toml but closed in its record")
                elif there != "licensed" and licensed:
                    errors.append(f"{rid}: {name} is licensed in its record but not in transport_stages.toml")
                elif there == "closed" and row.get("reason_code") != route.get("reason_code"):
                    errors.append(
                        f"{rid}: {name}: transport_stages.toml reason_code {row.get('reason_code')!r} "
                        f"!= record reason_code {route.get('reason_code')!r}"
                    )
            if promoted and not permanent and not licensed:
                errors.append(f"{rid}: promoted record leaves non-permanent route {name} closed")
        elif owner == "support_licensed":
            query, contrast = route.get("query"), route.get("contrast")
            if not query or not contrast:
                errors.append(f"{rid}: {name}: a support_licensed route names its query and contrast")
            cells = [c for c in support_cells if c.get("query") == query and c.get("contrast") == contrast]
            closed_rows = [
                c for c in support_closed_contrasts if c.get("query") == query and c.get("contrast") == contrast
            ]
            owning_rows.extend(cells + closed_rows)
            if cells and not licensed:
                errors.append(f"{rid}: {name} is licensed in support_licensed.toml but closed in its record")
            if licensed and not cells:
                errors.append(f"{rid}: {name} is licensed in its record but has no support_licensed.toml cell")
            if closed_rows and licensed:
                errors.append(f"{rid}: {name} is licensed in its record but still closed in support_closed.toml")
            if not licensed and not closed_rows:
                errors.append(
                    f"{rid}: {name} has no support_closed.toml row; add [[closed_contrast]] query = \"{query}\", "
                    f"contrast = \"{contrast}\", reason_code = \"{route.get('reason_code')}\", record = \"{rid}\""
                )
            for row in closed_rows:
                if row.get("record") != rid:
                    errors.append(f"{rid}: {name}: support_closed.toml row names record {row.get('record')!r}")
                if row.get("reason_code") != route.get("reason_code"):
                    errors.append(
                        f"{rid}: {name}: support_closed.toml reason_code {row.get('reason_code')!r} "
                        f"!= record reason_code {route.get('reason_code')!r}"
                    )
            if promoted and not permanent and not licensed:
                errors.append(f"{rid}: promoted record leaves non-permanent route {name} closed")

        # No nominal-only interval ships from implemented 2.2 work, anywhere.
        if implemented:
            if licensed and carries(route, NOMINAL):
                errors.append(f"{rid}: licensed route {name} carries {NOMINAL}")
            for row in owning_rows:
                if carries(row, NOMINAL):
                    errors.append(f"{rid}: {name}: its {owner} row carries {NOMINAL}")

    # Route inventory: every public symbol of a declared surface file is a
    # component of some route name in this record, or a value/descriptor type
    # listed in surface_values (classes/structs/enums only, never stale).
    surface = rec.get("surface")
    values = rec.get("surface_values")
    if values is not None and (not isinstance(values, list) or not all(isinstance(v, str) and v for v in values)):
        errors.append(f"{rid}: surface_values must be a list of symbol names")
        values = []
    values = values or []
    if values and surface is None:
        errors.append(f"{rid}: surface_values {', '.join(values)} listed without a surface")
    if surface is not None:
        if not isinstance(surface, list) or not all(isinstance(p, str) and p for p in surface):
            errors.append(f"{rid}: surface must be a list of source paths")
            surface = []
        components = {part for name in route_names for part in name.split(".")}
        declared: dict[str, tuple[str, str]] = {}
        for src in surface:
            path = root / src
            if not path.is_file() or path.suffix not in (".py", ".rs"):
                errors.append(f"{rid}: surface {src} is not a Python or Rust source file")
                continue
            for symbol, kind in surface_symbols(path).items():
                declared.setdefault(symbol, (kind, src))
                if symbol not in components and symbol not in values:
                    errors.append(
                        f"{rid}: surface symbol {symbol} ({src}) is not a component of any route name; "
                        + (
                            "add it to surface_values if it is a pure value/descriptor type, "
                            if kind == "type"
                            else ""
                        )
                        + "add a route for it (closed until evidenced), or make it private"
                    )
        for value in values:
            if value not in declared:
                errors.append(
                    f"{rid}: surface_values entry {value} is not a public symbol of any surface file "
                    f"({', '.join(surface)}); remove the stale exemption"
                )
            elif declared[value][0] != "type":
                errors.append(
                    f"{rid}: surface_values entry {value} ({declared[value][1]}) is a {declared[value][0]}, "
                    f"not a class/struct/enum; an entry point must be a route"
                )

if emit_path is not None:
    emit_path.write_text(
        "".join(
            f'[[fixture_evidence]]\nid = "{fid}"\nevidence_test = "{path}"\nevidence_assertion = "{assertion}"\n\n'
            for fid, path, assertion in evidence_rows
        )
    )
if errors:
    print("Promotion record gate FAILED:\n" + "\n".join(f" - {e}" for e in errors))
    sys.exit(1)
promoted_count = sum(1 for rec in records if rec.get("status") == "promoted")
licensed_count = sum(
    1 for rec in records for route in rec.get("routes") or [] if route.get("status") == "licensed"
)
print(
    f"Promotion records OK ({len(records)} records, {promoted_count} promoted; "
    f"{len(seen_routes)} routes, {licensed_count} licensed)"
)
