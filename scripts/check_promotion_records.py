"""Validate a versioned promotion registry (2.2 or 2.3).

A record freezes a cell's scientific surface before implementation. A route is
licensed only when its record carries the evidence for it: a promoted record
cites executed positive, negative and artifact fixtures (and budget/calibration
evidence where its search or interval claims require them); an in_progress
record may license a non-uncertainty point_only/none route once those same
fixture roles cite executed evidence. Once a record is in_progress, its closed
routes cite executed runtime-refusal tests, its refusal details equal the
namespaced detail literals in non-test source (raw, byte, f-string and continued
literals included; dynamically built details are refused), each (code, detail)
pair is emitted by live code that names the code, its declared public surface is
covered by its routes, and its search is metered by the shared SearchBudget
inside a loop. Every cited test must assert something, and one test backs one
fixture role. Every route has a row in its owning registry that agrees with the
record.

    python3 scripts/check_promotion_records.py [registry.toml] [--emit-evidence out.toml]
    python3 scripts/check_promotion_records.py --list-rules
    python3 scripts/check_promotion_records.py [registry.toml] --suggest

--emit-evidence writes every fixture and closed-route refusal that cites a test as
a [[fixture_evidence]] row, so scripts/gate_promotion.sh can execute it with
run_evidence_rows.py. --list-rules prints every rule id; each error message ends
with its `[rule-id]`, and scripts/promotion_selftest.py requires a self-test case
tagged with every id. --suggest lists, per record, the public Rust/pyo3/export
symbols found in its likely source files that no route, surface_values or
surface_internal entry covers (the fields a record would need to declare).

Optional record fields (in addition to the frozen ones):
  surface           python/rust facade files; top-level public names need a route
  surface_rust      rust files: every `pub fn/struct/enum/trait`, including methods of
                    inherent impls of public types, needs a route component,
                    surface_values (types) or surface_internal
  surface_values    [type names]: struct/enum symbols of a surface file that are data,
                    not routes. A listed type also covers the INHERENT methods and
                    associated functions of its `impl Type` blocks in the scanned
                    files, except methods starting with run/execute/evaluate/decide/
                    identify/prepare/estimate/consume/fit (surface_value_method_verb:
                    those, free functions and pyo3 items still need a route or
                    surface_internal)
  surface_pyo3      rust files: every #[pyfunction]/#[pyclass]/#[pymethods] fn likewise;
                    a #[pymethods] fn of a #[pyclass] listed in surface_values is
                    covered like a Rust inherent method (getters such as interval/plan
                    included), except executing-verb names (the Rust verbs plus
                    refresh and export), which need a route or surface_internal
  route `covers`    on a route: ["symbol", "Type.method", ...] extra public symbols
                    of the record's surface files that route accounts for, counted
                    exactly like a route-name component (route_covers_stale if an
                    entry is no public symbol; a warning is printed when the name
                    appears in no cited evidence closure nor the route's entry-point
                    file). Route names stay meaningful identifiers.
  surface_exports   python files whose `__all__` names listed in owned_exports are
                    covered likewise; owned_exports = names this workstream owns
  surface_internal  [{ name = "...", why = "..." }]: exempt Rust/pyo3 items that
                    must be #[doc(hidden)] (or an underscore pyo3 name)
  shared_evidence   true on a fixture: it may share its test with another role
                    (every fixture citing that test must set it)

Environment overrides (gate self-tests and analysis only):
  PROMOTION_TRANSPORT_STAGES   owning registry for transport_stages routes
  PROMOTION_SUPPORT_LICENSED   owning registry for licensed support_licensed routes
  PROMOTION_SUPPORT_CLOSED     owning registry for closed support_licensed routes
  PROMOTION_EXTRA_SOURCES      os.pathsep-separated extra files scanned as non-test
                               source for refusal-detail literals
  PROMOTION_DISABLE_RULES      comma-separated rule ids to switch off (mutation check)
  PROMOTION_SYNTHETIC_EVIDENCE 1: evidence files outside the repo are resolved
                               statically (self-test synthetic tests)
  PROMOTION_EVIDENCE_CACHE     directory caching cargo/pytest listing results
  PROMOTION_STATIC_ONLY        1: resolve evidence statically, without cargo or pytest
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import tomllib
from promotion_source import (
    Symbol,
    assertion_problems,
    assertion_texts,
    charge_in_loop,
    closure_code,
    crate_src,
    finite_refusal_helpers,
    input_exception_evidence_problems,
    nontest_rust,
    pyo3_items,
    python_literals,
    python_symbols,
    refusal_stage_literals,
    rust_literals,
    rust_pub_items,
    shared_namespace_problems,
    typed_input_exception_problems,
)
from test_evidence import (
    resolve_python_test,
    resolve_rust_test,
    rust_items,
    static_python_test,
    static_rust_test,
)

root = Path(__file__).resolve().parents[1]

# Every rule the checker can report, with what it enforces. `fail("<id>", ...)` is
# the only way to report, each message ends with `[<id>]`, and
# scripts/promotion_selftest.py needs a case tagged with every id (and, under
# --mutation-check, proves the case stops failing when the rule is switched off).
RULES = {
    # registry and frozen fields
    "registry_header": "registry is version 1, release matches its versioned filename",
    "frozen_field": "every frozen field is present and non-empty",
    "duplicate_record": "record ids are unique",
    "unknown_status": "status is frozen, in_progress, promoted or carried_forward",
    "claim_invalid": "inference_claim is a 2.2 claim (never nominal)",
    "coverage_required": "a calibrated claim allocates coverage record ids",
    "coverage_unexpected": "coverage ids are allocated only for a calibrated claim",
    "coverage_unknown": "a promoted record's coverage ids exist in the coverage registry",
    # bounds and search
    "bounds_cancellation": "bounds.cancellation is true",
    "bounds_limit": "a bounded search declares operation/depth/memory limits",
    "bounds_contract": "a bounded search runs under antecedent_core::SearchBudget",
    "search_flag": "search is declared true or false",
    "search_impl_shape": "search_impl is a list of paths",
    "search_impl_missing": "an implemented search declares search_impl",
    "search_impl_file": "each search_impl file is a Rust source file",
    "search_impl_names_budget": "each search_impl file names SearchBudget in non-test source",
    "search_impl_charge": "search_impl files call .charge( in non-test source",
    "search_charge_loop": "a .charge( on the budget sits in a looping or recursive fn that live code reaches",
    # refusals
    "refusal_code": "a refusal code is a registered runtime_refusal code",
    "refusal_when": "a refusal states its condition",
    "refusal_detail_shape": "a refusal detail is <namespace>.<snake_case>",
    "refusal_detail_duplicate": "refusal details are unique in a record",
    "refusal_namespace": "a record's refusal details share one namespace",
    "refusal_detail_missing": "a declared detail is a literal in non-test source",
    "refusal_detail_undeclared": "a namespaced literal in non-test source is declared",
    "refusal_dynamic": "no non-test source builds a namespaced detail dynamically",
    "refusal_pair_code": "the code of a (code, detail) pair is named where the detail is emitted",
    "refusal_dead_const": "a detail held in a const is used by live code",
    # fixtures and evidence
    "fixture_duplicate": "fixture ids are unique",
    "fixture_id_format": "fixture id is <workstream>.<name>.<role>",
    "fixture_intent": "a fixture states its intent",
    "evidence_pair": "evidence_test and evidence_assertion come together",
    "evidence_unresolved": "a cited test resolves to a collected, non-ignored test",
    "evidence_no_assertion": "a cited test body or helper contains an assertion",
    "evidence_should_panic": "a cited test is not #[should_panic]",
    "evidence_shared": "one test backs one fixture role unless every fixture sets shared_evidence",
    "budget_symbol": "a budget fixture exercises SearchBudget/SearchReceipt/SearchStop",
    "budget_asserts": "a budget fixture asserts on SearchStop or receipt fields",
    "promoted_evidence": "a promoted record cites evidence for every fixture",
    "fixture_roles": "required fixture roles are present",
    # routes
    "route_name": "route names are present and unique",
    "route_stage": "route stage is valid",
    "route_registry": "owning registry is known",
    "route_claim": "a route claim is a 2.2 claim",
    "route_status": "route status is closed or licensed",
    "closed_reason": "a closed route has a registered runtime reason_code",
    "closed_refusal_pair": "refusal_test and refusal_assertion come together",
    "closed_refusal_unresolved": "a route's refusal test resolves",
    "closed_refusal_reason_named": "a route's refusal test names its reason_code",
    "closed_refusal_required": "a closed route of implemented work cites a refusal test",
    "licensed_permanent": "a licensed route is not permanent_in_release",
    "licensed_uncertainty": "an uncertainty route is not licensed before promotion",
    "licensed_claim": "an early licensed route claims point_only or none",
    "licensed_unevidenced": "an early licensed route needs executed fixture evidence",
    "licensed_status": "a route is licensed only in_progress or promoted",
    "promoted_closed_route": "a promoted record leaves no non-permanent route closed",
    "nominal_route": "no licensed route carries estimator_grid_not_measured",
    "nominal_row": "no owning-registry row carries estimator_grid_not_measured",
    # owning registries
    "stages_missing": "a transport route has a transport_stages row",
    "stages_stage": "the row's stage equals the record's",
    "stages_licensed_only_there": "not licensed in transport_stages while closed in the record",
    "stages_licensed_only_here": "not licensed in the record while unlicensed in transport_stages",
    "stages_reason": "a closed row's reason_code equals the record's",
    "support_query_contrast": "a support route names its query and contrast",
    "support_licensed_only_there": "not licensed in support_licensed while closed in the record",
    "support_licensed_no_cell": "a licensed support route has a support_licensed cell",
    "support_licensed_still_closed": "a licensed support route is no longer closed",
    "support_no_closed_row": "a closed support route has a support_closed row",
    "support_closed_record": "the closed row names the record",
    "support_closed_reason": "the closed row's reason_code equals the record's",
    # public surface
    "surface_shape": "surface fields are lists of paths / names",
    "surface_file": "a surface file is a Python or Rust source file",
    "surface_symbol": "every public symbol of a surface is routed, a value or internal",
    "surface_values_shape": "surface_values is a list of names",
    "surface_values_orphan": "surface_values needs a declared surface",
    "surface_value_stale": "a surface_values entry is a public symbol",
    "surface_value_kind": "a surface_values entry is a class/struct/enum",
    "surface_value_method_verb": "a value type / pyclass does not cover an executing-verb method (run/execute/evaluate/decide/...)",
    "route_covers_shape": "a route's covers is a list of symbol names",
    "route_covers_stale": "a route's covers entry is a public symbol of a scanned surface file",
    "surface_internal_shape": "surface_internal entries carry a name and a why",
    "surface_internal_stale": "a surface_internal entry names a Rust/pyo3 item",
    "surface_internal_visible": "a surface_internal item is #[doc(hidden)]",
    "surface_owned_orphan": "surface_exports and owned_exports come together",
    "surface_export_stale": "an owned export is in the exports file's __all__",
}

args = sys.argv[1:]
if "--list-rules" in args:
    print("\n".join(RULES))
    sys.exit(0)
suggest = "--suggest" in args
if suggest:
    args.remove("--suggest")
emit_path = None
if "--emit-evidence" in args:
    at = args.index("--emit-evidence")
    emit_path = Path(args[at + 1])
    del args[at : at + 2]
registry_path = Path(args[0]) if args else root / "parity/promotion_2_2.toml"
registry = tomllib.loads(registry_path.read_text())
expected_release = {"promotion_2_2.toml": "2.2", "promotion_2_3.toml": "2.3"}.get(
    registry_path.name
)
if expected_release is None:
    expected_release = "2.2"  # Synthetic self-test registries retain the 2.2 contract.

DISABLED = {r for r in os.environ.get("PROMOTION_DISABLE_RULES", "").split(",") if r}
STATIC_ONLY = os.environ.get("PROMOTION_STATIC_ONLY") == "1" or suggest
SYNTHETIC = os.environ.get("PROMOTION_SYNTHETIC_EVIDENCE") == "1"
CACHE_DIR = os.environ.get("PROMOTION_EVIDENCE_CACHE")

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
# What an assertion in a budget fixture must observe: the stop or a receipt field.
BUDGET_OBSERVED = re.compile(
    r"SearchStop|SearchReceipt|operations_consumed|operations_limit|depth_limit|memory_limit|\bstop\b"
)
# Constructors/variants that carry a reason code without spelling it, so a pair
# emitted through them still names its code where the detail is emitted.
CODE_ALIASES = {
    "invalid_argument": ("invalid_input", "InvalidInput"),
    "route_not_supported": ("unsupported_input", "UnsupportedInput"),
}
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


def fail(rule: str, message: str) -> None:
    """Report one violation of `rule`; the message ends with `[rule]`."""
    if rule not in RULES:
        raise SystemExit(f"check_promotion_records: unknown rule id {rule!r}")
    if rule not in DISABLED:
        errors.append(f"{message} [{rule}]")


if registry.get("version") != 1 or registry.get("release") != expected_release:
    fail("registry_header", f"promotion registry requires version 1 and release {expected_release}")


# A surface_values type also covers the INHERENT methods and associated functions
# of `impl <Type>` blocks in the scanned files (trait impls are never scanned): a
# value type's own accessors and constructors are data access, not routes. Methods
# that start with an executing verb are not covered; they must be routes or
# surface_internal. Free functions and pyo3 items are never covered this way.
EXECUTING_VERBS = ("run", "execute", "evaluate", "decide", "identify", "prepare", "estimate", "consume", "fit")


# A #[pymethods] fn of a listed #[pyclass] is covered the same way, with two more
# executing verbs: `refresh` (re-runs against a new snapshot) and `export` (builds
# the wire artifact) act on data, they do not merely expose it. Getters such as
# `interval`, `plan`, `outcome`, `decision`, `seed` are plain accessors: covered.
PYO3_EXTRA_VERBS = ("refresh", "export")


def executing_verb(name: str, where: str = "method") -> str | None:
    """The executing verb a method name starts with (`decide`, `decide_x`), if any."""
    verbs = EXECUTING_VERBS + (PYO3_EXTRA_VERBS if where == "pyo3" else ())
    return next((v for v in verbs if name == v or name.startswith(v + "_")), None)


def value_method_state(sym: Symbol, value_types: set[str]) -> str | None:
    """None if `sym` is not a method of a listed value type (a Rust inherent
    method, or a #[pymethods] fn of a #[pyclass]); else "covered" or "verb" (an
    executing-verb method, which a value type cannot cover)."""
    if sym.where not in ("method", "pyo3") or sym.owner is None or sym.owner not in value_types:
        return None
    return "verb" if executing_verb(sym.name, sym.where) else "covered"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def outside_repo(path: str) -> bool:
    p = Path(path)
    return p.is_absolute() and not str(p.resolve()).startswith(str(root.resolve()) + os.sep)


def _synthetic_problems(path: str, assertion: str) -> list[str]:
    """Static resolution of a self-test evidence file that lives outside the repo
    (no cargo target, no pytest project): the test exists, is a test, not ignored."""
    p = Path(path)
    if p.suffix == ".py":
        return static_python_test(p, assertion)
    if p.suffix == ".rs":
        fns = [it for it in rust_items(p)[1] if it.kind == "fn" and it.name == assertion]
        if not fns:
            return [f"no fn {assertion} in {path}"]
        attrs = [re.sub(r"\s+", "", a) for a in fns[0].attrs]
        problems = [] if "#[test]" in attrs else [f"fn {assertion} has no #[test] attribute"]
        return problems + ([f"fn {assertion} is #[ignore]d"] if any(a.startswith("#[ignore") for a in attrs) else [])
    return [f"{path}: evidence must be a collected Rust or Python test"]


def _static_problems(path: str, assertion: str) -> list[str]:
    if path.endswith(".rs"):
        return static_rust_test(root / path, assertion).problems
    if path.endswith(".py"):
        return static_python_test(root / path, assertion)
    return [f"{path}: evidence must be a collected Rust or Python test"]


def resolve(path: str, assertion: str) -> list[str]:
    if SYNTHETIC and outside_repo(path):
        return _synthetic_problems(path, assertion)
    if STATIC_ONLY:
        return _static_problems(path, assertion)
    cached = None
    if CACHE_DIR:
        cached = Path(CACHE_DIR) / hashlib.sha1(f"{path}::{assertion}".encode()).hexdigest()
        if cached.is_file():
            return json.loads(cached.read_text())
    if path.endswith(".rs"):
        problems = resolve_rust_test(root / path, assertion)[1]
    elif path.endswith(".py"):
        problems = resolve_python_test(root / path, assertion)
    else:
        problems = [f"{path}: evidence must be a collected Rust or Python test"]
    if cached is not None:
        tmp = cached.with_suffix(f".{os.getpid()}")
        tmp.write_text(json.dumps(problems))
        tmp.replace(cached)
    return problems


def carries(value, needle: str) -> bool:
    """Whether any string anywhere in a TOML value contains `needle`."""
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(carries(v, needle) for v in value.values())
    if isinstance(value, list):
        return any(carries(v, needle) for v in value)
    return False


def camel(snake: str) -> str:
    return "".join(part.title() for part in snake.split("_"))


# ---------------------------------------------------------------- source corpus

_files: list[tuple[Path, bool]] | None = None
_raw: dict[Path, str] = {}


def source_files() -> list[tuple[Path, bool]]:
    global _files
    if _files is None:
        _files = [(p, True) for pattern in SOURCE_GLOBS for p in sorted(root.glob(pattern))]
        _files += [(p, False) for p in extra_sources]
    return _files


def raw(path: Path) -> str:
    if path not in _raw:
        _raw[path] = path.read_text(errors="ignore") if path.is_file() else ""
    return _raw[path]


def code_of(path: Path, in_crate: bool) -> str:
    """Non-test text of a source file (whole text for Python)."""
    return raw(path) if path.suffix == ".py" else nontest_rust(path, in_crate=in_crate)


def scan_details(namespaces: set[str]):
    """(found, dynamic): namespace -> detail -> [(path, in_crate, Lit)] for every
    literal in non-test source that is exactly `<ns>.<snake>` or starts
    `<ns>.<snake>:`, and namespace -> ["file:line"] for every literal that builds
    a detail dynamically (`<ns>.` followed by a format placeholder or ending
    there)."""
    found: dict[str, dict[str, list]] = {ns: {} for ns in namespaces}
    dynamic: dict[str, list[str]] = {ns: [] for ns in namespaces}
    if not namespaces:
        return found, dynamic
    alt = "|".join(map(re.escape, sorted(namespaces)))
    shape = re.compile(r"^(" + alt + r")\.([a-z0-9_]+)(?::|$)")
    dyn = re.compile(r"(?<![\w.])(" + alt + r")\.(?:\{|%|$)")
    hint = re.compile("|".join(re.escape(ns + ".") for ns in sorted(namespaces)))
    for path, in_crate in source_files():
        if not path.is_file() or not hint.search(raw(path)):
            continue
        if path.suffix == ".py":
            lits, finite_lines, stages = python_literals(path), set(), set()
        else:
            inferred, finite_lines = finite_refusal_helpers(path, in_crate=in_crate)
            stages = refusal_stage_literals(path, in_crate=in_crate)
            lits = [*rust_literals(path, in_crate=in_crate), *inferred]
        for lit in lits:
            if (lit.line, lit.value) in stages:
                continue
            m = shape.match(lit.value)
            if m:
                found[m.group(1)].setdefault(f"{m.group(1)}.{m.group(2)}", []).append(
                    (path, in_crate, lit)
                )
            for d in dyn.finditer(lit.value):
                if not (
                    lit.line in finite_lines
                    and lit.value == f"{d.group(1)}.{{detail}}: {{message}}"
                ):
                    dynamic[d.group(1)].append(f"{rel(path)}:{lit.line}")
    return found, dynamic


_refs: dict[str, dict[Path, int]] = {}


def const_refs(name: str) -> dict[Path, int]:
    """File -> occurrences of the identifier `name` in non-test source."""
    if name not in _refs:
        pattern = re.compile(rf"(?<![\w]){re.escape(name)}(?![\w])")
        refs = {}
        for path, in_crate in source_files():
            if path.is_file() and name in raw(path):
                n = len(pattern.findall(code_of(path, in_crate)))
                if n:
                    refs[path] = n
        _refs[name] = refs
    return _refs[name]


# ------------------------------------------------------------------- records

seen_records: set[str] = set()
seen_fixtures: set[str] = set()
seen_routes: set[str] = set()
evidence_rows: list[tuple[str, str, str]] = []
# (path, assertion) -> [(record, fixture id, role, shared_evidence)]
test_backers: dict[tuple[str, str], list[tuple[str, str, str, bool]]] = {}
records = registry.get("record", [])
namespaces_in_use = {
    refusal.get("detail", "").split(".")[0]
    for rec in records
    if rec.get("status") in IMPLEMENTED
    for refusal in [*(rec.get("refusals") or []), *(rec.get("input_exceptions") or [])]
    if "." in refusal.get("detail", "")
}
code_details, dynamic_details = scan_details(namespaces_in_use)
# Details each namespace's implemented records declare between them. Sibling records
# of one workstream may share a namespace (2.2B.X3's range record and its sampling
# uncertainty record both use `joint_sensitivity`): a literal in the namespace is
# declared when ANY implemented record declares it, while each record still emits
# every detail it declares itself (refusal_detail_missing).
namespace_declared: dict[str, set[str]] = {}
for _rec in records:
    if _rec.get("status") in IMPLEMENTED:
        for _refusal in [*(_rec.get("refusals") or []), *(_rec.get("input_exceptions") or [])]:
            _detail = _refusal.get("detail", "")
            if "." in _detail:
                namespace_declared.setdefault(_detail.split(".")[0], set()).add(_detail)


def check_evidence_body(who: str, path: str, assertion: str) -> bool:
    """A cited test must assert something and not pass on any panic. True if it does."""
    ok = True
    for rule, message in assertion_problems(root / path, assertion):
        fail(rule, f"{who}: {message}")
        ok = False
    return ok


def str_list(rid: str, key: str, value) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        fail("surface_shape", f"{rid}: {key} must be a list of non-empty strings")
        return []
    return value


for rec in records:
    rid = rec.get("id", "?")
    if rid in seen_records:
        fail("duplicate_record", f"{rid}: duplicate record id")
    seen_records.add(rid)
    for key in FROZEN:
        value = rec.get(key)
        if value is None or (isinstance(value, (str, list)) and not value):
            fail("frozen_field", f"{rid}: missing frozen field {key}")
    status = rec.get("status")
    if status not in STATUSES:
        fail("unknown_status", f"{rid}: unknown status {status!r}")
    promoted = status == "promoted"
    implemented = status in IMPLEMENTED
    claim = rec.get("inference_claim")
    if claim not in CLAIMS:
        fail("claim_invalid", f"{rid}: inference_claim {claim!r} is not a 2.2 claim ({', '.join(sorted(CLAIMS))})")
    cov = rec.get("coverage_records") or []
    if claim == "calibrated" and not cov:
        fail("coverage_required", f"{rid}: a calibrated interval must allocate its coverage record ids")
    if claim != "calibrated" and cov:
        fail("coverage_unexpected", f"{rid}: coverage records allocated for a {claim} claim")
    if promoted:
        for cid in cov:
            if cid not in coverage_ids:
                fail("coverage_unknown", f"{rid}: unknown coverage record {cid}")

    # Bounded computation is mandatory on every new search.
    bounds = rec.get("bounds") or {}
    if bounds.get("cancellation") is not True:
        fail("bounds_cancellation", f"{rid}: bounds.cancellation must be true")
    if rec.get("search") is True:
        for key in SEARCH_LIMITS:
            if not bounds.get(key):
                fail("bounds_limit", f"{rid}: a bounded search must declare bounds.{key}")
        if bounds.get("contract") != SEARCH_CONTRACT:
            fail("bounds_contract", f"{rid}: a bounded search must run under {SEARCH_CONTRACT}")
    elif rec.get("search") is not False:
        fail("search_flag", f"{rid}: search must be declared true or false")

    # The search is metered by the shared contract in code, not only pre-flighted:
    # every declared implementation file names SearchBudget and together they
    # charge it inside a loop or recursion.
    search_impl = rec.get("search_impl")
    if search_impl is not None and (
        not isinstance(search_impl, list) or not all(isinstance(p, str) and p for p in search_impl)
    ):
        fail("search_impl_shape", f"{rid}: search_impl must be a list of source paths")
        search_impl = []
    if rec.get("search") is True and implemented and not search_impl:
        fail(
            "search_impl_missing",
            f"{rid}: a search at status {status} must declare search_impl = [<.rs files>] whose non-test "
            f"source builds a SearchBudget and calls .charge( on it",
        )
    charged = False
    looped: list[str] = []
    for src in search_impl or []:
        path = root / src
        if not path.is_file() or path.suffix != ".rs":
            fail("search_impl_file", f"{rid}: search_impl {src} is not a Rust source file")
            continue
        in_crate = crate_src(path) is not None
        code = nontest_rust(path, in_crate=in_crate)
        if "SearchBudget" not in code:
            fail(
                "search_impl_names_budget",
                f"{rid}: search_impl {src} non-test source never names SearchBudget; "
                f"every declared implementation file runs under {SEARCH_CONTRACT}",
            )
        charged = charged or ".charge(" in code
        if "SearchBudget" in code or "SharedSearch" in code:
            looped += [f"{src}::{name}" for name in charge_in_loop(path, in_crate=in_crate)]
    if search_impl and not charged:
        fail(
            "search_impl_charge",
            f"{rid}: search_impl files lack .charge(; the search must charge {SEARCH_CONTRACT} "
            f"per step, not only pre-flight it",
        )
    elif search_impl and charged and not looped:
        fail(
            "search_charge_loop",
            f"{rid}: no search_impl fn charges a SearchBudget/SharedSearch inside a loop or recursion "
            f"that live code reaches (.charge( in a fn that loops, recurses, or is called from one, and "
            f"is itself pub, a trait-impl method, or called from a pub fn or another file); a single "
            f"charge is a pre-flight and a dead private fn meters nothing",
        )

    # A registered top-level code plus a unique, namespaced detail code: callers
    # switch on the pair, so neither may be prose or collide inside a record.
    details: set[str] = set()
    detail_code: dict[str, str] = {}
    for refusal in rec.get("refusals") or []:
        code, detail = refusal.get("code"), refusal.get("detail", "")
        if code not in runtime_codes:
            fail(
                "refusal_code",
                f"{rid}: refusal code {code!r} is not a registered runtime_refusal code",
            )
        if not refusal.get("when"):
            fail("refusal_when", f"{rid}: refusal {code!r} needs its condition")
        parts = detail.split(".")
        if len(parts) != 2 or not all(
            part.replace("_", "").isalnum() and part.islower() for part in parts
        ):
            fail(
                "refusal_detail_shape",
                f"{rid}: refusal detail {detail!r} must be <namespace>.<snake_case>",
            )
        if detail in details:
            fail(
                "refusal_detail_duplicate", f"{rid}: duplicate refusal detail {detail}"
            )
        details.add(detail)
        detail_code.setdefault(detail, code)
    namespaces = {d.split(".")[0] for d in details}
    for problem in shared_namespace_problems(rec, records, code_details, root):
        fail("refusal_namespace", f"{rid}: {problem}")
    exception_details = set()
    for entry in rec.get("input_exceptions", []):
        for problem in typed_input_exception_problems(entry, root):
            fail("refusal_pair_code", f"{rid}: {problem}")
        detail = entry.get("detail", "")
        if detail in exception_details or detail in details:
            fail(
                "refusal_detail_duplicate",
                f"{rid}: duplicate exception/refusal detail {detail}",
            )
        exception_details.add(detail)
        path, assertion = (
            entry.get("evidence_test", ""),
            entry.get("evidence_assertion", ""),
        )
        if not path or not assertion:
            fail(
                "evidence_pair",
                f"{rid}: class-only exception needs ordinary class/absent-code assertion",
            )
        else:
            problems = resolve(path, assertion)
            for problem in problems:
                fail("evidence_unresolved", f"{rid}: {problem}")
            for problem in input_exception_evidence_problems(
                entry, root / path, assertion
            ):
                fail("evidence_no_assertion", f"{rid}: {problem}")
            evidence_rows.append((f"{rid}.{detail}.input_exception", path, assertion))
    namespaces |= {d.split(".")[0] for d in exception_details}

    # Refusal boundary == code: once implemented, each declared detail is a
    # literal in non-test source, each namespaced literal there is declared, none
    # is built dynamically, and each (code, detail) pair is emitted by live code
    # that names the code.
    for ns in sorted(namespaces) if implemented else []:
        in_code = code_details.get(ns, {})
        local_details = {
            d for d in details | exception_details if d.startswith(ns + ".")
        }
        for detail in sorted(local_details - set(in_code)):
            fail(
                "refusal_detail_missing",
                f"{rid}: refusal detail {detail} is not emitted by non-test source "
                f'(no "{detail}" or "{detail}: ..." literal in {", ".join(SOURCE_GLOBS)})',
            )
        for detail in sorted(
            set(in_code) - details - namespace_declared.get(ns, set())
        ):
            places = ", ".join(f"{rel(p)}:{lit.line}" for p, _, lit in in_code[detail])
            fail(
                "refusal_detail_undeclared",
                f"{rid}: non-test source emits undeclared refusal detail {detail} "
                f"at {places}; add it to the record's refusals or remove it",
            )
        for place in dynamic_details.get(ns, []):
            fail(
                "refusal_dynamic",
                f'{rid}: dynamic refusal detail at {place} builds "{ns}." from a placeholder; '
                f"a dynamic detail cannot be checked, use a literal",
            )
        for detail in sorted(details & set(in_code)):
            code = detail_code[detail]
            uses = in_code[detail]
            files: dict[Path, bool] = {}
            live, dead = False, []
            for path, in_crate, lit in uses:
                files[path] = in_crate
                if lit.const is None:
                    live = True
                    continue
                refs = const_refs(lit.const)
                if sum(refs.values()) >= 2:  # the definition plus at least one use
                    live = True
                    files.update({p: c for p, c in source_files() if p in refs})
                else:
                    dead.append(lit.const)
            if not live:
                fail(
                    "refusal_dead_const",
                    f"{rid}: refusal detail {detail} exists only as const {', '.join(sorted(set(dead)))} "
                    f"that no non-test code uses; emit it or delete it",
                )
                continue
            proven_codes = {
                lit.reason_code for _, _, lit in uses if lit.reason_code is not None
            }
            if proven_codes and proven_codes != {code}:
                fail(
                    "refusal_pair_code",
                    f"{rid}: closed literal helper emits {sorted(proven_codes)} for {detail}, not {code}",
                )
            spellings = (str(code), camel(str(code)), *CODE_ALIASES.get(str(code), ()))
            named = re.compile(
                r"(?<![A-Za-z0-9_])(?:"
                + "|".join(map(re.escape, spellings))
                + r")(?![A-Za-z0-9_])"
            )
            if not any(named.search(code_of(p, c)) for p, c in files.items()):
                fail(
                    "refusal_pair_code",
                    f"{rid}: refusal ({code}, {detail}): no non-test file that emits {detail} "
                    f"(or uses the const holding it) names {code} or {camel(str(code))}; "
                    f"checked {', '.join(sorted(rel(p) for p in files))}",
                )

    # Fixtures first: a route's license depends on which roles cite executed evidence.
    workstream = str(rec.get("workstream", "")).lower()
    roles: set[str] = set()
    evidenced: set[str] = set()
    budget_evidenced = budget_asserting = False
    for fixture in rec.get("fixtures") or []:
        fid, role = fixture.get("id", ""), fixture.get("role")
        if fid in seen_fixtures:
            fail("fixture_duplicate", f"{rid}: duplicate fixture id {fid}")
        seen_fixtures.add(fid)
        if not fid.startswith(f"{workstream}.") or not fid.endswith(f".{role}"):
            fail("fixture_id_format", f"{rid}: fixture {fid} must be <workstream>.<name>.<role>")
        if not fixture.get("intent"):
            fail("fixture_intent", f"{rid}: fixture {fid} needs an intent")
        roles.add(role)
        path, assertion = fixture.get("evidence_test", ""), fixture.get("evidence_assertion", "")
        if bool(path) != bool(assertion):
            fail("evidence_pair", f"{rid}: fixture {fid} needs both evidence_test and evidence_assertion")
        elif path:
            problems = resolve(path, assertion)
            for p in problems:
                fail("evidence_unresolved", f"{rid}: {fid}: {p}")
            evidence_rows.append((fid, path, assertion))
            test_backers.setdefault((path, assertion), []).append(
                (rid, fid, str(role), fixture.get("shared_evidence") is True)
            )
            ok = not problems
            # Assertions are read from the source, so a test cargo cannot list here
            # (or a whole-repo static read) is still checked for proving something.
            found = (root / path).is_file() or outside_repo(path)
            if found and not check_evidence_body(f"{rid}: {fid}", path, assertion):
                ok = False
            if role == "budget" and rec.get("search") is True and ok:
                body = closure_code(root / path, assertion)
                if not any(symbol in body for symbol in SEARCH_SYMBOLS):
                    fail("budget_symbol", f"{rid}: budget fixture {fid} does not exercise {SEARCH_CONTRACT}")
                    ok = False
                else:
                    budget_evidenced = True
                    if any(BUDGET_OBSERVED.search(t) for t in assertion_texts(root / path, assertion)):
                        budget_asserting = True
            if ok:
                evidenced.add(role)
        elif promoted:
            fail("promoted_evidence", f"{rid}: promoted record lacks executed evidence for fixture {fid}")
    if rec.get("search") is True and implemented and budget_evidenced and not budget_asserting:
        fail(
            "budget_asserts",
            f"{rid}: no budget fixture asserts on SearchStop or a SearchReceipt field "
            f"(operations_consumed/operations_limit/depth_limit/memory_limit); naming the type is not observing the stop",
        )
    required = {"positive", "negative", "artifact"}
    if rec.get("search") is True:
        required.add("budget")
    if claim == "calibrated":
        required.add("calibration")
    missing = sorted(required - roles)
    if missing:
        fail("fixture_roles", f"{rid}: missing fixture roles {', '.join(missing)}")
    # Roles an in_progress record must evidence before licensing any route.
    early_roles = {"positive", "negative", "artifact"} | ({"budget"} if rec.get("search") is True else set())
    unevidenced = sorted(early_roles - evidenced)

    route_names: list[str] = []
    route_covers: dict[str, list[str]] = {}
    for route in rec.get("routes") or []:
        name = route.get("name")
        if not name or name in seen_routes:
            fail("route_name", f"{rid}: duplicate or missing route name {name!r}")
        seen_routes.add(name)
        route_names.append(name or "")
        route_cover = route.get("covers")
        if route_cover is not None:
            if not isinstance(route_cover, list) or not all(isinstance(c, str) and c for c in route_cover):
                fail("route_covers_shape", f"{rid}: {name}: covers must be a list of non-empty symbol names")
            else:
                route_covers[name or ""] = route_cover
        stage = route.get("stage")
        if stage not in STAGES:
            fail("route_stage", f"{rid}: {name}: invalid stage {stage!r}")
        owner = route.get("registry", "transport_stages")
        if owner not in REGISTRIES:
            fail("route_registry", f"{rid}: {name}: unknown owning registry {owner!r}")
        permanent = route.get("permanent_in_release") is True
        route_claim = route.get("claim")
        if route_claim is not None and route_claim not in CLAIMS:
            fail("route_claim", f"{rid}: {name}: claim {route_claim!r} is not a 2.2 claim")
        route_status = route.get("status")
        licensed = route_status == "licensed"
        if route_status == "closed":
            reason = route.get("reason_code")
            if reason not in runtime_codes:
                fail("closed_reason", f"{rid}: {name}: closed route needs a registered runtime reason_code")
            # A closed route of implemented work must be shown to refuse at runtime.
            rpath, rassert = route.get("refusal_test", ""), route.get("refusal_assertion", "")
            if bool(rpath) != bool(rassert):
                fail("closed_refusal_pair", f"{rid}: {name}: closed route needs both refusal_test and refusal_assertion")
            elif rpath:
                problems = resolve(rpath, rassert)
                for p in problems:
                    fail("closed_refusal_unresolved", f"{rid}: {name}: refusal evidence: {p}")
                evidence_rows.append((f"{name}.refusal", rpath, rassert))
                if (root / rpath).is_file() or outside_repo(rpath):
                    check_evidence_body(f"{rid}: {name}: refusal evidence", rpath, rassert)
                if not problems and reason:
                    body = closure_code(root / rpath, rassert)
                    named = re.compile(rf"(?<![A-Za-z0-9_])(?:{re.escape(reason)}|{re.escape(camel(reason))})(?![A-Za-z0-9_])")
                    if not named.search(body):
                        fail(
                            "closed_refusal_reason_named",
                            f"{rid}: {name}: refusal test {rpath}::{rassert} never names the "
                            f"route's reason_code {reason} (or {camel(reason)})",
                        )
            elif implemented:
                fail(
                    "closed_refusal_required",
                    f"{rid}: {name}: closed route of a record at status {status} needs refusal_test + "
                    f"refusal_assertion citing a test that calls it and observes {reason}",
                )
        elif licensed:
            if permanent:
                fail("licensed_permanent", f"{rid}: {name}: licensed but marked permanent_in_release (closed)")
            elif promoted:
                pass
            elif status == "in_progress":
                if stage == "uncertainty":
                    fail("licensed_uncertainty", f"{rid}: {name}: uncertainty route licensed before its record is promoted")
                if route_claim not in EARLY_CLAIMS:
                    fail(
                        "licensed_claim",
                        f"{rid}: {name}: licensed before promotion requires claim = \"point_only\" "
                        f"or \"none\" on the route (got {route_claim!r})",
                    )
                if unevidenced:
                    fail(
                        "licensed_unevidenced",
                        f"{rid}: {name}: licensed before promotion but the record cites no executed "
                        f"evidence for fixture role(s) {', '.join(unevidenced)}",
                    )
            else:
                fail("licensed_status", f"{rid}: {name}: licensed while its record is {status}")
        else:
            fail("route_status", f"{rid}: {name}: status must be closed or licensed")

        # Owning registry must hold a row that agrees with the record.
        owning_rows: list[dict] = []
        if owner == "transport_stages":
            row = stage_routes.get(name)
            if row is None:
                fail(
                    "stages_missing",
                    f"{rid}: {name} has no row in transport_stages.toml; add [[routes]] route = \"{name}\", "
                    f"stage = \"{stage}\", status = \"{'licensed' if licensed else 'closed'}\""
                    + ("" if licensed else f", reason_code = \"{route.get('reason_code')}\""),
                )
            else:
                owning_rows.append(row)
                there = row.get("status")
                if row.get("stage") != stage:
                    fail("stages_stage", f"{rid}: {name}: transport_stages.toml stage {row.get('stage')!r} != record stage {stage!r}")
                if there == "licensed" and not licensed:
                    fail("stages_licensed_only_there", f"{rid}: {name} is licensed in transport_stages.toml but closed in its record")
                elif there != "licensed" and licensed:
                    fail("stages_licensed_only_here", f"{rid}: {name} is licensed in its record but not in transport_stages.toml")
                elif there == "closed" and row.get("reason_code") != route.get("reason_code"):
                    fail(
                        "stages_reason",
                        f"{rid}: {name}: transport_stages.toml reason_code {row.get('reason_code')!r} "
                        f"!= record reason_code {route.get('reason_code')!r}",
                    )
            if promoted and not permanent and not licensed:
                fail("promoted_closed_route", f"{rid}: promoted record leaves non-permanent route {name} closed")
        elif owner == "support_licensed":
            query, contrast = route.get("query"), route.get("contrast")
            if not query or not contrast:
                fail("support_query_contrast", f"{rid}: {name}: a support_licensed route names its query and contrast")
            cells = [c for c in support_cells if c.get("query") == query and c.get("contrast") == contrast]
            closed_rows = [
                c for c in support_closed_contrasts if c.get("query") == query and c.get("contrast") == contrast
            ]
            owning_rows.extend(cells + closed_rows)
            if cells and not licensed:
                fail("support_licensed_only_there", f"{rid}: {name} is licensed in support_licensed.toml but closed in its record")
            if licensed and not cells:
                fail("support_licensed_no_cell", f"{rid}: {name} is licensed in its record but has no support_licensed.toml cell")
            if closed_rows and licensed:
                fail("support_licensed_still_closed", f"{rid}: {name} is licensed in its record but still closed in support_closed.toml")
            if not licensed and not closed_rows:
                fail(
                    "support_no_closed_row",
                    f"{rid}: {name} has no support_closed.toml row; add [[closed_contrast]] query = \"{query}\", "
                    f"contrast = \"{contrast}\", reason_code = \"{route.get('reason_code')}\", record = \"{rid}\"",
                )
            for row in closed_rows:
                if row.get("record") != rid:
                    fail("support_closed_record", f"{rid}: {name}: support_closed.toml row names record {row.get('record')!r}")
                if row.get("reason_code") != route.get("reason_code"):
                    fail(
                        "support_closed_reason",
                        f"{rid}: {name}: support_closed.toml reason_code {row.get('reason_code')!r} "
                        f"!= record reason_code {route.get('reason_code')!r}",
                    )
            if promoted and not permanent and not licensed:
                fail("promoted_closed_route", f"{rid}: promoted record leaves non-permanent route {name} closed")

        # No nominal-only interval ships from implemented 2.2 work, anywhere.
        if implemented:
            if licensed and carries(route, NOMINAL):
                fail("nominal_route", f"{rid}: licensed route {name} carries {NOMINAL}")
            for row in owning_rows:
                if carries(row, NOMINAL):
                    fail("nominal_row", f"{rid}: {name}: its {owner} row carries {NOMINAL}")

    # ---- route inventory: every public symbol of a declared surface is a component
    # of some route name in this record, a value/descriptor type listed in
    # surface_values (classes/structs/enums only, never stale; also covering the
    # non-verb inherent methods of that type), or a surface_internal item that is #[doc(hidden)] in Rust.
    components = {part for name in route_names for part in name.split(".")}
    surface = rec.get("surface")
    if surface is not None and not isinstance(surface, list):
        fail("surface_shape", f"{rid}: surface must be a list of source paths")
        surface = None
    if isinstance(surface, list) and not all(isinstance(p, str) and p for p in surface):
        fail("surface_shape", f"{rid}: surface must be a list of source paths")
        surface = []
    surface_rust = str_list(rid, "surface_rust", rec.get("surface_rust"))
    surface_pyo3 = str_list(rid, "surface_pyo3", rec.get("surface_pyo3"))
    surface_exports = str_list(rid, "surface_exports", rec.get("surface_exports"))
    owned = str_list(rid, "owned_exports", rec.get("owned_exports"))
    values = rec.get("surface_values")
    if values is not None and (not isinstance(values, list) or not all(isinstance(v, str) and v for v in values)):
        fail("surface_values_shape", f"{rid}: surface_values must be a list of symbol names")
        values = []
    values = values or []
    internal_raw = rec.get("surface_internal")
    internal: dict[str, str] = {}
    if internal_raw is not None and not isinstance(internal_raw, list):
        fail("surface_internal_shape", f"{rid}: surface_internal must be a list of {{ name, why }} tables")
        internal_raw = []
    for entry in internal_raw or []:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("why"):
            fail("surface_internal_shape", f"{rid}: surface_internal entry {entry!r} needs a name and a why")
        else:
            internal[entry["name"]] = entry["why"]
    has_surface = surface is not None or bool(surface_rust or surface_pyo3 or surface_exports)
    if values and not has_surface:
        fail("surface_values_orphan", f"{rid}: surface_values {', '.join(values)} listed without a surface")
    if bool(surface_exports) != bool(owned):
        fail(
            "surface_owned_orphan",
            f"{rid}: surface_exports and owned_exports go together (the exports file's __all__ names "
            f"this record owns); got surface_exports={surface_exports} owned_exports={owned}",
        )

    def in_crate_of(path: Path) -> bool:
        return crate_src(path) is not None

    declared: dict[str, tuple[str, str]] = {}  # name -> (kind, src)
    checked: list[tuple[str, Symbol]] = []  # (src, symbol) each must be covered
    surface_files: list[str] = []
    for src in surface or []:
        path = root / src
        if not path.is_file() or path.suffix not in (".py", ".rs"):
            fail("surface_file", f"{rid}: surface {src} is not a Python or Rust source file")
            continue
        surface_files.append(src)
        if path.suffix == ".py":
            symbols = list(python_symbols(path).values())
        else:  # a Rust facade: top-level public items and re-exports, not methods
            symbols = [s for s in rust_pub_items(path, in_crate=in_crate_of(path)) if s.where == "top"]
        checked += [(src, sym) for sym in symbols]
    for keyname, files, scan in (
        ("surface_rust", surface_rust, rust_pub_items),
        ("surface_pyo3", surface_pyo3, pyo3_items),
    ):
        for src in files:
            path = root / src
            if not path.is_file() or path.suffix != ".rs":
                fail("surface_file", f"{rid}: {keyname} {src} is not a Rust source file")
                continue
            surface_files.append(src)
            found_syms = scan(path, in_crate=in_crate_of(path))
            checked += [(src, sym) for sym in (found_syms if isinstance(found_syms, list) else found_syms.values())]
    for src in surface_exports:
        path = root / src
        if not path.is_file() or path.suffix != ".py":
            fail("surface_file", f"{rid}: surface_exports {src} is not a Python source file")
            continue
        surface_files.append(src)
        symbols = python_symbols(path)
        for name in owned:
            if name in symbols:
                checked.append((src, symbols[name]))
            else:
                fail("surface_export_stale", f"{rid}: owned_exports entry {name} is not in __all__ of {src}")
    # `covers`: public symbols a route accounts for besides its name components. An
    # entry is `symbol` (any owner, like a route-name component) or `Type.method`.
    covers = {c for lst in route_covers.values() for c in lst}
    for cname, lst in route_covers.items():
        for entry in lst:
            if not any(entry in (sym.name, f"{sym.owner}.{sym.name}") for _, sym in checked):
                fail(
                    "route_covers_stale",
                    f"{rid}: {cname}: covers entry {entry} is not a public symbol of any surface file "
                    f"({', '.join(surface_files) or 'none declared'}); remove the stale entry",
                )
    if covers:
        # Reach (a warning, not a rule): the covered name should appear in a cited
        # evidence test closure of the record or in the source file declaring the
        # route's own entry point.
        bodies = []
        for pth, assertion in [(f.get("evidence_test"), f.get("evidence_assertion")) for f in rec.get("fixtures") or []] + [
            (r.get("refusal_test"), r.get("refusal_assertion")) for r in rec.get("routes") or []
        ]:
            if pth and assertion and ((root / pth).is_file() or outside_repo(pth)):
                try:
                    bodies.append(closure_code(root / pth if not outside_repo(pth) else Path(pth), assertion))
                except (ValueError, SyntaxError, OSError):
                    continue
        for cname, lst in route_covers.items():
            parts = set(cname.split("."))
            entry_text = "".join(
                raw(root / src) for src in sorted({s2 for s2, y in checked if y.name in parts})
            )
            for entry in lst:
                word = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(entry.rsplit(".", 1)[-1]) + r"(?![A-Za-z0-9_])")
                if not (any(word.search(b) for b in bodies) or word.search(entry_text)):
                    print(
                        f"warning: {rid}: {cname}: covers entry {entry} appears in no cited evidence test closure "
                        f"and not in the route's own entry-point file; is it really reached by this route?",
                        file=sys.stderr,
                    )
    for src, sym in checked:
        if sym.name not in declared or (declared[sym.name][0] != "type" and sym.kind == "type"):
            declared[sym.name] = (sym.kind, src)
    value_types = {v for v in values if declared.get(v, ("", ""))[0] == "type"}
    for src, sym in checked:
        if sym.name in components or sym.name in values or sym.name in internal or sym.name in covers or (
            sym.owner and (f"{sym.owner}.{sym.name}" in covers or f"{sym.owner}.{sym.name}" in internal)
        ):
            continue
        state = value_method_state(sym, value_types)
        if state == "covered":
            continue
        if state == "verb":
            fail(
                "surface_value_method_verb",
                f"{rid}: {sym.owner}.{sym.name} ({src}) starts with the executing verb "
                f"'{executing_verb(sym.name, sym.where)}', so surface_values entry {sym.owner} does not cover it; "
                f"add a route (or a route `covers` entry) for it (closed until evidenced), list it in surface_internal "
                f"with a reason and mark it #[doc(hidden)] (or an underscore pyo3 name), or make it private",
            )
            continue
        fail(
            "surface_symbol",
            f"{rid}: surface symbol {sym.name} ({src}) is not a component of any route name; "
            + (f"it is a method of {sym.owner}: list {sym.owner} in surface_values to cover its accessors, " if sym.owner else "")
            + ("add it to surface_values if it is a pure value/descriptor type, " if sym.kind == "type" else "")
            + "add a route for it (closed until evidenced) or list it in a route's `covers`, "
            + ("list it in surface_internal with a why and mark it #[doc(hidden)], " if sym.where in ("top", "method", "pyo3") else "")
            + "or make it private",
        )
    for value in values:
        if surface_files and value not in declared:
            fail(
                "surface_value_stale",
                f"{rid}: surface_values entry {value} is not a public symbol of any surface file "
                f"({', '.join(surface_files)}); remove the stale exemption",
            )
        elif value in declared and declared[value][0] != "type":
            fail(
                "surface_value_kind",
                f"{rid}: surface_values entry {value} ({declared[value][1]}) is a {declared[value][0]}, "
                f"not a class/struct/enum; an entry point must be a route",
            )
    for name in internal:
        matches = [
            s for _, s in checked
            if (s.name == name or (s.owner and f"{s.owner}.{s.name}" == name))
            and s.where in ("top", "method", "pyo3")
        ]
        rust_scanned = [src for src in surface_files if src.endswith(".rs")]
        if not matches:
            fail(
                "surface_internal_stale",
                f"{rid}: surface_internal entry {name} names no Rust/pyo3 item of {', '.join(rust_scanned) or 'any surface_rust/surface_pyo3 file'}; "
                f"remove the stale exemption",
            )
        elif not all(s.hidden for s in matches):
            fail(
                "surface_internal_visible",
                f"{rid}: surface_internal entry {name} is a public item that is not #[doc(hidden)]; "
                f"mark it #[doc(hidden)] or make it pub(crate)",
            )

# ---- registry-wide: one test proves one role.
for (path, assertion), backers in sorted(test_backers.items()):
    distinct_roles = {role for _, _, role, _ in backers}
    if len(distinct_roles) > 1 and not all(shared for *_, shared in backers):
        who = ", ".join(f"{fid} ({role})" for _, fid, role, _ in backers)
        fail(
            "evidence_shared",
            f"{backers[0][0]}: test {path}::{assertion} backs more than one fixture role: {who}; cite a "
            f"distinct test per role or set shared_evidence = true on every fixture that shares it",
        )


# --------------------------------------------------------------------- suggest


def suggest_report() -> None:
    """Per record: public symbols of its likely Rust/pyo3/export files that no
    route, surface_values or surface_internal entry covers."""
    ns_all = {
        r.get("detail", "").split(".")[0] for rec in records for r in rec.get("refusals") or [] if "." in r.get("detail", "")
    }
    found, _ = scan_details(ns_all)
    py_files = sorted((root / "python/antecedent").glob("**/*.py"))
    for rec in records:
        rid = rec.get("id", "?")
        components = {part for r in rec.get("routes") or [] for part in r.get("name", "").split(".")}
        values = set(rec.get("surface_values") or [])
        internal = {e.get("name") for e in rec.get("surface_internal") or [] if isinstance(e, dict)}
        covers = {c.rsplit(".", 1)[-1] for r in rec.get("routes") or [] for c in r.get("covers") or []}
        covered = components | values | internal | covers
        ns = {r.get("detail", "").split(".")[0] for r in rec.get("refusals") or [] if "." in r.get("detail", "")}
        rs: set[Path] = {root / p for p in rec.get("search_impl") or [] if p.endswith(".rs")}
        rs |= {root / p for p in (rec.get("surface") or []) if p.endswith(".rs")}
        rs |= {root / p for p in rec.get("surface_rust") or []}
        for n in ns:
            for uses in found.get(n, {}).values():
                rs |= {p for p, c, _ in uses if p.suffix == ".rs" and "/python/src/" not in p.as_posix()}
        pyfaces = {root / p: python_symbols(root / p) for p in (rec.get("surface") or []) if p.endswith(".py")}
        names = {s for syms in pyfaces.values() for s in syms} | values
        pyo3: set[Path] = {root / p for p in rec.get("surface_pyo3") or []}
        for path in sorted((root / "python/src").glob("*.rs")):
            text = raw(path)
            if any(f"{n}." in text for n in ns) or any(re.search(rf"\b{re.escape(s)}\b", text) for s in names if len(s) > 4):
                pyo3.add(path)
        owned_exports: dict[Path, list[str]] = {}
        for path in py_files:
            if "__all__" not in raw(path) or path in pyfaces:
                continue
            try:
                syms = python_symbols(path)
            except SyntaxError:
                continue
            face_mods = {p.stem for p in pyfaces}
            imported = set()
            for m in re.finditer(r"from\s+\.(\w+)\s+import\s+\(?([^)\n]*(?:\n[^)\n]*)*?)\)?(?=\n\S|\Z)", raw(path)):
                if m.group(1) in face_mods:
                    imported |= set(re.findall(r"[A-Za-z_]\w*", m.group(2)))
            mine = sorted(n for n in syms if n in imported)
            if mine:
                owned_exports[path] = mine
        print(f"\n{rid} ({rec.get('status')})")
        items: dict[Path, list[Symbol]] = {
            path: rust_pub_items(path, in_crate=crate_src(path) is not None) for path in sorted(rs)
        }
        declared_types = {s.name for syms in items.values() for s in syms if s.kind == "type" and s.where == "top"}
        value_types = values & declared_types
        need_py = {}
        for path in sorted(pyo3):
            syms = pyo3_items(path, in_crate=crate_src(path) is not None)
            py_types = values & {s.name for s in syms if s.kind == "type"}
            gaps = sorted(
                {
                    f"{s.owner}.{s.name}" if s.owner else s.name
                    for s in syms
                    if s.name not in covered and value_method_state(s, value_types | py_types) != "covered"
                }
            )
            if gaps:
                need_py[rel(path)] = gaps
        print(f"  surface_rust candidates: {sorted(rel(p) for p in rs)}")
        n_covered = n_verb = n_free = 0
        all_syms = [s for syms in items.values() for s in syms]
        for path, syms in items.items():
            todo = [s for s in syms if s.name not in covered and value_method_state(s, value_types) != "covered"]
            free = sorted({s.name for s in todo if s.where == "top" and s.kind != "type"})
            verbs = sorted(f"{s.owner}.{s.name}" for s in todo if s.where == "method" and s.owner in value_types)
            # types not yet listed that would become values, with the methods each would cover
            cand: dict[str, int] = {}
            for s in todo:
                if s.where == "top" and s.kind == "type":
                    cand[s.name] = sum(
                        1
                        for t in all_syms
                        if t.where == "method" and t.owner == s.name and t.name not in covered and not executing_verb(t.name)
                    )
            # methods of a type not listed in surface_values and not a candidate here (declared in another file)
            orphans = sorted(
                {
                    f"{s.owner}.{s.name}"
                    for s in todo
                    if s.where == "method" and s.owner not in value_types and s.owner not in declared_types
                }
            )
            cand_verbs = sorted(
                f"{t.owner}.{t.name}"
                for t in all_syms
                if t.where == "method" and t.owner in cand and t.name not in covered and executing_verb(t.name)
            )
            if not (free or verbs or cand or orphans):
                continue
            print(f"    {rel(path)}:")
            if cand:
                print(
                    "      surface_values candidates (type: methods it would cover): "
                    + ", ".join(f"{t}: {n}" for t, n in sorted(cand.items()))
                )
                n_covered += sum(cand.values())
            if free:
                print(f"      free functions / other items still needing a route or surface_internal -> {free}")
                n_free += len(free)
            if verbs or cand_verbs:
                print(f"      executing-verb methods a value type cannot cover -> {sorted(verbs + cand_verbs)}")
                n_verb += len(verbs) + len(cand_verbs)
            if orphans:
                print(f"      methods of types declared outside these files (route or surface_internal) -> {orphans}")
                n_free += len(orphans)
        print(
            f"  summary: {n_covered} methods coverable by listing value types, "
            f"{n_free} free functions/items needing routes, {n_verb} executing-verb methods needing routes, "
            f"{sum(len(g) for g in need_py.values())} pyo3 items needing routes"
        )
        print(f"  surface_pyo3 candidates: {sorted(rel(p) for p in pyo3)}")
        for src, gaps in need_py.items():
            print(f"    {src}: uncovered pyo3 names -> {gaps}")
        for path, mine in owned_exports.items():
            gaps = sorted(n for n in mine if n not in covered)
            print(f"  surface_exports: {rel(path)} owned_exports={mine}")
            if gaps:
                print(f"    uncovered exports -> {gaps}")


if suggest:
    suggest_report()
    sys.exit(0)

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
