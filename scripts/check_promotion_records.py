"""Validate the 2.2 promotion records (parity/promotion_2_2.toml).

A record freezes a cell's scientific surface before implementation. Its routes
stay closed until the record is promoted; a promoted record must cite executed
positive, negative and artifact fixtures (and budget/calibration evidence where
its search or interval claims require them). A route that executes before its
record carries that evidence fails here.

    python3 scripts/check_promotion_records.py [registry.toml] [--emit-evidence out.toml]

--emit-evidence writes every fixture that cites a test as a [[fixture_evidence]]
row, so scripts/gate_promotion.sh can execute it with run_evidence_rows.py.
PROMOTION_TRANSPORT_STAGES / PROMOTION_SUPPORT_LICENSED override the owning
registries for gate self-tests.
"""

import os
import sys
from pathlib import Path

import tomllib
from test_evidence import closure, resolve_python_test, resolve_rust_test

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
# "nominal" is deliberately absent: no 2.2 interval ships unmeasured.
CLAIMS = {"point_only", "calibrated", "structural_envelope", "assumption_range", "none"}
STAGES = {"identify", "representation", "evaluate", "uncertainty", "prepare", "consume"}
REGISTRIES = {"transport_stages", "support_licensed"}
SEARCH_LIMITS = ("operation_limit", "depth_limit", "memory_limit")
# The one bounded-search contract (crates/antecedent-core/src/search.rs): it
# cannot be built without limits and observes cancellation and memory per charge.
SEARCH_CONTRACT = "antecedent_core::SearchBudget"
# Symbols whose appearance in a budget fixture shows it exercises that contract.
SEARCH_SYMBOLS = ("SearchBudget", "SearchReceipt", "SearchStop")

codes = {
    row["id"]: row
    for row in tomllib.loads((root / "parity/reason_codes.toml").read_text())["code"]
}
runtime_codes = {cid for cid, row in codes.items() if "runtime_refusal" in row.get("applies_to", [])}
stages_path = Path(os.environ.get("PROMOTION_TRANSPORT_STAGES", root / "parity/transport_stages.toml"))
support_path = Path(os.environ.get("PROMOTION_SUPPORT_LICENSED", root / "parity/support_licensed.toml"))
stage_routes = {
    row.get("route"): row for row in tomllib.loads(stages_path.read_text()).get("routes", [])
}
support_cells = tomllib.loads(support_path.read_text()).get("cell", [])
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


seen_records: set[str] = set()
seen_fixtures: set[str] = set()
seen_routes: set[str] = set()
evidence_rows: list[tuple[str, str, str]] = []
for rec in registry.get("record", []):
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
    if len({d.split(".")[0] for d in details}) > 1:
        errors.append(f"{rid}: refusal details must share one namespace")

    for route in rec.get("routes") or []:
        name = route.get("name")
        if not name or name in seen_routes:
            errors.append(f"{rid}: duplicate or missing route name {name!r}")
        seen_routes.add(name)
        if route.get("stage") not in STAGES:
            errors.append(f"{rid}: {name}: invalid stage {route.get('stage')!r}")
        owner = route.get("registry", "transport_stages")
        if owner not in REGISTRIES:
            errors.append(f"{rid}: {name}: unknown owning registry {owner!r}")
        permanent = route.get("permanent_in_release") is True
        if route.get("status") == "closed":
            if route.get("reason_code") not in runtime_codes:
                errors.append(f"{rid}: {name}: closed route needs a registered runtime reason_code")
        elif route.get("status") == "licensed":
            if permanent or not promoted:
                errors.append(f"{rid}: {name}: licensed before its record is promoted")
        else:
            errors.append(f"{rid}: {name}: status must be closed or licensed")
        # Owning registry must agree: nothing executes ahead of the record.
        if owner == "transport_stages":
            row = stage_routes.get(name)
            licensed_there = row is not None and row.get("status") == "licensed"
            if licensed_there and (permanent or not promoted):
                errors.append(f"{rid}: {name} is licensed in transport_stages.toml before promotion")
            if promoted and not permanent and not licensed_there:
                errors.append(f"{rid}: promoted route {name} has no licensed transport_stages row")
            if licensed_there and row.get("calibration_reason") == "estimator_grid_not_measured":
                errors.append(f"{rid}: {name} ships a 2.2 interval under estimator_grid_not_measured")
        elif owner == "support_licensed":
            query, contrast = route.get("query"), route.get("contrast")
            if not query or not contrast:
                errors.append(f"{rid}: {name}: a support_licensed route names its query and contrast")
            licensed_there = any(
                cell.get("query") == query and cell.get("contrast") == contrast for cell in support_cells
            )
            if licensed_there and (permanent or not promoted):
                errors.append(f"{rid}: {name} is licensed in support_licensed.toml before promotion")
            if promoted and not permanent and not licensed_there:
                errors.append(f"{rid}: promoted route {name} has no licensed support_licensed cell")

    workstream = str(rec.get("workstream", "")).lower()
    roles: set[str] = set()
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
            if role == "budget" and rec.get("search") is True and not problems:
                body = closure(root / path, assertion)
                if not any(symbol in body for symbol in SEARCH_SYMBOLS):
                    errors.append(f"{rid}: budget fixture {fid} does not exercise {SEARCH_CONTRACT}")
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
records = registry.get("record", [])
promoted = sum(1 for rec in records if rec.get("status") == "promoted")
print(
    f"Promotion records OK ({len(records)} records, {promoted} promoted; "
    f"{len(seen_routes)} routes closed until promotion)"
)
