"""Validate the transport stage registry independently of the analyze axes."""

import sys
from pathlib import Path

import tomllib
from test_evidence import (
    checked_execution_body_problems,
    closure,
    resolve_python_test,
    resolve_rust_test,
)

root = Path(__file__).resolve().parents[1]
# An explicit registry path exists for gate self-tests; evidence still resolves in this tree.
registry_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "parity/transport_stages.toml"
registry = tomllib.loads(registry_path.read_text())
errors = []
if registry.get("version") != 1 or registry.get("default") != "closed":
    errors.append("transport stages require version 1 and default=closed")
seen = set()
for route in registry.get("routes", []):
    name = route.get("route")
    if not name or name in seen:
        errors.append(f"duplicate or missing transport route: {name}")
    seen.add(name)
    if route.get("stage") not in {
        "identify",
        "representation",
        "evaluate",
        "uncertainty",
        "prepare",
        "consume",
    }:
        errors.append(f"invalid stage: {name}")
    if route.get("status") == "licensed":
        for key in (
            "graph",
            "evidence",
            "functional",
            "guarantee",
            "evidence_test",
            "evidence_assertion",
        ):
            if not route.get(key):
                errors.append(f"{name}: missing {key}")
        path, assertion = route.get("evidence_test"), route.get("evidence_assertion")
        if path and assertion:
            evidence_path = root / path
            problems: list[str] = []
            if evidence_path.suffix == ".rs":
                _, problems = resolve_rust_test(evidence_path, assertion)
            elif evidence_path.suffix == ".py":
                problems = resolve_python_test(evidence_path, assertion)
            else:
                problems = [f"{name}: evidence must be a collected Rust or Python test"]
            errors.extend(problems)
            # A public stage route executes through its retained checked plan:
            # the cited test drops its builder, executes, and inspects the plan.
            if not problems and name.startswith("antecedent.transport."):
                errors.extend(
                    f"{name}: {problem}"
                    for problem in checked_execution_body_problems(closure(evidence_path, assertion), assertion)
                )
        if (
            route.get("stage") == "identify"
            and route.get("guarantee") != "sound_incomplete"
        ):
            pinned = {
                "antecedent.transport.advanced.identify_classical": (
                    "complete_in_classical_evidence_scope", "classical_complete_source_experimental_family",
                    "https://arxiv.org/abs/1312.7485v1"),
                "antecedent.transport.advanced.identify_meta": (
                    "complete_in_classical_meta_evidence_scope", "classical_complete_multi_source_experimental_families",
                    "https://proceedings.mlr.press/v31/bareinboim13a.pdf"),
            }
            if pinned.get(name) != (route.get("guarantee"), route.get("evidence"), route.get("reference")):
                errors.append(f"{name}: completeness requires its pinned classical evidence scope")
            path, assertion = route.get("conformance_test"), route.get("conformance_assertion")
            if not path or not assertion:
                errors.append(f"{name}: completeness requires consuming Rust branch conformance")
            else:
                _, problems = resolve_rust_test(root / path, assertion)
                errors.extend(problems)
    elif route.get("status") != "closed" or not route.get("reason_code"):
        errors.append(f"{name}: expected licensed or a reason-backed closed contract")
FAMILIES = {
    f"transport_{name}"
    for name in (
        "direct_regression", "standardize_regression", "target_only", "recursive_district",
        "negative_witness", "catalog_binding", "complementary_sources", "support_local",
        "grid_joint_inference", "multisample_inference", "prepared_lifecycle",
        "artifact_acceptance", "budget_refusal",
    )
}
EVIDENCE_CLASSES = {
    "external_parity", "exact_scm_truth", "internal_cross_check",
    "theoretical_witness", "statistical_calibration",
    "inferential_plumbing", "resampling_property",
}
roles: dict[str, dict[str, tuple[str, str]]] = {}
for row in registry.get("fixture_evidence", []):
    rid, family, role = row.get("id"), row.get("family"), row.get("role")
    if family not in FAMILIES or role not in {"positive", "counterexample"}:
        errors.append(f"{rid}: unknown fixture family or role")
        continue
    if rid != f"{family}.{role}" or role in roles.setdefault(family, {}):
        errors.append(f"{rid}: id must be the unique <family>.<role>")
    if row.get("evidence_class") not in EVIDENCE_CLASSES or not row.get("limits"):
        errors.append(f"{rid}: requires a known evidence_class and stated limits")
    path, assertion = row.get("evidence_test", ""), row.get("evidence_assertion", "")
    roles[family][role] = (path, assertion)
    if path.endswith(".rs"):
        errors.extend(resolve_rust_test(root / path, assertion)[1])
    elif path.endswith(".py"):
        errors.extend(resolve_python_test(root / path, assertion))
    else:
        errors.append(f"{rid}: evidence must be a collected Rust or Python test")
for family in sorted(FAMILIES):
    pair = roles.get(family, {})
    if set(pair) != {"positive", "counterexample"}:
        errors.append(f"{family}: requires a positive and a counterexample row")
    elif pair["positive"] == pair["counterexample"]:
        errors.append(f"{family}: one assertion cannot be both positive and counterexample")
required = {
    "antecedent.transport.identify",
    "antecedent.transport.reload_lowered_expression",
}
if not required.issubset(seen):
    errors.append("missing current public transport stage")
coverage_ids = {
    row.get("id")
    for row in tomllib.loads((root / "parity/coverage_records.toml").read_text()).get("record", [])
}
for route in registry.get("routes", []):
    for cid in route.get("coverage") or []:
        if cid not in coverage_ids:
            errors.append(f"{route.get('route')}: unknown coverage record {cid}")
    if route.get("coverage") and route.get("calibration_reason"):
        errors.append(
            f"{route.get('route')}: a route cannot cite coverage records and also state "
            "calibration_reason; the records must measure the route's own estimator"
        )
    for key in ("scm", "tolerance", "calibration_reason"):
        value = route.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{route.get('route')}: {key} must be a non-empty string")
for row in registry.get("fixture_evidence", []):
    for key in ("scm", "tolerance"):
        value = row.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{row.get('id')}: {key} must be a non-empty string")
if errors:
    print("Transport stage gate FAILED:\n" + "\n".join(f" - {e}" for e in errors))
    sys.exit(1)
print(f"Transport stage contracts OK ({len(seen)} routes; unlisted routes closed)")
