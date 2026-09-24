#!/usr/bin/env python3
"""Build and validate the registry-driven 2.1 compiler migration inventory."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "parity/support_licensed.toml"
ROUTES = ROOT / "parity/licensed_routes.toml"
TRANSPORT_STAGES = ROOT / "parity/transport_stages.toml"
INVENTORY = ROOT / "parity/compiler_migration.toml"

MODEL_OPERATIONS = {
    "AnomalyAttribution", "ChangeAttribution", "Counterfactual",
    "InterferenceQuery", "NestedCounterfactualEffect",
}
COMPOSITIONS = {"TemporalMediationEffect", "TransportQuery"}
EXPRESSION_ESTIMATORS = {"functional.distribution", "functional.effect"}
EXPRESSION_IDENTIFIERS = {"general.id"}
KINDS = {"expression_evaluation", "specialized_estimation", "model_operation", "composition"}
MIGRATION_STATES = {"pending", "in_progress", "verified"}


def coordinate(cell: dict) -> str:
    return ":".join(str(cell[k]) for k in ("query", "graph_class", "structure", "inference", "validation"))


def classify(route: dict) -> tuple[str, str]:
    coordinate_parts = route["coordinate"].split(":")
    query = coordinate_parts[0]
    if query in MODEL_OPERATIONS:
        return "model_operation", "query is a typed model or design operation"
    if query == "InterventionResponse":
        if len(coordinate_parts) >= 3 and coordinate_parts[2] == "graph_posterior":
            return "composition", "route composes checked results across graph-posterior members"
        return "specialized_estimation", "route estimates an intervention-response functional"
    if len(coordinate_parts) >= 3 and coordinate_parts[2] == "graph_posterior":
        return "composition", "route composes checked results across graph-posterior members"
    if query in COMPOSITIONS:
        return "composition", "route composes evidence, graph members, or temporal operations"
    if route["estimator"] in EXPRESSION_ESTIMATORS or route["identifier"] in EXPRESSION_IDENTIFIERS:
        return "expression_evaluation", "route evaluates an identified probability expression"
    if query in {
        "AverageDerivative", "AverageEffect", "ConditionalEffect", "DirectionalDerivative",
        "Elasticity", "InterventionalDistribution", "MediationEffect", "NestedCounterfactualEffect",
        "PathSpecificEffect", "PointDerivative", "PulseEffect", "ResponseCurve", "ResponseJacobian",
        "SemiElasticity", "SustainedEffect",
    }:
        return "specialized_estimation", "route selects a named estimator family or procedure"
    raise ValueError(f"unclassified licensed route query {query!r}: {route['coordinate']}")


def load_registries() -> tuple[dict[str, dict], dict[str, dict]]:
    support_rows = tomllib.loads(SUPPORT.read_text())["cell"]
    route_rows = tomllib.loads(ROUTES.read_text())["route"]
    support = {coordinate(row): row for row in support_rows}
    routes = {row["coordinate"]: row for row in route_rows}
    if len(support) != len(support_rows):
        raise ValueError("support_licensed.toml contains duplicate licensed coordinates")
    if len(routes) != len(route_rows):
        raise ValueError("licensed_routes.toml contains duplicate route coordinates")
    return support, routes


def optional_estimator_routes(support: dict[str, dict], routes: dict[str, dict]) -> dict[str, dict]:
    """Each additionally licensed estimator is a distinct execution path.

    The route registry names the default selected estimator. The support axis
    licenses other high-level estimator choices on that same coordinate, whose
    identifier is resolved only during preparation. They must not inherit the
    default route's compiler evidence.
    """
    out: dict[str, dict] = {}
    for base, cell in support.items():
        selected = routes[base]["estimator"]
        choices = cell.get("estimators", [])
        if len(choices) != len(set(choices)) or selected not in choices:
            raise ValueError(f"{base}: invalid licensed estimator axis")
        for estimator in choices:
            if estimator == selected:
                continue
            key = f"{base}::estimator={estimator}"
            out[key] = {
                "coordinate": key,
                "base_coordinate": base,
                "identifier": "selected_at_preparation",
                "estimator": estimator,
            }
    return out


def load_transport_stage_routes() -> dict[str, dict]:
    """Public stage routes are licensed separately from the analyze Cartesian matrix."""
    rows = tomllib.loads(TRANSPORT_STAGES.read_text()).get("routes", [])
    public = {
        f"transport-stage:{row['route']}": row
        for row in rows
        if row.get("status") == "licensed" and row.get("route", "").startswith("antecedent.transport.")
    }
    if len(public) != sum(
        row.get("status") == "licensed" and row.get("route", "").startswith("antecedent.transport.")
        for row in rows
    ):
        raise ValueError("transport_stages.toml contains duplicate licensed public transport routes")
    return public


def classify_transport_stage(route: dict) -> tuple[str, str]:
    stage = route["stage"]
    name = route["route"]
    if stage == "uncertainty":
        return "specialized_estimation", "public transport route provides an estimator-specific uncertainty procedure"
    if stage in {"identify", "prepare", "consume"}:
        return "composition", f"public transport {stage} route composes checked proof, binding, or artifact contracts"
    if stage == "representation":
        return "expression_evaluation", "public transport route retains and reloads a lowered expression"
    if stage == "evaluate":
        if "grid" in name.lower():
            return "composition", "public transport route evaluates a checked family of response-grid members"
        return "expression_evaluation", "public transport route evaluates a bound target distribution expression"
    raise ValueError(f"unclassified public transport stage {stage!r}: {name}")


def render() -> str:
    support, routes = load_registries()
    optional_routes = optional_estimator_routes(support, routes)
    stage_routes = load_transport_stage_routes()
    try:
        previous_rows = tomllib.loads(INVENTORY.read_text()).get("route", [])
    except FileNotFoundError:
        previous_rows = []
    previous = {row.get("coordinate"): row for row in previous_rows}
    if support.keys() != routes.keys():
        missing_routes = sorted(support.keys() - routes.keys())
        missing_support = sorted(routes.keys() - support.keys())
        raise ValueError(f"licensed registry mismatch: no route={missing_routes[:8]}, no support={missing_support[:8]}")
    rows = [
        "# Generated by scripts/compiler_migration.py; edit route registries, then regenerate.",
        "# Status is deliberately pending until the retained checked execution path is verified.",
        "",
    ]
    for key in sorted(routes):
        route = routes[key]
        kind, reason = classify(route)
        prior = previous.get(key, {})
        same_route = (
            prior.get("identifier") == route["identifier"]
            and prior.get("estimator") == route["estimator"]
            and prior.get("kind") == kind
            and prior.get("classification_reason") == reason
        )
        # Keep verified progress when regenerating unchanged coordinates. New or
        # changed route meanings return to pending/unverified automatically.
        status = prior.get("migration_status", "pending") if same_route else "pending"
        checked = prior.get("checked_execution", "unverified") if same_route else "unverified"
        builder_independent = prior.get("builder_independent", "unverified") if same_route else "unverified"
        semantics = prior.get("execution_semantics", "unverified") if same_route else "unverified"
        rows.extend([
            "[[route]]",
            f'coordinate = "{key}"',
            f'query = "{key.split(":", 1)[0]}"',
            f'identifier = "{route["identifier"]}"',
            f'estimator = "{route["estimator"]}"',
            f'kind = "{kind}"',
            f'classification_reason = "{reason}"',
            f'migration_status = "{status}"',
            f'checked_execution = "{checked}"',
            f'builder_independent = "{builder_independent}"',
            f'execution_semantics = "{semantics}"',
            "",
        ])
        if same_route and prior.get("evidence_test") and prior.get("evidence_assertion"):
            # Evidence references are user-maintained proof links, retained only
            # while the registry coordinate and route meaning remain unchanged.
            insert_at = len(rows) - 1
            rows[insert_at:insert_at] = [
                f'evidence_test = "{prior["evidence_test"]}"',
                f'evidence_assertion = "{prior["evidence_assertion"]}"',
            ]
    for key in sorted(optional_routes):
        route = optional_routes[key]
        kind, reason = classify(route)
        prior = previous.get(key, {})
        same_route = (
            prior.get("base_coordinate") == route["base_coordinate"]
            and prior.get("estimator") == route["estimator"]
            and prior.get("kind") == kind
            and prior.get("classification_reason") == reason
        )
        status = prior.get("migration_status", "pending") if same_route else "pending"
        checked = prior.get("checked_execution", "unverified") if same_route else "unverified"
        builder_independent = prior.get("builder_independent", "unverified") if same_route else "unverified"
        semantics = prior.get("execution_semantics", "unverified") if same_route else "unverified"
        rows.extend([
            "[[route]]",
            f'coordinate = "{key}"',
            'source_registry = "parity/support_licensed.toml"',
            f'base_coordinate = "{route["base_coordinate"]}"',
            f'query = "{key.split(":", 1)[0]}"',
            'identifier = "selected_at_preparation"',
            f'estimator = "{route["estimator"]}"',
            f'kind = "{kind}"',
            f'classification_reason = "{reason}"',
            f'migration_status = "{status}"',
            f'checked_execution = "{checked}"',
            f'builder_independent = "{builder_independent}"',
            f'execution_semantics = "{semantics}"',
            "",
        ])
        if same_route and prior.get("evidence_test") and prior.get("evidence_assertion"):
            insert_at = len(rows) - 1
            rows[insert_at:insert_at] = [
                f'evidence_test = "{prior["evidence_test"]}"',
                f'evidence_assertion = "{prior["evidence_assertion"]}"',
            ]
    for key in sorted(stage_routes):
        route = stage_routes[key]
        kind, reason = classify_transport_stage(route)
        prior = previous.get(key, {})
        same_route = (
            prior.get("route") == route["route"]
            and prior.get("stage") == route["stage"]
            and prior.get("kind") == kind
            and prior.get("classification_reason") == reason
        )
        status = prior.get("migration_status", "pending") if same_route else "pending"
        checked = prior.get("checked_execution", "unverified") if same_route else "unverified"
        builder_independent = prior.get("builder_independent", "unverified") if same_route else "unverified"
        semantics = prior.get("execution_semantics", "unverified") if same_route else "unverified"
        evidence_test = route.get("evidence_test") or route.get("conformance_test")
        evidence_assertion = route.get("evidence_assertion") or route.get("conformance_assertion")
        rows.extend([
            "[[route]]",
            f'coordinate = "{key}"',
            'source_registry = "parity/transport_stages.toml"',
            f'route = "{route["route"]}"',
            f'stage = "{route["stage"]}"',
            f'kind = "{kind}"',
            f'classification_reason = "{reason}"',
            f'migration_status = "{status}"',
            f'checked_execution = "{checked}"',
            f'builder_independent = "{builder_independent}"',
            f'execution_semantics = "{semantics}"',
            f'evidence_test = "{evidence_test}"',
            f'evidence_assertion = "{evidence_assertion}"',
            "",
        ])
    return "\n".join(rows)


def validate(release_gate: bool = False, verify_progress: bool = False) -> list[str]:
    issues: list[str] = []
    support, routes = load_registries()
    try:
        manifest = tomllib.loads(INVENTORY.read_text())
    except FileNotFoundError:
        return [f"missing {INVENTORY.relative_to(ROOT)}"]
    inventory_rows = manifest.get("route", [])
    inventory = {row.get("coordinate"): row for row in inventory_rows}
    stage_routes = load_transport_stage_routes()
    optional_routes = optional_estimator_routes(support, routes)
    expected_inventory_keys = routes.keys() | optional_routes.keys() | stage_routes.keys()
    if len(inventory) != len(inventory_rows):
        issues.append("compiler inventory contains duplicate coordinates")
    if support.keys() != routes.keys():
        issues.append("licensed support and licensed route registries have different coordinates")
    if inventory.keys() != expected_inventory_keys:
        issues.append(
            f"inventory coordinate mismatch: missing={sorted(expected_inventory_keys - inventory.keys())[:8]}, "
            f"stale={sorted(inventory.keys() - expected_inventory_keys)[:8]}"
        )
    route_checks = {key: (route, "analyze") for key, route in routes.items()}
    route_checks.update({key: (route, "optional_estimator") for key, route in optional_routes.items()})
    route_checks.update({key: (route, "transport_stage") for key, route in stage_routes.items()})
    for key, (route, source) in route_checks.items():
        row = inventory.get(key)
        if row is None:
            continue
        try:
            if source in {"analyze", "optional_estimator"}:
                expected_kind, expected_reason = classify(route)
            else:
                expected_kind, expected_reason = classify_transport_stage(route)
        except (KeyError, ValueError) as exc:
            issues.append(str(exc))
            continue
        if row.get("kind") not in KINDS or row.get("kind") != expected_kind:
            issues.append(f"{key}: missing, unknown, or incorrect route classification")
        if row.get("classification_reason") != expected_reason:
            issues.append(f"{key}: classification mapping/reason drift; regenerate inventory")
        if source in {"analyze", "optional_estimator"}:
            if row.get("identifier") != route.get("identifier") or row.get("estimator") != route.get("estimator"):
                issues.append(f"{key}: identifier or estimator drift; regenerate inventory")
            if source == "optional_estimator" and (
                row.get("base_coordinate") != route["base_coordinate"]
                or row.get("source_registry") != "parity/support_licensed.toml"
            ):
                issues.append(f"{key}: optional estimator source drift; regenerate inventory")
        else:
            if row.get("route") != route.get("route") or row.get("stage") != route.get("stage"):
                issues.append(f"{key}: public transport stage route drift; regenerate inventory")
            evidence_test = route.get("evidence_test") or route.get("conformance_test")
            evidence_assertion = route.get("evidence_assertion") or route.get("conformance_assertion")
            if row.get("evidence_test") != evidence_test or row.get("evidence_assertion") != evidence_assertion:
                issues.append(f"{key}: registered transport evidence citation drift; regenerate inventory")
        if row.get("migration_status") not in MIGRATION_STATES:
            issues.append(f"{key}: missing or unknown migration_status")
        if row.get("checked_execution") not in {"unverified", "verified"}:
            issues.append(f"{key}: missing or unknown checked_execution status")
        if row.get("builder_independent") not in {"unverified", "verified"}:
            issues.append(f"{key}: missing or unknown builder_independent status")
        if row.get("execution_semantics") not in {"unverified", "checked_plan", "builder_or_convention"}:
            issues.append(f"{key}: missing or unknown execution_semantics status")
        evidence_test = row.get("evidence_test")
        evidence_assertion = row.get("evidence_assertion")
        verified = row.get("migration_status") == "verified"
        if verified and (
            row.get("checked_execution") != "verified"
            or row.get("builder_independent") != "verified"
            or row.get("execution_semantics") != "checked_plan"
        ):
            issues.append(
                f"{key}: verified status requires checked execution, builder independence, "
                "and checked-plan semantics"
            )
        if row.get("migration_status") == "pending" and (
            row.get("checked_execution") != "unverified"
            or row.get("builder_independent") != "unverified"
            or row.get("execution_semantics") != "unverified"
        ):
            issues.append(f"{key}: pending route cannot carry verified execution claims")
        if verified and (not evidence_test or not evidence_assertion):
            issues.append(f"{key}: verified route must name evidence_test and evidence_assertion")
        if verified and evidence_test and evidence_assertion:
            evidence_problems = validate_route_evidence(evidence_test, evidence_assertion)
            issues.extend(f"{key}: {problem}" for problem in evidence_problems)
        if release_gate and (
            row.get("migration_status") != "verified"
            or row.get("checked_execution") != "verified"
            or row.get("builder_independent") != "verified"
            or row.get("execution_semantics") != "checked_plan"
        ):
            issues.append(f"{key}: 2.1 release blocked; checked plan execution independent of builders is not verified")
    if (release_gate or verify_progress) and not issues:
        run_route_evidence(inventory.values(), issues)
    return issues


def validate_route_evidence(test_path: str, assertion: str) -> list[str]:
    """Check the named test is collected/compiled and its helper closure proves discard + execution."""
    sys.path.insert(0, str((ROOT / "scripts").resolve()))
    import test_evidence

    path = (ROOT / test_path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return ["evidence_test must be a repository-relative path"]
    if path.suffix == ".py":
        problems = test_evidence.resolve_python_test(path, assertion, ROOT)
        body = test_evidence.python_closure(path, assertion)
    else:
        _full_name, problems = test_evidence.resolve_rust_test(path, assertion, ROOT)
        body = test_evidence.closure(path, assertion)
    if problems:
        return [f"evidence assertion {assertion!r} is not an executing test: {problem}" for problem in problems]
    return validate_evidence_body(body, assertion)


def validate_evidence_body(body: str, assertion: str) -> list[str]:
    """Require concrete source signals for builder discard, plan inspection, and execution."""
    builder_name = r"(?:[A-Za-z_][A-Za-z0-9_]*)?builder[A-Za-z0-9_]*"
    discarded = (
        re.search(rf"\bdrop\s*\(\s*{builder_name}\s*\)", body, re.I)
        or re.search(rf"\bdel\s+{builder_name}\b", body, re.I)
        or re.search(rf"\b{builder_name}\s*=\s*None\b", body, re.I)
    )
    if not discarded:
        return [f"evidence assertion {assertion!r} does not explicitly discard its builder before execution"]
    if not re.search(r"\b(?:execute|estimate|run)\s*\(", body):
        return [f"evidence assertion {assertion!r} does not execute the prepared plan"]
    if not re.search(r"\b(?:program|plan|lowering)\b", body, re.I):
        return [f"evidence assertion {assertion!r} does not inspect a retained program or plan"]
    return []


def run_route_evidence(rows, issues: list[str]) -> None:
    """Execute each distinct verified evidence test on the 2.1 release gate."""
    import test_evidence

    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("migration_status") != "verified":
            continue
        key = (row["evidence_test"], row["evidence_assertion"])
        if key in seen:
            continue
        seen.add(key)
        path = ROOT / key[0]
        if path.suffix == ".py":
            problems = test_evidence.resolve_python_test(path, key[1], ROOT)
            if problems:
                issues.append(f"{row['coordinate']}: evidence test could not be resolved: {'; '.join(problems)}")
                continue
            rel = path.resolve().relative_to((ROOT / "python").resolve())
            command = [
                "uv", "run", "--quiet", "--project", ".", "pytest", "-q",
                "-p", "no:cacheprovider", f"{rel}::{key[1]}",
            ]
            cwd = ROOT / "python"
        else:
            target_root = test_evidence.target_root(path)
            if target_root is None:
                issues.append(f"{row['coordinate']}: evidence test has no Cargo target")
                continue
            _, crate, target = target_root
            full_name, problems = test_evidence.resolve_rust_test(path, key[1], ROOT)
            if problems or full_name is None:
                issues.append(f"{row['coordinate']}: evidence test could not be resolved: {'; '.join(problems)}")
                continue
            command = ["cargo", "test", "-q", "-p", crate, *target, "--", full_name, "--exact"]
            cwd = ROOT
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        if result.returncode:
            detail = (result.stdout + result.stderr)[-2500:]
            issues.append(f"{row['coordinate']}: evidence test failed ({' '.join(command)}):\n{detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="regenerate the checked-in inventory")
    parser.add_argument("--release-gate", action="store_true", help="require every licensed route to be verified")
    parser.add_argument("--progress", action="store_true", help="show closure counts by execution family")
    parser.add_argument(
        "--verify-progress",
        action="store_true",
        help="execute every currently verified route's cited test without requiring full release closure",
    )
    args = parser.parse_args()
    if args.write:
        INVENTORY.write_text(render())
    issues = validate(args.release_gate, args.verify_progress)
    if issues:
        print("compiler migration gate failed:", file=sys.stderr)
        for issue in issues[:80]:
            print(f"- {issue}", file=sys.stderr)
        if len(issues) > 80:
            print(f"- ... and {len(issues) - 80} more", file=sys.stderr)
        return 1
    total = len(tomllib.loads(INVENTORY.read_text())["route"])
    print(f"compiler migration inventory: ok ({total} licensed routes classified, including {len(load_transport_stage_routes())} public transport stages)")
    if args.progress:
        rows = tomllib.loads(INVENTORY.read_text())["route"]
        for kind in sorted(KINDS):
            counts = Counter(row["migration_status"] for row in rows if row["kind"] == kind)
            print(
                f"{kind}: verified={counts['verified']} "
                f"in_progress={counts['in_progress']} pending={counts['pending']}"
            )
    if args.release_gate:
        print("compiler migration release gate: all licensed routes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
