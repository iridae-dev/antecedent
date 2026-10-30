#!/usr/bin/env python3
"""Cross-check checked-recipe obligations with the support matrix.

A checked recipe owes a route obligation and a test obligation. Identification
and calibration stay on the explicit license in ``support_licensed.toml``. A
coordinate that matches ``support_closed.toml`` stays closed, including when a
typed-impossibility rule also matches. Obligations for a licensed cell are that
cell's estimators and checked-execution citations. An estimator the support
matrix does not list is not a license. A matrix estimator that is only the
calibration join is not a recipe obligation.

``scripts/gate_checked_execution.py`` still fails when a licensed route bypasses
its retained program. This script does not replace that check.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

AXIS = ("query", "graph_class", "structure", "inference", "validation")
RULE_AXES = (
    ("queries", "query"),
    ("graph_classes", "graph_class"),
    ("structures", "structure"),
    ("inferences", "inference"),
    ("validations", "validation"),
)
IDENTIFICATION = (
    "evidence_kind",
    "evidence_test",
    "evidence_assertion",
    "known_truth_fixture",
)
CALIBRATION = ("calibration", "calibration_reason", "calibration_estimators")
OBLIGATION_KEYS = frozenset({"routes", "tests"})


def coordinate(cell: dict) -> str:
    return ":".join(str(cell.get(key)) for key in AXIS)


def rule_values(rule: dict, name: str) -> tuple:
    raw = rule.get(name)
    if not raw:
        return ()
    return tuple(raw)


def rule_key(rule: dict) -> tuple:
    return tuple(rule_values(rule, name) for name, _cell in RULE_AXES) + (rule.get("reason"),)


def rule_matches(rule: dict, cell: dict) -> bool:
    for rule_key_name, cell_key in RULE_AXES:
        allowed = rule.get(rule_key_name)
        if not allowed:
            continue
        if cell.get(cell_key) not in allowed:
            return False
    return True


def standing(cell: dict, closed_rules: list[dict], na_rules: list[dict], licensed: set[str]) -> str:
    """Closed wins over a license and over ``not_applicable``."""
    if any(rule_matches(rule, cell) for rule in closed_rules):
        return "closed"
    if any(rule_matches(rule, cell) for rule in na_rules):
        return "not_applicable"
    if coordinate(cell) in licensed:
        return "licensed"
    return "unlicensed"


def emit_obligations(row: dict) -> dict:
    """Route and test obligations copied from a licensed row.

    Identification and calibration fields on ``row`` are not copied.
    """
    routes = []
    for estimator in row.get("estimators") or []:
        routes.append({key: row.get(key) for key in AXIS} | {"estimator": estimator})
    tests = []
    for entry in row.get("checked_execution") or []:
        if not isinstance(entry, dict):
            tests.append(entry)
            continue
        tests.append(
            {
                "estimator": entry.get("estimator"),
                "test": entry.get("test"),
                "assertion": entry.get("assertion"),
            }
        )
    return {"routes": routes, "tests": tests}


def witness(rule: dict, axes: dict[str, list[str]]) -> dict:
    cell = {}
    for rule_key_name, cell_key in RULE_AXES:
        allowed = rule.get(rule_key_name)
        pool = axes[cell_key]
        cell[cell_key] = allowed[0] if allowed else pool[0]
    return cell


def _strings(values) -> list[str] | None:
    if not isinstance(values, list) or not values:
        return None
    if not all(isinstance(value, str) and value.strip() for value in values):
        return None
    return values


def cross_check(
    licensed_rows: list[dict],
    matrix_licensed: list[dict],
    closed_rules: list[dict],
    na_rules: list[dict],
    *,
    closed_snapshot: list[dict] | None = None,
    na_snapshot: list[dict] | None = None,
    axes: dict[str, list[str]] | None = None,
    cite=None,
) -> list[str]:
    """Problems in the obligation cross-check. Empty means the records agree."""
    problems: list[str] = []
    if closed_snapshot is not None and [rule_key(rule) for rule in closed_rules] != [
        rule_key(rule) for rule in closed_snapshot
    ]:
        problems.append("support_closed.toml does not match support_matrix.json closed_rules")
    if na_snapshot is not None and [rule_key(rule) for rule in na_rules] != [
        rule_key(rule) for rule in na_snapshot
    ]:
        problems.append("support_n_a.toml does not match support_matrix.json na_rules")

    matrix_by: dict[str, dict] = {}
    for cell in matrix_licensed:
        key = coordinate(cell)
        if key in matrix_by:
            problems.append(f"{key}: support matrix lists this licensed coordinate twice")
        matrix_by[key] = cell
    licensed_keys = {coordinate(row) for row in licensed_rows}
    if len(licensed_keys) != len(licensed_rows):
        problems.append("support_licensed.toml lists a coordinate twice")
    missing = sorted(set(matrix_by) - licensed_keys)
    extra = sorted(licensed_keys - set(matrix_by))
    for key in missing:
        problems.append(f"{key}: support matrix licenses a route the license table does not")
    for key in extra:
        query = key.split(":", 1)[0]
        problems.append(f"{query}: license table names {key}, which the support matrix does not")

    for row in licensed_rows:
        key = coordinate(row)
        query = str(row.get("query"))
        label = f"{query} ({key})"
        status = standing(row, closed_rules, na_rules, licensed_keys)
        if status == "closed":
            problems.append(f"{label} stays closed; a license does not open it")
            continue
        if status == "not_applicable":
            problems.append(f"{label} is not_applicable and is not a license")
            continue
        matrix = matrix_by.get(key)
        if matrix is None:
            continue
        estimators = _strings(row.get("estimators"))
        matrix_estimators = _strings(matrix.get("estimators"))
        route_estimator = matrix.get("route_estimator")
        if estimators is None or matrix_estimators is None or not isinstance(route_estimator, str):
            problems.append(f"{label}: licensed obligations need estimators and a route estimator")
            continue
        obligations = emit_obligations(row)
        if set(obligations) != OBLIGATION_KEYS:
            problems.append(
                f"{label}: a recipe emitted something other than route and test obligations"
            )
        for field in IDENTIFICATION + CALIBRATION:
            if field in obligations:
                problems.append(f"{label}: recipe obligations include explicit {field}")
        named = [item.get("estimator") for item in obligations["routes"]]
        if named != estimators:
            problems.append(f"{label} is licensed for {estimators}")
        if route_estimator not in estimators:
            problems.append(f"{label} is licensed for route estimator {route_estimator}")
        invented = [item for item in estimators if item not in matrix_estimators]
        if invented:
            problems.append(f"{label} is licensed on the support matrix for {matrix_estimators}")
        tests = obligations["tests"]
        test_names = []
        for entry in tests:
            if not isinstance(entry, dict):
                problems.append(f"{label}: checked execution entry is not a citation")
                continue
            estimator, test_rel, assertion = (
                entry.get("estimator"),
                entry.get("test"),
                entry.get("assertion"),
            )
            if not all(
                isinstance(value, str) and value.strip()
                for value in (estimator, test_rel, assertion)
            ):
                problems.append(f"{label} is licensed for {estimators}")
                continue
            test_names.append(estimator)
            if estimator not in matrix_estimators:
                problems.append(
                    f"{label} is licensed on the support matrix for {matrix_estimators}"
                )
            if cite is not None:
                for problem in cite(test_rel, assertion):
                    problems.append(f"{label}::estimator={estimator}: {problem}")
        if sorted(test_names) != sorted(estimators):
            problems.append(f"{label} is licensed for {estimators}")
        extras = [item for item in matrix_estimators if item not in estimators]
        for estimator in extras:
            if any(
                isinstance(entry, dict) and entry.get("estimator") == estimator for entry in tests
            ):
                problems.append(
                    f"{label} is licensed for {estimators}; {estimator} stays on the matrix record"
                )

    if axes:
        for rule in closed_rules:
            cell = witness(rule, axes)
            status = standing(cell, closed_rules, na_rules, licensed_keys)
            if status != "closed":
                problems.append(
                    f"{cell.get('query')}: a closed reason stays closed, standing {status}"
                )
            if status == "not_applicable":
                problems.append(f"{cell.get('query')}: a closed cell was relabeled not_applicable")
        for combo in _unlicensed_search(axes, closed_rules, na_rules, licensed_keys):
            if standing(combo, closed_rules, na_rules, licensed_keys) != "unlicensed":
                problems.append(f"{combo.get('query')}: an unlicensed route was relabeled")
            break
    return problems


def _unlicensed_search(axes, closed_rules, na_rules, licensed_keys):
    from itertools import product

    keys = list(AXIS)
    pools = [axes[key] for key in keys]
    for values in product(*pools):
        cell = dict(zip(keys, values, strict=False))
        if standing(cell, closed_rules, na_rules, licensed_keys) == "unlicensed":
            yield cell
            return


def matrix_alignment_problems(root: Path, cite=None) -> list[str]:
    """Load the records the support-matrix gate already reads and cross-check them."""
    licensed_rows = tomllib.loads(
        (root / "parity" / "support_licensed.toml").read_text(encoding="utf-8")
    ).get("cell", [])
    closed_rules = tomllib.loads(
        (root / "parity" / "support_closed.toml").read_text(encoding="utf-8")
    ).get("closed", [])
    na_rules = tomllib.loads(
        (root / "parity" / "support_n_a.toml").read_text(encoding="utf-8")
    ).get("n_a", [])
    axes_doc = tomllib.loads((root / "parity" / "support_axes.toml").read_text(encoding="utf-8"))
    matrix = json.loads(
        (root / "crates" / "antecedent" / "data" / "support_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    axes = {
        "query": list(axes_doc["queries"]),
        "graph_class": list(axes_doc["graph_classes"]),
        "structure": list(axes_doc["structures"]),
        "inference": list(axes_doc["inferences"]),
        "validation": list(axes_doc["validations"]),
    }
    return cross_check(
        licensed_rows,
        matrix.get("licensed") or [],
        closed_rules,
        na_rules,
        closed_snapshot=matrix.get("closed_rules") or [],
        na_snapshot=matrix.get("na_rules") or [],
        axes=axes,
        cite=cite,
    )


def main() -> None:
    from _repo import REPO_ROOT, ensure_scripts_on_path

    ensure_scripts_on_path()
    import test_evidence  # noqa: E402

    problems = matrix_alignment_problems(
        REPO_ROOT, cite=test_evidence.checked_execution_citation_problems
    )
    if problems:
        print(f"FAIL: recipe obligations: {len(problems)} cross-check failure(s)", flush=True)
        for problem in problems:
            print(f"FAIL: {problem}", flush=True)
        raise SystemExit(1)
    print("recipe obligations match the support matrix", flush=True)


if __name__ == "__main__":
    main()
