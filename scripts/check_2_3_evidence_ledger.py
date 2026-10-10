#!/usr/bin/env python3
"""Audit the hand-maintained 2.3 route-to-evidence ledger.

Frozen cells may allocate pending evidence. Implemented cells may not.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "parity/2_3_evidence_ledger.toml"
RECORDS = ROOT / "parity/promotion_2_3.toml"
STAGES = ROOT / "parity/transport_stages.toml"

ORACLE_KINDS = {
    "semantic_contract",
    "enumerated_finite_law",
    "closed_form",
    "paper_example",
    "external_package",
    "pending_independent_oracle",
}


def problems(cells: list[dict], records: list[dict], stages: list[dict]) -> list[str]:
    errors: list[str] = []
    expected = {(record["id"], route["name"]): (record, route)
                for record in records for route in record["routes"]}
    stage_by_route: dict[str, list[dict]] = {}
    for stage in stages:
        stage_by_route.setdefault(stage["route"], []).append(stage)
    seen: set[tuple[str, str]] = set()
    for cell in cells:
        key = (cell.get("id"), cell.get("route"))
        if key in seen:
            errors.append(f"{key}: duplicate ledger row")
            continue
        seen.add(key)
        if key not in expected:
            errors.append(f"{key}: no matching 2.3 promotion route")
            continue
        record, route = expected[key]
        prefix = f"{record['id']} {route['name']}"
        if cell.get("work_package") != record["work_package"]:
            errors.append(f"{prefix}: work_package disagrees with promotion record")
        for field, wanted in (
            ("parity_registry", "parity/promotion_2_3.toml"),
            ("support_registry", "parity/transport_stages.toml"),
            ("executing_gate", "scripts/gate_promotion.sh"),
        ):
            if cell.get(field) != wanted:
                errors.append(f"{prefix}: {field} must be {wanted}")
        matches = stage_by_route.get(route["name"], [])
        if not any(stage.get("stage") == route["stage"]
                   and stage.get("status") == route["status"]
                   and (route["status"] != "closed"
                        or stage.get("reason_code") == route["reason_code"])
                   for stage in matches):
            errors.append(f"{prefix}: no matching stage status and reason code")
        fixtures = {(fixture["id"], fixture["role"]) for fixture in record["fixtures"]}
        if (cell.get("positive_fixture"), "positive") not in fixtures:
            errors.append(f"{prefix}: positive_fixture is not owned by record")
        if (cell.get("artifact_consumer_fixture"), "artifact") not in fixtures:
            errors.append(f"{prefix}: artifact_consumer_fixture is not owned by record")
        provenance = cell.get("algorithm_provenance")
        if not isinstance(provenance, list) or not provenance or any(
            not isinstance(path, str) or not path.startswith("provenance/")
            and not path.startswith("pending:provenance/") for path in provenance
        ):
            errors.append(f"{prefix}: algorithm_provenance must list record paths or pending paths")
            provenance = []
        for path in provenance:
            if not path.startswith("pending:") and not (ROOT / path).is_file():
                errors.append(f"{prefix}: missing algorithm provenance {path}")
        if cell.get("oracle_kind") not in ORACLE_KINDS:
            errors.append(f"{prefix}: unknown oracle_kind")
        for field in ("oracle_inputs", "expected_values", "tolerance", "calibration",
                      "provider_trust", "exact_request_verification"):
            if not isinstance(cell.get(field), str) or not cell[field].strip():
                errors.append(f"{prefix}: {field} is required")
        coverage = record.get("coverage_records", [])
        if coverage and any(item not in cell.get("calibration", "") for item in coverage):
            errors.append(f"{prefix}: calibration omits a promotion coverage coordinate")
        if not coverage and cell.get("calibration", "").startswith("cov."):
            errors.append(f"{prefix}: undeclared calibration coordinate")
        if record["status"] in {"in_progress", "promoted"}:
            if any(path.startswith("pending:") for path in provenance):
                errors.append(f"{prefix}: implemented cell has pending provenance")
            for field in ("oracle_kind", "oracle_inputs", "expected_values", "tolerance"):
                if str(cell.get(field, "")).startswith("pending"):
                    errors.append(f"{prefix}: implemented cell has pending {field}")
            if cell.get("provider_trust") == "native_route_pending_evidence":
                errors.append(f"{prefix}: implemented cell has pending provider trust")
    for key in expected.keys() - seen:
        errors.append(f"{key}: promotion route missing from evidence ledger")
    return errors


def self_test() -> None:
    record = {
        "id": "2.3A.F.test", "work_package": "A0", "status": "frozen",
        "routes": [{"name": "test.route", "stage": "consume", "status": "closed",
                    "reason_code": "cell_not_licensed"}],
        "fixtures": [{"id": "test.positive", "role": "positive"},
                     {"id": "test.artifact", "role": "artifact"}],
    }
    stage = {"route": "test.route", "stage": "consume", "status": "closed",
             "reason_code": "cell_not_licensed"}
    cell = {
        "id": "2.3A.F.test", "route": "test.route", "work_package": "A0",
        "algorithm_provenance": ["pending:provenance/test.toml"],
        "parity_registry": "parity/promotion_2_3.toml",
        "support_registry": "parity/transport_stages.toml",
        "positive_fixture": "test.positive", "oracle_kind": "pending_independent_oracle",
        "oracle_inputs": "pending inputs", "expected_values": "pending values",
        "tolerance": "pending tolerance", "calibration": "point_only_no_interval",
        "executing_gate": "scripts/gate_promotion.sh",
        "artifact_consumer_fixture": "test.artifact",
        "provider_trust": "native_route_pending_evidence",
        "exact_request_verification": "not_applicable_no_external_provider",
    }
    assert not problems([cell], [record], [stage])
    assert any("duplicate" in item for item in problems([cell, cell], [record], [stage]))
    assert any("missing from evidence ledger" in item for item in problems([], [record], [stage]))
    assert any("no matching stage" in item for item in problems([cell], [record], []))
    assert any("pending provenance" in item for item in problems(
        [cell], [{**record, "status": "in_progress"}], [stage]))
    print("2.3 evidence ledger self-test: ok")


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(sys.argv) != 1:
        raise SystemExit("usage: check_2_3_evidence_ledger.py [--self-test]")
    ledger = tomllib.loads(LEDGER.read_text())
    if ledger.get("version") != 1 or ledger.get("release") != "2.3":
        print("2.3 evidence ledger: unsupported header", file=sys.stderr)
        return 1
    records = tomllib.loads(RECORDS.read_text())["record"]
    stages = tomllib.loads(STAGES.read_text())["routes"]
    errors = problems(ledger["cell"], records, stages)
    if errors:
        print("\n".join(f"FAIL: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"2.3 evidence ledger: {len(ledger['cell'])} owned routes checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
