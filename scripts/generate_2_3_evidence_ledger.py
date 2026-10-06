#!/usr/bin/env python3
"""Generate or check the frozen 2.3 evidence allocation ledger.

Pending oracle values are explicit blockers, never evidence of promotion.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "parity/promotion_2_3.toml"
LEDGER = ROOT / "parity/2_3_evidence_ledger.toml"


def quote(value: str) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def build() -> str:
    records = tomllib.loads(REGISTRY.read_text())["record"]
    lines = [
        '# Allocations for frozen 2.3 routes. "pending" fields block in_progress status.',
        'version = 1',
        'release = "2.3"',
    ]
    for record in records:
        rid = record["id"]
        positive = next(f["id"] for f in record["fixtures"] if f["role"] == "positive")
        artifact = next(f["id"] for f in record["fixtures"] if f["role"] == "artifact")
        calibration = record.get("coverage_records", [])
        if rid == "2.3A.F15.joint_distribution_semantics":
            oracle = "enumerated_finite_law"
            inputs = "two aligned, equally weighted draws (0,0) and (1,2)"
            expected = "means=(0.5,1); population covariance=0.5; E[XY]=1"
            tolerance = "absolute 1e-12"
            provenance = "provenance/f15_joint_distribution.toml (semantic contract; no estimator paper)"
        else:
            oracle = "pending_independent_oracle"
            inputs = "pending exact fixture inputs"
            expected = "pending independent expected values"
            tolerance = "pending predeclared tolerance"
            provenance = f"provenance/{rid.split('.', 1)[1].replace('.', '_')}.toml"
        for route in record["routes"]:
            lines += [
                "", "[[cell]]",
                f'id = {quote(rid)}',
                f'route = {quote(route["name"])}',
                f'work_package = {quote(record["work_package"])}',
                f'algorithm_provenance = {quote(provenance)}',
                'parity_registry = "parity/promotion_2_3.toml"',
                'support_registry = "parity/transport_stages.toml"',
                f'positive_fixture = {quote(positive)}',
                f'oracle_kind = {quote(oracle)}',
                f'oracle_inputs = {quote(inputs)}',
                f'expected_values = {quote(expected)}',
                f'tolerance = {quote(tolerance)}',
                f'calibration = {quote(", ".join(calibration) if calibration else "exact_or_point_only_no_interval")}',
                'executing_gate = "scripts/gate_promotion.sh"',
                f'artifact_consumer_fixture = {quote(artifact)}',
                'provider_trust = "native_route_pending_evidence"',
                'exact_request_verification = "not_applicable_no_external_provider"',
            ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build()
    if args.check:
        if not LEDGER.exists() or LEDGER.read_text() != expected:
            print("2.3 evidence ledger is out of date")
            return 1
        records = tomllib.loads(REGISTRY.read_text())["record"]
        for record in records:
            if record["status"] == "in_progress" and record["id"] != "2.3A.F15.joint_distribution_semantics":
                print(f'{record["id"]}: independent oracle values are pending')
                return 1
        print(f"2.3 evidence ledger: {len(tomllib.loads(expected)['cell'])} routes allocated")
    else:
        LEDGER.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
