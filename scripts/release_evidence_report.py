#!/usr/bin/env python3
"""Report promotion evidence and enforce explicit closed-cell dispositions.

This inventories cited evidence, not execution or calibration. A consistent report
does not certify a release. The promotion runner and release-candidate gates still
own those verdicts. Reads the one registry, parity/promotion.toml.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import tomllib
from pathlib import Path

from _repo import REPO_ROOT as REPO


def build_report(root: Path) -> dict:
    dispositions = tomllib.loads(
        (root / "parity/promotion_carryovers.toml").read_text(encoding="utf-8")
    )
    if dispositions.get("version") != 1:
        raise ValueError("unsupported carryover disposition version")
    cells = dispositions.get("cell", [])
    by_id = {c["id"]: c for c in cells}
    if len(by_id) != len(cells):
        raise ValueError("duplicate carryover disposition")
    registry = "parity/promotion.toml"
    data = tomllib.loads((root / registry).read_text(encoding="utf-8"))
    if "release" in data:
        raise ValueError(f"{registry}: release field is not a claim")
    records = data.get("record", [])
    all_records: dict[str, dict] = {}
    for record in records:
        rid = record["id"]
        if rid in all_records:
            raise ValueError(f"duplicate promotion record {rid}")
        all_records[rid] = record
    carried = {rid for rid, r in all_records.items() if r["status"] == "carried_forward"}
    if set(by_id) != carried:
        raise ValueError(
            f"carryover disposition mismatch: missing={sorted(carried - set(by_id))}; "
            f"stale={sorted(set(by_id) - carried)}"
        )
    for rid in sorted(carried):
        record, disposition = all_records[rid], by_id[rid]
        gates = disposition.get("failed_gates", [])
        if not gates or any(not isinstance(g, str) or not g.strip() for g in gates):
            raise ValueError(f"{rid}: precise failed gates required")
        for key in ("detail", "corrective_action"):
            if not isinstance(disposition.get(key), str) or not disposition[key].strip():
                raise ValueError(f"{rid}: {key} required")
        if not record.get("owners"):
            raise ValueError(f"{rid}: corrective owners required")
        routes = record.get("routes", [])
        if not routes or any(r.get("status") != "closed" for r in routes):
            raise ValueError(f"{rid}: every carried route must remain closed")
        for route in routes:
            for key in ("reason_code", "refusal_test", "refusal_assertion"):
                if not route.get(key):
                    raise ValueError(f"{rid}: {route['name']}: unchanged refusal {key} required")
            if not any(r.get("code") == route["reason_code"] for r in record.get("refusals", [])):
                raise ValueError(f"{rid}: {route['name']}: refusal reason absent from record")
    summaries = []
    for record in records:
        summary = {
            "id": record["id"],
            "capability": record.get("capability", ""),
            "status": record["status"],
            "inference_claim": record["inference_claim"],
            "owners": record.get("owners", []),
            "coverage_records": record.get("coverage_records", []),
            "fixtures": record.get("fixtures", []),
            "routes": record.get("routes", []),
        }
        if record["id"] in by_id:
            summary["carryover"] = by_id[record["id"]]
            summary["refusals"] = record.get("refusals", [])
        summaries.append(summary)
    return {
        "report_kind": "promotion_evidence_inventory",
        "execution_verified": False,
        "calibration_verified": False,
        "release_certified": False,
        "registry": registry,
        "records": summaries,
    }


def render(report: dict) -> str:
    lines = ["Promotion evidence inventory (execution and calibration not verified)"]
    records = report["records"]
    counts = {
        state: sum(r["status"] == state for r in records)
        for state in sorted({r["status"] for r in records})
    }
    lines += [
        "",
        f"{report['registry']}; {len(records)} records; {counts}",
    ]
    for record in records:
        lines.append(
            f"  {record['id']}: {record['status']}; "
            f"claim={record['inference_claim']}; "
            f"{len(record['fixtures'])} cited fixtures; "
            f"{len(record['routes'])} routes; "
            f"{len(record['coverage_records'])} allocated coverage records"
        )
        if "carryover" not in record:
            continue
        disposition = record["carryover"]
        lines += [
            f"    failed gates: {', '.join(disposition['failed_gates'])}",
            f"    gap: {disposition['detail']}",
            f"    owners: {', '.join(record['owners'])}",
            f"    corrective action: {disposition['corrective_action']}",
        ]
        for route in record["routes"]:
            details = [r["detail"] for r in record["refusals"] if r["code"] == route["reason_code"]]
            lines.append(
                f"    unchanged closed route: {route['name']} -> "
                f"{route['reason_code']} "
                f"({', '.join(details)}); refusal evidence: "
                f"{route['refusal_test']}::{route['refusal_assertion']}"
            )
    lines += [
        "",
        (
            "Inventory consistent; release certification remains owned by the "
            "executed-evidence and release-candidate gates."
        ),
    ]
    return "\n".join(lines)


def self_test() -> int:
    # Pin synthetic closed records: promotion in the live registry must not
    # remove the very broken-input controls this checker is meant to exercise.
    # No measurement, build or release gate runs.
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "parity").mkdir()
        (root / "parity/promotion.toml").write_text(
            "\n".join(
                f'''[[record]]
id = "{rid}"
status = "carried_forward"
inference_claim = "none"
owners = ["synthetic_owner"]
routes = [{{ name = "synthetic.{index}", status = "closed", reason_code = "cell_not_licensed", refusal_test = "synthetic.rs", refusal_assertion = "refuses_{index}" }}]
refusals = [{{ code = "cell_not_licensed", detail = "synthetic.closed" }}]
'''
                for index, rid in enumerate(
                    ["sensitivity.joint_uncertainty", "synthetic.transport"]
                )
            ),
            encoding="utf-8",
            newline="\n",
        )
        (root / "parity/promotion_carryovers.toml").write_text(
            """version = 1
[[cell]]
id = "sensitivity.joint_uncertainty"
failed_gates = ["public_interval_producer", "independent_public_interval_consumer"]
detail = "Synthetic public interval gap."
corrective_action = "Implement and exercise both synthetic routes."
[[cell]]
id = "synthetic.transport"
failed_gates = ["synthetic_transport"]
detail = "Synthetic transport gap."
corrective_action = "Add independent transport evidence."
""",
            encoding="utf-8",
            newline="\n",
        )
        report = build_report(root)
        assert report["records"]
        assert not any(
            report[k] for k in ("execution_verified", "calibration_verified", "release_certified")
        )
        assert "parity/promotion.toml;" in render(report)
        baseline = (root / "parity/promotion_carryovers.toml").read_text(encoding="utf-8")
        correction = tomllib.loads(baseline)["cell"][1]["corrective_action"]
        for before, after, expected in [
            ('id = "sensitivity.joint_uncertainty"', 'id = "unknown"', "mismatch"),
            (
                'failed_gates = ["public_interval_producer", "independent_public_interval_consumer"]',
                "failed_gates = []",
                "precise failed gates",
            ),
            (
                f"corrective_action = {json.dumps(correction)}",
                'corrective_action = ""',
                "corrective_action required",
            ),
        ]:
            assert before in baseline
            (root / "parity/promotion_carryovers.toml").write_text(
                baseline.replace(before, after, 1), encoding="utf-8", newline="\n"
            )
            try:
                build_report(root)
            except ValueError as error:
                assert expected in str(error), str(error)
            else:
                raise AssertionError(f"mutation accepted: {expected}")
        (root / "parity/promotion_carryovers.toml").write_text(
            baseline, encoding="utf-8", newline="\n"
        )
        duplicate = baseline[
            baseline.index("[[cell]]") : baseline.index("[[cell]]", baseline.index("[[cell]]") + 1)
        ]
        (root / "parity/promotion_carryovers.toml").write_text(
            baseline + duplicate, encoding="utf-8", newline="\n"
        )
        try:
            build_report(root)
        except ValueError as error:
            assert "duplicate carryover disposition" in str(error)
        else:
            raise AssertionError("duplicate disposition accepted")
        (root / "parity/promotion_carryovers.toml").write_text(
            baseline, encoding="utf-8", newline="\n"
        )
        registry = root / "parity/promotion.toml"
        original = registry.read_text(encoding="utf-8")
        # All carried routes must be closed regardless of the report filter.
        carried_at = original.index('id = "sensitivity.joint_uncertainty"')
        closed_at = original.index('status = "closed"', carried_at)
        registry.write_text(
            original[:closed_at]
            + 'status = "licensed"'
            + original[closed_at + len('status = "closed"') :],
            encoding="utf-8",
            newline="\n",
        )
        try:
            build_report(root)
        except ValueError as error:
            assert "every carried route must remain closed" in str(error)
        else:
            raise AssertionError("licensed carried route accepted")
        registry.write_text(original, encoding="utf-8", newline="\n")
        # Losing an executable refusal citation must fail even when the route
        # still has a reason code and remains closed.
        first = next(
            r for r in tomllib.loads(original)["record"] if r["status"] == "carried_forward"
        )
        assertion = first["routes"][0]["refusal_assertion"]
        carried_at = original.index(f'id = "{first["id"]}"')
        token = f'refusal_assertion = "{assertion}"'
        at = original.index(token, carried_at)
        registry.write_text(
            original[:at] + 'refusal_assertion = ""' + original[at + len(token) :],
            encoding="utf-8",
            newline="\n",
        )
        try:
            build_report(root)
        except ValueError as error:
            assert "unchanged refusal refusal_assertion required" in str(error)
        else:
            raise AssertionError("absent refusal assertion accepted")
        registry.write_text(original, encoding="utf-8", newline="\n")
    print("release_evidence_report self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        report = build_report(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"release evidence inventory: FAIL: {error}\n")
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
