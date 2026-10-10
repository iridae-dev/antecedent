#!/usr/bin/env python3
"""Report versioned promotion evidence and enforce explicit closed-cell dispositions.

This inventories cited evidence, not execution or calibration. A consistent report
does not certify a release. The promotion runner and release-candidate gates still
own those verdicts. Default: both retained 2.2 and current 2.3 registries.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[1]


def build_report(root: Path, releases: list[str]) -> dict:
    dispositions = tomllib.loads(
        (root / "parity/promotion_carryovers.toml").read_text()
    )
    if dispositions.get("version") != 1:
        raise ValueError("unsupported carryover disposition version")
    cells = dispositions.get("cell", [])
    by_id = {c["id"]: c for c in cells}
    if len(by_id) != len(cells):
        raise ValueError("duplicate carryover disposition")
    registries = {}
    all_records = {}
    # Validate both registries even for a filtered report, so a stale disposition
    # cannot hide behind a release filter.
    for release in ("2.2", "2.3"):
        registry = f"parity/promotion_{release.replace('.', '_')}.toml"
        data = tomllib.loads((root / registry).read_text())
        if data.get("release") != release:
            raise ValueError(f"{registry}: release mismatch")
        records = data.get("record", [])
        registries[release] = (registry, records)
        for record in records:
            rid = record["id"]
            if rid in all_records:
                raise ValueError(f"duplicate promotion record {rid}")
            all_records[rid] = record
    carried = {
        rid for rid, r in all_records.items() if r["status"] == "carried_forward"
    }
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
            if (
                not isinstance(disposition.get(key), str)
                or not disposition[key].strip()
            ):
                raise ValueError(f"{rid}: {key} required")
        if not record.get("owners"):
            raise ValueError(f"{rid}: corrective owners required")
        routes = record.get("routes", [])
        if not routes or any(r.get("status") != "closed" for r in routes):
            raise ValueError(f"{rid}: every carried route must remain closed")
        for route in routes:
            for key in ("reason_code", "refusal_test", "refusal_assertion"):
                if not route.get(key):
                    raise ValueError(
                        f"{rid}: {route['name']}: unchanged refusal {key} required"
                    )
            if not any(
                r.get("code") == route["reason_code"]
                for r in record.get("refusals", [])
            ):
                raise ValueError(
                    f"{rid}: {route['name']}: refusal reason absent from record"
                )
    output = []
    for release in releases:
        registry, records = registries[release]
        summaries = []
        for record in records:
            summary = {
                "id": record["id"],
                "work_package": record["work_package"],
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
        output.append({"release": release, "registry": registry, "records": summaries})
    return {
        "report_kind": "promotion_evidence_inventory",
        "execution_verified": False,
        "calibration_verified": False,
        "release_certified": False,
        "releases": output,
    }


def render(report: dict) -> str:
    lines = ["Promotion evidence inventory (execution and calibration not verified)"]
    for release in report["releases"]:
        records = release["records"]
        counts = {
            state: sum(r["status"] == state for r in records)
            for state in sorted({r["status"] for r in records})
        }
        lines += [
            "",
            (
                f"{release['release']}: {release['registry']}; "
                f"{len(records)} records; {counts}"
            ),
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
                details = [
                    r["detail"]
                    for r in record["refusals"]
                    if r["code"] == route["reason_code"]
                ]
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
    # Overlay only registry inputs: no measurement, build or release gate runs.
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "parity").mkdir()
        paths = [
            "promotion_2_2.toml",
            "promotion_2_3.toml",
            "promotion_carryovers.toml",
        ]
        for name in paths:
            (root / "parity" / name).write_text((REPO / "parity" / name).read_text())
        report = build_report(root, ["2.2", "2.3"])
        assert len(report["releases"]) == 2
        assert not any(
            report[k]
            for k in ("execution_verified", "calibration_verified", "release_certified")
        )
        assert "2.3: parity/promotion_2_3.toml" in render(report)
        assert len(build_report(root, ["2.2"])["releases"]) == 1
        assert len(build_report(root, ["2.3"])["releases"]) == 1
        baseline = (root / "parity/promotion_carryovers.toml").read_text()
        correction = tomllib.loads(baseline)["cell"][1]["corrective_action"]
        for before, after, expected in [
            ('id = "2.3A.X4.joint_bayesian_transport"', 'id = "unknown"', "mismatch"),
            (
                'failed_gates = ["whole_posterior_calibration"]',
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
                baseline.replace(before, after, 1)
            )
            try:
                build_report(root, ["2.2"])
            except ValueError as error:
                assert expected in str(error), str(error)
            else:
                raise AssertionError(f"mutation accepted: {expected}")
        (root / "parity/promotion_carryovers.toml").write_text(baseline)
        duplicate = baseline[
            baseline.index("[[cell]]") : baseline.index(
                "[[cell]]", baseline.index("[[cell]]") + 1
            )
        ]
        (root / "parity/promotion_carryovers.toml").write_text(baseline + duplicate)
        try:
            build_report(root, ["2.3"])
        except ValueError as error:
            assert "duplicate carryover disposition" in str(error)
        else:
            raise AssertionError("duplicate disposition accepted")
        (root / "parity/promotion_carryovers.toml").write_text(baseline)
        registry = root / "parity/promotion_2_3.toml"
        original = registry.read_text()
        # All carried routes must be closed regardless of the report filter.
        registry.write_text(
            original.replace('status = "closed"', 'status = "licensed"', 1)
        )
        try:
            build_report(root, ["2.2"])
        except ValueError as error:
            assert "every carried route must remain closed" in str(error)
        else:
            raise AssertionError("licensed carried route accepted")
        registry.write_text(original)
        # Losing an executable refusal citation must fail even when the route
        # still has a reason code and remains closed.
        first = next(
            r
            for r in tomllib.loads(original)["record"]
            if r["status"] == "carried_forward"
        )
        assertion = first["routes"][0]["refusal_assertion"]
        registry.write_text(
            original.replace(
                f'refusal_assertion = "{assertion}"', 'refusal_assertion = ""', 1
            )
        )
        try:
            build_report(root, ["2.3"])
        except ValueError as error:
            assert "unchanged refusal refusal_assertion required" in str(error)
        else:
            raise AssertionError("absent refusal assertion accepted")
        registry.write_text(original)
    print("release_evidence_report self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-version", choices=("2.2", "2.3", "all"), default="all"
    )
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    releases = (
        ["2.2", "2.3"] if args.release_version == "all" else [args.release_version]
    )
    try:
        report = build_report(args.root, releases)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"release evidence inventory: FAIL: {error}\n")
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
