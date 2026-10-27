#!/usr/bin/env python3
"""Check promotion prerequisites and feature-package links."""

from __future__ import annotations

import sys
import tomllib

from _repo import REPO_ROOT as ROOT

RECORDS = ROOT / "parity/promotion.toml"
FEATURES = ROOT / "parity/feature_packages.toml"


def problems(records: list[dict]) -> list[str]:
    current = {record["id"]: record for record in records}
    errors: list[str] = []
    for record in records:
        rid = record["id"]
        dependencies = record.get("prerequisite_records", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) for item in dependencies
        ):
            errors.append(f"{rid}: prerequisite_records must be a list of record IDs")
            continue
        if len(dependencies) != len(set(dependencies)):
            errors.append(f"{rid}: duplicate prerequisite")
        for dependency in dependencies:
            if dependency == rid:
                errors.append(f"{rid}: self prerequisite")
            elif dependency not in current:
                errors.append(f"{rid}: unknown prerequisite {dependency}")

    active: set[str] = set()
    visited: set[str] = set()

    def visit(rid: str) -> None:
        if rid in active:
            errors.append(f"{rid}: cyclic prerequisites")
            return
        if rid in visited or rid not in current:
            return
        active.add(rid)
        for dependency in current[rid].get("prerequisite_records", []):
            if dependency in current:
                visit(dependency)
        active.remove(rid)
        visited.add(rid)

    for rid in current:
        visit(rid)
    return errors


def feature_problems(features: list[dict], records: list[dict]) -> list[str]:
    ids = [feature["id"] for feature in features]
    current = {record["id"] for record in records}
    known = set(ids)
    errors: list[str] = []
    if len(ids) != len(known):
        errors.append("feature ids must be unique")
    for feature in features:
        fid = feature["id"]
        dependencies = feature.get("depends_on")
        if not isinstance(dependencies, list) or any(dep not in known for dep in dependencies):
            errors.append(f"{fid}: dependencies must name feature ids")
            continue
        if fid in dependencies:
            errors.append(f"{fid}: self dependency")
        if feature.get("record") != fid:
            errors.append(f"{fid}: record must be the feature id")
        elif fid not in current:
            errors.append(f"{fid}: promotion record is missing")
        if feature.get("record_state") != "frozen":
            errors.append(f"{fid}: record_state must be frozen")
        if not feature.get("purpose"):
            errors.append(f"{fid}: purpose is required")

    graph = {feature["id"]: feature.get("depends_on", []) for feature in features}
    active: set[str] = set()
    visited: set[str] = set()

    def visit(fid: str) -> None:
        if fid in active:
            errors.append(f"{fid}: cyclic feature dependencies")
            return
        if fid in visited or fid not in graph:
            return
        active.add(fid)
        for dependency in graph[fid]:
            visit(dependency)
        active.remove(fid)
        visited.add(fid)

    for fid in graph:
        visit(fid)
    return errors


def self_test() -> None:
    base = {"id": "transport.base", "prerequisite_records": []}
    child = {"id": "distribution.child", "prerequisite_records": ["transport.base"]}
    assert not problems([base, child])
    assert any(
        "unknown prerequisite" in item
        for item in problems([{**child, "prerequisite_records": ["missing"]}])
    )
    assert any(
        "self prerequisite" in item
        for item in problems([{**child, "prerequisite_records": ["distribution.child"]}])
    )
    assert any(
        "cyclic" in item
        for item in problems(
            [
                {**base, "prerequisite_records": ["distribution.child"]},
                child,
            ]
        )
    )
    features = [
        {
            "id": "distribution.child",
            "purpose": "child",
            "depends_on": ["transport.base"],
            "record": "distribution.child",
            "record_state": "frozen",
        },
        {
            "id": "transport.base",
            "purpose": "base",
            "depends_on": [],
            "record": "transport.base",
            "record_state": "frozen",
        },
    ]
    assert not feature_problems(features, [base, child])
    assert any(
        "cyclic" in item
        for item in feature_problems(
            [{**features[0], "depends_on": ["distribution.child"]}, features[1]],
            [base, child],
        )
    )
    committed_records = tomllib.loads(RECORDS.read_text(encoding="utf-8"))["record"]
    committed_features = tomllib.loads(FEATURES.read_text(encoding="utf-8"))["feature"]
    assert not problems(committed_records)
    assert not feature_problems(committed_features, committed_records)
    print("prerequisite self-test: ok")


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(sys.argv) != 1:
        raise SystemExit("usage: check_prerequisites.py [--self-test]")
    records = tomllib.loads(RECORDS.read_text(encoding="utf-8"))["record"]
    features = tomllib.loads(FEATURES.read_text(encoding="utf-8"))["feature"]
    issues = problems(records) + feature_problems(features, records)
    if issues:
        print("\n".join(f"FAIL: {issue}" for issue in issues), file=sys.stderr)
        return 1
    print(f"prerequisites: {len(records)} records and {len(features)} feature packages checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
