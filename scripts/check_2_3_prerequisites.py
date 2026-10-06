#!/usr/bin/env python3
"""Check frozen 2.3 cell dependencies before any route can be promoted."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "parity/promotion_2_2.toml"
NEW = ROOT / "parity/promotion_2_3.toml"
FEATURES = ROOT / "parity/2_3_feature_packages.toml"


def problems(old: list[dict], new: list[dict]) -> list[str]:
    older = {record["id"]: record for record in old}
    current = {record["id"]: record for record in new}
    errors: list[str] = []
    for record in new:
        rid = record["id"]
        dependencies = record.get("prerequisite_records")
        if not isinstance(dependencies, list) or any(not isinstance(x, str) for x in dependencies):
            errors.append(f"{rid}: prerequisite_records must be a list of record IDs")
            continue
        if not dependencies and not record.get("prerequisite_reason"):
            errors.append(f"{rid}: no prerequisite needs an explicit reason")
        if len(dependencies) != len(set(dependencies)):
            errors.append(f"{rid}: duplicate prerequisite")
        if record["workstream"].startswith("X") and not any(dep in older for dep in dependencies):
            errors.append(f"{rid}: scientific A cell needs a named 2.2 prerequisite")
        for dependency in dependencies:
            if dependency == rid:
                errors.append(f"{rid}: self prerequisite")
            elif dependency in older:
                if older[dependency]["status"] not in {"promoted", "in_progress"}:
                    errors.append(f"{rid}: 2.2 prerequisite {dependency} is not an executing base")
            elif dependency not in current:
                errors.append(f"{rid}: unknown prerequisite {dependency}")

    active: set[str] = set()
    visited: set[str] = set()

    def visit(rid: str) -> None:
        if rid in active:
            errors.append(f"{rid}: cyclic 2.3 prerequisites")
            return
        if rid in visited:
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


def feature_problems(features: list[dict], new: list[dict]) -> list[str]:
    expected = {f"F{number}" for number in range(1, 26)}
    ids = [feature["id"] for feature in features]
    current = {record["id"] for record in new}
    errors: list[str] = []
    if set(ids) != expected or len(ids) != 25:
        errors.append("feature package registry must name F1–F25 exactly once")
    planned = [feature["record"] for feature in features]
    if len(planned) != len(set(planned)):
        errors.append("feature promotion record IDs must be distinct")
    for feature in features:
        fid = feature["id"]
        dependencies = feature.get("depends_on")
        if not isinstance(dependencies, list) or any(dep not in expected for dep in dependencies):
            errors.append(f"{fid}: dependencies must name F1–F25")
            continue
        if fid in dependencies:
            errors.append(f"{fid}: self dependency")
        state = feature.get("record_state")
        if state not in {"planned", "frozen"}:
            errors.append(f"{fid}: record_state must be planned or frozen")
        elif (feature["record"] in current) != (state == "frozen"):
            errors.append(f"{fid}: record_state disagrees with promotion_2_3.toml")
        if not feature.get("package") or not feature.get("purpose"):
            errors.append(f"{fid}: package and purpose are required")

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
    old = [{"id": "2.2A.X.test", "status": "promoted"}]
    base = {"id": "2.3A.X.test", "workstream": "X", "prerequisite_records": ["2.2A.X.test"]}
    assert not problems(old, [base])
    for change, expected in (
        ({"prerequisite_records": []}, "needs a named 2.2 prerequisite"),
        ({"prerequisite_records": ["missing"]}, "unknown prerequisite missing"),
        ({"prerequisite_records": ["2.3A.X.test"]}, "self prerequisite"),
    ):
        assert any(expected in item for item in problems(old, [{**base, **change}]))
    child = {"id": "2.3A.F.test", "workstream": "F", "prerequisite_records": ["2.3A.X.test"]}
    assert not problems(old, [base, child])
    assert any("cyclic" in item for item in problems(old, [
        {**base, "prerequisite_records": ["2.2A.X.test", "2.3A.F.test"]}, child
    ]))
    features = tomllib.loads(FEATURES.read_text())["feature"]
    current = tomllib.loads(NEW.read_text())["record"]
    assert not feature_problems(features, current)
    assert any("exactly once" in item for item in feature_problems(features[:-1], current))
    changed = [{**feature} for feature in features]
    changed[0]["depends_on"] = [changed[-1]["id"]]
    assert any("cyclic" in item for item in feature_problems(changed, current))
    print("2.3 prerequisite self-test: ok")


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(sys.argv) != 1:
        raise SystemExit("usage: check_2_3_prerequisites.py [--self-test]")
    old = tomllib.loads(OLD.read_text())["record"]
    new = tomllib.loads(NEW.read_text())["record"]
    features = tomllib.loads(FEATURES.read_text())["feature"]
    issues = problems(old, new) + feature_problems(features, new)
    if issues:
        print("\n".join(f"FAIL: {issue}" for issue in issues), file=sys.stderr)
        return 1
    print(f"2.3 prerequisites: {len(new)} records and {len(features)} feature packages checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
