#!/usr/bin/env python3
"""Check frozen 2.3 cell dependencies before any route can be promoted."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "parity/promotion_2_2.toml"
NEW = ROOT / "parity/promotion_2_3.toml"


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
    print("2.3 prerequisite self-test: ok")


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(sys.argv) != 1:
        raise SystemExit("usage: check_2_3_prerequisites.py [--self-test]")
    old = tomllib.loads(OLD.read_text())["record"]
    new = tomllib.loads(NEW.read_text())["record"]
    issues = problems(old, new)
    if issues:
        print("\n".join(f"FAIL: {issue}" for issue in issues), file=sys.stderr)
        return 1
    print(f"2.3 prerequisites: {len(new)} records checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
