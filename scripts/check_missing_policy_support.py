#!/usr/bin/env python3
"""Fail closed on the declared missing-data support registry.

parity/missing_policy_support.toml is the single statement of which `missing=` policies each
route cell (query kind x data modality x inference mode) executes. The facade reads it at build
time; this check keeps it honest without building anything. It fails when

  (a) a row names a policy outside the vocabulary of `MissingPolicy::NAMES`
      (crates/antecedent-data/src/sample_policy.rs), or a query kind outside `CausalQueryKind`;
  (b) a default or next attempt is not among the row's supported policies, or a row lists
      `imputation_estimators` without supporting multiple imputation (or the reverse);
  (c) two rows cover the same cell;
  (d) docs/guides/missing-data.md does not carry the row's line in its route-cell table
      (`| `Query` | modality | inference | `default` | `p1`, `p2` |`), or carries a route-cell
      line the registry does not have.

    python3 scripts/check_missing_policy_support.py
    python3 scripts/check_missing_policy_support.py --self-test
"""

from __future__ import annotations

import re
import sys
import tomllib

from _repo import REPO_ROOT as ROOT

REGISTRY = ROOT / "parity/missing_policy_support.toml"
POLICY_SOURCE = ROOT / "crates/antecedent-data/src/sample_policy.rs"
QUERY_SOURCE = ROOT / "crates/antecedent-core/src/query/mod.rs"
GUIDE = ROOT / "docs/guides/missing-data.md"
MODALITIES = {"tabular", "series", "event", "multi_env", "panel"}
INFERENCES = ("frequentist", "bayesian")
FAMILIES = {"static", "randomized", "panel", "temporal"}
DOC_LINE = re.compile(r"^\| `([A-Za-z]+)` \| ([a-z_]+) \| ([a-z]+) \|")


def policy_names(source: str) -> set[str]:
    match = re.search(r"pub const NAMES: \[&'static str; \d+\] = \[(.*?)\];", source, re.S)
    return set(re.findall(r'"([a-z_]+)"', match.group(1))) if match else set()


def query_kinds(source: str) -> set[str]:
    match = re.search(r"pub enum CausalQueryKind \{(.*?)\n\}", source, re.S)
    return set(re.findall(r"^\s+([A-Z][A-Za-z]+),", match.group(1), re.M)) if match else set()


def doc_line(row: dict, inference: str) -> str:
    supported = ", ".join(f"`{name}`" for name in row["supported"])
    return (
        f"| `{row['query']}` | {row['modality']} | {inference} | `{row['default']}` | {supported} |"
    )


def problems(registry: dict, policies: set[str], kinds: set[str], guide: str) -> list[str]:
    errors: list[str] = []
    if registry.get("schema") != 1:
        errors.append("FAIL: schema must be 1")
    seen: set[tuple[str, str, str]] = set()
    expected_lines: set[str] = set()
    for index, row in enumerate(registry.get("route", []), start=1):
        where = f"route {index}"
        for key in ("query", "modality", "family", "default", "next_attempt"):
            if not isinstance(row.get(key), str):
                errors.append(f"FAIL: {where}: `{key}` must be a string")
        for key in ("inference", "supported", "imputation_estimators"):
            if not isinstance(row.get(key), list):
                errors.append(f"FAIL: {where}: `{key}` must be an array")
        if errors:
            continue
        if row["query"] not in kinds:
            errors.append(f"FAIL: {where}: unknown query kind {row['query']}")
        if row["modality"] not in MODALITIES:
            errors.append(f"FAIL: {where}: unknown modality {row['modality']}")
        if row["family"] not in FAMILIES:
            errors.append(f"FAIL: {where}: unknown family {row['family']}")
        for name in row["supported"]:
            if name not in policies:
                errors.append(f"FAIL: {where}: unknown policy {name}")
        for key in ("default", "next_attempt"):
            if row[key] not in row["supported"]:
                errors.append(f"FAIL: {where}: {key} {row[key]} is not supported")
        imputes = "multiple_imputation" in row["supported"]
        if imputes == (not row["imputation_estimators"]):
            errors.append(
                f"FAIL: {where}: imputation_estimators must be listed exactly when "
                "multiple_imputation is supported"
            )
        for inference in row["inference"]:
            if inference not in INFERENCES:
                errors.append(f"FAIL: {where}: unknown inference {inference}")
                continue
            cell = (row["query"], row["modality"], inference)
            if cell in seen:
                errors.append(f"FAIL: {where}: two rows cover {':'.join(cell)}")
            seen.add(cell)
            line = doc_line(row, inference)
            expected_lines.add(line)
            if line not in guide:
                errors.append(f"FAIL: {where}: {GUIDE.name} lacks the line `{line}`")
    for line in guide.splitlines():
        if DOC_LINE.match(line) and line.strip() not in expected_lines:
            errors.append(f"FAIL: {GUIDE.name} lists a route cell the registry lacks: `{line}`")
    return errors


def self_test() -> int:
    registry = {
        "schema": 1,
        "route": [
            {
                "query": "AverageEffect",
                "modality": "tabular",
                "inference": ["frequentist"],
                "family": "static",
                "default": "complete_case",
                "supported": ["error", "complete_case"],
                "next_attempt": "complete_case",
                "imputation_estimators": [],
            }
        ],
    }
    policies = {"error", "complete_case", "multiple_imputation"}
    kinds = {"AverageEffect"}
    good = doc_line(registry["route"][0], "frequentist") + "\n"
    failures = 0
    if problems(registry, policies, kinds, good):
        print("self-test: a good registry failed")
        failures += 1
    if not problems(registry, policies, kinds, ""):
        print("self-test: a missing doc line passed")
        failures += 1
    bad = dict(registry["route"][0], supported=["bogus"])
    if not problems({"schema": 1, "route": [bad]}, policies, kinds, good):
        print("self-test: an unknown policy passed")
        failures += 1
    stray = good + "| `Survival` | tabular | frequentist | `error` | `error` |\n"
    if not problems(registry, policies, kinds, stray):
        print("self-test: a stray doc line passed")
        failures += 1
    duplicate = {"schema": 1, "route": [registry["route"][0], registry["route"][0]]}
    if not problems(duplicate, policies, kinds, good):
        print("self-test: a duplicate cell passed")
        failures += 1
    print("self-test: ok" if failures == 0 else f"self-test: {failures} failure(s)")
    return 1 if failures else 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    registry = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))
    errors = problems(
        registry,
        policy_names(POLICY_SOURCE.read_text(encoding="utf-8")),
        query_kinds(QUERY_SOURCE.read_text(encoding="utf-8")),
        GUIDE.read_text(encoding="utf-8"),
    )
    for error in errors:
        print(error)
    if errors:
        return 1
    print(f"missing-policy support: {len(registry.get('route', []))} rows ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
