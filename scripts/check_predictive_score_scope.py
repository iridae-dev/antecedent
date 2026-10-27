#!/usr/bin/env python3
"""Predictive score interval scope: documentation, examples, limits and records agree.

The interval of an expected predictive score is published only for a case count inside the
range a shipped coverage record measured for the declared dependence. Four places state that
range and must say the same thing:

  design      the measurement designs in
              crates/antecedent/tests/predictive_score_interval_calibration.rs (base case
              counts by dependence and the larger iid grids; a grid measures half the base,
              the base and twice the base)
  records     the gated records of the shipped coverage registry (parity/coverage_records.toml)
  docs        the scope table of docs/predictive-assessment.md (design range and the range the
              shipped records cover) and the Python docstring of `expected_score_interval`
  notes       the release notes, which state the supported range of the interval

The audit fails when

  design_range_mismatch    a documented measurement-design range differs from the design range
  record_outside_design    a shipped record measures a case count the design does not
  shipped_range_mismatch   the documented shipped-record range differs from the gated records
  unqualified_upward       a statement that cases are supported "upward" with no upper bound
  limit_below_design       the documented case bound is below the largest measured case count

    python3 scripts/check_predictive_score_scope.py              # audit; exit 1 on a finding
    python3 scripts/check_predictive_score_scope.py --self-test

Environment: PREDICTIVE_SCOPE_ROOT (tree to read, default the repository).
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

from _repo import REPO_ROOT

ROOT = Path(os.environ.get("PREDICTIVE_SCOPE_ROOT", REPO_ROOT))
DEPENDENCES = ("iid", "cluster", "serial")
DESIGN_FILE = "crates/antecedent/tests/predictive_score_interval_calibration.rs"
DOC_FILE = "docs/predictive-assessment.md"
PYTHON_FILE = "python/antecedent/predictive_assessment.py"
LAW_FILE = "crates/antecedent-validate/src/predictive_assessment/law.rs"
RECORDS_FILE = "parity/coverage_records.toml"
QUERY = "PredictiveScoreInterval"
UPWARD = re.compile(r"cases?\s+upward|upward\s+of\s+[\d,]+\s+cases", re.IGNORECASE)
RANGE = re.compile(r"(\d[\d,]*)\s+to\s+(\d[\d,]*)")


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def integer(text: str) -> int:
    return int(text.replace(",", "").replace("_", ""))


def span(text: str) -> tuple[int, int] | None:
    """Smallest and largest case count of every `N to M` outside parentheses."""
    outside = re.sub(r"\([^)]*\)", "", text)
    pairs = [(integer(a), integer(b)) for a, b in RANGE.findall(outside)]
    return (min(a for a, _ in pairs), max(b for _, b in pairs)) if pairs else None


def design_ranges(source: str) -> dict[str, tuple[int, int]]:
    """Case-count range each dependence is measured over: half the smallest base to twice the
    largest, from `base_cases` and the `<dependence>_*_<n>_cases_l95` designs."""
    bases: dict[str, set[int]] = {name: set() for name in DEPENDENCES}
    for name, count in re.findall(r"Design::(Iid|Cluster|Serial)\s*=>\s*([\d_]+)", source):
        bases[name.lower()].add(integer(count))
    for name, count in re.findall(r"fn (iid|cluster|serial)_\w*?_(\d+)_cases_l95\b", source):
        bases[name].add(int(count))
    return {name: (min(found) // 2, max(found) * 2) for name, found in bases.items() if found}


def record_ranges(
    registry: dict,
) -> tuple[dict[str, tuple[int, int]], list[tuple[str, int]]]:
    """Gated case-count range per dependence, and every measured (dependence, n) grid point."""
    gated: dict[str, tuple[int, int]] = {}
    points: list[tuple[str, int]] = []
    for record in registry.get("record", []):
        if record.get("query") != QUERY:
            continue
        dependence = record["dependence"]
        for point in record.get("grid") or [record]:
            points.extend([(dependence, point["n_min"]), (dependence, point["n_max"])])
        if record.get("role") == "gated":
            low, high = gated.get(dependence, (record["n_min"], record["n_max"]))
            gated[dependence] = (min(low, record["n_min"]), max(high, record["n_max"]))
    return gated, points


def table_rows(doc: str) -> dict[str, list[str]]:
    """Cells of the scope table, by the dependence named in the first cell."""
    rows: dict[str, list[str]] = {}
    for line in doc.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        match = re.fullmatch(r"`(iid|cluster|serial)`", cells[0]) if cells else None
        if line.startswith("|") and match:
            rows[match.group(1)] = cells
    return rows


def fmt(count: int) -> str:
    return f"{count:,}"


def audit() -> list[str]:
    findings: list[str] = []
    design = design_ranges(read(DESIGN_FILE))
    gated, points = record_ranges(tomllib.loads(read(RECORDS_FILE)))
    if set(design) != set(DEPENDENCES):
        findings.append(
            f"design_range_mismatch: no design range for {set(DEPENDENCES) - set(design)}"
        )

    for dependence, count in points:
        low, high = design.get(dependence, (0, 0))
        if not low <= count <= high:
            findings.append(
                f"record_outside_design: a {dependence} record measures {fmt(count)} cases, outside "
                f"the design range {fmt(low)} to {fmt(high)}"
            )

    doc = read(DOC_FILE)
    rows = table_rows(doc)
    for dependence in DEPENDENCES:
        cells = rows.get(dependence)
        if cells is None or len(cells) < 4:
            findings.append(f"design_range_mismatch: {DOC_FILE} has no scope row for {dependence}")
            continue
        documented, shipped = span(cells[2]), span(cells[3])
        if documented != design.get(dependence):
            findings.append(
                f"design_range_mismatch: {DOC_FILE} states {documented} for {dependence}, "
                f"the designs measure {design.get(dependence)}"
            )
        if shipped != gated.get(dependence):
            findings.append(
                f"shipped_range_mismatch: {DOC_FILE} states {shipped} for {dependence}, "
                f"the shipped gated records cover {gated.get(dependence)}"
            )

    docstring = read(PYTHON_FILE)
    for dependence, (low, high) in design.items():
        if f"{dependence}`` {fmt(low)} to {fmt(high)}" not in docstring:
            findings.append(
                f"design_range_mismatch: the docstring of expected_score_interval does not "
                f"state the {dependence} design range {fmt(low)} to {fmt(high)}"
            )

    notes = sorted((ROOT / "docs" / "release-notes").glob("*.md"))
    for path in [*notes, ROOT / DOC_FILE, ROOT / PYTHON_FILE]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if UPWARD.search(line):
                findings.append(f"unqualified_upward: {path.relative_to(ROOT)}:{number}")

    limit = re.search(r"MAX_CASES: usize = ([\d_]+)", read(LAW_FILE))
    largest = max((high for _, high in design.values()), default=0)
    if limit and integer(limit.group(1)) < largest:
        findings.append(f"limit_below_design: MAX_CASES {limit.group(1)} < {fmt(largest)}")
    return findings


def self_test() -> int:
    source = "Design::Iid => 800,\nDesign::Cluster => 1_600,\nDesign::Serial => 3_200,\n"
    source += "fn iid_crps_mean_3200_cases_l95() {}\nfn iid_crps_mean_51200_cases_l95() {}\n"
    expected = {
        "iid": (400, 102_400),
        "cluster": (800, 3_200),
        "serial": (1_600, 6_400),
    }
    assert design_ranges(source) == expected, design_ranges(source)
    assert span("400 to 1,600; 6,400 to 25,600 (so 10,000 to 30,000)") == (400, 25_600)
    assert UPWARD.search("400, 800 and 1,600 cases upward")
    assert not UPWARD.search("biased upward by the sample size")
    registry = {
        "record": [
            {
                "query": QUERY,
                "dependence": "iid",
                "role": "gated",
                "n_min": 400,
                "n_max": 1600,
                "grid": [{"n_min": 400, "n_max": 400}, {"n_min": 1600, "n_max": 1600}],
            },
            {
                "query": QUERY,
                "dependence": "iid",
                "role": "named_boundary",
                "n_min": 400,
                "n_max": 1600,
            },
        ]
    }
    gated, points = record_ranges(registry)
    assert gated == {"iid": (400, 1600)} and ("iid", 1600) in points
    print("check_predictive_score_scope self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    findings = audit()
    for finding in findings:
        print(f"FAIL {finding}")
    if not findings:
        print("predictive score scope: design, records, documentation and notes agree")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
