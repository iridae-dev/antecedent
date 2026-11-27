#!/usr/bin/env python3
"""Every publisher the compiler can reach owes a scientific check and a calibration declaration.

A publisher is an `EstimatorId`, a `PlanRoute` the lowering table or a program family
introduces, a support-matrix cell, a graphless row, or an `owner` row of
parity/compiled_path.toml. The declaration is a calibration list, a coverage record the
unit names or the unit index (parity/unit_index.json) links to it, or an explicit
`no_interval_reported` / `estimator_grid_not_measured` reason. A unit that publishes no
sampling interval declares nothing. A name with neither is a missing publisher: the claim
is owed. The declaration is an annotation; it never decides whether a unit runs. A flagged
pass is a different failure and is not decided here.
"""

from __future__ import annotations

import functools
import json
import re
import sys
from pathlib import Path

import gate_support
from _repo import load_toml

ROOT = Path(__file__).resolve().parents[1]
NO_INTERVAL = "no_interval_reported"
UNMEASURED = "estimator_grid_not_measured"
REASONS = {NO_INTERVAL, UNMEASURED}
# A structural envelope or an assumption range is deterministic; it has no coverage.
INTERVAL_ROUTES = {"interval"}
MEASURED_KEYS = (
    "calibration",
    "calibration_records",
    "coverage_records",
    "allocated_coverage_records",
    "inherited_coverage_records",
    "candidate_coverage_records",
)
FAMILY_ROW = re.compile(r'^\s+([A-Z][A-Za-z0-9]*) => \w+, unit "([^"]*)";', re.M)
LOWERING_ROW = re.compile(r"^\s+([A-Z][A-Za-z0-9]*) => (\w+) as \w+,", re.M)
# Panel temporal routes share the temporal cell table: a route with no supported
# coordinate is the closed refusal. Every other route names a cell, a graphless row,
# a program family, or a capability `plan_routes` entry.
MATRIX_SIBLING = {"temporal"}
OWED = "claim is owed"
INTERVAL_WORD = re.compile(
    r"(?<![a-z])(intervals?|bands?|quantiles?|(confidence|credible)[ _](sets?|regions?))(?![a-z])"
)


def publishes_interval(row: dict) -> bool:
    claims = " ".join(str(row.get(key) or "") for key in ("inference_claim", "claim")).lower()
    if INTERVAL_WORD.search(claims):
        return True
    return any(route.get("claim") in INTERVAL_ROUTES for route in row.get("routes") or [])


@functools.cache
def _index(root: Path) -> dict[tuple[str, str], int]:
    """(kind, id) -> how many coverage records the unit index links to the unit."""
    path = root / "parity" / "unit_index.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(u["kind"], u["id"]): len(u["calibration"]) for u in data["units"]}


def claim_measured(root: Path, kind: str, ident: str) -> bool:
    """A coverage record measures the unit (parity/unit_index.json)."""
    return bool(_index(root).get((kind, ident)))


def contradicted(root: Path, kind: str, ident: str, row: dict) -> str | None:
    """A unit that states it measured nothing while a coverage record measures it."""
    if row.get("calibration_reason") != UNMEASURED:
        return None
    count = _index(root).get((kind, ident), 0)
    if not count:
        return None
    return f"{kind} {ident}: states {UNMEASURED} but {count} coverage record(s) measure it"


def _declaration(row: dict, measured: bool = False) -> bool:
    if measured or any(row.get(key) for key in MEASURED_KEYS):
        return True
    if row.get("calibration_reason") in REASONS or row.get("interval") == ["none"]:
        return True
    return not publishes_interval(row)


def _check(row: dict) -> bool:
    if row.get("evidence_test") and row.get("evidence_assertion"):
        return True
    if row.get("fixtures"):
        return True
    return bool(row.get("known_truth_test") or row.get("independent_check"))


def unit_claim_problems(root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    cells = load_toml(root / "parity/support_cells.toml").get("cell", [])
    graphless = load_toml(root / "parity/support_graphless.toml").get("support", [])
    records = {
        record["id"]: record
        for record in load_toml(root / "parity/capability_records.toml").get("record", [])
    }
    def declared(kind: str, ident: str, row: dict) -> bool:
        return _declaration(row, claim_measured(root, kind, ident))

    def publisher(unit: str) -> bool:
        record = records.get(unit)
        return record is not None and _check(record) and declared("capability", unit, record)

    named: set[str] = set()
    for cell in cells:
        named.update(cell.get("estimators") or [])
        ident = gate_support.cell_ident(cell)
        if not _check(cell) or not declared("matrix", ident, cell):
            problems.append(f"cell {ident}: {OWED}")
        problems.extend(filter(None, [contradicted(root, "matrix", ident, cell)]))
    for row in graphless:
        named.update(row.get("estimators") or [])
        ident = gate_support.graphless_ident(row)
        if not _check(row) or not declared("graphless", ident, row):
            problems.append(f"graphless {ident}: {OWED}")
        problems.extend(filter(None, [contradicted(root, "graphless", ident, row)]))
    text = (root / "crates/antecedent/src/strategy_table/ids.rs").read_text(encoding="utf-8")
    for wire in gate_support.parse_estimators(text):
        if wire not in named:
            problems.append(f"estimator {wire}: {OWED}")
    families = (root / "crates/antecedent/src/analysis/lowering/program_families.rs").read_text(
        encoding="utf-8"
    )
    family_names = {name for name, _unit in FAMILY_ROW.findall(families)}
    for name, unit in FAMILY_ROW.findall(families):
        if not publisher(unit):
            problems.append(f"route {name}: {OWED}")
    lowering = (root / "crates/antecedent/src/analysis/lowering/registry.rs").read_text(
        encoding="utf-8"
    )
    bound = {
        row["lowering"]
        for row in load_toml(root / "parity/supported_routes.toml").get("route", [])
    }
    graphless_routes = {row.get("route") for row in graphless if isinstance(row.get("route"), str)}
    emitted = set(bound) | graphless_routes | family_names
    for ident, record in records.items():
        problems.extend(filter(None, [contradicted(root, "capability", ident, record)]))
        if publisher(ident):
            emitted.update(record.get("plan_routes") or [])
    modules: dict[str, set[str]] = {}
    for name, module in LOWERING_ROW.findall(lowering):
        modules.setdefault(module, set()).add(name)
    for name, module in LOWERING_ROW.findall(lowering):
        if name in emitted:
            continue
        if module in MATRIX_SIBLING and modules[module] & bound:
            continue
        problems.append(f"route {name}: {OWED}")
    manifest = load_toml(root / "parity/compiled_path.toml")
    for section in ("row", "python", "verdict"):
        for row in manifest.get(section, []):
            if row.get("disposition") != "owner":
                continue
            unit = str(row.get("unit") or "")
            if unit in family_names or unit in bound or unit in graphless_routes:
                continue
            if not publisher(unit):
                problems.append(f"owner {row['path']}: {OWED}")
    return problems


def _self_test() -> None:
    interval = {"inference_claim": "95% interval"}
    assert not _declaration(interval)
    assert _declaration(interval, measured=True)
    assert _declaration({**interval, "calibration_reason": UNMEASURED})
    assert _declaration({**interval, "calibration_reason": NO_INTERVAL})
    for key in MEASURED_KEYS:
        assert _declaration({**interval, key: ["cov.x"]}), key
    assert _declaration({"interval": ["none"], "inference_claim": "interval"})
    assert _declaration({"inference_claim": "point estimate"})
    assert _declaration({"inference_claim": "identified set of offsets on the dataset"})
    assert not _declaration({"inference_claim": "a credible set"})
    assert not _declaration({"inference_claim": "effect_pointwise_95_student_interval"})
    assert not _declaration({"routes": [{"claim": "interval"}]})
    assert _declaration({"routes": [{"claim": "assumption_range"}]})
    assert _declaration({"routes": [{"claim": "structural_envelope"}]})
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "parity").mkdir()
        (root / "parity" / "unit_index.json").write_text(json.dumps({"units": [
            {"kind": "graphless", "id": "a:b", "calibration": [0]},
            {"kind": "graphless", "id": "a:c", "calibration": []},
        ]}), encoding="utf-8")
        unmeasured = {"calibration_reason": UNMEASURED}
        assert contradicted(root, "graphless", "a:b", unmeasured)
        assert not contradicted(root, "graphless", "a:c", unmeasured)
        assert not contradicted(root, "graphless", "a:b", {"calibration_reason": NO_INTERVAL})
        assert not contradicted(root, "graphless", "a:d", unmeasured)
    print("unit claim gate self-test ok")


def main() -> int:
    if "--self-test" in sys.argv:
        _self_test()
        return 0
    found = unit_claim_problems()
    if found:
        print(f"unit claims: {len(found)} publisher(s) owe a claim", file=sys.stderr)
        for problem in found:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print("unit claims: every publisher names a check and a calibration declaration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
