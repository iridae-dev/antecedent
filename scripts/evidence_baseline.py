#!/usr/bin/env python3
"""The evidence each supported item had before unified evidence records (ADR 0031).

    python3 scripts/evidence_baseline.py            # write parity/evidence_baseline.json
    python3 scripts/evidence_baseline.py --check    # fail when it differs from the refs

Reads every registry through `git show <ref>:<path>`, without checking anything out, at
each baseline ref: the v3.0.0 release and the commit before the migration. An item is a
supported matrix cell, graphless row or capability by the derived status embedded at that
ref. Its scientific evidence is each check it cites with the recorded passes of that check;
its calibration evidence is every coverage record attached to it by any of the
mechanisms in use at that ref. The file is generated once and never edited; the
monotonicity gate (`scripts/gate_evidence_closure.py`) reads it until 3.1.0 ships.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "parity" / "evidence_baseline.json"
REFS = ("v3.0.0", "9de9f6ceab")
MATRIX = "crates/antecedent/data/support_matrix.json"
STATUS = "crates/antecedent-io-core/data/unit_support_status.json"
CELLS = "parity/support_cells.toml"
GRAPHLESS = "parity/support_graphless.toml"
CAPABILITIES = "parity/capability_records.toml"
SCIENTIFIC = "parity/scientific_records.toml"
COVERAGE = "parity/coverage_records.toml"
ROUTES = "parity/calibration_routes.toml"
SOURCES = (MATRIX, STATUS, CELLS, GRAPHLESS, CAPABILITIES, SCIENTIFIC, COVERAGE, ROUTES)
CELL_AXES = ("query", "graph_class", "structure", "inference", "validation")
CAPABILITY_LISTS = (
    "coverage_records",
    "allocated_coverage_records",
    "inherited_coverage_records",
    "candidate_coverage_records",
)
# Independent roles in use at either ref (`gate_support.INDEPENDENT_FIXTURE_ROLES` plus the
# older `positive`): the baseline errs toward requiring more checks to be kept.
INDEPENDENT_ROLES = {"truth", "parity", "positive"}


def show(ref: str | None, path: str) -> bytes | None:
    """`path` at `ref`, or in the working tree when `ref` is `None`."""
    if ref is None:
        file = ROOT / path
        return file.read_bytes() if file.is_file() else None
    done = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, check=False
    )
    return done.stdout if done.returncode == 0 else None


def cell_ident(cell: dict) -> str:
    return "/".join(str(cell.get(axis)) for axis in CELL_AXES)


def graphless_ident(row: dict) -> str:
    return f"{row['family']}:{row['design']}:{row['method']}:{row['inference_claim']}"


def measures(group: str, test: str) -> bool:
    import calibration_groups

    head, _, test_filter = group.partition(": ")
    return calibration_groups.measures(head, test_filter, {"test": test})


def recorded_passes(cited: str, by_leaf: dict[str, list[str]]) -> list[str]:
    """Recorded scientific passes of the check `path::name`: the selector its file and name
    give, inside any further inline modules of that test target. A check whose file is not a
    known test layout has none."""
    import evidence_fingerprint

    path, _, name = cited.partition("::")
    try:
        selector = evidence_fingerprint.locate(ROOT, {"evidence_test": path, "evidence_assertion": name})["selector"]
    except (OSError, ValueError, KeyError):
        return []
    target, _, rest = selector.rpartition("::")
    return [s for s in by_leaf.get(name.rsplit("::", 1)[-1], [])
            if s == selector or (s.startswith(target + "::") and s.endswith("::" + rest))]


def capability_checks(record: dict) -> list[str]:
    checks = []
    for fixture in record.get("fixtures") or []:
        if fixture.get("role") in INDEPENDENT_ROLES and fixture.get("evidence_test"):
            if fixture.get("evidence_assertion"):
                checks.append(f"{fixture['evidence_test']}::{fixture['evidence_assertion']}")
    return checks


def _suite(value: object) -> str | None:
    return value if isinstance(value, str) and value and value != "unallocated" else None


def unit_targets(record: dict, cells: dict[str, dict], rows: dict[str, dict]) -> list[tuple[str, str]]:
    """The units a capability record's evidence attaches to: the cells and rows its
    `matrix_units` and `graphless_units` select, or the record itself."""
    import gate_support

    if not (record.get("matrix_units") or record.get("graphless_units")):
        return [("capability", str(record["id"]))]
    found = [("matrix", ident) for selector in record.get("matrix_units") or []
             for ident, cell in cells.items()
             if gate_support.selector_matches(selector, cell, gate_support.CELL_AXES)]
    found += [("graphless", ident) for selector in record.get("graphless_units") or []
              for ident, row in rows.items()
              if gate_support.selector_matches(selector, row, gate_support.GRAPHLESS_AXES)]
    return found


def attachments(toml: dict[str, dict], records: list[tuple[str, str]]) -> dict[tuple[str, str], set[str]]:
    """Every coverage record each unit's claim sources attach, by any mechanism in use: a
    cell's `calibration` list, a capability record's lists and suites, and a calibration
    route's groups."""
    cells = {cell_ident(c): c for c in toml[CELLS].get("cell", [])}
    rows = {graphless_ident(r): r for r in toml[GRAPHLESS].get("support", [])}
    caps = {str(r["id"]): r for r in toml[CAPABILITIES].get("record", [])}
    out: dict[tuple[str, str], set[str]] = {}
    for ident, cell in cells.items():
        out.setdefault(("matrix", ident), set()).update(str(x) for x in cell.get("calibration") or [])
    for key, measured in capability_measurements(list(caps.values()), cells, rows, records).items():
        out.setdefault(key, set()).update(measured)
    for route in toml[ROUTES].get("route", []):
        groups = [str(g) for g in route.get("groups") or []]
        if not groups:
            continue
        produced = {rid for rid, test in records if any(measures(g, test) for g in groups)}
        record = caps.get(str(route.get("record", "")))
        targets = unit_targets(record, cells, rows) if record else []
        for name in route.get("routes") or []:
            if str(name).startswith("graphless."):
                for ident, r in rows.items():
                    prefix = f"graphless.{r['family']}.{r['design']}.{r['method']}"
                    if name == prefix or str(name).startswith(prefix + "."):
                        targets.append(("graphless", ident))
        for key in targets:
            out.setdefault(key, set()).update(produced)
    return out


def current_toml() -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """The working tree's registries and its coverage records as (id, test)."""
    toml = {p: tomllib.loads((ROOT / p).read_text(encoding="utf-8")) if (ROOT / p).is_file() else {}
            for p in SOURCES if p.endswith(".toml")}
    records = [(str(r["id"]), str(r.get("test", ""))) for r in toml[COVERAGE].get("record", [])]
    return toml, records


def capability_measurements(caps: list[dict], cells: dict[str, dict], rows: dict[str, dict],
                            records: list[tuple[str, str]]) -> dict[tuple[str, str], set[str]]:
    """The coverage records each capability record attaches, by the unit it attaches them to,
    as `claim_passes.write_claim_index` attaches them: a record naming `matrix_units` or
    `graphless_units` attaches to those units, any other to itself; it attaches its named
    lists, every record of a file only it names as a suite (its own top-level suite always),
    and every record of a test an inference output names."""
    suites: dict[str, tuple[list[str], list[str]]] = {}
    owners: dict[str, set[str]] = {}
    for record in caps:
        files, tests = [], []
        top = _suite(record.get("measurement_suite"))
        if top:
            files.append(top.split("::", 1)[0])
        for output in record.get("inference_outputs") or []:
            suite = _suite(output.get("measurement_suite"))
            if suite:
                (tests if "::" in suite else files).append(suite)
        suites[str(record["id"])] = (files, tests)
        for path in files:
            owners.setdefault(path, set()).add(str(record["id"]))
    out: dict[tuple[str, str], set[str]] = {}
    for record in caps:
        rid = str(record["id"])
        files, tests = suites[rid]
        owned = {path for path in files if owners.get(path) == {rid}}
        top = _suite(record.get("measurement_suite"))
        if top:
            owned.add(top.split("::", 1)[0])
        measured = {str(x) for key in CAPABILITY_LISTS for x in record.get(key) or []}
        measured |= {ident for ident, test in records
                     if test.split("::", 1)[0] in owned
                     or any(test == suite or test.startswith(suite + "::") for suite in tests)}
        for key in unit_targets(record, cells, rows):
            out.setdefault(key, set()).update(measured)
    return out


def extract(ref: str | None) -> tuple[dict, dict[tuple[str, str], dict]]:
    raw = {path: show(ref, path) for path in SOURCES}
    commit = subprocess.check_output(["git", "rev-parse", ref or "HEAD"], cwd=ROOT, text=True).strip()
    source = {
        "commit": commit,
        "files": {p: hashlib.sha256(b).hexdigest() for p, b in raw.items() if b is not None},
    }
    toml = {p: tomllib.loads(raw[p].decode()) if raw[p] else {} for p in SOURCES if p.endswith(".toml")}
    matrix = json.loads(raw[MATRIX])["supported"]
    status = json.loads(raw[STATUS])
    by_leaf: dict[str, list[str]] = {}
    for r in toml[SCIENTIFIC].get("record", []):
        by_leaf.setdefault(str(r["test"]).rsplit("::", 1)[-1], []).append(str(r["test"]))
    records = [(str(r["id"]), str(r.get("test", ""))) for r in toml[COVERAGE].get("record", [])]
    record_ids = {rid for rid, _ in records}

    items: dict[tuple[str, str], dict] = {}

    def evidence(checks: list[str], calibration: set[str]) -> dict:
        return {
            "scientific": {check: recorded_passes(check, by_leaf) for check in sorted(set(checks))},
            "calibration": sorted(calibration & record_ids),
        }

    cells = {cell_ident(c): c for c in toml[CELLS].get("cell", [])}
    for row in matrix:
        if row.get("status") != "supported":
            continue
        ident = cell_ident(row)
        cell = cells.get(ident, {})
        checks = []
        if cell.get("evidence_test") and cell.get("evidence_assertion"):
            checks.append(f"{cell['evidence_test']}::{cell['evidence_assertion']}")
        items[("matrix", ident)] = evidence(checks, set())

    rows = {graphless_ident(r): r for r in toml[GRAPHLESS].get("support", [])}
    for row in status.get("graphless", []):
        if row.get("status") != "supported":
            continue
        ident = graphless_ident(row)
        source_row = rows.get(ident, {})
        checks = [
            str(source_row[f])
            for f in ("independent_check", "known_truth_test")
            if isinstance(source_row.get(f), str) and "::" in source_row[f]
        ]
        items[("graphless", ident)] = evidence(checks, set())

    caps = {str(r["id"]): r for r in toml[CAPABILITIES].get("record", [])}
    for row in status.get("capabilities", []):
        if row.get("status") != "supported":
            continue
        record = caps.get(str(row["id"]), {})
        items[("capability", str(row["id"]))] = evidence(capability_checks(record), set())
    for key, measured in attachments(toml, records).items():
        if key in items:
            items[key]["calibration"] = sorted(set(items[key]["calibration"]) | (measured & record_ids))
    return source, items


def build() -> dict:
    sources, merged = {}, {}
    for ref in REFS:
        source, items = extract(ref)
        sources[ref] = source
        for key, evidence in items.items():
            merged.setdefault(key, {})[ref] = evidence
    record_index = sorted({rid for per in merged.values() for e in per.values() for rid in e["calibration"]})
    position = {rid: i for i, rid in enumerate(record_index)}
    out_items = []
    for (kind, ident), per in sorted(merged.items()):
        out_items.append(
            {
                "kind": kind,
                "id": ident,
                "evidence": {
                    ref: {
                        "scientific": e["scientific"],
                        "calibration": [position[rid] for rid in e["calibration"]],
                    }
                    for ref, e in per.items()
                },
            }
        )
    return {
        "about": "ADR 0031 evidence baseline; generated by scripts/evidence_baseline.py; never edited",
        "sources": sources,
        "records": record_index,
        "items": out_items,
    }


def render(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"


def load(root: Path = ROOT) -> dict:
    """The baseline with calibration record positions resolved to ids."""
    data = json.loads((root / "parity" / "evidence_baseline.json").read_text(encoding="utf-8"))
    ids = data["records"]
    for item in data["items"]:
        for evidence in item["evidence"].values():
            evidence["calibration"] = [ids[i] for i in evidence["calibration"]]
    return data


def main() -> int:
    text = render(build())
    if "--check" in sys.argv:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            print("parity/evidence_baseline.json differs from the baseline refs", file=sys.stderr)
            return 1
        print("evidence baseline: current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    data = json.loads(text)
    kinds: dict[str, int] = {}
    for item in data["items"]:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data['items'])} items {kinds}, {len(data['records'])} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
