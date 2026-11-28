#!/usr/bin/env python3
"""One evidence record per executable unit (ADR 0031): `parity/unit_index.json`.

    python3 scripts/unit_index.py            # write parity/unit_index.json
    python3 scripts/unit_index.py --check    # fail when it is stale
    python3 scripts/unit_index.py --diff     # the differential report against parity/claims/

A unit is a support-matrix cell, a construction on a cell (`<cell>#<key>`, from
`CONSTRUCTION_CHECKS` in crates/antecedent/src/analysis/builder/finish.rs), a graphless
row or a capability record that is not an alias of cells or rows. Its record is a straight
join, computed and never edited:

  claim        digest of the unit's claim-source row
  scientific   the recorded passes of the checks its claim source cites, and the passes
               whose test executed the unit (parity/evidence_links.json)
  calibration  the coverage records whose test executed the unit, and the records a
               resolution assigns it (parity/evidence_resolutions.toml)
  state        derived support status (embedded snapshot) and calibration state

Support reads only `scientific`; `calibration` annotates.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "parity" / "unit_index.json"
FINISH = ROOT / "crates/antecedent/src/analysis/builder/finish.rs"
CONSTRUCTION_ROW = re.compile(r'\(\s*"([a-z_]+)",\s*"([^"]+::[^"]+)",\s*\)', re.S)
MATRIX = ROOT / "crates/antecedent/data/support_matrix.json"
STATUS = ROOT / "crates/antecedent-io-core/data/unit_support_status.json"
NOT_MEASURED = "not_measured"


def construction_checks() -> dict[str, str]:
    """Construction key -> the check `path::name` that carries a cell's support to it."""
    text = FINISH.read_text(encoding="utf-8")
    block = text[text.index("const CONSTRUCTION_CHECKS"):]
    block = block[: block.index("];")]
    return dict(CONSTRUCTION_ROW.findall(block))


def read(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text(encoding="utf-8"))


def derived_states() -> dict[tuple[str, str], str]:
    import evidence_baseline as eb

    states: dict[tuple[str, str], str] = {}
    for row in json.loads(MATRIX.read_text(encoding="utf-8")).get("supported", []):
        states[("matrix", eb.cell_ident(row))] = str(row.get("status"))
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    for row in status.get("graphless", []):
        states[("graphless", eb.graphless_ident(row))] = str(row.get("status"))
    for row in status.get("capabilities", []):
        states[("capability", str(row["id"]))] = str(row.get("status"))
    return states


def calibration_state(records: list[str], by_id: dict[str, dict], reason: str | None) -> str:
    if not records:
        return reason or NOT_MEASURED
    present = [by_id[r] for r in records if r in by_id]
    if not present:
        return NOT_MEASURED
    if any(rec.get("boundary") for rec in present):
        return "measured_below_nominal"
    return "calibrated"


def selector(test: str, name: str) -> str | None:
    """The recorded-pass selector of a cited check, as the support gate resolves it."""
    from evidence_fingerprint import locate

    try:
        return locate(ROOT, {"evidence_test": test, "evidence_assertion": name})["selector"]
    except (OSError, ValueError, KeyError):
        return None


def cited_passes() -> dict[tuple[str, str], set[str]]:
    """Unit -> the selectors of the checks its claim source cites."""
    import gate_support as gs

    out: dict[tuple[str, str], set[str]] = {}

    def cite(key: tuple[str, str], test: object, name: object) -> None:
        found = selector(str(test), str(name)) if test and name else None
        out.setdefault(key, set())
        if found:
            out[key].add(found)

    for cell in read("parity/support_cells.toml").get("cell", []):
        cite(("matrix", gs.cell_ident(cell)), cell.get("evidence_test"), cell.get("evidence_assertion"))
    for row in read("parity/support_graphless.toml").get("support", []):
        citation = gs.graphless_citation(row)
        if citation:
            cite(("graphless", gs.graphless_ident(row)), citation[1], citation[2])
    for record in read("parity/capability_records.toml").get("record", []):
        for fixture in gs.capability_citations(record)[0]:
            cite(("capability", record["id"]), fixture["evidence_test"], fixture["evidence_assertion"])
    return out


def build() -> dict:
    import evidence_links as el
    import evidence_residue as er

    sources = er.claim_sources()
    aliases = {key for key, row in sources.items()
               if key[0] == "capability" and (row.get("matrix_units") or row.get("graphless_units"))}
    units: dict[tuple[str, str], dict] = {
        key: {"scientific": set(), "calibration": set()} for key in sources if key not in aliases
    }
    for key, selectors in cited_passes().items():
        if key in units:
            units[key]["scientific"].update(selectors)
    constructions = construction_checks()
    links = el.load() if (ROOT / "parity/evidence_links.json").is_file() else {"passes": {}, "records": {}}
    for table, half in (("passes", "scientific"), ("records", "calibration")):
        for ident, entry in links[table].items():
            for kind, unit in [*entry.get("executed", []), *entry.get("cited", [])]:
                key = (kind, unit)
                if not er.credits(entry, key):
                    continue
                if kind == "construction":
                    cell, _, construction = unit.partition("#")
                    if construction not in constructions or ("matrix", cell) not in sources:
                        continue
                    units.setdefault(key, {"scientific": set(), "calibration": set()})
                if key in units:
                    units[key][half].add(ident)
    for key in [k for k in units if k[0] == "construction"]:
        test, _, name = constructions[key[1].partition("#")[2]].rpartition("::")
        found = selector(test, name)
        if found:
            units[key]["scientific"].add(found)
    for res in er.load_resolutions():
        if res.get("kind") == "retired":
            continue
        evidence = str(res.get("evidence", ""))
        targets = [er.unit_key(str(u)) for u in res.get("units") or []]
        if evidence.startswith("record:"):
            for key in targets:
                if key in units:
                    units[key]["calibration"].add(evidence.removeprefix("record:"))
        elif evidence.startswith("pass:"):
            for key in targets:
                if key in units:
                    units[key]["scientific"].add(evidence.removeprefix("pass:"))

    coverage = {str(r["id"]): r for r in read("parity/coverage_records.toml").get("record", [])}
    states = derived_states()
    passes = sorted({p for u in units.values() for p in u["scientific"]})
    records = sorted({r for u in units.values() for r in u["calibration"]})
    pass_at = {p: i for i, p in enumerate(passes)}
    record_at = {r: i for i, r in enumerate(records)}
    out = []
    for key in sorted(units):
        kind, ident = key
        unit = units[key]
        row = sources.get(key) if kind != "construction" else None
        reason = row.get("calibration_reason") if row else None
        calibration = sorted(r for r in unit["calibration"] if r in coverage)
        out.append({
            "kind": kind,
            "id": ident,
            "claim": er.unit_digest(sources, key),
            "scientific": [pass_at[p] for p in sorted(unit["scientific"])],
            "calibration": [record_at[r] for r in calibration],
            "state": {
                "support": states.get(key, "derived_by_construction" if kind == "construction" else "not_supported"),
                "calibration": calibration_state(calibration, coverage, reason),
            },
        })
    return {
        "about": "ADR 0031 unit index; generated by scripts/unit_index.py; never edited",
        "passes": passes,
        "records": records,
        "units": out,
    }


def render(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"


def load(root: Path = ROOT) -> dict[tuple[str, str], dict]:
    """(kind, id) -> unit record with pass and record positions resolved to ids."""
    data = json.loads((root / "parity" / "unit_index.json").read_text(encoding="utf-8"))
    out = {}
    for unit in data["units"]:
        out[(unit["kind"], unit["id"])] = {
            **unit,
            "scientific": [data["passes"][i] for i in unit["scientific"]],
            "calibration": [data["records"][i] for i in unit["calibration"]],
        }
    return out


def differential(index: dict[tuple[str, str], dict]) -> dict[str, list[str]]:
    """Per unit, what parity/claims/ attached that the index does not: the old mechanisms'
    evidence the new join would drop. Empty before the old mechanisms are deleted."""
    from claim_passes import claim_path

    variants: dict[str, set[str]] = {}
    for (kind, ident), unit in index.items():
        if kind == "construction":
            variants.setdefault(ident.partition("#")[0], set()).update(unit["calibration"])
    report: dict[str, list[str]] = {}
    for (kind, ident), unit in index.items():
        if kind == "construction":
            continue
        path = claim_path(ROOT, kind, ident)
        if not path.is_file():
            continue
        claim = json.loads(path.read_text(encoding="utf-8"))
        records = set(unit["calibration"]) | (variants.get(ident, set()) if kind == "matrix" else set())
        dropped = [f"pass:{p}" for p in claim.get("assertions") or [] if p not in unit["scientific"]]
        dropped += [f"record:{r}" for r in claim.get("measurements") or [] if r not in records]
        if dropped:
            report[f"{kind}:{ident}"] = dropped
    return report


def main() -> int:
    text = render(build())
    if "--check" in sys.argv:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            print("parity/unit_index.json is stale: run python3 scripts/unit_index.py", file=sys.stderr)
            return 1
        print("unit index: current")
        return 0
    if "--diff" in sys.argv:
        OUT.write_text(text, encoding="utf-8")
        report = differential(load())
        dropped = sum(len(v) for v in report.values())
        print(f"differential: {len(report)} unit(s) would drop {dropped} attachment(s)")
        if "--json" in sys.argv:
            print(json.dumps(report, indent=1, sort_keys=True))
        return 1 if report else 0
    OUT.write_text(text, encoding="utf-8")
    data = json.loads(text)
    measured = sum(1 for u in data["units"] if u["calibration"])
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data['units'])} units, {len(data['passes'])} passes, "
          f"{len(data['records'])} records, {measured} units with calibration records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
