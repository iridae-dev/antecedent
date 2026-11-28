#!/usr/bin/env python3
"""One evidence record per executable unit (ADR 0031): `parity/unit_index.json`.

    python3 scripts/unit_index.py            # write parity/unit_index.json
    python3 scripts/unit_index.py --check    # fail when it is stale
    python3 scripts/unit_index.py --diff     # fail when the index drops evidence parity/claims/ attached

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

Support reads only `scientific`; `calibration` annotates. A coverage record whose test ran
under trace but reached no supported unit is `unadmitted`, with the units its test refused:
it measures a construction no unit admits and backs no claim.
"""

from __future__ import annotations

import json
import re
import subprocess
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


def calibration_states() -> dict[str, str]:
    """`<kind>:<id>` -> calibration state of every unit; it reads no support status, so the
    embedded status snapshot can carry it."""
    return {f"{u['kind']}:{u['id']}": u["state"]["calibration"] for u in build(support=False)["units"]}


def build(support: bool = True) -> dict:
    import evidence_links as el
    import evidence_residue as er

    sources = er.claim_sources()
    aliases = {key for key, row in sources.items() if er.is_alias(key, row)}
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
            for kind, unit in er.reached_units(entry):
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
    coverage = {str(r["id"]): r for r in read("parity/coverage_records.toml").get("record", [])}
    # A record whose test ran a unit still awaiting support measured it when its estimator
    # coordinate names that unit.
    for ident, entry in links["records"].items():
        estimator = str(coverage.get(ident, {}).get("estimator", ""))
        for kind, unit in (tuple(u) for u in entry.get("refused", [])):
            if kind == "capability" and unit == estimator and (kind, unit) in units:
                units[(kind, unit)]["calibration"].add(ident)
    for key in [k for k in units if k[0] == "construction"]:
        test, _, name = constructions[key[1].partition("#")[2]].rpartition("::")
        found = selector(test, name)
        if found:
            units[key]["scientific"].add(found)
    for res in er.load_resolutions():
        # A narrowing names the unit whose claim no longer holds the evidence; a retirement none.
        if res.get("kind") not in ("same", "split", "merged"):
            continue
        targets = [er.unit_key(str(u)) for u in res.get("units") or []]
        for evidence in er.evidence_of(res):
            half, _, ident = evidence.partition(":")
            for key in targets:
                if key in units and half in ("record", "pass"):
                    units[key]["calibration" if half == "record" else "scientific"].add(ident)

    states = derived_states() if support else {}
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
    held = set(records)
    unadmitted = {
        ident: sorted(f"{k}:{u}" for k, u in (tuple(x) for x in entry.get("refused", [])))
        for ident, entry in sorted(links["records"].items())
        if ident in coverage and ident not in held and not entry.get("declared")
        and not {tuple(u) for u in entry.get("executed", [])} - {tuple(u) for u in entry.get("refused", [])}
    }
    return {
        "about": "ADR 0031 unit index; generated by scripts/unit_index.py; never edited",
        "passes": passes,
        "records": records,
        "unadmitted": unadmitted,
        "units": out,
    }


def render(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"


def load(root: Path = ROOT) -> dict[tuple[str, str], dict]:
    """(kind, id) -> unit record with pass and record positions resolved to ids."""
    return resolve(json.loads((root / "parity" / "unit_index.json").read_text(encoding="utf-8")))


def matrix_records(root: Path = ROOT) -> dict[str, list[str]]:
    """Cell id -> the coverage records the index links to the cell and its constructions."""
    out: dict[str, set[str]] = {}
    if not (root / "parity" / "unit_index.json").is_file():
        return {}
    for (kind, ident), unit in load(root).items():
        if kind in ("matrix", "construction"):
            out.setdefault(ident.partition("#")[0], set()).update(unit["calibration"])
    return {cell: sorted(records) for cell, records in out.items()}


def resolve(data: dict) -> dict[tuple[str, str], dict]:
    """[`load`] over an index already in memory, such as a fresh [`build`]."""
    out = {}
    for unit in data["units"]:
        out[(unit["kind"], unit["id"])] = {
            **unit,
            "scientific": [data["passes"][i] for i in unit["scientific"]],
            "calibration": [data["records"][i] for i in unit["calibration"]],
        }
    return out


def committed_claims() -> dict[tuple[str, str], dict]:
    """`(kind, id-with-colons-as-underscores)` -> the claim file at HEAD.

    The worktree may already have deleted `parity/claims/`. The report is against the
    committed files, and it has to be empty before that deletion is committed.
    """
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "parity/claims"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    paths = [line for line in listed.stdout.splitlines() if line.endswith(".json")]
    if not paths:
        return {}
    proc = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=ROOT, input="".join(f"HEAD:{path}\n" for path in paths).encode(),
        capture_output=True, check=True,
    )
    out: dict[tuple[str, str], dict] = {}
    pos = 0
    blob = proc.stdout
    for path in paths:
        newline = blob.index(b"\n", pos)
        header = blob[pos:newline].decode()
        pos = newline + 1
        if header.endswith(" missing"):
            continue
        size = int(header.rsplit(" ", 1)[1])
        kind, _, name = path.removeprefix("parity/claims/").partition("/")
        out[(kind, name.removesuffix(".json"))] = json.loads(blob[pos:pos + size])
        pos += size + 1
    return out


def differential(index: dict[tuple[str, str], dict]) -> dict[str, list[str]]:
    """Evidence a committed claim attached that the index dropped.

    A resolution keeps its evidence when a merged, split or same target holds it, or when
    a narrowing leaves it held by some unit. Empty before `parity/claims/` is deleted.
    """
    import evidence_residue as er

    claims = committed_claims()
    coverage = {str(r["id"]) for r in read("parity/coverage_records.toml").get("record", [])}
    by_item: dict[str, list[dict]] = {}
    for res in er.load_resolutions():
        by_item.setdefault(str(res.get("item", "")), []).append(res)
    held: dict[str, dict[str, set[tuple[str, str]]]] = {"record": {}, "pass": {}}
    for key, unit in index.items():
        for rid in unit["calibration"]:
            held["record"].setdefault(rid, set()).add(key)
        for pid in unit["scientific"]:
            held["pass"].setdefault(pid, set()).add(key)

    def settled(item: str, evidence: str) -> bool:
        half, _, ident = evidence.partition(":")
        for res in by_item.get(item, []):
            if evidence not in er.evidence_of(res) and "*" not in er.evidence_of(res):
                continue
            if res.get("kind") in ("same", "split", "merged"):
                targets = {er.unit_key(str(u)) for u in res.get("units") or []}
                if held[half].get(ident, set()) & targets:
                    return True
            elif res.get("kind") == "narrowed" and ident in held[half]:
                return True
            elif res.get("kind") == "retired":
                return True
        return False

    variants: dict[str, set[str]] = {}
    for (kind, ident), unit in index.items():
        if kind == "construction":
            variants.setdefault(ident.partition("#")[0], set()).update(unit["calibration"])
    report: dict[str, list[str]] = {}
    for (kind, ident), unit in index.items():
        if kind == "construction":
            continue
        item = f"{kind}:{ident}"
        claim = claims.get((kind, ident.replace(":", "_")))
        if claim is None:
            continue
        records = set(unit["calibration"]) | (variants.get(ident, set()) if kind == "matrix" else set())
        dropped = [f"pass:{p}" for p in claim.get("assertions") or []
                   if p not in unit["scientific"] and not settled(item, f"pass:{p}")]
        dropped += [f"record:{r}" for r in claim.get("measurements") or []
                    if r in coverage and r not in records and not settled(item, f"record:{r}")]
        if dropped:
            report[item] = dropped
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
        report = differential(resolve(json.loads(text)))
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
