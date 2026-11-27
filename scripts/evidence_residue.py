#!/usr/bin/env python3
"""Baseline evidence that no execution link or resolution covers (ADR 0031).

Read through `python3 scripts/evidence_links.py residue`. For every item supported at a
baseline ref (`parity/evidence_baseline.json`), each scientific pass and coverage record
that backed it must, today, link to the item's unit (`parity/evidence_links.json`) or be
covered by a resolution (`parity/evidence_resolutions.toml`) bound to the current digests
of both sides. What is left is the residue, classified:

  lost       the pass or record no longer exists
  moved      it exists and links only to other units
  uncovered  it exists and links to no unit at all
  ambiguous  its test executed several units of the item's kind, declared none, and was
             handed no result of the item's unit
  refused    its test executed the item's unit, and the unit refused
  retired    the item's unit no longer has a claim source

A resolution names a baseline item, the evidence it settles (`*` for all of it), the units
it now belongs to, a kind (same, split, merged, narrowed, retired), a reason and an author.
It goes stale when the baseline item or any named unit's claim source changes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RESOLUTIONS = ROOT / "parity" / "evidence_resolutions.toml"
RESOLUTION_KINDS = ("same", "split", "merged", "narrowed", "retired")
RESIDUE_KINDS = ("lost", "moved", "uncovered", "ambiguous", "refused", "retired")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def claim_sources() -> dict[tuple[str, str], dict]:
    """(kind, id) -> the current claim-source row of every unit."""
    import evidence_baseline as eb

    def read(path: str) -> dict:
        return tomllib.loads((ROOT / path).read_text(encoding="utf-8"))

    rows: dict[tuple[str, str], dict] = {}
    for cell in read(eb.CELLS).get("cell", []):
        rows[("matrix", eb.cell_ident(cell))] = cell
    for row in read(eb.GRAPHLESS).get("support", []):
        rows[("graphless", eb.graphless_ident(row))] = row
    for record in read(eb.CAPABILITIES).get("record", []):
        rows[("capability", str(record["id"]))] = record
    return rows


def unit_key(text: str) -> tuple[str, str]:
    kind, _, ident = text.partition(":")
    return kind, ident


def item_digest(item: dict) -> str:
    return digest({"kind": item["kind"], "id": item["id"], "evidence": item["evidence"]})


def unit_digest(sources: dict[tuple[str, str], dict], unit: tuple[str, str]) -> str | None:
    if unit[0] == "construction":
        cell = unit[1].split("#", 1)[0]
        row = sources.get(("matrix", cell))
        return digest({"construction": unit[1], "cell": row}) if row is not None else None
    row = sources.get(unit)
    return digest(row) if row is not None else None


def load_resolutions() -> list[dict]:
    if not RESOLUTIONS.is_file():
        return []
    return list(tomllib.loads(RESOLUTIONS.read_text(encoding="utf-8")).get("resolution", []))


def resolution_problems(resolutions: list[dict], items: dict[tuple[str, str], dict],
                        sources: dict[tuple[str, str], dict]) -> list[str]:
    problems = []
    for n, res in enumerate(resolutions, 1):
        where = f"resolution {n} ({res.get('item', '?')})"
        for field in ("item", "evidence", "kind", "units", "reason", "author", "baseline_digest", "unit_digests"):
            if field not in res:
                problems.append(f"{where}: missing {field}")
        if res.get("kind") not in RESOLUTION_KINDS:
            problems.append(f"{where}: kind must be one of {', '.join(RESOLUTION_KINDS)}")
        item = items.get(unit_key(str(res.get("item", ""))))
        if item is None:
            problems.append(f"{where}: names no baseline item")
            continue
        if res.get("baseline_digest") != item_digest(item):
            problems.append(f"{where}: stale, the baseline item changed")
        units = [unit_key(str(u)) for u in res.get("units") or []]
        if res.get("kind") == "retired" and units:
            problems.append(f"{where}: a retirement names no units")
        if res.get("kind") != "retired" and not units:
            problems.append(f"{where}: names no units")
        expected = {f"{k}:{i}": unit_digest(sources, (k, i)) for k, i in units}
        if any(v is None for v in expected.values()):
            problems.append(f"{where}: names a unit with no claim source")
        elif res.get("unit_digests", {}) != expected:
            problems.append(f"{where}: stale, a named unit's claim source changed")
        if not str(res.get("reason", "")).strip():
            problems.append(f"{where}: needs a reason")
    return problems


def residue() -> tuple[list[dict], list[str]]:
    import evidence_baseline as eb
    import evidence_links as el

    baseline = eb.load()
    links = el.load()
    sources = claim_sources()
    science = {str(r["test"]) for r in tomllib.loads(
        (ROOT / "parity/scientific_records.toml").read_text(encoding="utf-8")).get("record", [])}
    coverage = {str(r["id"]) for r in tomllib.loads(
        (ROOT / "parity/coverage_records.toml").read_text(encoding="utf-8")).get("record", [])}
    items = {(i["kind"], i["id"]): i for i in baseline["items"]}
    resolutions = load_resolutions()
    problems = resolution_problems(resolutions, items, sources)
    settled: dict[tuple[str, str], set[str]] = {}
    for res in resolutions:
        settled.setdefault(unit_key(str(res["item"])), set()).add(str(res["evidence"]))

    out = []
    for key, item in sorted(items.items()):
        kind, ident = key
        done = settled.get(key, set())
        if "*" in done:
            continue
        if key not in sources:
            out.append({"item": f"{kind}:{ident}", "evidence": "*", "residue": "retired", "links": []})
            continue
        passes = {p for e in item["evidence"].values() for ps in e["scientific"].values() for p in ps}
        records = {r for e in item["evidence"].values() for r in e["calibration"]}
        for evidence, exists, entry in (
            *((f"pass:{p}", p in science, links["passes"].get(p, {})) for p in sorted(passes)),
            *((f"record:{r}", r in coverage, links["records"].get(r, {})) for r in sorted(records)),
        ):
            if evidence in done:
                continue
            reached = [tuple(u) for field in ("executed", "cited") for u in entry.get(field, [])]
            if key in reached or (kind == "matrix" and any(
                    u[0] == "construction" and u[1].split("#", 1)[0] == ident for u in reached)):
                same_kind = {u for u in reached if u[0] == kind}
                handed = {tuple(u) for u in entry.get("returned", [])}
                if len(same_kind) > 1 and not entry.get("cited") and key not in handed:
                    out.append({"item": f"{kind}:{ident}", "evidence": evidence, "residue": "ambiguous",
                                "links": sorted(f"{k}:{i}" for k, i in same_kind)})
                continue
            if not exists:
                label = "lost"
            elif key in {tuple(u) for u in entry.get("refused", [])}:
                label = "refused"
            elif reached:
                label = "moved"
            else:
                label = "uncovered"
            out.append({"item": f"{kind}:{ident}", "evidence": evidence, "residue": label,
                        "links": sorted(f"{k}:{i}" for k, i in set(reached))})
    return out, problems


def main(args) -> int:
    rows, problems = residue()
    for problem in problems:
        print(f"evidence resolutions: {problem}", file=sys.stderr)
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=1))
    else:
        by = Counter((r["residue"], r["item"].split(":", 1)[0], r["evidence"].split(":", 1)[0]) for r in rows)
        print(f"evidence residue: {len(rows)} unmatched baseline evidence")
        for (label, kind, evidence), n in sorted(by.items()):
            print(f"  {label:9} {kind:10} {evidence:6} {n}")
    return 1 if rows or problems else 0
