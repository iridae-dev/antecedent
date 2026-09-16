#!/usr/bin/env python3
"""Which external oracles each licensed support-matrix cell rests on.

Two kinds are kept apart and never merged:

* **cell-level**: the licensed row's own evidence is `frozen_external_oracle`:
  its cited test compares the cell's output with a pinned upstream run
  recorded in `known_truth_fixture` (gate_support_matrix.sh requires that test
  to consume the fixture).
* **component-level**: a component the cell's execution ran is checked against
  an external oracle elsewhere. The components come from
  `parity/licensed_routes.toml`, which the licensed compiler test generates from
  each cell's executed logical plan (identifier and estimator ids). A parity row
  with an external evidence kind declares the components its oracle test
  exercises in `oracle_components` (`estimator:<id>`, `identifier:<id>`); the
  join is mechanical. `discovery:<GraphClass>` components link an `accepted`
  cell of that class to the discovery algorithms whose output class it is, for
  analyses whose accepted graph came from that discovery strategy. A component
  link never says the cell's own output was compared with an oracle.

    python3 scripts/external_evidence.py counts
    python3 scripts/external_evidence.py check
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSED = ROOT / "parity/support_licensed.toml"
ROUTES = ROOT / "parity/licensed_routes.toml"
AXES = ROOT / "parity/support_axes.toml"
EXTERNAL_KINDS = {"frozen_external_oracle", "behavioral_parity"}
COMPONENT_KINDS = ("estimator", "identifier", "discovery")


def coordinate(cell: dict) -> str:
    return ":".join(
        cell[k] for k in ("query", "graph_class", "structure", "inference", "validation")
    )


def _manifests() -> list[Path]:
    return sorted(
        p
        for p in (ROOT / "parity").glob("*.toml")
        if p.name not in {"support_licensed.toml", "licensed_routes.toml", "coverage_records.toml"}
    )


def component_rows() -> dict[str, list[dict]]:
    """`kind:id` -> the parity rows whose external oracle exercises it."""
    out: dict[str, list[dict]] = {}
    for path in _manifests():
        for row in tomllib.loads(path.read_text()).get("capabilities", []):
            for component in row.get("oracle_components") or []:
                out.setdefault(str(component), []).append({**row, "_manifest": path.name})
    return out


def routes() -> dict[str, dict]:
    if not ROUTES.is_file():
        return {}
    return {r["coordinate"]: r for r in tomllib.loads(ROUTES.read_text()).get("route", [])}


def cell_links(cell: dict, route: dict | None, components: dict[str, list[dict]]) -> dict:
    """{'cell': fixture or None, 'components': [(row id, component)], 'discovery': [...]}."""
    links: dict = {"cell": None, "components": [], "discovery": []}
    if cell.get("evidence_kind") in EXTERNAL_KINDS:
        links["cell"] = cell.get("known_truth_fixture")
    if route:
        for kind in ("identifier", "estimator"):
            key = f"{kind}:{route.get(kind)}"
            for row in components.get(key, []):
                links["components"].append((row["id"], key))
    if cell.get("structure") == "accepted":
        key = f"discovery:{cell['graph_class']}"
        for row in components.get(key, []):
            links["discovery"].append((row["id"], key))
    return links


def render(links: dict) -> str:
    parts = []
    if links["cell"]:
        parts.append(f"cell output compared with `{links['cell']}`")
    if links["components"]:
        rows = ", ".join(f"`{rid}` ({comp})" for rid, comp in links["components"])
        parts.append(f"component oracles: {rows}")
    if links["discovery"]:
        rows = ", ".join(f"`{rid}`" for rid, _ in links["discovery"])
        parts.append(f"discovery oracles when the accepted graph is discovered: {rows}")
    return "; ".join(parts) or "none"


def counts() -> dict[str, int]:
    cells = tomllib.loads(LICENSED.read_text()).get("cell", [])
    comps, rts = component_rows(), routes()
    out = {"cell_level": 0, "component_level_only": 0, "discovery_only": 0, "neither": 0}
    for cell in cells:
        links = cell_links(cell, rts.get(coordinate(cell)), comps)
        if links["cell"]:
            out["cell_level"] += 1
        elif links["components"]:
            out["component_level_only"] += 1
        elif links["discovery"]:
            out["discovery_only"] += 1
        else:
            out["neither"] += 1
    return out


def check() -> list[str]:
    problems: list[str] = []
    cells = tomllib.loads(LICENSED.read_text()).get("cell", [])
    rts = routes()
    licensed = {coordinate(c) for c in cells}
    if not rts:
        problems.append(f"{ROUTES.relative_to(ROOT)} is missing or empty")
    for missing in sorted(licensed - set(rts)):
        problems.append(f"licensed cell {missing} has no route in parity/licensed_routes.toml")
    for stale in sorted(set(rts) - licensed):
        problems.append(f"parity/licensed_routes.toml routes {stale}, which is not licensed")
    ran = {kind: {r.get(kind) for r in rts.values()} for kind in ("identifier", "estimator")}
    graph_classes = set(tomllib.loads(AXES.read_text()).get("graph_classes", []))
    for component, rows in component_rows().items():
        kind, _, ident = component.partition(":")
        for row in rows:
            label = f"{row['_manifest']} {row.get('id')}"
            if row.get("evidence_kind") not in EXTERNAL_KINDS:
                problems.append(
                    f"{label}: oracle_components on a {row.get('evidence_kind')!r} row; only "
                    "an external oracle row can back a component"
                )
            if not row.get("known_truth_fixture"):
                problems.append(f"{label}: oracle_components without the fixture its oracle froze")
        if kind not in COMPONENT_KINDS or not ident:
            problems.append(f"oracle component {component!r} is not kind:id ({COMPONENT_KINDS})")
        elif kind in ran and ident not in ran[kind]:
            problems.append(
                f"oracle component {component!r} is not a {kind} any licensed cell runs "
                "(parity/licensed_routes.toml)"
            )
        elif kind == "discovery" and ident not in graph_classes:
            problems.append(f"oracle component {component!r} names no graph class")
    return problems


def main(argv: list[str]) -> int:
    if argv == ["counts"]:
        for key, value in counts().items():
            print(f"{key}: {value}")
        return 0
    if argv == ["check"]:
        problems = check()
        for p in problems:
            print(f" - {p}")
        return 1 if problems else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
