#!/usr/bin/env python3
"""Generate the `licensed.<query>.<graph class>` route records of the route manifest.

Every (query, graph class) pair with a licensed cell in parity/support_licensed.toml has one
`[[route]]` record in parity/capabilities/route_manifest.toml. The long-standing records carry
hand-written uncertainty text and are never touched. A pair without a hand-written record gets
a generated one: the structures, inferences and validations, the licensed coordinates and the
estimator wire ids all come from the licensed table, and the uncertainty text defers to the
method records, so the record cannot disagree with the table.

    python3 scripts/generate_licensed_route_records.py          # write
    python3 scripts/generate_licensed_route_records.py --check  # fail on drift

A generated record is recognised by GENERATED_UNCERTAINTY; edit a record by hand (and drop that
marker) to take it out of the generator's hands.
"""

from __future__ import annotations

import collections
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "parity/capabilities/route_manifest.toml"
LICENSED = ROOT / "parity/support_licensed.toml"
METHODS_MARK = "# ---- generated method records"
GENERATED_UNCERTAINTY = "per-estimator uncertainty procedure is stated by the method records of this route"
STRUCTURES = ("explicit", "accepted", "graph_posterior")
INFERENCES = ("Frequentist", "Bayesian")
VALIDATIONS = ("none", "cheap", "full")
BLOCK = re.compile(r"^\[\[route\]\]\n(?:[^\n]+\n)+", re.MULTILINE)


def record(query: str, graph_class: str, cells: list[dict]) -> str:
    structures = [s for s in STRUCTURES if any(c["structure"] == s for c in cells)]
    inferences = [i for i in INFERENCES if any(c["inference"] == i for c in cells)]
    validations = [v for v in VALIDATIONS if any(c["validation"] == v for c in cells)]
    coordinates = [f"{c['structure']}:{c['inference']}:{c['validation']}" for c in cells]
    estimators: list[str] = []
    for cell in cells:
        for estimator in cell["estimators"]:
            if estimator not in estimators:
                estimators.append(estimator)
    design = "temporal" if graph_class.startswith("Temporal") else "static"
    fields = {
        "id": f"licensed.{query}.{graph_class}",
        "target": query,
        "graph_class": graph_class,
        "design_time_model": f"{design}; structure={','.join(structures)}",
        "inference": ",".join(inferences),
        "uncertainty": (
            f"validation={','.join(validations)}; {GENERATED_UNCERTAINTY}; "
            "calibration pending, estimator_grid_not_measured, no boundary record"
        ),
        "decision_ports": "unlisted",
        "artifact_consumer": "unlisted",
        "status": "licensed",
        "owning_module": "crates/antecedent/src/analysis/execute/compile.rs",
        "notes": (
            f"Licensed coordinates ({len(cells)}): {', '.join(coordinates)}. "
            f"Estimators: {', '.join(estimators)}. Full coordinates and estimator wire-ids: "
            "parity/licensed_routes.toml and parity/support_licensed.toml. Coordinates outside "
            "this list are listed in parity/support_closed.toml."
        ),
    }
    return "[[route]]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in fields.items()) + "\n"


def render(text: str) -> str:
    licensed = tomllib.loads(LICENSED.read_text(encoding="utf-8"))["cell"]
    by_pair: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for cell in licensed:
        by_pair[(cell["query"], cell["graph_class"])].append(cell)
    hand_written = set()
    kept = text
    for block in list(BLOCK.finditer(text)):
        body = block.group(0)
        ident = re.search(r'^id = "([^"]+)"', body, re.MULTILINE)
        if ident is None or not ident.group(1).startswith("licensed."):
            continue
        if GENERATED_UNCERTAINTY in body:
            kept = kept.replace(body + "\n", "", 1) if body + "\n" in kept else kept.replace(body, "", 1)
        else:
            hand_written.add(ident.group(1))
    generated = "".join(
        record(query, graph_class, cells)
        for (query, graph_class), cells in sorted(by_pair.items())
        if f"licensed.{query}.{graph_class}" not in hand_written
    )
    at = kept.index(METHODS_MARK)
    return kept[:at] + generated + kept[at:]


def main() -> int:
    text = MANIFEST.read_text(encoding="utf-8")
    tomllib.loads(text)
    new = render(text)
    tomllib.loads(new)
    if "--check" in sys.argv:
        if new != text:
            print("route_manifest.toml licensed route records are stale; run "
                  "python3 scripts/generate_licensed_route_records.py")
            return 1
        print("licensed route records match the licensed table")
        return 0
    MANIFEST.write_text(new, encoding="utf-8")
    routes = tomllib.loads(new)["route"]
    licensed = [r for r in routes if r["id"].startswith("licensed.")]
    generated = [r for r in licensed if GENERATED_UNCERTAINTY in r["uncertainty"]]
    print(f"{len(licensed)} licensed route records, {len(generated)} generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
