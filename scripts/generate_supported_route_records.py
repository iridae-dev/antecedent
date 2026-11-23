#!/usr/bin/env python3
"""Generate the `supported.<query>.<graph class>` route records the route manifest lacks.

Every (query, graph class) pair with a supported cell in parity/support_cells.toml has one
`[[route]]` record in parity/capabilities/route_manifest.toml. An existing record is never
touched. A pair without a record gets a generated one, appended at the end of the file: the
structures and inferences, the supported coordinates and the estimator wire ids all come from
the supported table, so the record cannot disagree with it. A `supported.*` record whose pair has
no supported cell is stale.

    python3 scripts/generate_supported_route_records.py          # write
    python3 scripts/generate_supported_route_records.py --check  # fail on a missing or stale record
"""

from __future__ import annotations

import collections
import json
import sys
import tomllib
from pathlib import Path

from _repo import load_toml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "parity/capabilities/route_manifest.toml"
SUPPORTED = ROOT / "parity/support_cells.toml"
STRUCTURES = ("explicit", "accepted", "graph_posterior")
INFERENCES = ("Frequentist", "Bayesian")


def record(query: str, graph_class: str, cells: list[dict]) -> str:
    structures = [s for s in STRUCTURES if any(c["structure"] == s for c in cells)]
    inferences = [i for i in INFERENCES if any(c["inference"] == i for c in cells)]
    coordinates = [f"{c['structure']}:{c['inference']}:{c['validation']}" for c in cells]
    estimators: list[str] = []
    for cell in cells:
        for estimator in cell["estimators"]:
            if estimator not in estimators:
                estimators.append(estimator)
    design = "temporal" if graph_class.startswith("Temporal") else "static"
    fields = {
        "id": f"supported.{query}.{graph_class}",
        "target": query,
        "graph_class": graph_class,
        "design_time_model": f"{design}; structure={','.join(structures)}",
        "inference": ",".join(inferences),
        "decision_ports": "unlisted",
        "artifact_consumer": "unlisted",
        "status": "supported",
        "owning_module": "crates/antecedent/src/analysis/execute/compile.rs",
        "notes": (
            f"Supported coordinates ({len(cells)}): {', '.join(coordinates)}. "
            f"Estimators: {', '.join(estimators)}. Full coordinates and estimator wire-ids: "
            "parity/supported_routes.toml and parity/support_cells.toml. Coordinates outside "
            "this list are listed in parity/support_refusal_reasons.toml."
        ),
    }
    return "[[route]]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in fields.items()) + "\n"


def pairs_of(text: str) -> tuple[dict[tuple[str, str], list[dict]], set[str]]:
    """The supported cells by (query, graph class), and the `supported.*` ids already recorded."""
    by_pair: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for cell in load_toml(SUPPORTED)["cell"]:
        by_pair[(cell["query"], cell["graph_class"])].append(cell)
    recorded = {
        r["id"] for r in tomllib.loads(text)["route"] if r["id"].startswith("supported.")
    }
    return by_pair, recorded


def render(text: str) -> str:
    by_pair, recorded = pairs_of(text)
    generated = "".join(
        record(query, graph_class, cells)
        for (query, graph_class), cells in sorted(by_pair.items())
        if f"supported.{query}.{graph_class}" not in recorded
    )
    return text.rstrip("\n") + "\n\n" + generated.rstrip("\n") + "\n" if generated else text


def stale(text: str) -> list[str]:
    by_pair, recorded = pairs_of(text)
    return sorted(recorded - {f"supported.{q}.{g}" for q, g in by_pair})


def main() -> int:
    text = MANIFEST.read_text(encoding="utf-8")
    new = render(text)
    tomllib.loads(new)
    if "--check" in sys.argv:
        problems = []
        if new != text:
            problems.append("a supported (query, graph class) pair has no route record")
        problems += [f"{ident} has no supported cell" for ident in stale(text)]
        if problems:
            print("route_manifest.toml supported route records are out of date: " + "; ".join(problems))
            print("run python3 scripts/generate_supported_route_records.py")
            return 1
        print("supported route records match the supported table")
        return 0
    MANIFEST.write_text(new, encoding="utf-8")
    supported = [r for r in tomllib.loads(new)["route"] if r["id"].startswith("supported.")]
    print(f"{len(supported)} supported route records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
