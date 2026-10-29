#!/usr/bin/env python3
"""Method-family scenario matrix: every family x scenario axis as delivered, unavailable or gap.

Source inspection only; no test is executed. The matrix joins parity/counterparts.toml
(design x inference quadrants, uncertainty, graph class), parity/missing_policy_support.toml and
the curated cells of parity/scenario_matrix.toml. A cell is `delivered` only when it names an
exact compiled, non-ignored test (or a conformance fixture) that exists; a file named without a
test, a test whose name says smoke or limit, and a measurement-only test do not count. A cell no
registry derives and the curated file does not record is a gap.

A cell that executes and is tested but owes a calibration measurement is `pending_measurement`,
never a gap, and only through parity/pending_measurement.toml: the entry names the calibration facet
of scripts/calibration_surface.list whose records will close it and the kind of measurement owed.
An entry without a defined facet or a known kind, an entry that names no open cell, and an axis
other than design_inference or uncertainty are findings, and the cell stays a gap.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import tomllib
from _repo import REPO_ROOT
from test_evidence import static_python_test, static_rust_test

DELIVERED, UNAVAILABLE, GAP = "delivered", "unavailable", "gap"
# A cell whose only open element is a calibration measurement the pending frozen candidate owes.
PENDING = "pending_measurement"
OWED_KINDS = (
    "interval_coverage",
    "posterior_calibration",
    "selected_decision_reliability",
    "refuter_error_rates",
    "band_coverage",
)
PENDING_AXES = ("design_inference", "uncertainty")
QUADRANTS = tuple(
    f"{design}.{inference}"
    for design in ("static", "temporal")
    for inference in ("frequentist", "bayesian")
)
GRAPH_CLASSES = {
    "dag": {"Dag"},
    "latent_or_equivalence_class": {"Admg", "Cpdag", "Pag", "CoDetermined"},
    "temporal_graph": {"TemporalDag", "TemporalAdmg", "TemporalCpdag", "TemporalPag"},
    "graphless": None,
}
DERIVED_AXES = {
    "design_inference": ("Static/temporal x frequentist/Bayesian", QUADRANTS),
    "uncertainty": ("Uncertainty reported", QUADRANTS),
    "graph_class": ("Model and graph class", tuple(GRAPH_CLASSES)),
    "missing_data_policy": (
        "Declared missing-data policy of the family's target",
        ("missing_data_policy",),
    ),
}
# Derived axes whose registry gaps a curated cell may settle; the quadrant axes (design x inference,
# uncertainty) stay tied to their counterpart rows and their measured closure.
OVERRIDABLE_AXES = ("graph_class", "missing_data_policy")
# The uncertainty axis may additionally be settled by a curated cell, but only where the counterpart
# row executes with its claim whole and reports no interval (note "uncertainty=none"): a family whose
# product is not a sampled estimate says so as unavailable with its reason. A cell open for another
# reason, such as an owed calibration, is not settled that way.
UNCERTAINTY_NONE = "uncertainty=none"
NOT_EVIDENCE = re.compile(r"smoke|limit", re.IGNORECASE)


def _check_token(root: Path, token: str) -> tuple[bool, str]:
    """(is an exact live test or fixture, why not)."""
    token = token.strip()
    rel, sep, name = token.partition("::")
    path = root / rel
    if not sep:
        if rel.startswith("conformance/") and (path / "expected.json").is_file():
            return True, ""
        return False, "names a file or directory, not an exact test or fixture"
    if NOT_EVIDENCE.search(name):
        return False, "test name says smoke or limit, which is not delivered coverage"
    if not path.is_file():
        return False, f"{rel} does not exist"
    if path.suffix == ".rs":
        problems = static_rust_test(path, name).problems
    elif path.suffix == ".py":
        problems = static_python_test(path, name)
    else:
        return False, f"{rel} is not a Rust or Python test source"
    return (not problems), "; ".join(problems)


def _cell(family, axis, value, status, evidence="", note="", **extra) -> dict:
    return {
        "family": family,
        "axis": axis,
        "value": value,
        "status": status,
        "evidence": evidence,
        "note": note,
        **extra,
    }


def _quadrant_cells(root: Path, family: str, rows: list[dict], checked: dict) -> list[dict]:
    cells = []
    by = {(r["design"], r["inference"]): r for r in rows}
    for quadrant in QUADRANTS:
        design, inference = quadrant.split(".")
        row = by.get((design, inference))
        if row is None:
            for axis in ("design_inference", "uncertainty"):
                cells.append(
                    _cell(family, axis, quadrant, GAP, note="no counterpart row records the quadrant")
                )
            continue
        if row["status"] == "not_applicable_with_theory_reason":
            for axis in ("design_inference", "uncertainty"):
                cells.append(
                    _cell(family, axis, quadrant, UNAVAILABLE, note=row["reason"], row=row["id"])
                )
            continue
        ok, why = checked.setdefault(
            row["execution"], _check_token(root, row["execution"])
        )
        delivered = row["closure"] == "delivered" and row["gap"] == "none" and ok
        note = row["reason"] or ("" if ok else why)
        cells.append(
            _cell(
                family, "design_inference", quadrant,
                DELIVERED if delivered else GAP,
                row["execution"] if delivered else "", note, row=row["id"],
            )
        )
        reported = row["uncertainty"] != "none"
        cells.append(
            _cell(
                family, "uncertainty", quadrant,
                DELIVERED if delivered and reported else GAP,
                row["execution"] if delivered and reported else "",
                f"uncertainty={row['uncertainty']}" if delivered else note, row=row["id"],
            )
        )
    return cells


def _graph_cells(root: Path, family: str, rows: list[dict], checked: dict) -> list[dict]:
    live = [r for r in rows if r["status"] != "not_applicable_with_theory_reason"]
    ok_rows = [
        r for r in live
        if r["closure"] == "delivered" and r["gap"] == "none"
        and checked.setdefault(r["execution"], _check_token(root, r["execution"]))[0]
    ]
    suffix = lambda r: r["route"].rsplit(".", 1)[-1]  # noqa: E731
    graphed = {
        cls
        for r in live
        for cls, members in GRAPH_CLASSES.items()
        if members and suffix(r) in members
    }
    cells = []
    for cls, members in GRAPH_CLASSES.items():
        if members is None:
            routed = [r for r in live if not any(suffix(r) in m for m in GRAPH_CLASSES.values() if m)]
            ok = [r for r in ok_rows if r in routed]
            if ok:
                cells.append(_cell(family, "graph_class", cls, DELIVERED, ok[0]["execution"],
                                   "executes without a graph argument", row=ok[0]["id"]))
            elif graphed and not routed:
                cells.append(_cell(family, "graph_class", cls, UNAVAILABLE,
                                   note="every recorded route of this family takes a graph"))
            else:
                cells.append(_cell(family, "graph_class", cls, GAP,
                                   note="no delivered graphless counterpart with an exact test"))
            continue
        ok = [r for r in ok_rows if suffix(r) in members]
        if ok:
            cells.append(_cell(family, "graph_class", cls, DELIVERED, ok[0]["execution"],
                               f"route {ok[0]['route']}", row=ok[0]["id"]))
        elif not graphed:
            cells.append(_cell(family, "graph_class", cls, UNAVAILABLE,
                               note="the family executes without a graph; no graph class applies"))
        else:
            cells.append(_cell(family, "graph_class", cls, GAP,
                               note="no delivered route of this class with an exact test"))
    return cells


def _curated(root: Path, family, axis_id, value, src, checked, findings) -> dict:
    key = (family, axis_id, value)
    token = src.get("test") or src.get("fixture") or ""
    status = src["status"]
    if status not in (DELIVERED, UNAVAILABLE, GAP):
        findings.append(f"{key}: unknown status {status!r}")
        status = GAP
    if status == DELIVERED:
        ok, why = checked.setdefault(token, _check_token(root, token)) if token \
            else (False, "no test or fixture")
        if not ok:
            findings.append(f"{key}: delivered but {why}: {token}")
            status = GAP
    elif status == UNAVAILABLE:
        if not str(src.get("reason", "")).strip():
            findings.append(f"{key}: unavailable without a reason")
        if token and not checked.setdefault(token, _check_token(root, token))[0]:
            findings.append(f"{key}: refusal evidence does not resolve: {token}")
    return _cell(family, axis_id, value, status, token if status != GAP else "",
                 src.get("what") or src.get("reason", ""))


def surface_facets(root: Path) -> set[str]:
    """The facet names scripts/calibration_surface.list defines (a path, key or key_all line)."""
    facets: set[str] = set()
    path = root / "scripts/calibration_surface.list"
    if not path.is_file():
        return facets
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts or parts[0].startswith("#") or parts[0] in ("allow", "testmod"):
            continue
        facets.add(parts[1] if parts[0] in ("key", "key_all") and len(parts) > 1 else parts[0])
    return facets


def _apply_pending(root: Path, cells: list[dict], findings: list[str]) -> None:
    """Turn the gap cells a registered entry owes to calibration into `pending_measurement`."""
    path = root / "parity/pending_measurement.toml"
    if not path.is_file():
        return
    entries = tomllib.loads(path.read_text(encoding="utf-8")).get("pending", [])
    facets = surface_facets(root)
    seen: set[str] = set()
    for entry in entries:
        row = entry.get("row", "")
        where = f"pending_measurement {row!r}"
        if row in seen:
            findings.append(f"{where}: duplicated")
            continue
        seen.add(row)
        facet, owed = str(entry.get("facet", "")).strip(), entry.get("owed")
        axes = entry.get("axes") or list(PENDING_AXES)
        if not facet:
            findings.append(f"{where}: no calibration facet that will close it")
        elif facet not in facets:
            findings.append(f"{where}: facet {facet} is not defined in scripts/calibration_surface.list")
        if owed not in OWED_KINDS:
            findings.append(f"{where}: owed {owed!r} is not one of {', '.join(OWED_KINDS)}")
        if any(axis not in PENDING_AXES for axis in axes):
            findings.append(f"{where}: axes {axes} are not among {', '.join(PENDING_AXES)}")
        if not facet or facet not in facets or owed not in OWED_KINDS:
            continue
        hit = [
            c for c in cells
            if c.get("row") == row and c["status"] == GAP and c["axis"] in axes
        ]
        if not hit:
            findings.append(f"{where}: names no open cell on {axes}; remove the entry")
        for c in hit:
            c["status"] = PENDING
            c["facet"], c["owed"] = facet, owed
            c["gap_kind"] = ""


def build(root: Path = REPO_ROOT) -> dict:
    root = root.resolve()
    counterparts = tomllib.loads((root / "parity/counterparts.toml").read_text(encoding="utf-8"))
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in counterparts["row"]:
        if row["kind"] == "method":
            by_family[row["method"]].append(row)
    curated = tomllib.loads((root / "parity/scenario_matrix.toml").read_text(encoding="utf-8"))
    axes = {a["id"]: a for a in curated.get("axis", [])}
    missing = tomllib.loads((root / "parity/missing_policy_support.toml").read_text(encoding="utf-8"))
    missing_queries = {
        r["query"]: r for r in missing.get("route", []) if len(r.get("supported", [])) > 1
    }
    findings: list[str] = []
    recorded: dict[tuple, dict] = {}
    derived_overrides: dict[tuple, dict] = {}
    for cell in curated.get("cell", []):
        key = (cell.get("family"), cell.get("axis"), cell.get("value"))
        if cell.get("family") not in by_family:
            findings.append(f"curated cell names unknown family: {key}")
        elif cell.get("axis") in OVERRIDABLE_AXES + ("uncertainty",):
            if cell.get("value") not in DERIVED_AXES[cell["axis"]][1]:
                findings.append(f"curated cell names unknown axis value: {key}")
            elif cell.get("status") == GAP:
                findings.append(f"curated cell on a derived axis must be delivered or unavailable: {key}")
            elif key in recorded:
                findings.append(f"curated cell is duplicated: {key}")
            else:
                derived_overrides[key] = cell
        elif cell.get("axis") not in axes or cell.get("value") not in axes[cell["axis"]]["values"]:
            findings.append(f"curated cell names unknown axis value: {key}")
        elif key in recorded:
            findings.append(f"curated cell is duplicated: {key}")
        else:
            recorded[key] = cell
    checked: dict = {}
    cells: list[dict] = []
    for family, rows in sorted(by_family.items()):
        cells += _quadrant_cells(root, family, rows, checked)
        cells += _graph_cells(root, family, rows, checked)
        hit = next((missing_queries[r["target"]] for r in rows if r["target"] in missing_queries), None)
        cells.append(
            _cell(family, "missing_data_policy", "missing_data_policy",
                  DELIVERED if hit else GAP,
                  "crates/antecedent/tests/missing_policy_support.rs::every_route_cell_matches_the_declared_support"
                  if hit else "",
                  f"supported policies {hit['supported']}" if hit
                  else "no route cell for this family's target declares more than the error policy")
        )
        notapplicable = all(r["sampling_design"].startswith("none:") for r in rows)
        for axis_id, axis in axes.items():
            for value in axis["values"]:
                key = (family, axis_id, value)
                src = recorded.get(key)
                if src is None:
                    if notapplicable:
                        cells.append(_cell(family, axis_id, value, UNAVAILABLE,
                                           note=rows[0]["sampling_design"]))
                    else:
                        cells.append(_cell(family, axis_id, value, GAP,
                                           note="no test recorded for this scenario"))
                    continue
                cells.append(_curated(root, family, axis_id, value, src, checked, findings))
    # A derived cell that the registries leave a gap may be settled by a curated cell that names
    # an exact test: delivered, or unavailable with the reason and the refusal that shows it.
    for index, c in enumerate(cells):
        key = (c["family"], c["axis"], c["value"])
        settles = c["axis"] in OVERRIDABLE_AXES or (
            c["axis"] == "uncertainty" and c["note"] == UNCERTAINTY_NONE
        )
        if settles and c["status"] == GAP and key in derived_overrides:
            cells[index] = _curated(root, *key, derived_overrides[key], checked, findings)
        elif key in derived_overrides and c["axis"] == "uncertainty":
            findings.append(f"curated uncertainty cell does not settle an open 'uncertainty=none' cell: {key}")
    for c in cells:
        if c["status"] != GAP:
            continue
        note = c["note"]
        c["gap_kind"] = (
            "execution_cites_file_not_exact_test"
            if "names a file" in note or "does not exist" in note
            else "no_test_recorded"
            if note.startswith("no test recorded")
            else "no_delivered_route_or_policy"
            if note.startswith(("no delivered", "no route cell", "no counterpart"))
            else "narrowed_promise"
        )
    _apply_pending(root, cells, findings)
    counts = Counter(c["status"] for c in cells)
    per_axis: dict[str, Counter] = defaultdict(Counter)
    for c in cells:
        per_axis[c["axis"]][c["status"]] += 1
    return {
        "schema": 1,
        "execution_verified": False,
        "families": sorted(by_family),
        "axes": {
            **{k: {"title": t, "values": list(v)} for k, (t, v) in DERIVED_AXES.items()},
            **{k: {"title": a["title"], "values": a["values"]} for k, a in axes.items()},
        },
        "counts": dict(counts),
        "by_axis": {k: dict(v) for k, v in sorted(per_axis.items())},
        "cells": cells,
        "findings": findings,
    }


def gaps(matrix: dict) -> list[dict]:
    return [c for c in matrix["cells"] if c["status"] == GAP]


def pending(matrix: dict) -> list[dict]:
    return [c for c in matrix["cells"] if c["status"] == PENDING]


if __name__ == "__main__":
    result = build()
    print(result["counts"], len(result["findings"]), file=sys.stderr)
    for finding in result["findings"]:
        print(finding)
