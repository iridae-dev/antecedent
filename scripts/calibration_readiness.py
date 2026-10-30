#!/usr/bin/env python3
"""Audit whether currently unmeasured licensed coordinates have calibration wiring.

This is a static readiness check only. It reads the support registry, evidence
test sources, and the calibration gate's explicit dry-run group list. It never
invokes a calibration test, writes a record, or edits a coverage registry.

    python3 scripts/calibration_readiness.py [--check]
"""

from __future__ import annotations

import argparse
import collections
import re
import tomllib

from _repo import REPO_ROOT as ROOT
from test_evidence import rust_items


def _slug(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


SUPPORT = ROOT / "parity/support_licensed.toml"
OUTPUT = ROOT / "parity/calibration_readiness.md"


def cells_by_coordinate() -> dict[tuple[str, str, str, str], list[dict]]:
    cells = tomllib.loads(SUPPORT.read_text(encoding="utf-8"))["cell"]
    grouped: dict[tuple[str, str, str, str], list[dict]] = collections.defaultdict(list)
    for cell in cells:
        if cell.get("calibration_reason") == "estimator_grid_not_measured":
            key = tuple(cell[field] for field in ("query", "graph_class", "structure", "inference"))
            grouped[key].append(cell)
    return dict(sorted(grouped.items()))


def calibration_groups() -> list[tuple[str, str]]:
    """Read group names from gate dry-run; no test command is executed."""
    import subprocess

    env = dict(__import__("os").environ, ANTECEDENT_CALIBRATION_DRY_RUN="1")
    env.pop("ANTECEDENT_CALIBRATION_SHARD", None)
    env.pop("ANTECEDENT_JOINT_RESPONSE_STABLE_CANDIDATE", None)
    result = subprocess.run(
        ["bash", "scripts/gate_calibration.sh"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    groups = []
    for line in result.stdout.splitlines():
        prefix, separator, label = line.partition(": ")
        if separator and prefix.startswith("group "):
            head, separator, test_filter = label.partition(": ")
            groups.append((head, test_filter if separator else ""))
    if not groups:
        raise RuntimeError("calibration gate dry-run returned no groups")
    return groups


def _phrase(label: str, phrase: str, *, static_graph: bool = False) -> bool:
    # Underscore-delimited components are tokens: dag does not match cpdag or
    # the dag part of temporal_dag. Query/graph names may themselves use underscores.
    normalized = re.sub(r"[^a-z0-9_]+", "_", label.lower())
    token = _slug(phrase)
    for match in re.finditer(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", normalized):
        if static_graph and normalized[:match.start()].endswith("temporal_"):
            continue
        return True
    return False


def _candidate(key: tuple[str, str, str, str], label: str) -> bool:
    query, graph, _structure, inference = key
    return (_phrase(label, query) and _phrase(label, graph, static_graph=graph in {"Dag", "Cpdag", "Pag", "Admg"})
            and _phrase(label, inference))


def candidate_routes(key: tuple[str, str, str, str], groups: list[tuple[str, str]]) -> str:
    """List name-matched groups; these are candidates, never proof of fit."""
    labels = [f"{head}: {filter_}" if filter_ else head for head, filter_ in groups]
    candidates = [label for label in labels if _candidate(key, label)]
    return "<br>".join(candidates) if candidates else "none (no query/graph/inference candidate group)"


def has_record_emitter(key: tuple[str, str, str, str], groups: list[tuple[str, str]]) -> bool:
    """Find syntactic wiring in the exact ignored test; not a scientific match."""
    structure = key[2]
    for head, test_filter in groups:
        if not _candidate(key, f"{head} {test_filter}"):
            continue
        if structure == "accepted" and not _phrase(test_filter, "accepted"):
            continue
        if structure == "graph_posterior" and not _phrase(test_filter, "graph_posterior"):
            continue
        if structure == "explicit" and any(_phrase(test_filter, marker) for marker in ("accepted", "graph_posterior", "class_posterior")):
            continue
        test_file = ROOT / "crates" / "antecedent" / "tests" / f"{head}.rs"
        if not test_file.is_file() or not test_filter:
            continue
        masked, items = rust_items(test_file)
        functions = [item for item in items if item.kind == "fn" and item.name == test_filter]
        if len(functions) != 1:
            continue
        test = functions[0]
        attrs = [re.sub(r"\s+", " ", attr) for attr in test.attrs]
        if not any(re.fullmatch(r'#\[ignore\s*=\s*"calibration: run via scripts/gate_calibration\.sh"\s*\]', attr) for attr in attrs):
            continue
        if not any(re.fullmatch(r"#\[test\s*\]", attr) for attr in attrs):
            continue
        if _reaches_bound_record_emitter(masked, items, test):
            return True
    return False


def _reaches_bound_record_emitter(masked, items, test) -> bool:
    """Syntactic local call graph; braces in literals/comments cannot end a body."""
    functions = {}
    duplicate = set()
    for item in items:
        if item.kind != "fn":
            continue
        lo, hi = item.body
        if lo < 0 or hi <= lo or hi > len(masked.skeleton) or masked.skeleton[hi - 1] != "}":
            continue
        if item.name in functions:
            duplicate.add(item.name)
        functions[item.name] = (masked.code[lo:hi], masked.skeleton[lo:hi])
    if test.name in duplicate or test.name not in functions:
        return False
    for name in duplicate:
        functions.pop(name, None)
    pending = [test.name]
    seen = set()
    bodies = []
    while pending:
        name = pending.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        code, skeleton = functions[name]
        bodies.append(skeleton)
        pending.extend(called for called in functions if re.search(rf"(?<![\w:]){re.escape(called)}\s*\(", skeleton))
    chain = "\n".join(bodies)
    test_code, _ = functions[test.name]
    exact = f'"{test.name}"' in test_code or any(
        re.search(rf"\b{re.escape(name)}\b", functions[test.name][1])
        for name in re.findall(rf'\b(?:const|let)\s+([A-Za-z_]\w*)\s*:\s*&str\s*=\s*"{re.escape(test.name)}"', masked.code)
    )
    return ("CoverageTally::for_record" in chain and exact
            and bool(re.search(r"RecordKey\s*\{\s*test(?:\s*[, :])", chain))
            and bool(re.search(r"\b(?:bind|bind_all)\s*\(", chain))
            and bool(re.search(r"(?:\.record|\brecord_pair|\brecord)\s*\(", chain)))


def render() -> str:
    coordinates = cells_by_coordinate()
    groups = calibration_groups()
    lines = [
        "# Calibration readiness audit",
        "",
        "Generated by `scripts/calibration_readiness.py`. This is a readiness inventory, not a calibration run or a coverage record.",
        "The gate is queried in explicit dry-run mode only. A point-truth test is not treated as a repeated-sampling calibration DGP.",
        "",
        f"**{sum(len(rows) for rows in coordinates.values())} unmeasured cells across {len(coordinates)} coordinates.**",
        "",
        "Each row records the support route, declared truth fixture, estimator target, semantically similar gate groups, and why exact record emission is or is not wired. Similar group names are candidates only; they do not establish a matching DGP or target.",
        "",
        "| Coordinate (query / graph / structure / inference) | Cells / variants | Procedural evidence route and assertion | Declared truth fixture | Interval estimator / target | Candidate calibration group | Record readiness and next step |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, rows in coordinates.items():
        variants = sorted({str(row.get("validation", "none")) for row in rows})
        cells = f"{len(rows)} ({', '.join(variants)})"
        routes = sorted({f"{r['evidence_test']}::{r['evidence_assertion']}" for r in rows})
        fixtures = sorted({str(r.get("known_truth_fixture", "none declared")) for r in rows})
        estimators = sorted({est for row in rows for est in row.get("estimators", [])})
        route_text = "<br>".join(routes)
        fixture_text = "<br>".join(fixtures)
        estimator_text = "<br>".join(estimators)
        group_text = candidate_routes(key, groups)
        has_truth_fixture = any(row.get("known_truth_fixture") for row in rows)
        is_ready = has_record_emitter(key, groups)
        if is_ready:
            readiness = "Candidate wiring found in an ignored gate test; verify its DGP, construction and exact interval target before any future run. No calibration was run by this audit."
        else:
            truth_step = (
                "validate the declared fixture's exact truth for this interval estimand, then derive a repeated-sampling DGP"
                if has_truth_fixture
                else "add a known-truth repeated-sampling DGP and exact interval estimand truth"
            )
            readiness = f"Matching calibration wiring not established: {truth_step}; review candidate groups and bind a verified `CoverageTally::for_record` emission to this exact procedure and estimator target."
        coordinate = " / ".join(key)
        lines.append(
            f"| {coordinate} | {cells} | {route_text} | {fixture_text} | {estimator_text} | {group_text} | {readiness} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify the checked-in audit is current"
    )
    args = parser.parse_args()
    report = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != report:
            parser.error("parity/calibration_readiness.md is stale")
    else:
        OUTPUT.write_text(report, encoding="utf-8", newline="\n")
    print(f"Calibration readiness audit: {len(cells_by_coordinate())} coordinates; no calibration tests executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
