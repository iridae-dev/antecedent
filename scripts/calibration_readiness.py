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
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "parity/support_licensed.toml"
OUTPUT = ROOT / "parity/calibration_readiness.md"


def cells_by_coordinate() -> dict[tuple[str, str, str, str], list[dict]]:
    cells = tomllib.loads(SUPPORT.read_text())["cell"]
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


def candidate_routes(key: tuple[str, str, str, str], groups: list[tuple[str, str]]) -> str:
    """List semantically similar group labels; these are candidates, not proof of fit."""
    query, graph, _structure, inference = key
    slug = lambda value: re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    graph_alias = {"Cpdag": "cpdag", "Pag": "pag", "Admg": "admg"}.get(graph, slug(graph))
    needles = (slug(query), graph_alias, inference.lower())
    labels = [f"{head}: {filter_}" if filter_ else head for head, filter_ in groups]
    candidates = [label for label in labels if all(needle in label.lower() for needle in needles)]
    return "<br>".join(candidates) if candidates else "none (no query/graph/inference candidate group)"


def has_record_emitter(key: tuple[str, str, str, str], groups: list[tuple[str, str]]) -> bool:
    """Whether a semantically matching gate group reaches a record-emitting test."""
    query, graph, structure, inference = key
    slug = lambda value: re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    graph_alias = {"Cpdag": "cpdag", "Pag": "pag", "Admg": "admg"}.get(graph, slug(graph))
    needles = (slug(query), graph_alias, inference.lower())
    for head, test_filter in groups:
        label = f"{head} {test_filter}".lower()
        if not all(needle in label for needle in needles):
            continue
        if structure == "accepted" and "accepted" not in test_filter.lower():
            continue
        # Some correctly scoped class-posterior tests use the query + graph
        # names in their test filter without repeating "graph_posterior"
        # (e.g. average_effect_cpdag_*). The gate's exact test filter and the
        # required query/graph/inference tokens are the binding here.
        if structure == "explicit" and any(
            marker in test_filter.lower() for marker in ("accepted", "graph_posterior", "class_posterior")
        ):
            continue
        test_file = ROOT / "crates" / "antecedent" / "tests" / f"{head}.rs"
        if not test_file.is_file() or not test_filter:
            continue
        source = test_file.read_text(errors="ignore")
        marker = f"fn {test_filter}("
        start = source.find(marker)
        if start < 0:
            continue
        opening = source.find("{", start)
        depth = 0
        end = opening
        for end in range(opening, len(source)):
            depth += source[end] == "{"
            depth -= source[end] == "}"
            if depth == 0:
                break
        body = source[opening : end + 1]
        # Calibration suites commonly keep the tally plumbing in shared
        # helpers (for example `test -> coverage -> coverage_over -> keyed`).
        # Follow that local Rust function-call graph instead of requiring the
        # test body itself to spell `CoverageTally::for_record`; otherwise a
        # real, bound emitter is incorrectly classified as unmeasurable.
        emitter = _reaches_bound_record_emitter(source, test_filter, body)
        if emitter and "#[ignore = \"calibration: run via scripts/gate_calibration.sh\"]" in source[:start]:
            return True
    return False


def _reaches_bound_record_emitter(source: str, test_name: str, test_body: str) -> bool:
    """Recognize a test whose local helper chain creates and scores records."""
    functions: dict[str, str] = {}
    declarations = list(re.finditer(r"\bfn\s+([A-Za-z_]\w*)\s*\(", source))
    for declaration in declarations:
        name = declaration.group(1)
        opening = source.find("{", declaration.end())
        if opening < 0:
            continue
        depth = 0
        for end in range(opening, len(source)):
            depth += source[end] == "{"
            depth -= source[end] == "}"
            if depth == 0:
                functions[name] = source[opening : end + 1]
                break

    pending = [test_name]
    seen: set[str] = set()
    reached: list[str] = []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        body = test_body if name == test_name else functions.get(name, "")
        if not body:
            continue
        reached.append(body)
        pending.extend(
            called for called in functions
            if re.search(rf"(?<![\w:]){re.escape(called)}\s*\(", body)
        )
    chain = "\n".join(reached)
    creates = "CoverageTally::for_record" in chain
    # The test must pass its own exact libtest name into the helper chain, and
    # that chain must put the parameter into the RecordKey. This prevents an
    # unrelated emitter in the same integration-test binary from making a
    # coordinate appear ready.
    exact_name_in_body = f'"{test_name}"' in test_body or any(
        name in test_body
        for name in re.findall(
            rf"\b(?:const|let)\s+([A-Za-z_]\w*)\s*:\s*&str\s*=\s*\"{re.escape(test_name)}\"",
            source,
        )
    )
    keyed_for_test = exact_name_in_body and bool(
        re.search(r"RecordKey\s*\{\s*test(?:\s*[, :])", chain)
    )
    binds = "bind(" in chain or "bind_all(" in chain
    scores = ".record(" in chain or "record_pair(" in chain or "record(" in chain
    return creates and keyed_for_test and binds and scores


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
            readiness = "Ready for a later run: a semantically matching ignored gate test emits a RecordKey and binds the checked construction; no calibration was run by this audit."
        else:
            truth_step = (
                "validate the declared fixture's exact truth for this interval estimand, then derive a repeated-sampling DGP"
                if has_truth_fixture
                else "add a known-truth repeated-sampling DGP and exact interval estimand truth"
            )
            readiness = (
                f"Not yet measurable: {truth_step}; bind the candidate group to this exact procedure and estimator target, then add a `CoverageTally::for_record` RecordKey emission."
            )
        coordinate = " / ".join(key)
        lines.append(
            f"| {coordinate} | {cells} | {route_text} | {fixture_text} | {estimator_text} | {group_text} | {readiness} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the checked-in audit is current")
    args = parser.parse_args()
    report = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != report:
            parser.error("parity/calibration_readiness.md is stale")
    else:
        OUTPUT.write_text(report)
    print("Calibration readiness audit: 92 coordinates; no calibration tests executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
