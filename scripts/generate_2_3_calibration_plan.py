#!/usr/bin/env python3
"""Inventory the 2.2 calibration backlog without weakening its reason codes."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "parity/support_licensed.toml"
OUTPUT = ROOT / "parity/2_3_calibration_plan.md"
AXES = ("query", "graph_class", "structure", "inference")


def disposition(row: dict) -> tuple[str, str]:
    if row["structure"] == "explicit":
        if (row["query"], row["graph_class"], row["inference"]) != (
            "NestedCounterfactualEffect", "Dag", "Frequentist"
        ):
            raise ValueError("unreviewed fixed-graph backlog coordinate")
        return (
            "point_only_no_interval",
            "The fixed Markovian natural-direct-effect route publishes a point only. "
            "Build a separate whole-estimator interval and repeated-row known-SCM grid "
            "before claiming sampling coverage; retain its current point truth fixture.",
        )
    if row["structure"] != "graph_posterior":
        raise ValueError("unreviewed backlog structure")
    quantity = row["query"]
    if quantity in {"ResponseCurve", "InterventionResponse"}:
        oracle = "a finite dose/horizon grid with per-atom truth and a declared pointwise or simultaneous target"
    elif quantity == "ConditionalEffect":
        oracle = "a finite conditional-stratum SCM with support and per-atom conditional-effect truth"
    else:
        oracle = "an enumerated graph/SCM family with per-atom ATE and unidentified-mass truth"
    if row["inference"] == "Bayesian":
        calibration = "posterior predictive/decision calibration for the actual mixed-atom claim"
    else:
        calibration = "whole-method repeated sampling, including the graph-weight and data dependence rule"
    return (
        "graph_mixture_no_whole_method_calibration",
        f"Freeze {oracle}; then measure {calibration}. Current atom values and supplied graph weights do not license an aggregate interval or posterior probability guarantee.",
    )


def render() -> str:
    cells = [
        row for row in tomllib.loads(SOURCE.read_text())["cell"]
        if row.get("calibration_reason") == "estimator_grid_not_measured"
    ]
    counts = Counter(tuple(row[a] for a in AXES) for row in cells)
    if len(cells) != 40 or len(counts) != 18:
        raise ValueError(f"backlog changed to {len(cells)} cells / {len(counts)} coordinates; review plan")
    cells.sort(key=lambda row: (row["structure"] != "explicit", *(row[a] for a in AXES), row.get("validation", "")))
    lines = [
        "# 2.3 X11 calibration backlog plan",
        "",
        "Generated from `support_licensed.toml` without changing its `estimator_grid_not_measured` markers. These are 40 cells across 18 distinct coordinates. The fixed-DAG point route is reviewed first; graph-posterior rows remain pending. A typed reason below explains why the missing calibration does not license an inferential claim. It is not a waiver or a coverage record.",
        "",
        "| Priority | Query | Graph | Structure | Inference | Validation | Estimator | Typed reason | Required next evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in cells:
        reason, next_evidence = disposition(row)
        lines.append(
            "| " + " | ".join((
                "fixed graph first" if row["structure"] == "explicit" else "graph posterior later",
                row["query"], row["graph_class"], row["structure"], row["inference"],
                row.get("validation", "none"), ", ".join(row.get("estimators", [])),
                f"`{reason}`", next_evidence,
            )) + " |"
        )
    lines += [
        "",
        "No row above is relabeled in `support_licensed.toml`. The graph-posterior work starts only after a whole-method oracle and calibration design are frozen at each graph/query/provider coordinate; completion counts are never posterior probabilities, and unidentified or unevaluated mass cannot be renormalized away.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != content:
            parser.error("2_3_calibration_plan.md is stale")
    else:
        OUTPUT.write_text(content)
    print("2.3 calibration plan: 40 cells / 18 coordinates, original reasons retained")


if __name__ == "__main__":
    main()
