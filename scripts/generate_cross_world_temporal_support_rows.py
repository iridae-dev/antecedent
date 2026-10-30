#!/usr/bin/env python3
"""Generate the licensed rows of the temporal cross-world effects (counterfactual, interventional
distribution, nested natural direct effect, path-specific effect) on a TemporalDag, TemporalCpdag and
TemporalPag, supplied, accepted or as a graph posterior.

The rows are a deterministic function of the table below. They live between two marker lines of
parity/support_licensed.toml; `--check` fails when the file differs from the generated block and
`--write` replaces (or appends) the block. `--routes` prints the matching licensed_routes entries.

    python3 scripts/generate_cross_world_temporal_support_rows.py --check
    python3 scripts/generate_cross_world_temporal_support_rows.py --write

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSED = ROOT / "parity" / "support_licensed.toml"
BEGIN = "# BEGIN generated rows: scripts/generate_cross_world_temporal_support_rows.py"
END = "# END generated rows: scripts/generate_cross_world_temporal_support_rows.py"

FIXTURE = "conformance/estimate/temporal_cross_world"
TEST = "crates/antecedent/tests/cross_world_temporal_truth.rs"
EVIDENCE = "temporal_cross_world_cells_match_the_linear_mechanism_under_every_suite"
CHECKED = "temporal_cross_world_cells_execute_from_the_retained_plan_without_the_builder"
QUERIES = ["Counterfactual", "InterventionalDistribution", "NestedCounterfactualEffect", "PathSpecificEffect"]
CLASSES = ["TemporalDag", "TemporalCpdag", "TemporalPag"]
ESTIMATOR = "temporal.linear.adjustment"


def validations(graph: str, structure: str) -> list[str]:
    if structure == "graph_posterior" and graph == "TemporalDag":
        return ["none", "cheap", "full"]
    return ["cheap", "full"]


def route(graph: str, structure: str) -> tuple[str, str]:
    identifier = "temporal.backdoor.unfolded" if graph == "TemporalDag" else "generalized.adjustment"
    lowering = "GraphPosteriorCrossWorld" if structure == "graph_posterior" else (
        "TemporalCrossWorldDag" if graph == "TemporalDag" else "TemporalCrossWorldClass"
    )
    return identifier, lowering


LIMITATIONS = (
    "Linear unfolded point of a temporal cross-world effect on a {graph} {structure_text} with one "
    "lag: the nested direct effect, the path-specific effect through the mediator, the total effect "
    "of a counterfactual and the interventional-mean contrast, read from the temporal linear "
    "adjustment slopes under the back-door set of the declared graph{class_text}. Known truth: the "
    "closed forms of the one-lag linear mechanism M = 0.5 X(t-1) + e_m, Y = 1.5 X(t-1) + 4 M + e_y "
    "(conformance/estimate/temporal_cross_world): a nested direct effect of 1.5, a mediated path "
    "effect of 2 and a total effect and interventional-mean contrast of 3.5, on seeded Gaussian "
    "disturbances. Validation runs the functional refuter state on the series: the circular-shift "
    "placebo treatment and placebo outcome, circular-window subset stability and circular-block "
    "bootstrap stability, whose verdicts hold the planted-violation and clean-twin tests; the "
    "pre-trend, anticipation and lagged negative-control refuters reject the clean mechanism at "
    "no more than their size across seeded series and flag a planted anticipation (the outcome and "
    "mediator read the treatment of the same period) on every seed. The sampling weights of a "
    "resampled series are read by the slope fits. The verdicts are diagnostics, not a calibration "
    "claim. A mapped coefficient prior and a nonempty back-door set refuse. Interval: a shared "
    "circular-block bootstrap of series times when replicates are requested (Frequentist) or the "
    "slope posterior draws (Bayesian); no coverage record measures them."
)


def quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def cells() -> list[tuple[str, str, str, str, str]]:
    out = []
    for graph in CLASSES:
        for structure in ("explicit", "accepted", "graph_posterior"):
            for query in QUERIES:
                for inference in ("Frequentist", "Bayesian"):
                    for validation in validations(graph, structure):
                        out.append((query, graph, structure, inference, validation))
    return out


def rows() -> list[str]:
    out = []
    for query, graph, structure, inference, validation in cells():
        estimator = ("bayesian.temporal.gcomp" if query == "Counterfactual" and graph == "TemporalDag"
                     and structure in ("explicit", "accepted") and inference == "Bayesian"
                     else ESTIMATOR)
        structure_text = {
            "explicit": "supplied",
            "accepted": "accepted",
            "graph_posterior": "graph posterior (one atom per sampled template, mixed by posterior probability)",
        }[structure]
        class_text = (
            "; a class graph runs on every completion and publishes the completion-weighted mean"
            if graph != "TemporalDag"
            else ""
        )
        text = LIMITATIONS.format(graph=graph, structure_text=structure_text, class_text=class_text)
        out.append(
            "\n".join(
                [
                    "[[cell]]",
                    f'query = "{query}"',
                    f'graph_class = "{graph}"',
                    f'structure = "{structure}"',
                    f'inference = "{inference}"',
                    f'validation = "{validation}"',
                    f'estimators = ["{estimator}"]',
                    "staged = true",
                    'evidence_kind = "internal_known_truth"',
                    f'known_truth_fixture = "{FIXTURE}"',
                    f'evidence_test = "{TEST}"',
                    f'evidence_assertion = "{EVIDENCE}"',
                    "checked_execution = [",
                    f'  {{ estimator = "{estimator}", test = "{TEST}", assertion = "{CHECKED}" }},',
                    "]",
                    f"limitations = {quote(text)}",
                    'calibration_reason = "estimator_grid_not_measured"',
                ]
            )
        )
    return out


def route_entries() -> str:
    out = []
    for query, graph, structure, inference, validation in cells():
        estimator = ("bayesian.temporal.gcomp" if query == "Counterfactual" and graph == "TemporalDag"
                     and structure in ("explicit", "accepted") and inference == "Bayesian"
                     else ESTIMATOR)
        identifier, lowering = route(graph, structure)
        out.append(
            f'\n[[route]]\ncoordinate = "{query}:{graph}:{structure}:{inference}:{validation}"\n'
            f'identifier = "{identifier}"\nestimator = "{estimator}"\nlowering = "{lowering}"\n'
        )
    return "".join(out)


def block() -> str:
    return BEGIN + "\n\n" + "\n\n".join(rows()) + "\n\n" + END + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in {"--check", "--write", "--routes"}:
        print(__doc__)
        return 2
    if argv[0] == "--routes":
        print(route_entries(), end="")
        return 0
    text = LICENSED.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        new = head + block().rstrip("\n") + tail
    else:
        new = text.rstrip("\n") + "\n\n" + block()
    if argv[0] == "--check":
        if new != text:
            print("parity/support_licensed.toml differs from the generated temporal cross-world rows")
            return 1
        print("ok")
        return 0
    LICENSED.write_text(new, encoding="utf-8")
    print(f"wrote {len(rows())} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
