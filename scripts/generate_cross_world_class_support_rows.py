#!/usr/bin/env python3
"""Generate the licensed rows of the counterfactual and nested-effect cells on mixed and class graphs.

The rows are a deterministic function of the table below. They live between two marker lines of
parity/support_licensed.toml; `--check` fails when the file differs from the generated block and
`--write` replaces (or appends) the block.

    python3 scripts/generate_cross_world_class_support_rows.py --check
    python3 scripts/generate_cross_world_class_support_rows.py --write

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSED = ROOT / "parity" / "support_licensed.toml"
BEGIN = "# BEGIN generated rows: scripts/generate_cross_world_class_support_rows.py"
END = "# END generated rows: scripts/generate_cross_world_class_support_rows.py"

FIXTURE = "conformance/estimate/cross_world_class_truth"
TEST = "crates/antecedent/tests/cross_world_class_truth.rs"
EVIDENCE = "class_cross_world_effects_match_the_exact_functionals_of_the_count_tables"
CHECKED = "class_cross_world_effects_execute_from_the_retained_plan_without_the_builder"
EVIDENCE_VALIDATED = "class_cross_world_cheap_and_full_validation_refutes_the_exact_functionals"
CHECKED_VALIDATED = "class_cross_world_validation_executes_from_the_retained_plan_without_the_builder"

# (query, graph class, estimator, validations licensed)
CELLS = [
    ("Counterfactual", "Admg", "functional.effect", ["none", "cheap", "full"]),
    ("Counterfactual", "TemporalAdmg", "functional.effect", ["none", "cheap", "full"]),
    ("Counterfactual", "Pag", "functional.effect", ["none"]),
    ("Counterfactual", "Cpdag", "gcm.fit", ["none", "cheap", "full"]),
    ("NestedCounterfactualEffect", "Admg", "functional.effect", ["none", "cheap", "full"]),
    ("NestedCounterfactualEffect", "TemporalAdmg", "functional.effect", ["none", "cheap", "full"]),
    ("NestedCounterfactualEffect", "Pag", "functional.effect", ["none", "cheap", "full"]),
    ("NestedCounterfactualEffect", "Cpdag", "functional.effect", ["none", "cheap", "full"]),
]

POSTERIOR_TEST = "crates/antecedent/tests/cross_world_posterior_truth.rs"
POSTERIOR_EVIDENCE = "nested_graph_posterior_is_the_weighted_mean_of_the_atoms_closed_forms_under_every_suite"
POSTERIOR_CHECKED = "nested_graph_posterior_executes_from_the_retained_plan_without_the_builder"
POSTERIOR_GRAPHS = ["Dag", "Admg", "TemporalAdmg", "Cpdag", "Pag"]
POSTERIOR_FIXTURE: dict[str, str] = {}
POSTERIOR_LIMITATIONS = (
    "Natural direct effect over a graph posterior of {kind} atoms: the path-specific effect along the "
    "edges that avoid the mediator, answered on every atom by the executor of the atom's kind and mixed "
    "by posterior probability over the atoms that identify it. An atom the identifier refuses is "
    "unidentified posterior mass and is not redistributed; the status resolves to graph-dependent. Known "
    "truth: the posterior-weighted mean of each atom's own closed form, evaluated in rational arithmetic "
    "on an enumerated count table ({fixture}); the refused atom keeps 0.2 of the mass. Cheap and full "
    "validation run each atom's own refuter suite against that atom's own estimate and mix the reports "
    "by posterior weight; a mixed verdict passes only if every contributing atom passes. The verdicts are "
    "diagnostics, not a calibration claim. Frequentist: point only. Bayesian: the draws of the "
    "contributing atoms pooled draw by draw, whose interval is not calibrated."
)

LIMITATIONS = {
    "Counterfactual": {
        "Admg": (
            "Population-level cross-world mean E[Y_x | X = x'] - E[Y | X = x'] on a supplied or "
            "accepted ADMG, decided by ID* from the observational joint over finite-discrete "
            "variables and evaluated on the empirical cell table; latent confounding breaks "
            "unit-level noise abduction, so this is not a per-unit counterfactual. Known truth: the "
            "front-door effect on the treated, evaluated in rational arithmetic on an enumerated "
            "binary count table (conformance/estimate/cross_world_class_truth). An unidentified "
            "query (a bow arc) is refused with the typed ID* witness, which is not a proof of "
            "non-identifiability."
        ),
        "TemporalAdmg": (
            "The same population-level cross-world mean read on the contemporaneous projection of a "
            "temporal ADMG, with the same enumerated front-door count table as known truth "
            "(conformance/estimate/cross_world_class_truth)."
        ),
        "Pag": (
            "The population-level cross-world mean read through every maximal-ancestral-graph "
            "completion of a PAG and every ADMG each stands for; here the candidates agree and one "
            "functional is published. Known truth: the unconfounded contrast on an enumerated "
            "count table (conformance/estimate/cross_world_class_truth). A candidate that leaves "
            "the query unidentified refuses it."
        ),
        "Cpdag": (
            "A CPDAG assumes causal sufficiency, so the per-unit counterfactual route runs on every "
            "DAG completion. Known truth: the exact slope of a noiseless line, 2 for x -> y and 0 for "
            "y -> x, so the completions disagree and the enumeration-weighted mean is 1 with the "
            "identified set [0, 2] (conformance/estimate/cross_world_class_truth)."
        ),
    },
    "NestedCounterfactualEffect": {
        "Admg": (
            "Natural direct effect as the path-specific effect along the edges that avoid the "
            "mediator on a supplied or accepted ADMG, identified by the path-specific identifier "
            "(the edge g-formula without latent confounding on the outcome's ancestors, Shpitser's "
            "district formula with it) and evaluated by the finite-discrete plug-in. Known truth: "
            "the mediation formula on an enumerated binary count table, evaluated in rational "
            "arithmetic (conformance/estimate/cross_world_class_truth). A mediator-outcome latent "
            "confounder leaves the effect unidentified and it is refused."
        ),
        "TemporalAdmg": (
            "The same path-specific natural direct effect read on the contemporaneous projection of "
            "a temporal ADMG, with the same enumerated count table as known truth "
            "(conformance/estimate/cross_world_class_truth)."
        ),
        "Pag": (
            "The natural direct effect read through the ADMGs that share the m-separation model of "
            "each MAG completion and reported only when every one is identified; here one functional "
            "is published. Known truth: the mediation formula on an enumerated count table "
            "(conformance/estimate/cross_world_class_truth). A PAG with an invisible edge that leaves "
            "a candidate unidentified refuses the effect."
        ),
        "Cpdag": (
            "The natural direct effect as the path-specific effect along the edges that avoid the "
            "mediator, identified in every DAG completion of the equivalence class; here the class is "
            "one DAG. Known truth: the edge g-formula of a two-mediator model on an enumerated count "
            "table (conformance/estimate/cross_world_class_truth)."
        ),
    },
}

VALIDATION = (
    " Cheap and full validation run the functional refuter state: the placebo treatment, the placebo "
    "outcome and bootstrap stability under full, and, on a supplied or accepted ADMG, the cross-world "
    "sensitivity range, which passes the front-door table and flags a planted weak effect as "
    "sign-fragile; on the other classes the range is reported as a refuter that did not run, with the "
    "reason. The verdicts are diagnostics, not a calibration claim."
)

TAIL = (
    " Frequentist: point only, no interval is reported. Bayesian: the Dirichlet-weighted "
    "Bayesian-bootstrap posterior of the identified functional, whose interval is not calibrated. "
    "Accepted structures execute as supplied ones; acceptance does not establish the cross-world "
    "assumptions."
)


def quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def rows() -> list[str]:
    out: list[str] = []
    for query, graph, estimator, validations in CELLS:
        for structure in ("explicit", "accepted"):
            for inference in ("Frequentist", "Bayesian"):
                for validation in validations:
                    evidence = EVIDENCE if validation == "none" else EVIDENCE_VALIDATED
                    checked = CHECKED if validation == "none" else CHECKED_VALIDATED
                    limitations = LIMITATIONS[query][graph] + TAIL
                    if validation != "none":
                        limitations += VALIDATION
                    reason = "no_interval_reported" if inference == "Frequentist" else "estimator_grid_not_measured"
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
                                f'evidence_assertion = "{evidence}"',
                                "checked_execution = [",
                                f'  {{ estimator = "{estimator}", test = "{TEST}", assertion = "{checked}" }},',
                                "]",
                                f"limitations = {quote(limitations)}",
                                f'calibration_reason = "{reason}"',
                            ]
                        )
                    )
    for graph in POSTERIOR_GRAPHS:
        fixture = POSTERIOR_FIXTURE.get(graph, FIXTURE)
        for inference in ("Frequentist", "Bayesian"):
            for validation in ("none", "cheap") if graph == "Dag" else ("none", "cheap", "full"):
                reason = "no_interval_reported" if inference == "Frequentist" else "estimator_grid_not_measured"
                out.append(
                    "\n".join(
                        [
                            "[[cell]]",
                            'query = "NestedCounterfactualEffect"',
                            f'graph_class = "{graph}"',
                            'structure = "graph_posterior"',
                            f'inference = "{inference}"',
                            f'validation = "{validation}"',
                            'estimators = ["functional.effect"]',
                            "staged = true",
                            'evidence_kind = "internal_known_truth"',
                            f'known_truth_fixture = "{fixture}"',
                            f'evidence_test = "{POSTERIOR_TEST}"',
                            f'evidence_assertion = "{POSTERIOR_EVIDENCE}"',
                            "checked_execution = [",
                            f'  {{ estimator = "functional.effect", test = "{POSTERIOR_TEST}", assertion = "{POSTERIOR_CHECKED}" }},',
                            "]",
                            f"limitations = {quote(POSTERIOR_LIMITATIONS.format(kind=graph, fixture=fixture))}",
                            f'calibration_reason = "{reason}"',
                        ]
                    )
                )
    return out


def block() -> str:
    return BEGIN + "\n\n" + "\n\n".join(rows()) + "\n\n" + END + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in {"--check", "--write"}:
        print(__doc__)
        return 2
    text = LICENSED.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        new = head + block().rstrip("\n") + tail
    else:
        new = text.rstrip("\n") + "\n\n" + block()
    if argv[0] == "--check":
        if new != text:
            print("parity/support_licensed.toml differs from the generated cross-world class rows")
            return 1
        print("ok")
        return 0
    LICENSED.write_text(new, encoding="utf-8")
    print(f"wrote {len(rows())} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
