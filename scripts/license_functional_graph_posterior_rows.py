#!/usr/bin/env python3
"""Write the licensed rows of the functional cells over a graph posterior.

Three families: the path-restricted natural effect, the interventional distribution and the
mediation decomposition, each over a graph posterior of Dag, ADMG, temporal-ADMG, CPDAG and PAG
atoms, on both inference sides. The rows of a family share one fixture and one evidence test
file; only the graph class, inference mode and validation differ. The block sits between marker
comments at the end of parity/support_licensed.toml and is rewritten in place, so the script is
idempotent:

    python3 scripts/license_functional_graph_posterior_rows.py          # rewrite the block
    python3 scripts/license_functional_graph_posterior_rows.py --check  # fail if it is stale

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSED = ROOT / "parity" / "support_licensed.toml"
SELF = "scripts/license_functional_graph_posterior_rows.py"

FIXTURE = "conformance/estimate/functional_graph_posterior_mixtures"
TEST = "crates/antecedent/tests/functional_graph_posterior_mixture_truth.rs"

CLASSES = ("Dag", "Admg", "TemporalAdmg", "Cpdag", "Pag")
INFERENCES = ("Frequentist", "Bayesian")
VALIDATIONS = ("none", "cheap", "full")

# family -> (query, test stem, what the query is, the fixture's mixtures)
FAMILIES = {
    "PathSpecificEffect": (
        "path_specific",
        "A path-restricted natural effect (the effect of the treatment on the outcome through one "
        "selected mediator node)",
        "The fixture's atoms are the Markovian DAG that generated the table (edge g-formula, "
        "0.15625), the DAG whose sibling parent of the treatment is adjusted (0.1875), a DAG with "
        "no directed path through the selected child and an ADMG whose latent common cause of the "
        "selected and the complementary child is a recanting district (both unidentified mass); a "
        "CPDAG atom whose completions disagree publishes the mean of its identified completions "
        "with its identified set; a PAG with every edge visible beside one whose treatment edges "
        "are invisible carries the whole conditional value.",
    ),
    "InterventionalDistribution": (
        "distribution",
        "An interventional distribution P(Y | do(T = 1)) of a binary outcome",
        "The fixture's atoms are the DAG whose treatment-mediator edge is forward (sum over the "
        "mediator of P(x | t) P(y | t, x), P(Y = 1) = 0.6875), the DAG whose mediator is a "
        "sibling parent of the treatment (adjusted, 0.625), an ADMG whose front-door structure "
        "reads the front-door functional, a CPDAG whose completions disagree (mean 0.65625 with "
        "the identified set) and a PAG with every edge visible beside one whose treatment edges "
        "are invisible (unidentified mass). The mixture is the posterior-weighted mean "
        "probability of each outcome level.",
    ),
    "MediationEffect": (
        "mediation",
        "A mediation decomposition into the total, natural direct and pure natural indirect "
        "effect",
        "The fixture's atoms are, on a table with a covariate that confounds treatment, mediator "
        "and outcome, the graph that adjusts the covariate (total 0.15625, natural direct "
        "0.15625, pure natural indirect 0.0625), the graph that omits it (unadjusted 0.2656, "
        "0.2503, 0.1091) and a graph in which a descendant of the treatment confounds mediator "
        "and outcome (unidentified mass); the Dag atoms read the linear table whose parent "
        "regressions are solved exactly (direct 1, indirect 6 against a confounder-omitted "
        "direct 1.5); a PAG with every edge visible beside one with invisible treatment edges "
        "carries the whole conditional value. Each component is mixed over the atoms that "
        "identify it, never total minus direct.",
    ),
}


def estimator(query: str, klass: str) -> str:
    if query == "InterventionalDistribution":
        return "functional.distribution"
    if query == "MediationEffect" and klass == "Dag":
        return "mediation.linear"
    return "functional.effect"


def limitations(query: str, klass: str, inference: str, validation: str) -> str:
    _, what, atoms = FAMILIES[query]
    inference_text = (
        "Frequentist: each atom is evaluated by its own plug-in, with the row bootstrap of the "
        "whole procedure for uncertainty when replicates are requested."
        if inference == "Frequentist"
        else "Bayesian: the atoms run in one context and their draws are pooled draw by draw; "
        "the posterior mean is checked within the fixture's declared band."
    )
    validation_text = (
        "Cheap and full validation stay refused on the none cell. "
        if validation == "none"
        else f"Validation {validation} runs each atom's own refuter suite against that atom's "
        "own estimate and mixes the reports of each refuter by posterior weight; a mixed verdict "
        "passes only if every contributing atom passes (checked against the verdicts the same "
        "refuters give when each atom runs alone), and a refuter that ran on only part of the "
        "mass says so (refute.envelope.partial_coverage). The permutation placebos are compared "
        "for presence only, because each context draws its own permutation. The verdicts are "
        "checks, not a calibration claim. "
    )
    return (
        f"{what} over a graph posterior of {klass} atoms, unconditional finite-discrete table "
        f"(a linear table for the Dag mediation atoms), validation {validation}. Every atom is "
        "answered by the executor the query has on a supplied structure of the atom's kind and "
        "the atoms mix by posterior probability: the value is the posterior-weighted mean over "
        "the atoms that identify and evaluate it; an atom the identifier refuses is unidentified "
        "posterior mass and is not redistributed, stated in the result, which resolves its "
        "status to graph-dependent. " + atoms + " " + inference_text + " The mixture equals the "
        "hand-weighted mean of each atom's identified functional evaluated in exact rational "
        "arithmetic from the table (conformance/estimate/functional_graph_posterior_mixtures; "
        "the count tables are those of the single-graph fixtures, whose generators enumerate the "
        "structural noise cells). " + validation_text + "The components of a mediation are mixed one by one and no estimate.mediation.joint is disclosed over a graph posterior. A graph posterior atom cannot hold a "
        "bow (a directed and a bidirected edge on one pair); such an atom is refused when the "
        "posterior is built. No sampling interval is claimed to cover; no coverage record "
        "measures this construction."
    )


def rows() -> str:
    out: list[str] = []
    for query, (stem, _, _) in FAMILIES.items():
        test = f"{stem}_graph_posterior_executes_from_the_retained_plan_without_the_builder"
        evidence = {
            "path_specific": "path_specific_graph_posterior_is_the_weighted_mean_of_the_atoms_identified_functionals_under_every_suite",
            "distribution": "interventional_distribution_graph_posterior_is_the_weighted_mean_of_the_atoms_identified_laws_under_every_suite",
            "mediation": "mediation_graph_posterior_is_the_weighted_mean_of_the_atoms_components_under_every_suite",
        }[stem]
        for klass in CLASSES:
            for inference in INFERENCES:
                for validation in VALIDATIONS:
                    est = estimator(query, klass)
                    out.append(
                        "[[cell]]\n"
                        f'query = "{query}"\n'
                        f'graph_class = "{klass}"\n'
                        'structure = "graph_posterior"\n'
                        f'inference = "{inference}"\n'
                        f'validation = "{validation}"\n'
                        f'estimators = ["{est}"]\n'
                        "staged = true\n"
                        'evidence_kind = "internal_known_truth"\n'
                        f'known_truth_fixture = "{FIXTURE}"\n'
                        f'evidence_test = "{TEST}"\n'
                        f'evidence_assertion = "{evidence}"\n'
                        "checked_execution = [\n"
                        f'  {{ estimator = "{est}", test = "{TEST}", assertion = "{test}" }},\n'
                        "]\n"
                        f'limitations = "{limitations(query, klass, inference, validation)}"\n'
                        'calibration_reason = "estimator_grid_not_measured"\n'
                    )
    return "\n".join(out)


def markers() -> tuple[str, str]:
    return (
        f"# generated:functional_graph_posterior:begin ({SELF})\n",
        "# generated:functional_graph_posterior:end\n",
    )


def apply(text: str) -> str:
    begin, end = markers()
    fresh = begin + "\n" + rows() + end
    if begin in text:
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        return head + fresh + tail
    return text.rstrip("\n") + "\n\n" + fresh


def main() -> int:
    text = LICENSED.read_text(encoding="utf-8")
    updated = apply(text)
    if "--check" in sys.argv:
        if updated != text:
            print("parity/support_licensed.toml: the functional graph-posterior block is stale")
            return 1
        return 0
    if updated != text:
        LICENSED.write_text(updated, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
