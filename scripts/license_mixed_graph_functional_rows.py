#!/usr/bin/env python3
"""Write the licensed rows of the functional cells on supplied mixed graphs and classes.

Three blocks: the path-specific effect on a supplied ADMG, TemporalAdmg, CPDAG and PAG
(`path_specific_mixed`), the interventional distribution on a supplied CPDAG and PAG
(`distribution_classes`), and the mediation decomposition on a supplied ADMG, TemporalAdmg, CPDAG
and PAG (`mediation_mixed`). The rows of a block share one fixture and one evidence test file; only
the graph class, structure source and inference mode differ. Each block sits between marker
comments at the end of parity/support_licensed.toml and is rewritten in place, so the script is
idempotent:

    python3 scripts/license_mixed_graph_functional_rows.py          # rewrite the blocks
    python3 scripts/license_mixed_graph_functional_rows.py --check  # fail if a block is stale

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSED = ROOT / "parity" / "support_licensed.toml"
SELF = "scripts/license_mixed_graph_functional_rows.py"

STRUCTURES = ("explicit", "accepted")
INFERENCES = ("Frequentist", "Bayesian")

PATH_FIXTURE = "conformance/estimate/path_specific_mixed_graphs"
PATH_TEST = "crates/antecedent/tests/path_specific_mixed_graph_route_evidence.rs"

# graph class -> (test stem, what the graph is, the identification the fixture case exercises)
PATH_CLASSES = {
    "Admg": (
        "admg",
        "a supplied ADMG",
        "The fixture case is T -> A, T -> B, A -> Y, B -> Y with a latent common cause of A and Y "
        "(A <-> Y): the selected child shares a district with the outcome, so the path-specific "
        "identifier reads the district formula, not the total-effect or adjustment functional "
        "(truth 0.125 against a total effect of 0.25).",
    ),
    "TemporalAdmg": (
        "temporal_admg",
        "the contemporaneous projection of a supplied TemporalAdmg",
        "The projection keeps the static ADMG functional, so the cell reads the same fixture case "
        "as the static ADMG (truth 0.125 against a total effect of 0.25).",
    ),
    "Cpdag": (
        "cpdag",
        "a supplied CPDAG",
        "The fixture case is the Markovian graph with every edge compelled, so the class has one "
        "completion and the effect is that completion's edge g-formula (truth 0.15625 against a "
        "total effect of 0.375). A class whose completions disagree publishes the enumeration-"
        "weighted mean with the identified set over the identified completions, and a completion "
        "with no directed path through the selected child is unidentified mass; the fixture's "
        "collider class pins the set [0.15625, 0.1875] with one third unidentified mass.",
    ),
    "Pag": (
        "pag",
        "a supplied PAG",
        "The fixture case is the Markovian graph with every mark decided and every edge out of the "
        "treatment and into the outcome visible, so every ADMG sharing the m-separation model of "
        "its MAG completion yields one functional and the effect is a point (truth 0.15625 against "
        "a total effect of 0.375). A PAG whose candidates disagree publishes the identified set and "
        "a single refused candidate refuses the effect.",
    ),
}

DIST_FIXTURE = "conformance/estimate/interventional_distribution_classes"
DIST_TEST = "crates/antecedent/tests/interventional_distribution_class_route_evidence.rs"

DIST_CLASSES = {
    "Cpdag": (
        "cpdag",
        "a supplied or accepted CPDAG",
        "The fixture case is the Markovian graph with every edge compelled, so the class has one "
        "completion and the distribution is a point (P(Y=1 | do(T=1)) = 0.6875). A class whose "
        "completions disagree leaves the distribution partially identified and publishes, atom by "
        "atom, the lowest and highest probability over the identified completions with the "
        "enumeration-weighted mean beside them (estimate.distribution.class_envelope; the "
        "fixture's two-completion class pins [0.625, 0.6875] for the atom Y=1 and mean 0.65625), "
        "never a probability over completions; a completion the algorithm refuses is "
        "unidentified mass.",
    ),
    "Pag": (
        "pag",
        "a supplied or accepted PAG",
        "The fixture case is the Markovian graph with every mark decided and visible, so every "
        "ADMG sharing the m-separation model of its MAG completion yields one functional and the "
        "distribution is a point (P(Y=1 | do(T=1)) = 0.6875). A PAG whose candidates disagree "
        "publishes the per-atom envelope, and one refused candidate refuses the distribution, "
        "since no functional is then correct for every model the PAG represents.",
    ),
}


MED_FIXTURE = "conformance/estimate/mediation_mixed_graphs"
MED_TEST = "crates/antecedent/tests/mediation_mixed_graph_route_evidence.rs"

MED_CLASSES = {
    "Admg": (
        "admg",
        "a supplied or accepted ADMG",
        "The fixture case is a covariate W with a latent common cause of W and the outcome "
        "(W <-> Y), W -> T, W -> M, T -> M, T -> Y, M -> Y: total 0.15625, natural direct "
        "0.15625, pure natural indirect 0.0625. A frequentist bootstrap replicate evaluates the "
        "three components at one resampled table and estimate.mediation.joint discloses the "
        "paired draws (means checked against the enumerated components).",
    ),
    "TemporalAdmg": (
        "temporal_admg",
        "the contemporaneous projection of a supplied or accepted TemporalAdmg",
        "The projection keeps the static ADMG decomposition, so the cell reads the same fixture "
        "case as the static ADMG (total 0.15625, natural direct 0.15625, pure natural indirect "
        "0.0625) and discloses the same paired joint.",
    ),
    "Cpdag": (
        "cpdag",
        "a supplied or accepted CPDAG",
        "The fixture case is the Markovian graph C -> T, W -> M, T -> M, T -> Y, M -> Y with every "
        "edge compelled, so the class has one completion (total 0.25, natural direct 0.15625, "
        "pure natural indirect 0.1875). Each component is the class result of the path-specific "
        "executor; the frequentist components are resampled separately and no joint is "
        "disclosed, while the Bayesian-bootstrap draws are paired and estimate.mediation.joint "
        "discloses them.",
    ),
    "Pag": (
        "pag",
        "a supplied or accepted PAG",
        "The fixture case is the same Markovian graph with every mark decided and visible, so "
        "every ADMG sharing the m-separation model of its MAG completion yields one functional "
        "per component (total 0.25, natural direct 0.15625, pure natural indirect 0.1875). Each "
        "component is the class result of the path-specific executor; the Bayesian cell publishes the "
        "components' posterior means without draws, and no estimate.mediation.joint is disclosed "
        "over a PAG.",
    ),
}


def med_limitations(klass: str, inference: str) -> str:
    _, what, case = MED_CLASSES[klass]
    return (
        "Mediation decomposition of one treatment on one outcome through the declared mediators, "
        f"on {what}, validation none, unconditional finite-discrete table: the total, natural "
        "direct and pure natural indirect effect E[Y(a0, M(a1))] - E[Y(a0)], each identified by "
        f"the path-specific identifier and evaluated by the finite-discrete plug-in. {case} "
        f"{_inference_text(inference)} The components equal the exact enumerated truths of the "
        "generating structural model on the retained exact-law table (conformance/estimate/"
        "mediation_mixed_graphs; the independent generator evaluates the outcome equation under "
        "each treatment and mediator-treatment argument, not a mediation formula); the pure "
        "indirect effect is never total minus direct, and a mediator-outcome confounder the graph "
        "states is refused as not identified. Cheap and full validation (the mediation-native "
        "suite), graph-posterior structures and conditional queries stay refused. No sampling "
        "interval is claimed to cover; no coverage record measures this construction."
    )


def path_limitations(klass: str, inference: str) -> str:
    _, what, case = PATH_CLASSES[klass]
    return (
        "Natural path-restricted effect of one treatment on one outcome through a selected set of "
        f"mediator nodes, on {what}, validation none, unconditional finite-discrete table. {case} "
        f"{_inference_text(inference)} The effect equals the exact enumerated cross-world truth "
        "of the generating structural model on the retained exact-law table (conformance/"
        "estimate/path_specific_mixed_graphs; the independent generator evaluates the structural "
        "equations under the cross-world assignment, not an identification formula), and a latent "
        "common cause of the selected and the complementary child is refused as a recanting "
        "district. Cheap and full validation, graph-posterior structures and conditional queries "
        "stay refused. No sampling interval is claimed to cover; no coverage record measures this "
        "construction."
    )


def dist_limitations(klass: str, inference: str) -> str:
    _, what, case = DIST_CLASSES[klass]
    return (
        f"Unconditional finite-discrete interventional distribution on {what}, validation none. "
        f"{case} {_inference_text(inference)} The probabilities equal the exact enumerated "
        "interventional law of the generating structural model on the retained exact-law table "
        "(conformance/estimate/interventional_distribution_classes; the independent generator "
        "evaluates the outcome equation under the intervention, not an identification formula). "
        "Cheap and full validation, conditional queries (IDC) and graph-posterior structures stay "
        "refused. No sampling interval is claimed to cover; no coverage record measures this "
        "construction."
    )


def _inference_text(inference: str) -> str:
    if inference == "Frequentist":
        return (
            "Frequentist: the identified functional is evaluated by the finite-discrete plug-in, "
            "with a bootstrap of the rows for uncertainty when replicates are requested."
        )
    return (
        "Bayesian: the posterior is the Dirichlet row posterior of the same functional; the "
        "posterior mean is checked within the fixture's declared band."
    )


def block_rows(
    query: str,
    classes: dict,
    fixture: str,
    test_file: str,
    estimator: str,
    stem: str,
    limitations,
) -> str:
    out: list[str] = []
    for klass in classes:
        short = classes[klass][0]
        for structure in STRUCTURES:
            for inference in INFERENCES:
                test = f"{stem}_{short}_{structure}_{inference.lower()}_none"
                out.append(
                    "[[cell]]\n"
                    f'query = "{query}"\n'
                    f'graph_class = "{klass}"\n'
                    f'structure = "{structure}"\n'
                    f'inference = "{inference}"\n'
                    'validation = "none"\n'
                    f'estimators = ["{estimator}"]\n'
                    "staged = true\n"
                    'evidence_kind = "internal_known_truth"\n'
                    f'known_truth_fixture = "{fixture}"\n'
                    f'evidence_test = "{test_file}"\n'
                    f'evidence_assertion = "{test}"\n'
                    "checked_execution = [\n"
                    f'  {{ estimator = "{estimator}", test = "{test_file}", assertion = "{test}" }},\n'
                    "]\n"
                    f'limitations = "{limitations(klass, inference)}"\n'
                    'calibration_reason = "estimator_grid_not_measured"\n'
                )
    return "\n".join(out)


BLOCKS = {
    "mediation_mixed": lambda: block_rows(
        "MediationEffect",
        MED_CLASSES,
        MED_FIXTURE,
        MED_TEST,
        "functional.effect",
        "mediation",
        med_limitations,
    ),
    "path_specific_mixed": lambda: block_rows(
        "PathSpecificEffect",
        PATH_CLASSES,
        PATH_FIXTURE,
        PATH_TEST,
        "functional.effect",
        "path_specific",
        path_limitations,
    ),
    "distribution_classes": lambda: block_rows(
        "InterventionalDistribution",
        DIST_CLASSES,
        DIST_FIXTURE,
        DIST_TEST,
        "functional.distribution",
        "distribution",
        dist_limitations,
    ),
}


def markers(name: str) -> tuple[str, str]:
    return (
        f"# generated:{name}:begin ({SELF})\n",
        f"# generated:{name}:end\n",
    )


def apply(text: str) -> str:
    for name, rows in BLOCKS.items():
        begin, end = markers(name)
        fresh = begin + "\n" + rows() + end
        if begin in text:
            head, rest = text.split(begin, 1)
            _, tail = rest.split(end, 1)
            text = head + fresh + tail
        else:
            text = text.rstrip("\n") + "\n\n" + fresh
    return text


def main() -> int:
    text = LICENSED.read_text(encoding="utf-8")
    # Blocks written under an earlier script name are adopted in place.
    text = text.replace(
        "# generated:path_specific_mixed:begin (scripts/license_path_specific_mixed.py)\n",
        markers("path_specific_mixed")[0],
    )
    updated = apply(text)
    if "--check" in sys.argv:
        if updated != LICENSED.read_text(encoding="utf-8"):
            print("parity/support_licensed.toml: a mixed-graph functional block is stale")
            return 1
        return 0
    if updated != LICENSED.read_text(encoding="utf-8"):
        LICENSED.write_text(updated, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
