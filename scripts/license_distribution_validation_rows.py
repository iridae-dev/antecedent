#!/usr/bin/env python3
"""Write the licensed rows of cheap and full validation of an interventional distribution.

The cells are the interventional distribution on a supplied or accepted ADMG, TemporalAdmg,
CPDAG and PAG under cheap and full validation, for both inference modes. The mixed graphs read the
front-door fixture and the classes the class fixture; one evidence test file holds a test per
cell. The block sits between marker comments at the end of parity/support_licensed.toml and is
rewritten in place, so the script is idempotent:

    python3 scripts/license_distribution_validation_rows.py          # rewrite the block
    python3 scripts/license_distribution_validation_rows.py --check  # fail if the block is stale

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSED = ROOT / "parity" / "support_licensed.toml"
SELF = "scripts/license_distribution_validation_rows.py"
TEST = "crates/antecedent/tests/distribution_validation_route_evidence.rs"
NAME = "distribution_validation"

STRUCTURES = ("explicit", "accepted")
INFERENCES = ("Frequentist", "Bayesian")
VALIDATIONS = ("cheap", "full")

FRONT_DOOR = "conformance/estimate/admg_frontdoor_distribution"
CLASSES = "conformance/estimate/interventional_distribution_classes"

# graph class -> (test stem, fixture, what the graph is, what the fixture case exercises)
GRAPHS = {
    "Admg": (
        "admg",
        FRONT_DOOR,
        "a supplied or accepted ADMG",
        "The fixture case is the front-door graph T -> M -> Y with a latent common cause of T and "
        "Y (T <-> Y): P(Y=1 | do(T=1)) = 0.625, from the SCM, not from the front-door formula.",
    ),
    "TemporalAdmg": (
        "temporal_admg",
        FRONT_DOOR,
        "the contemporaneous projection of a supplied or accepted TemporalAdmg",
        "The projection keeps the static ADMG functional, so the cell reads the same front-door "
        "case (P(Y=1 | do(T=1)) = 0.625).",
    ),
    "Cpdag": (
        "cpdag",
        CLASSES,
        "a supplied or accepted CPDAG",
        "The fixture case is the Markovian graph with every edge compelled, so the class has one "
        "completion and the distribution is a point (P(Y=1 | do(T=1)) = 0.6875).",
    ),
    "Pag": (
        "pag",
        CLASSES,
        "a supplied or accepted PAG",
        "The fixture case is the Markovian graph with every mark decided and visible, so every "
        "ADMG sharing the m-separation model of its MAG completion yields one functional and the "
        "distribution is a point (P(Y=1 | do(T=1)) = 0.6875).",
    ),
}


def inference_text(inference: str) -> str:
    if inference == "Frequentist":
        return "Frequentist: the identified functional is evaluated by the finite-discrete plug-in. "
    return (
        "Bayesian: the posterior is the Dirichlet row posterior of the same functional and its "
        "mean is checked within a 0.02 band. "
    )


def validation_text(validation: str) -> str:
    suite = (
        "the distribution placebo and subset stability"
        if validation == "cheap"
        else "the distribution placebo, subset stability and bootstrap stability"
    )
    return (
        f"Validation {validation} refits the compiled program on perturbed copies of the table "
        f"({suite}; the Bayesian cell runs the prior sensitivity in place of the stability "
        "refuters), each a joint statistic over the vector of atom probabilities. The atoms are "
        "levels of a law, so the treatment-keyed null refuters do not apply. On the exact-law "
        "table scaled to several times the rows the distribution placebo gives the observed marginal "
        "share of the outcome counted from the rows, and every stability refit stays at the "
        "enumerated law, each within 0.02 with every verdict passing; the verdicts are checks, "
        "not a calibration claim. A resample that empties a cell the functional reads has that "
        "refuter refused with its reason, and a high-leverage twelve-row table is flagged or "
        "named in that diagnostic. "
    )


def limitations(klass: str, inference: str, validation: str) -> str:
    _, _, what, case = GRAPHS[klass]
    return (
        f"Unconditional finite-discrete interventional distribution on {what}, validation "
        f"{validation}. {case} {inference_text(inference)}{validation_text(validation)}"
        "Conditional queries (IDC) and graph-posterior structures stay refused. No sampling "
        "interval is claimed to cover; no coverage record measures this construction."
    )


def rows() -> str:
    out: list[str] = []
    for klass, (short, fixture, _, _) in GRAPHS.items():
        for structure in STRUCTURES:
            for inference in INFERENCES:
                for validation in VALIDATIONS:
                    test = f"distribution_{short}_{structure}_{inference.lower()}_{validation}"
                    out.append(
                        "[[cell]]\n"
                        'query = "InterventionalDistribution"\n'
                        f'graph_class = "{klass}"\n'
                        f'structure = "{structure}"\n'
                        f'inference = "{inference}"\n'
                        f'validation = "{validation}"\n'
                        'estimators = ["functional.distribution"]\n'
                        "staged = true\n"
                        'evidence_kind = "internal_known_truth"\n'
                        f'known_truth_fixture = "{fixture}"\n'
                        f'evidence_test = "{TEST}"\n'
                        f'evidence_assertion = "{test}"\n'
                        "checked_execution = [\n"
                        f'  {{ estimator = "functional.distribution", test = "{TEST}", '
                        f'assertion = "{test}" }},\n'
                        "]\n"
                        f'limitations = "{limitations(klass, inference, validation)}"\n'
                        'calibration_reason = "estimator_grid_not_measured"\n'
                    )
    return "\n".join(out)


def apply(text: str) -> str:
    begin = f"# generated:{NAME}:begin ({SELF})\n"
    end = f"# generated:{NAME}:end\n"
    fresh = begin + "\n" + rows() + end
    if begin in text:
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        return head + fresh + tail
    return text.rstrip("\n") + "\n\n" + fresh


def main() -> int:
    current = LICENSED.read_text(encoding="utf-8")
    updated = apply(current)
    if "--check" in sys.argv:
        if updated != current:
            print("parity/support_licensed.toml: the distribution validation block is stale")
            return 1
        return 0
    if updated != current:
        LICENSED.write_text(updated, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
