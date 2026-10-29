#!/usr/bin/env python3
"""Generate the licensed rows of the static-DAG derivative cheap and full validation cells.

Each family below names the cells it licenses (a cartesian selector over the support axes), the
fixture whose closed-form oracle is the truth, and the executing test that consumes the fixture,
builds every cell of the family and runs it from its retained checked operation. The rows are
written between the generated markers at the end of parity/support_licensed.toml; the script is
deterministic and `--check` fails when the file differs from the table.

    python3 scripts/generate_derivative_refuter_rows.py          # rewrite the generated block
    python3 scripts/generate_derivative_refuter_rows.py --check  # fail on drift

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSED = ROOT / "parity" / "support_licensed.toml"
BEGIN = "# generated:derivative_refuter_rows:begin (scripts/generate_derivative_refuter_rows.py)"
END = "# generated:derivative_refuter_rows:end"

STATIC = {
    "fixture": "conformance/response/derivative_refuter_states",
    "test": "crates/antecedent/tests/derivative_refuter_known_truth.rs",
    "assertion": "derivative_refuter_states_known_truth",
}
TEMPORAL = {
    "fixture": "conformance/estimate/temporal_derivative_refuter_states",
    "test": "crates/antecedent/tests/temporal_derivative_refuter_known_truth.rs",
    "assertion": "temporal_derivative_refuter_states_known_truth",
}

STATIC_TEXT = (
    "Cheap and full validation of a {what} on a supplied or accepted Dag, executed through "
    "{estimator}. The refuter state checks the derivative itself: support at the evaluation point "
    "(an evaluation point beyond the observed treatment range fails the check){difference}, and "
    "under full a whole-procedure row bootstrap of the derivative under independent subjects whose "
    "95% range must contain the published value (failed replicates are kept). {bayesian}Known "
    "truth: the linear law of the fixture, whose derivative is its coefficient; the law is linear, "
    "so the fixture cannot detect curvature errors{own}. ATE-shaped contrast refuters do not "
    "apply. The verdicts are checks of this route, not a calibration claim; no coverage record "
    "measures the intervals and calibration is not claimed."
)
STATIC_BAYES = (
    "Bayesian studies add the prior and posterior predictive checks of the outcome regression "
    "(a residual law that is not Gaussian fails the posterior predictive check; the prior "
    "predictive check may report that the default prior is diffuse relative to the data, a "
    "statement about the prior) and, under full, prior sensitivity refitting the whole "
    "derivative at each grid point. "
)
DIFFERENCE = (
    ", the central finite difference of the identified intervention levels against the published "
    "derivative on its reported scale (within a declared relative band of the closed-form value "
    "of a linear structural law)"
)
NO_DIFFERENCE = (
    " (a multivariate point is Supported only when it lies within the marginal ranges and no "
    "farther from its nearest observed treatment row than 95% of the rows are from their own "
    "nearest other row, in standard-deviation units); the contrast-shaped finite difference has "
    "no plug-in for a vector-valued derivative and is not reported"
)
TEMPORAL_TEXT = (
    "Cheap and full validation of a temporal {what} on a supplied or accepted TemporalDag at one "
    "horizon, executed through {estimator}. The state checks support at the evaluation point (a "
    "point beyond the observed series fails the check){difference} and runs the TemporalDag "
    "Pulse-native suite on the Pulse witness at that horizon (treatment placebo and data subset; "
    "full adds, under Frequentist inference, the circular-block bootstrap stability). {bayesian}"
    "Known truth: the lag-one linear law of the fixture, whose derivative is its coefficient; the "
    "law is linear, so the fixture cannot detect curvature errors. The verdicts are checks of this "
    "route, not a calibration claim; no coverage record measures the intervals and calibration is "
    "not claimed."
)
TEMPORAL_BAYES = (
    "Bayesian studies add the prior and posterior predictive checks of the witness regression and, "
    "under full, prior sensitivity. "
)
TEMPORAL_DIFFERENCE = (
    ", and for an identity-scale first derivative the unit-step difference of the identified "
    "levels (the Pulse witness effect) against the published slope"
)

CLASS = {
    "fixture": "conformance/response/class_derivative_states",
    "test": "crates/antecedent/tests/class_derivative_known_truth.rs",
    "assertion": "class_derivative_routes_and_refuter_states_known_truth",
}
CLASS_TEXT = (
    "A {what} on a supplied or accepted Cpdag, Pag or latent-confounded Admg at validation {validation}, "
    "executed through {estimator}. Every identified completion of a Cpdag or Pag (every DAG of the "
    "Markov class, every MAG completion of a PAG read as its most-confounded ADMG) is identified by "
    "one adjustment set valid for the derivative, and an Admg is identified by the adjustment "
    "criterion on its proper back-door graph, so a latent common cause stays in the graph; the "
    "estimator is fitted on that set and is checked against the exact rational least-squares "
    "structural slope of a confounded linear law (the confounded slope is not what a route "
    "publishes). {states}Front-door-identified and general-ID derivatives, completions that "
    "disagree, and temporal ADMGs are not covered by this row's evidence. The verdicts are checks "
    "of this route, not a calibration claim; no coverage record measures the intervals and "
    "calibration is not claimed."
)
CLASS_STATES = {
    "none": "",
    "cheap": (
        "The refuter state checks support at the evaluation point (an evaluation point beyond the "
        "observed treatment range fails the check){difference}. {bayesian}"
    ),
    "full": (
        "The refuter state checks support at the evaluation point (an evaluation point beyond the "
        "observed treatment range fails the check){difference}, and a whole-procedure row bootstrap "
        "of the derivative whose 95% range must contain the published value. {bayesian}"
    ),
}
CLASS_DIFFERENCE = (
    ", and the central finite difference of the identified intervention levels against the "
    "published derivative on its reported scale"
)
CLASS_NO_DIFFERENCE = (
    " (a multivariate point is Supported only when it lies within the marginal ranges and no "
    "farther from its nearest observed treatment row than 95% of the rows are from their own "
    "nearest other row); the contrast-shaped finite difference has no plug-in for a "
    "vector-valued derivative and is not reported"
)
CLASS_BAYES = (
    "Bayesian studies add the prior and posterior predictive checks of the outcome regression "
    "and, under full, prior sensitivity. "
)

TEMPORAL_ADMG = {
    "fixture": "conformance/response/temporal_admg_derivative_states",
    "test": "crates/antecedent/tests/temporal_admg_derivative_known_truth.rs",
    "assertion": "temporal_admg_derivative_routes_and_refuter_states_known_truth",
}
TEMPORAL_ADMG_TEXT = (
    "A temporal {what} on a supplied or accepted TemporalAdmg at one horizon at validation "
    "{validation}, executed through {estimator}. The ADMG is read as the unrolled window of its "
    "template and identified by the adjustment criterion on its proper back-door graph, so a "
    "latent common cause stays in the graph; the estimator is fitted on that set and is checked "
    "against the exact rational least-squares structural slope of a confounded lag-one linear law "
    "(the confounded slope is not what a route publishes). {states}Several horizons, "
    "front-door-identified and general-ID derivatives are not covered by this row's evidence. "
    "The verdicts are checks of this route, not a calibration claim; no coverage record measures "
    "the intervals and calibration is not claimed."
)
TEMPORAL_ADMG_STATES = {
    "none": "",
    "cheap": (
        "The refuter state checks support at the evaluation point (a point beyond the observed "
        "series fails the check){difference} and the E-value sensitivity of the Pulse witness. "
        "{bayesian}"
    ),
    "full": (
        "The refuter state checks support at the evaluation point (a point beyond the observed "
        "series fails the check){difference} and runs the E-value and nonparametric sensitivity "
        "of the Pulse witness. {bayesian}"
    ),
}
TEMPORAL_ADMG_DIFFERENCE = (
    ", and for an identity-scale first derivative the unit-step difference of the identified "
    "levels against the published slope"
)
TEMPORAL_ADMG_BAYES = (
    "Bayesian studies add the prior and posterior predictive checks of the witness regression "
    "and, under full, prior sensitivity. "
)

TEMPORAL_CLASS = {
    "fixture": "conformance/estimate/temporal_derivative_refuter_states",
    "test": "crates/antecedent/tests/temporal_class_derivative_known_truth.rs",
    "assertion": "temporal_class_derivative_refuter_states_known_truth",
}
TEMPORAL_CLASS_TEXT = (
    "Cheap and full validation of a temporal {what} on a supplied or accepted TemporalCpdag or "
    "TemporalPag at one horizon, executed through {estimator}. The class of the fixture's lagged "
    "template has one completion (a TemporalPag with a visible treatment edge), which is "
    "identified by the temporal back-door and carries one Pulse witness; the state checks support "
    "at the evaluation point (a point beyond the observed series fails the check){difference} and "
    "runs the TemporalDag Pulse-native suite on the completion's witness (treatment placebo and "
    "data subset; full adds, under Frequentist inference, the circular-block bootstrap "
    "stability). {bayesian}Several completions, whose witnesses are not mixed, and several "
    "horizons are not covered by this row's evidence. Known truth: the lag-one linear law of the "
    "fixture, whose derivative is its coefficient. The verdicts are checks of this route, not a "
    "calibration claim; no coverage record measures the intervals and calibration is not claimed."
)

GRAPH_POSTERIOR = {
    "fixture": "conformance/response/graph_posterior_derivative_states",
    "test": "crates/antecedent/tests/graph_posterior_derivative_known_truth.rs",
    "assertion": "graph_posterior_derivative_mixtures_and_refuter_states_known_truth",
}
GRAPH_POSTERIOR_TEXT = (
    "A {what} over a graph posterior of Dag atoms at validation {validation}, executed through "
    "{estimator}. Every atom is answered by the executor the query has on a supplied Dag and the "
    "atoms mix by posterior probability coordinate by coordinate: the value is the posterior-weighted "
    "mean over the atoms that identify and evaluate it; an atom the identifier refuses (a cyclic "
    "sample) is unidentified posterior mass, stated in the result, which resolves its status to "
    "graph-dependent. The fixture's atoms (one without and one with a back-door confounder) have "
    "the same structural derivative, so the mixture is checked against the exact derivative of the "
    "linear law whatever the weights, and the masses against the atoms' weights. {states}An atom "
    "whose graph has no directed path from the treatment to the outcome is answered with the "
    "route's data-based slope, not a structural zero, and is not covered by this row's evidence. "
    "The uncertainty of a mixture of several atoms is withheld with the stated reason when no "
    "replicates are requested. The verdicts are checks of this route, not a calibration claim; no "
    "coverage record measures the intervals and calibration is not claimed."
)
GRAPH_POSTERIOR_STATES = {
    "none": "",
    "cheap": (
        "Cheap runs each atom's own derivative refuter state against that atom's own estimate (support "
        "at the evaluation point{difference}) and mixes the reports of each refuter by the posterior "
        "weight of the atoms that produced them; a mixed verdict passes only if every contributing "
        "atom passes. {bayesian}"
    ),
    "full": (
        "Full adds, per atom, the whole-procedure row bootstrap stability of the derivative to the "
        "atom's own state (support at the evaluation point{difference}) and mixes the reports of "
        "each refuter by the posterior weight of the atoms that produced them; a mixed verdict "
        "passes only if every contributing atom passes. {bayesian}"
    ),
}
GRAPH_POSTERIOR_DIFFERENCE = (
    ", the central finite difference of the identified levels for a scalar derivative"
)
GRAPH_POSTERIOR_BAYES = (
    "Bayesian studies add the prior and posterior predictive checks and, under full, prior "
    "sensitivity. "
)

TEMPORAL_GRAPH_POSTERIOR = {
    "fixture": "conformance/estimate/temporal_derivative_refuter_states",
    "test": "crates/antecedent/tests/temporal_graph_posterior_derivative_known_truth.rs",
    "assertion": "temporal_graph_posterior_derivative_routes_and_refuter_states_known_truth",
}
TEMPORAL_GP_TEXT = (
    "A temporal {what} over a graph posterior of {atoms} atoms at one horizon at validation "
    "{validation}, executed through {estimator}. Every atom (a stationary template with its lag "
    "mask) is answered by the temporal slope design of that template and the atoms mix by "
    "posterior probability; an atom the identifier refuses (a cyclic sample) is unidentified "
    "posterior mass, stated in the result, which resolves its status to graph-dependent. The "
    "fixture's lag-one template has the structural derivative of its linear law, so the mixture "
    "is checked against the exact derivative and the masses against the atoms' weights. {states}"
    "Several atoms with different derivatives, several horizons and several completions of a class "
    "atom are not covered by this row's evidence. Known truth: the lag-one linear law of the "
    "fixture, whose derivative is its coefficient. The verdicts are checks of this route, not a "
    "calibration claim; no coverage record measures the intervals and calibration is not claimed."
)
TEMPORAL_GP_STATES = {
    "none": "",
    "cheap": (
        "Cheap runs each atom's own derivative refuter state against that atom's own estimate "
        "(support at the evaluation point{difference}) and the TemporalDag Pulse-native suite on "
        "its Pulse witness, and mixes the reports of each refuter by the posterior weight of the "
        "atoms that produced them. {bayesian}"
    ),
    "full": (
        "Full adds, per atom, the treatment placebo, data subset and circular-block stability "
        "checks (Frequentist) of the Pulse witness to the atom's own state (support at the "
        "evaluation point{difference}) and mixes the reports of each refuter by the posterior "
        "weight of the atoms that produced them. {bayesian}"
    ),
}
TEMPORAL_GP_DIFFERENCE = (
    ", and for an identity-scale first derivative the unit-step difference of the identified levels"
)
TEMPORAL_GP_BAYES = (
    "Bayesian studies add the prior and posterior predictive checks of the witness regression "
    "and, under full, prior sensitivity. "
)

CLASS_GP = {
    "fixture": "conformance/response/class_derivative_states",
    "test": "crates/antecedent/tests/class_graph_posterior_derivative_known_truth.rs",
    "assertion": "class_graph_posterior_derivative_routes_and_refuter_states_known_truth",
}
CLASS_GP_TEXT = (
    "A {what} over a graph posterior of Cpdag, Pag, latent-confounded Admg or TemporalAdmg (read as its contemporaneous ADMG projection) atoms at validation "
    "{validation}, executed through {estimator}. Every atom is answered by the executor the query "
    "has on a supplied structure of its kind (each identified completion of a class is adjusted "
    "by one valid set; an Admg is identified by the adjustment criterion on its proper back-door "
    "graph) and the atoms mix by posterior probability; the fixture's single atom is checked "
    "against the exact rational least-squares structural slope of a confounded linear law (the "
    "confounded slope is not what a route publishes). {states}Several atoms of different kinds, "
    "lost posterior mass of a class posterior, front-door-identified and general-ID atoms are not "
    "covered by this row's evidence. The verdicts are checks of this route, not a calibration "
    "claim; no coverage record measures the intervals and calibration is not claimed."
)

SELECTOR = {
    "structure": ["explicit", "accepted"],
    "inference": ["Frequentist", "Bayesian"],
    "validation": ["cheap", "full"],
}
STATIC_FAMILIES = [
    ("PointDerivative", "point derivative of order one", "response.kennedy_dr", False),
    ("Elasticity", "elasticity", "response.kennedy_dr", False),
    ("SemiElasticity", "semi-elasticity", "response.kennedy_dr", False),
    ("AverageDerivative", "observed-weight average derivative", "response.riesz_ade", False),
    ("DirectionalDerivative", "directional derivative", "response.gam_derivative", True),
    ("ResponseJacobian", "response Jacobian", "response.gam_derivative", True),
]
TEMPORAL_QUERIES = [
    ("PointDerivative", "point derivative of order one"),
    ("Elasticity", "elasticity"),
    ("SemiElasticity", "semi-elasticity"),
    ("AverageDerivative", "observed-weight average derivative"),
    ("DirectionalDerivative", "directional derivative"),
    ("ResponseJacobian", "response Jacobian"),
]
TEMPORAL_ESTIMATOR = {
    "Frequentist": "temporal.response.gcomp",
    "Bayesian": "response.temporal.bayesian",
}

FAMILIES: list[dict] = []
for query, what, estimator, multivariate in STATIC_FAMILIES:
    FAMILIES.append(
        {
            **CLASS_GP,
            "query": query,
            "graph_class": ["Cpdag", "Pag", "Admg", "TemporalAdmg"],
            "structure": ["graph_posterior"],
            "validation": ["none", "cheap", "full"],
            "estimator": {"Frequentist": estimator, "Bayesian": estimator},
            "what": what,
            "text": CLASS_GP_TEXT,
            "multivariate": multivariate,
        }
    )
for query, what in TEMPORAL_QUERIES:
    FAMILIES.append(
        {
            **TEMPORAL_GRAPH_POSTERIOR,
            "query": query,
            "graph_class": ["TemporalDag"],
            "structure": ["graph_posterior"],
            "validation": ["none", "cheap", "full"],
            "estimator": TEMPORAL_ESTIMATOR,
            "what": what,
            "text": TEMPORAL_GP_TEXT,
            "atoms": "TemporalDag",
            "has_difference": query in {"PointDerivative", "AverageDerivative", "ResponseJacobian"},
        }
    )
    FAMILIES.append(
        {
            **TEMPORAL_GRAPH_POSTERIOR,
            "query": query,
            "graph_class": ["TemporalCpdag", "TemporalPag"],
            "structure": ["graph_posterior"],
            "validation": ["cheap", "full"],
            "estimator": TEMPORAL_ESTIMATOR,
            "what": what,
            "text": TEMPORAL_GP_TEXT,
            "atoms": "TemporalCpdag or TemporalPag",
            "has_difference": query in {"PointDerivative", "AverageDerivative", "ResponseJacobian"},
        }
    )
for query, what, estimator, multivariate in STATIC_FAMILIES:
    FAMILIES.append(
        {
            **GRAPH_POSTERIOR,
            "query": query,
            "graph_class": ["Dag"],
            "structure": ["graph_posterior"],
            "validation": ["none", "cheap", "full"],
            "estimator": {"Frequentist": estimator, "Bayesian": estimator},
            "what": what,
            "text": GRAPH_POSTERIOR_TEXT,
            "multivariate": multivariate,
        }
    )
for query, what in TEMPORAL_QUERIES:
    FAMILIES.append(
        {
            **TEMPORAL_CLASS,
            "query": query,
            "graph_class": ["TemporalCpdag", "TemporalPag"],
            "estimator": TEMPORAL_ESTIMATOR,
            "what": what,
            "text": TEMPORAL_CLASS_TEXT,
            "bayes": TEMPORAL_BAYES,
            "difference": TEMPORAL_DIFFERENCE,
            "own": "",
        }
    )
for query, what in TEMPORAL_QUERIES:
    FAMILIES.append(
        {
            **TEMPORAL_ADMG,
            "query": query,
            "graph_class": ["TemporalAdmg"],
            "validation": ["none", "cheap", "full"],
            "estimator": TEMPORAL_ESTIMATOR,
            "what": what,
            "text": TEMPORAL_ADMG_TEXT,
            "has_difference": query in {"PointDerivative", "AverageDerivative", "ResponseJacobian"},
        }
    )
for query, what, estimator, multivariate in STATIC_FAMILIES:
    FAMILIES.append(
        {
            **CLASS,
            "query": query,
            "graph_class": ["Cpdag", "Pag", "Admg"],
            "validation": ["none", "cheap", "full"],
            "estimator": {"Frequentist": estimator, "Bayesian": estimator},
            "what": what,
            "text": CLASS_TEXT,
            "multivariate": multivariate,
        }
    )
for query, what, estimator, multivariate in STATIC_FAMILIES:
    FAMILIES.append(
        {
            **STATIC,
            "query": query,
            "graph_class": ["Dag"],
            "estimator": {"Frequentist": estimator, "Bayesian": estimator},
            "what": what,
            "text": STATIC_TEXT,
            "bayes": STATIC_BAYES,
            "difference": NO_DIFFERENCE if multivariate else DIFFERENCE,
            "own": "" if multivariate else ", and the finite-difference check has no planted violation of its own",
        }
    )
for query, what in TEMPORAL_QUERIES:
    FAMILIES.append(
        {
            **TEMPORAL,
            "query": query,
            "graph_class": ["TemporalDag"],
            "estimator": TEMPORAL_ESTIMATOR,
            "what": what,
            "text": TEMPORAL_TEXT,
            "bayes": TEMPORAL_BAYES,
            "difference": TEMPORAL_DIFFERENCE,
            "own": "",
        }
    )


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_row(family: dict, cell: tuple[str, str, str, str, str]) -> str:
    query, graph_class, structure, inference, validation = cell
    estimator = family["estimator"][inference]
    if family["text"] is CLASS_TEXT:
        states = CLASS_STATES[validation].format(
            difference=CLASS_NO_DIFFERENCE if family["multivariate"] else CLASS_DIFFERENCE,
            bayesian=CLASS_BAYES if inference == "Bayesian" else "",
        )
        limitations = family["text"].format(
            what=family["what"], estimator=estimator, validation=validation, states=states
        )
    elif family["text"] is CLASS_GP_TEXT:
        states = CLASS_STATES[validation].format(
            difference=CLASS_NO_DIFFERENCE if family["multivariate"] else CLASS_DIFFERENCE,
            bayesian=CLASS_BAYES if inference == "Bayesian" else "",
        )
        limitations = family["text"].format(
            what=family["what"], estimator=estimator, validation=validation, states=states
        )
    elif family["text"] is GRAPH_POSTERIOR_TEXT:
        states = GRAPH_POSTERIOR_STATES[validation].format(
            difference="" if family["multivariate"] else GRAPH_POSTERIOR_DIFFERENCE,
            bayesian=GRAPH_POSTERIOR_BAYES if inference == "Bayesian" else "",
        )
        limitations = family["text"].format(
            what=family["what"], estimator=estimator, validation=validation, states=states
        )
    elif family["text"] is TEMPORAL_GP_TEXT:
        states = TEMPORAL_GP_STATES[validation].format(
            difference=TEMPORAL_GP_DIFFERENCE if family["has_difference"] else "",
            bayesian=TEMPORAL_GP_BAYES if inference == "Bayesian" else "",
        )
        limitations = family["text"].format(
            what=family["what"],
            estimator=estimator,
            validation=validation,
            states=states,
            atoms=family["atoms"],
        )
    elif family["text"] is TEMPORAL_ADMG_TEXT:
        states = TEMPORAL_ADMG_STATES[validation].format(
            difference=TEMPORAL_ADMG_DIFFERENCE if family["has_difference"] else "",
            bayesian=TEMPORAL_ADMG_BAYES if inference == "Bayesian" else "",
        )
        limitations = family["text"].format(
            what=family["what"], estimator=estimator, validation=validation, states=states
        )
    else:
        limitations = family["text"].format(
            what=family["what"],
            estimator=estimator,
            bayesian=family["bayes"] if inference == "Bayesian" else "",
            difference=family["difference"],
            own=family["own"],
        )
    test, assertion = family["test"], family["assertion"]
    lines = [
        "[[cell]]",
        f"query = {quote(query)}",
        f"graph_class = {quote(graph_class)}",
        f"structure = {quote(structure)}",
        f"inference = {quote(inference)}",
        f"validation = {quote(validation)}",
        f"estimators = [{quote(estimator)}]",
        "staged = true",
        'evidence_kind = "internal_known_truth"',
        f"known_truth_fixture = {quote(family['fixture'])}",
        f"evidence_test = {quote(test)}",
        f"evidence_assertion = {quote(assertion)}",
        "checked_execution = [",
        f"  {{ estimator = {quote(estimator)}, test = {quote(test)}, "
        f"assertion = {quote(assertion)} }},",
        "]",
        f"limitations = {quote(limitations)}",
        'calibration_reason = "estimator_grid_not_measured"',
    ]
    return "\n".join(lines)


def cells(family: dict):
    return product(
        [family["query"]],
        family["graph_class"],
        family.get("structure", SELECTOR["structure"]),
        SELECTOR["inference"],
        family.get("validation", SELECTOR["validation"]),
    )


def render_block() -> str:
    rows = [render_row(family, cell) for family in FAMILIES for cell in cells(family)]
    return BEGIN + "\n\n" + "\n\n".join(rows) + "\n\n" + END + "\n"


def splice(text: str, block: str) -> str:
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        return head.rstrip("\n") + "\n\n" + block + ("\n" + tail.lstrip("\n") if tail.strip() else "")
    return text.rstrip("\n") + "\n\n" + block


def main(argv: list[str]) -> int:
    current = LICENSED.read_text(encoding="utf-8")
    updated = splice(current, render_block())
    if argv[:1] == ["--check"]:
        if updated != current:
            print("parity/support_licensed.toml: generated derivative refuter block drifted")
            return 1
        print("derivative refuter rows OK")
        return 0
    if updated != current:
        LICENSED.write_text(updated, encoding="utf-8", newline="\n")
    print(f"wrote {sum(len(list(cells(f))) for f in FAMILIES)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
