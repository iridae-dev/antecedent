#!/usr/bin/env python3
"""Generate the licensed attribution rows that share one evidence family.

An anomaly or change attribution on an accepted structure is the attribution of the structure the
accepted record names: the record is the structure's provenance, not a different graph. Each row
below therefore cites the independently derived truth of the supplied-structure cell (the
`conformance/estimate/staged_attribution` fixture) and a test that builds the cell with an
`AcceptedGraph`, plus a checked-execution test that drops its builder and runs the retained plan.

The rows are written between two marker comments in `parity/support_licensed.toml`; the file is
read, edited and replaced in one step, so a concurrent edit elsewhere in it is never rewritten.

    python3 scripts/generate_attribution_support_rows.py          # rewrite the block
    python3 scripts/generate_attribution_support_rows.py --check  # fail when it drifts

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSED = ROOT / "parity" / "support_licensed.toml"
BEGIN = "# BEGIN generated attribution rows (scripts/generate_attribution_support_rows.py)\n"
END = "# END generated attribution rows\n"
ANCHOR = '[[cell]]\nquery = "TransportQuery"\n'

FIXTURE = "conformance/estimate/staged_attribution"

# (graph class, evidence test, evidence assertion, checked-execution test, assertion, note)
FAMILIES = [
    (
        "Dag",
        "crates/antecedent/tests/staged_attribution_transport_interference.rs",
        "accepted_dag_attribution_known_truth",
        "crates/antecedent/tests/checked_attribution_route_evidence.rs",
        "accepted_dag_attribution_plans_are_sealed_for_both_inference_modes",
        "an accepted Dag",
    ),
    (
        "TemporalDag",
        "crates/antecedent/tests/temporal_attribution_route_evidence.rs",
        "temporal_accepted_attribution_known_truth",
        "crates/antecedent/tests/temporal_attribution_route_evidence.rs",
        "temporal_accepted_attribution_known_truth",
        "an accepted TemporalDag unfolded to a lag-aligned static SCM",
    ),
]
# (query, inference, estimator, what the row reports)
COORDINATES = [
    ("AnomalyAttribution", "Frequentist", "gcm.fit", "GCM anomaly scores"),
    ("ChangeAttribution", "Frequentist", "gcm.fit", "GCM distribution-change Shapley, two time-range populations"),
    ("AnomalyAttribution", "Bayesian", "gcm.fit.bayesian", "Bayesian-bootstrap posterior of target mean anomaly scores"),
    ("ChangeAttribution", "Bayesian", "gcm.attribution.bayesian", "Bayesian-bootstrap posterior of mechanism-change Shapley contributions"),
]


def calibration_reason(graph_class: str, inference: str) -> str:
    # A static frequentist attribution reports no interval; every other cell reports one
    # (a posterior interval, or a circular-block bootstrap one on request) whose coverage is
    # not measured.
    if graph_class == "Dag" and inference == "Frequentist":
        return "no_interval_reported"
    return "estimator_grid_not_measured"


REFUTER_FIXTURE = "conformance/estimate/attribution_refuter_states"
REFUTER_TEST = "crates/antecedent/tests/attribution_refuter_known_truth.rs"
REFUTER_ASSERTION = "anomaly_attribution_refuter_states_known_truth"
CHECKED = "crates/antecedent/tests/checked_attribution_route_evidence.rs"
# (structure, inference, estimator, checked-execution assertion): the retained plan of the
# anomaly route runs the same program at every validation, so the checked citation is the one of
# the validation-none cell of the same structure and inference.
TEMPORAL_TEST = "crates/antecedent/tests/temporal_attribution_route_evidence.rs"
TEMPORAL_ASSERTION = "temporal_anomaly_attribution_refuter_states_known_truth"
# (graph class, structure, inference, estimator, checked test, checked assertion, evidence assertion)
VALIDATED_ANOMALY = [
    ("Dag", "explicit", "Frequentist", "gcm.fit", CHECKED, "all_frequentist_static_dag_attribution_plans_are_sealed_and_dependency_refused"),
    ("Dag", "explicit", "Bayesian", "gcm.fit.bayesian", CHECKED, "all_bayesian_static_dag_attribution_plans_are_sealed_and_dependency_refused"),
    ("Dag", "accepted", "Frequentist", "gcm.fit", CHECKED, "accepted_dag_attribution_plans_are_sealed_for_both_inference_modes"),
    ("Dag", "accepted", "Bayesian", "gcm.fit.bayesian", CHECKED, "accepted_dag_attribution_plans_are_sealed_for_both_inference_modes"),
    ("TemporalDag", "explicit", "Frequentist", "gcm.fit", TEMPORAL_TEST, "temporal_attribution_frequentist_known_truth"),
    ("TemporalDag", "explicit", "Bayesian", "gcm.fit.bayesian", TEMPORAL_TEST, "temporal_attribution_bayesian_known_truth"),
    ("TemporalDag", "accepted", "Frequentist", "gcm.fit", TEMPORAL_TEST, "temporal_accepted_attribution_known_truth"),
    ("TemporalDag", "accepted", "Bayesian", "gcm.fit.bayesian", TEMPORAL_TEST, "temporal_accepted_attribution_known_truth"),
]


def validated_rows() -> list[str]:
    out = []
    for graph_class, structure, inference, estimator, checked_test, checked in VALIDATED_ANOMALY:
        for validation in ("cheap", "full"):
            out.append(
                "\n".join(
                    [
                        "[[cell]]",
                        'query = "AnomalyAttribution"',
                        f'graph_class = "{graph_class}"',
                        f'structure = "{structure}"',
                        f'inference = "{inference}"',
                        f'validation = "{validation}"',
                        f'estimators = ["{estimator}"]',
                        "staged = true",
                        'evidence_kind = "internal_known_truth"',
                        f'known_truth_fixture = "{REFUTER_FIXTURE}"',
                        f'evidence_test = "{REFUTER_TEST}"',
                        f'evidence_assertion = "{REFUTER_ASSERTION if graph_class == "Dag" else TEMPORAL_ASSERTION}"',
                        "checked_execution = [",
                        f'  {{ estimator = "{estimator}", test = "{checked_test}", assertion = "{checked}" }},',
                        "]",
                        f'limitations = "Anomaly attribution on a {structure} {graph_class} at validation {validation}: the whole vector of mean scores and per-component Shapley contributions is refitted by the same compiled program on random row subsets (the held-out-population check of the fitted mechanisms) and, under full, on bootstrap resamples, each refuter reporting one joint chi-square statistic. Held to a planted high-leverage row that the subset refuter must detect and a clean table it must pass (conformance/estimate/attribution_refuter_states, derived by leverage and least-squares slope in exact rationals). A rows-independent subset cannot flag a row-order violation, and the verdicts are diagnostics, not a calibration claim. ATE-shaped contrast refuters do not apply; the Bayesian study runs the same data refuters and no prior-sensitivity state. A temporal series is windowed circularly (every period equally often) and block-resampled as gathered series. Change attribution at cheap/full is a refusal."',
                        f'calibration_reason = "{calibration_reason(graph_class, inference)}"',
                        "",
                    ]
                )
                + "\n"
            )
    return out


CLASS_FIXTURE = "conformance/estimate/class_attribution_truth"
CLASS_REFUTER_FIXTURE = "conformance/estimate/class_attribution_refuter_states"
CLASS_TEST = "crates/antecedent/tests/class_attribution_known_truth.rs"
CLASS_CHECKED = "class_attribution_executes_from_retained_plan_after_builder_is_dropped"
POSTERIOR_CHECKED = "class_graph_posterior_attribution_executes_from_retained_plan_after_builder_is_dropped"
CLASS_GRAPHS = ["Admg", "TemporalAdmg", "Cpdag", "Pag", "TemporalCpdag", "TemporalPag"]
POSTERIOR_GRAPHS = ["Dag", "Admg", "TemporalAdmg", "Cpdag", "Pag", "TemporalDag", "TemporalCpdag", "TemporalPag"]
TEMPORAL = {"TemporalAdmg", "TemporalDag", "TemporalCpdag", "TemporalPag"}
CLASS_NOUN = {
    "Admg": "an Admg (attributed to the noise blocks of its ancestral sub-graph)",
    "TemporalAdmg": "a TemporalAdmg (the template unrolled over the window its longest lag spans)",
    "Cpdag": "a Cpdag (member by member over its DAG completions)",
    "Pag": "a Pag (member by member over its MAG completions, each read as its most-confounded mixed graph)",
    "TemporalCpdag": "a TemporalCpdag (member by member over its temporal DAG completions)",
    "TemporalPag": "a TemporalPag (member by member over its temporal MAG completions)",
    "Dag": "a Dag atom",
    "TemporalDag": "a TemporalDag atom",
}


def class_calibration_reason(graph_class: str, inference: str) -> str:
    # A frequentist class or posterior attribution is a point value (a requested mechanism-refit
    # interval is a separate, unlicensed construction); a Bayesian one reports a posterior
    # interval whose coverage is not measured.
    if inference == "Frequentist":
        return "no_interval_reported"
    return "estimator_grid_not_measured"


def estimator_of(query: str, inference: str) -> str:
    if inference == "Frequentist":
        return "gcm.fit"
    return "gcm.fit.bayesian" if query == "AnomalyAttribution" else "gcm.attribution.bayesian"


def class_row(query, graph_class, structure, inference, validation, fixture, assertion, checked, limitations):
    estimator = estimator_of(query, inference)
    return (
        "\n".join(
            [
                "[[cell]]",
                f'query = "{query}"',
                f'graph_class = "{graph_class}"',
                f'structure = "{structure}"',
                f'inference = "{inference}"',
                f'validation = "{validation}"',
                f'estimators = ["{estimator}"]',
                "staged = true",
                'evidence_kind = "internal_known_truth"',
                f'known_truth_fixture = "{fixture}"',
                f'evidence_test = "{CLASS_TEST}"',
                f'evidence_assertion = "{assertion}"',
                "checked_execution = [",
                f'  {{ estimator = "{estimator}", test = "{CLASS_TEST}", assertion = "{checked}" }},',
                "]",
                f'limitations = "{limitations}"',
                f'calibration_reason = "{class_calibration_reason(graph_class, inference)}"',
                "",
            ]
        )
        + "\n"
    )


def class_rows() -> list[str]:
    out = []
    for graph_class in CLASS_GRAPHS:
        for structure in ("explicit", "accepted"):
            for inference in ("Frequentist", "Bayesian"):
                for query in ("AnomalyAttribution", "ChangeAttribution"):
                    anomaly = query == "AnomalyAttribution"
                    what = "anomaly scores and Shapley noise contributions" if anomaly else "mechanism-swap Shapley contributions to the change of the mean"
                    out.append(
                        class_row(
                            query, graph_class, structure, inference, "none", CLASS_FIXTURE,
                            "class_anomaly_attribution_known_truth" if anomaly else "class_change_attribution_known_truth",
                            CLASS_CHECKED,
                            f"The {what} on {CLASS_NOUN[graph_class]}, "
                            + ("supplied" if structure == "explicit" else "accepted (the record is the structure\'s provenance, not a different graph)")
                            + ". Held to a derivation from the definitions in exact rationals (conformance/estimate/class_attribution_truth): the least-squares mechanism of each node, the abducted residuals as noise, the two-sided Gaussian tail and the exact Shapley value over the members\' players"
                            + ("" if anomaly else " (the mean-level mechanism-swap game, to 5 standard errors of the 2000 Monte Carlo draws)")
                            + "; the class value is the weighted mean of the members that identify, the identified set their per-component range, and a member whose mechanism a hedge leaves unidentified is a typed refusal counted as unidentified mass. "
                            + ("The Bayesian row-weight posterior is held to its closed-form Dirichlet moments; members that disagree keep their posteriors on their atoms and publish none. " if inference == "Bayesian" else "")
                            + "A series is read on the lag-aligned design (the first rows repeat the first period) and a contemporaneous edge of a template repeats in every period. Completion masses are enumeration weights, not posterior probabilities. cheap/full validation and graph posteriors are separate cells.",
                        )
                    )
                if True:
                    for validation in ("cheap", "full"):
                        out.append(
                            class_row(
                                "AnomalyAttribution", graph_class, structure, inference, validation, CLASS_REFUTER_FIXTURE,
                                "class_anomaly_attribution_refuter_states_known_truth", CLASS_CHECKED,
                                f"Anomaly attribution on {CLASS_NOUN[graph_class]} at validation {validation}: the whole vector of mean scores and per-component contributions is refitted by the same compiled program on random row subsets (the held-out-population check of the fitted mechanisms) and, under full, on bootstrap resamples, each refuter reporting one joint chi-square statistic. Held to a planted high-leverage row that the subset refuter must detect and a clean table that every refuter must pass (conformance/estimate/class_attribution_refuter_states, derived by leverage, least-squares coefficients and robust z-scores in exact rationals). The verdicts are diagnostics, not a calibration claim; ATE-shaped contrast refuters do not apply. Change attribution at cheap/full is a refusal.",
                            )
                        )
    return out


def posterior_rows() -> list[str]:
    out = []
    for graph_class in POSTERIOR_GRAPHS:
        for inference in ("Frequentist", "Bayesian"):
            for query in ("AnomalyAttribution", "ChangeAttribution"):
                validations = ("none", "cheap", "full") if query == "AnomalyAttribution" else ("none",)
                for validation in validations:
                    anomaly = query == "AnomalyAttribution"
                    none = validation == "none"
                    out.append(
                        class_row(
                            query, graph_class, "graph_posterior", inference, validation,
                            CLASS_FIXTURE if none else CLASS_REFUTER_FIXTURE,
                            ("graph_posterior_attribution_known_truth" if none else "graph_posterior_anomaly_attribution_refuter_states_known_truth"),
                            POSTERIOR_CHECKED,
                            (
                                f"{'Anomaly' if anomaly else 'Change'} attribution over a graph posterior whose atoms are {CLASS_NOUN[graph_class]}. "
                                + (
                                    "Held to a derivation from the definitions in exact rationals (conformance/estimate/class_attribution_truth): each atom attributed as the structure it stands for, a class atom sharing its probability equally among its completions, the published contribution the posterior-weighted mean over the atoms that attribute it, the identified set their per-component range, and a mask that is no structure of the posterior\'s kind (or an atom a hedge leaves unidentified) unidentified posterior mass that is not redistributed, the status graph-dependent. Atoms\' posteriors are conditional on their structures and never mixed: the posterior is published only when every atom publishes the very same draws, and is held to its closed-form Dirichlet moments. "
                                    if none
                                    else "At validation " + validation + " the whole vector is refitted by the same compiled posterior program on random row subsets and, under full, on bootstrap resamples; held to a planted high-leverage row that the subset refuter must detect and a clean table that every refuter must pass (conformance/estimate/class_attribution_refuter_states). The verdicts are diagnostics, not a calibration claim. "
                                )
                                + "Posterior probabilities are the declared weights of the atoms; change attribution at cheap/full is a refusal."
                            ),
                        )
                    )
    return out


def render() -> str:
    out = [BEGIN]
    for graph_class, test, assertion, checked_test, checked_assertion, noun in FAMILIES:
        for query, inference, estimator, what in COORDINATES:
            out.append(
                "\n".join(
                    [
                        "[[cell]]",
                        f'query = "{query}"',
                        f'graph_class = "{graph_class}"',
                        'structure = "accepted"',
                        f'inference = "{inference}"',
                        'validation = "none"',
                        f'estimators = ["{estimator}"]',
                        "staged = true",
                        'evidence_kind = "internal_known_truth"',
                        f'known_truth_fixture = "{FIXTURE}"',
                        f'evidence_test = "{test}"',
                        f'evidence_assertion = "{assertion}"',
                        "checked_execution = [",
                        f'  {{ estimator = "{estimator}", test = "{checked_test}", assertion = "{checked_assertion}" }},',
                        "]",
                        f'limitations = "{what} on {noun}, attributed exactly as a supplied structure: the accepted record is the structure\'s provenance, not a different graph. Held to the same independently derived truth as the supplied-structure cell (a planted outlier row ranks first; a planted intercept shift of 5 is the total change). Parametric identification (gcm.parametric). cheap/full validation, graph-posterior and class graphs are separate cells."',
                        f'calibration_reason = "{calibration_reason(graph_class, inference)}"',
                        "",
                    ]
                )
                + "\n"
            )
    out.extend(validated_rows())
    out.extend(class_rows())
    out.extend(posterior_rows())
    out.append(END)
    return "".join(out)


def spliced(text: str, block: str) -> str:
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        if END not in rest:
            raise SystemExit("generated block has no end marker")
        tail = rest.split(END, 1)[1]
        return head + block + tail
    if ANCHOR not in text:
        raise SystemExit("anchor cell not found")
    head, tail = text.split(ANCHOR, 1)
    return head + block + "\n" + ANCHOR + tail


def main() -> int:
    check = sys.argv[1:] == ["--check"]
    if sys.argv[1:] and not check:
        print(__doc__)
        return 2
    text = LICENSED.read_text(encoding="utf-8")
    new = spliced(text, render())
    if check:
        if new != text:
            print("parity/support_licensed.toml: the generated attribution rows drifted; rerun the generator")
            return 1
        print("generated attribution rows ok")
        return 0
    if new != text:
        fd, tmp = tempfile.mkstemp(dir=LICENSED.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(new)
        os.replace(tmp, LICENSED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
