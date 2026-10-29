#!/usr/bin/env python3
"""Generate the licensed transport and interference cells that read a Dag, an accepted
structure, a Cpdag, a Pag or a graph posterior, and their cheap and full validation states.

The rows are a table, not hand-written text: every row shares one shape (query, graph class,
structure, inference, validation) and is evidenced by the executing known-truth tests in
`crates/antecedent/tests/transport_interference_cell_evidence.rs`, against the fixtures named
below. The generated block sits between two marker comments at the end of
`parity/support_licensed.toml` and nothing else in the file is touched.

    python3 scripts/generate_transport_interference_cells.py            # rewrite the block
    python3 scripts/generate_transport_interference_cells.py --check    # fail when it drifts
    python3 scripts/generate_transport_interference_cells.py --list     # one coordinate per line

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSED = ROOT / "parity" / "support_licensed.toml"
BEGIN = "# generated:transport-interference-cells:begin (scripts/generate_transport_interference_cells.py)"
END = "# generated:transport-interference-cells:end"

TEST = "crates/antecedent/tests/transport_interference_cell_evidence.rs"
INFERENCES = ("Frequentist", "Bayesian")
VALIDATIONS = ("none", "cheap", "full")
CHEAP_FULL = ("cheap", "full")
ALL_CLASSES = ("Dag", "Admg", "TemporalAdmg", "Cpdag", "Pag")

# (query, graph classes, structures, validations) for every group of cells this script owns.
GROUPS = (
    ("TransportQuery", ("Dag",), ("explicit", "accepted"), VALIDATIONS),
    ("TransportQuery", ("Cpdag", "Pag"), ("explicit", "accepted"), VALIDATIONS),
    ("TransportQuery", ("Admg", "TemporalAdmg"), ("accepted",), VALIDATIONS),
    ("TransportQuery", ALL_CLASSES, ("graph_posterior",), VALIDATIONS),
    ("TransportQuery", ("Admg", "TemporalAdmg"), ("explicit",), CHEAP_FULL),
    ("InterferenceQuery", ALL_CLASSES, ("accepted",), VALIDATIONS),
    ("InterferenceQuery", ("Admg", "TemporalAdmg", "Cpdag", "Pag"), ("explicit",), VALIDATIONS),
    ("InterferenceQuery", ("Dag",), ("explicit",), CHEAP_FULL),
    ("InterferenceQuery", ALL_CLASSES, ("graph_posterior",), VALIDATIONS),
)

ESTIMATOR = {
    ("TransportQuery", "Frequentist"): "transport.trial_ipw",
    ("TransportQuery", "Bayesian"): "transport.trial_bayesian_bootstrap",
    ("InterferenceQuery", "Frequentist"): "interference.ht_hajek",
    ("InterferenceQuery", "Bayesian"): "interference.bayesian_gaussian",
}
FIXTURE = {
    ("TransportQuery", "Frequentist"): "conformance/estimate/staged_transport",
    ("TransportQuery", "Bayesian"): "conformance/estimate/bayesian_trial_transport",
    ("InterferenceQuery", "Frequentist"): "conformance/response/randomized_interference",
    ("InterferenceQuery", "Bayesian"): "conformance/estimate/bayesian_interference",
}
CHECKED = {
    "TransportQuery": "transport_cells_execute_from_the_retained_operation_after_builder_drop",
    "InterferenceQuery": "interference_cells_execute_from_the_retained_operation_after_builder_drop",
}

TRANSPORT_POSTERIOR = {
    "Dag": "transport_dag_posterior_cells_mix_the_atoms_sid_certifies",
    "Admg": "transport_admg_posterior_cells_state_the_unidentified_and_unevaluable_mass",
    "TemporalAdmg": "transport_temporal_admg_posterior_cells_mix_the_atoms_sid_certifies",
    "Cpdag": "transport_cpdag_posterior_cells_run_the_class_envelope_of_sid",
    "Pag": "transport_pag_posterior_cells_state_the_mass_sid_does_not_certify",
}
INTERFERENCE_POSTERIOR = {
    kind: f"interference_{snake}_posterior_cells_carry_the_design_contrast_and_the_atom_mass"
    for kind, snake in (
        ("Dag", "dag"),
        ("Admg", "admg"),
        ("TemporalAdmg", "temporal_admg"),
        ("Cpdag", "cpdag"),
        ("Pag", "pag"),
    )
}


def evidence_assertion(query: str, graph: str, structure: str, validation: str) -> str:
    if query == "TransportQuery":
        if structure == "graph_posterior":
            return TRANSPORT_POSTERIOR[graph]
        if graph == "Dag":
            return "transport_dag_cells_publish_the_pinned_contrast_and_the_hand_counted_suite"
        if graph in ("Cpdag", "Pag"):
            return (
                "transport_cpdag_and_pag_cells_publish_the_pinned_contrast_and_the_hand_counted_suite"
            )
        if structure == "accepted":
            return "transport_admg_and_temporal_admg_accepted_cells_publish_the_pinned_contrast"
        return (
            "transport_admg_and_temporal_admg_explicit_cheap_and_full_run_the_hand_counted_suite"
        )
    if structure == "graph_posterior":
        return INTERFERENCE_POSTERIOR[graph]
    if structure == "accepted":
        return "interference_accepted_cells_publish_the_enumerated_contrast_and_the_design_suite"
    if graph == "Dag":
        return "interference_dag_explicit_cheap_and_full_run_the_design_suite"
    return "interference_admg_temporal_admg_cpdag_and_pag_explicit_cells_publish_the_enumerated_contrast"


SUITE_TEXT = {
    "none": "",
    "cheap": (
        " Cheap validation runs the transport-native suite: the supplied selection probabilities "
        "are tested against the observed trial count and the supplied treatment probabilities "
        "against the observed arm counts of the trial rows (Poisson-binomial z tests{series}), "
        "and the effective sample size of the selection-odds and arm weights is reported as a "
        "share of the trial rows; the pinned fixture counts two trial rows where the supplied "
        "probabilities imply two, one treated trial row where they imply one, and equal weights, "
        "{planted} ATE-shaped contrast refuters do not apply."
    ),
    "full": (
        " Full validation adds to the cheap transport-native suite a whole-procedure bootstrap "
        "of the transported contrast (rows, or circular blocks on one series) whose 95% "
        "percentile range must contain the published value; replicates whose resampled table "
        "has no trial row, no target row or a single arm are failed and not counted. ATE-shaped "
        "contrast refuters do not apply."
    ),
}
INTERFERENCE_SUITE_TEXT = {
    "none": "",
    "cheap": (
        " Cheap validation runs the interference-native suite: the realized assignment against "
        "the declared design (a Poisson-binomial z test of the treated count under Bernoulli "
        "assignment) and the smallest exposure probability; the pinned designs count one treated "
        "unit of two and two of four where the declared probabilities imply the same, and a "
        "planted design that implies 7.2 treated of eight when three are treated fails the "
        "calibration. ATE-shaped contrast refuters do not apply."
    ),
    "full": (
        " Full validation adds a re-randomization test that the contrast is unbiased around zero "
        "under the sharp null with the outcomes held fixed (the enumerated two-unit design has "
        "re-drawn contrasts of mean zero and variance 18); the Gaussian posterior has no "
        "re-randomization. ATE-shaped contrast refuters do not apply."
    ),
}


def transport_limitations(graph: str, structure: str, inference: str, validation: str) -> str:
    series = graph == "TemporalAdmg"
    parts: list[str] = []
    if structure == "graph_posterior":
        parts.append(
            "A TransportQuery over a graph posterior reads every posterior atom as a selection "
            "diagram with the declared selection targets and decides it by sID on its own graph. "
            "An atom with a certified formula is an identified atom that retains it; an atom sID "
            "does not certify is unidentified posterior mass, and a certified recursive "
            "factorization the trial-to-target weighting cannot evaluate is unevaluable mass. "
            "The atoms mix by posterior probability and neither lost mass is redistributed: the "
            "published contrast is the transported contrast conditional on the atoms that "
            "identify and evaluate, and the result states the unidentified and unevaluable mass "
            "and resolves its status to graph-dependent when either exceeds the tolerance."
        )
        if graph in ("Cpdag", "Pag"):
            parts.append(
                "An atom of a Cpdag or Pag posterior names a class of selection diagrams, so it "
                "is run through the class envelope of sID (every DAG completion of a Cpdag read "
                "as the ADMG with no bidirected edge, every ADMG sharing the m-separation model "
                "of a MAG completion of a Pag): the atom is identified when some member "
                "certifies a formula and unidentified mass when none does, and the members the "
                "class leaves uncertified are disclosed on the result. Completion weights inside "
                "an atom are enumeration weights, not posterior probabilities."
            )
    elif graph == "Dag":
        parts.append(
            "A TransportQuery on a Dag reads the Dag as the selection diagram of a model with no "
            "latent confounding: its ADMG embedding (every edge directed, no bidirected edge) "
            "with the declared selection nodes, identified by the implemented sID subset (Direct, "
            "or an S-admissible standardization)."
        )
    elif graph in ("Cpdag", "Pag"):
        parts.append(
            "A TransportQuery reads a selection diagram, which is an ADMG. A Cpdag or Pag stands "
            "for several diagrams that can disagree on the transport formula, so sID decides "
            "every member on its own selection diagram with the declared selection targets: each "
            "DAG completion of a Cpdag (read as the ADMG with no bidirected edge), and every ADMG "
            "that shares the m-separation model of each MAG completion of a Pag. Members that "
            "certify the same formula identify the transported quantity by it; members that "
            "certify different formulas leave it partially identified, one case per distinct "
            "formula (the open chain a - y - z with z as the selection target has three "
            "completions, two direct and one standardizing); a member sID does not certify is "
            "unidentified mass. Every certified case is estimated by the one trial-to-target "
            "weighting, which reads a formula only through its certificate, so the identified "
            "set over the evaluated cases is the one transported contrast and the full "
            "bootstrap reports one range for it; a certified recursive factorization the "
            "weighting cannot evaluate is unevaluable mass. Completion weights are enumeration "
            "weights, not posterior probabilities."
        )
    else:
        parts.append(
            f"A TransportQuery on an accepted {graph} reads the supplied selection diagram "
            "(identified by the implemented sID subset: Direct, or an S-admissible "
            "standardization) exactly as the explicit one."
        )
    if graph in ("Dag", "Cpdag", "Pag") and structure == "accepted":
        parts.append("The structure is supplied as an accepted graph.")
    parts.append(
        "The contrast is estimated by the trial-to-target weighting of the supplied selection "
        "and treatment probabilities (binary Dahabreh IPW from caller-supplied trial membership "
        "and known probabilities)."
    )
    if inference == "Frequentist":
        if series:
            parts.append(
                "The rows are read as one series, so the standard error is the Newey-West "
                "(Bartlett) long-run variance of the per-row influence at the default lag; "
                "the pinned four-row design has influences (+6, -2, -2, -2) and the lag-two "
                "long-run variance 112/3."
            )
        else:
            parts.append(
                "The interval is the delta-method standard error of the ratio of means over "
                "iid rows (square root of 48 over 2 on the pinned design)."
            )
    else:
        if series:
            parts.append(
                "The posterior weights circular blocks of the series; only its mean is pinned "
                "by the independent row-Dirichlet moments, its spread is not."
            )
        else:
            parts.append(
                "The posterior is the Dirichlet(1, ..., 1) reweighting of the observed trial "
                "rows conditional on the supplied probabilities, whose moments the pinned "
                "fixture gives in closed form (mean 2, variance 16/3); fitted-probability "
                "uncertainty is not propagated."
            )
    parts.append(
        "RecursiveFactorization and NotCertified refuse. The calibration slot stays unmeasured "
        "(estimator_grid_not_measured): no coverage of the reported interval is claimed."
    )
    text = " ".join(parts)
    planted = (
        "and on one series four rows carry too little information for the batch-means test to "
        "reject a planted selection probability of 0.9 (its p-value is 0.377)."
        if series
        else "and a planted selection probability of 0.9 fails its calibration."
    )
    suite = SUITE_TEXT[validation].replace(
        "{series}", ", with batch-means variance on one dependent series" if series else ""
    ).replace("{planted}", planted)
    return text + suite


def interference_limitations(graph: str, structure: str, inference: str, validation: str) -> str:
    parts: list[str] = []
    if structure == "graph_posterior":
        parts.append(
            "An InterferenceQuery over a graph posterior executes the exposure contrast once "
            "under the declared assignment design. The contrast is identified by the known "
            "assignment mechanism and the exposure mapping on the fixed network, so no edge of "
            "any atom enters the identification and the posterior cannot move the contrast, its "
            "sampling interval or its posterior. Every atom that is a member of the posterior's "
            "graph class is an identified atom with the one design identification, an atom that "
            "is not a member (a cyclic mask or an unreadable mark) is unidentified posterior "
            "mass, and the structural mixture and the mass axes are those of every graph-posterior "
            "route; the posterior's variables must equal the unit table's."
        )
    else:
        kind = {
            "Dag": "an explicit Dag",
            "Admg": "an Admg",
            "TemporalAdmg": "a TemporalAdmg read as its contemporaneous projection",
            "Cpdag": "a Cpdag",
            "Pag": "a Pag",
        }[graph]
        parts.append(
            f"An InterferenceQuery on {kind}"
            + (" supplied as an accepted structure" if structure == "accepted" else "")
            + " states the exposure contrast under that structure class. The contrast is "
            "identified by the known assignment mechanism and the exposure mapping on the fixed "
            "network (the assignment is independent of the potential outcomes by design), so no "
            "edge of the structure enters the identification: a latent confounder or an "
            "unresolved orientation cannot move it, and the structure binds only the unit schema "
            "(its width must equal the unit table's)."
        )
    if inference == "Frequentist":
        parts.append(
            "Evidence: the licensed estimator is the Young-bound Horvitz-Thompson / Hajek "
            "contrast under Bernoulli assignment with the NeighborCount exposure mapping on a "
            "caller-supplied fixed network, checked against the full enumeration of the pinned "
            "two-unit design (Horvitz-Thompson -6, Hajek -3, conservative variance 204). Other "
            "assignment designs and exposure mappings run through separate estimators that are "
            "not part of this cell's evidence."
        )
    else:
        parts.append(
            "Evidence: the licensed estimator is the conjugate Gaussian exposure posterior with "
            "variance fixed to one and independent Normal priors, checked against the stated "
            "model y = 2 + 3 own + 4 treated neighbors (contrast -1 within 0.08) on the pinned "
            "four-unit network."
        )
    parts.append(
        "The calibration slot stays unmeasured (estimator_grid_not_measured): no coverage of "
        "the reported interval is claimed."
    )
    return " ".join(parts) + INTERFERENCE_SUITE_TEXT[validation]


def toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_row(query: str, graph: str, structure: str, inference: str, validation: str) -> str:
    estimator = ESTIMATOR[(query, inference)]
    limitations = (
        transport_limitations(graph, structure, inference, validation)
        if query == "TransportQuery"
        else interference_limitations(graph, structure, inference, validation)
    )
    lines = [
        "[[cell]]",
        f"query = {toml_string(query)}",
        f"graph_class = {toml_string(graph)}",
        f"structure = {toml_string(structure)}",
        f"inference = {toml_string(inference)}",
        f"validation = {toml_string(validation)}",
        f"estimators = [{toml_string(estimator)}]",
        "staged = true",
        'evidence_kind = "internal_known_truth"',
        f"known_truth_fixture = {toml_string(FIXTURE[(query, inference)])}",
        f"evidence_test = {toml_string(TEST)}",
        f"evidence_assertion = {toml_string(evidence_assertion(query, graph, structure, validation))}",
        "checked_execution = [",
        f"  {{ estimator = {toml_string(estimator)}, test = {toml_string(TEST)}, "
        f"assertion = {toml_string(CHECKED[query])} }},",
        "]",
        f"limitations = {toml_string(limitations)}",
        'calibration_reason = "estimator_grid_not_measured"',
    ]
    return "\n".join(lines)


def coordinates() -> list[tuple[str, str, str, str, str]]:
    seen: list[tuple[str, str, str, str, str]] = []
    for query, graphs, structures, validations in GROUPS:
        for graph, structure, inference, validation in product(
            graphs, structures, INFERENCES, validations
        ):
            seen.append((query, graph, structure, inference, validation))
    if len(seen) != len(set(seen)):
        raise SystemExit("the generator table lists a cell twice")
    return seen


def render_block() -> str:
    rows = [render_row(*coordinate) for coordinate in coordinates()]
    return BEGIN + "\n\n" + "\n\n".join(rows) + "\n\n" + END + "\n"


def current_text() -> str:
    return LICENSED.read_text(encoding="utf-8")


def splice(text: str, block: str) -> str:
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        return head.rstrip("\n") + "\n\n" + block.rstrip("\n") + tail
    return text.rstrip("\n") + "\n\n" + block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for coordinate in coordinates():
            print(":".join(coordinate))
        return 0
    block = render_block()
    text = current_text()
    expected = splice(text, block)
    if args.check:
        if expected != text:
            print("parity/support_licensed.toml: the generated transport/interference block drifted")
            return 1
        print(f"transport/interference cells OK ({len(coordinates())} generated)")
        return 0
    if expected != text:
        LICENSED.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {len(coordinates())} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
