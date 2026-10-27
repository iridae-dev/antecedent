#!/usr/bin/env python3
"""Triage of the provenance records that lack a completed bounded method review.

Writes ``parity/provenance_review_triage.toml`` (``--write``) or checks that the committed
file still matches the records (default).  The classification is a *reading aid*, not a review:

* A record with a ``[method_review]`` table is ``reviewed`` and keeps that table's own scope.
* A record that lists primary papers is ``substantive_method`` until its implementation is
  read against them; the triage records only the shared-source group, the modules and the
  number of admitted method specifications that link it.  Nothing here says the
  implementation follows the paper.
* A record with no paper is classified from the judgment written in the record itself (the
  comment and ``implementation_deviations``) plus the explicit tables below; each such
  record states why the classification holds.  An empty ``papers`` list is not by itself
  accepted as justification.

Priority orders the review work: a record linked by admitted licensed method specifications
that can change a numerical result ranks above contracts, closed variants and bookkeeping.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from _repo import REPO_ROOT

OUT = REPO_ROOT / "parity" / "provenance_review_triage.toml"

CLOSED = (
    "closed variant",
    "closed gate",
    "intentionally implements nothing",
    "intentionally publishes nothing",
)
CONTRACT_PREFIXES = (
    "composition.",
    "composition_verification.",
    "provider_callback.",
    "interface_contract.",
    "external.",
    "evidence_ledger.",
    "recalculation.",
    "distribution.",
    "scientific_bridge.",
    "support.",
    "measurement.model_contract",
    "inference.measured_scalar",
    "design.evidence_obligations",
    "design.evidence_plan",
    "design.external_signal",
    "design.proposal_receipt",
    "design.proposal_arrival",
    "design.ranking_artifact",
    "design.study_candidate",
    "design.rollout_state",
    "design.identification_repair",
    "design.conditional_study_ranking",
    "design.prior_signal_planning",
    "decision.admissibility",
    "decision.contract",
    "decision.scenario_claims",
    "decision.structural_policy",
    "decision.structural_robustness",
    "decision.transported_source",
    "decision.sensitivity_composition",
    "adaptive_inference.analysis_contract",
    "adaptive_inference.procedure_matrix",
    "adaptive_inference.guarantees_closed",
    "cohort_decision.law_valued_request",
    "identify.auto",
    "identify.envelope",
    "identify.intervention_support",
    "transport.handoff_artifact",
    "transport.provider_source_admission",
    "transport.revalidation",
    "transport.assumption_range",
    "estimate.compact_runtime_export",
    "discovery.per_population",
    "discovery.selective_reuse",
    "temporal.effect_constancy_consumers",
    "sensitivity.source_lineage",
    "response.joint_panel_reference",
    "validate.dbn_posterior",
    "estimate.calibration_coverage",
    "predictive_assessment.overlap_check",
)
# No-paper records whose numerics are derived in the repository from textbook identities.  They
# carry a numerical result, so they are review targets even though no paper introduces them.
DERIVED = {
    "adaptive_inference.procedure_law": "exact finite law of a batched adaptive Bernoulli procedure by enumeration; correctness is by construction of the recursion and needs an independent enumeration, not a paper",
    "adaptive_inference.procedure_repeated_measures": "Gaussian design-effect and unit-summary variance in closed form; textbook cluster variance algebra",
    "joint_law.aligned_nonlinear_functional": "delta method over an aligned joint law",
    "joint_law.composed_interval": "delta method and law of total variance over an aligned joint law",
    "joint_law.known_weight_mixture": "mixture mean and variance with known weights; law of total variance",
    "joint_law.linear_moment_propagation": "linear moment propagation; textbook",
    "joint_law.transported_composition": "delta method over a composed transported law",
    "cohort_decision.interval": "Student-t interval of a linear combination of OLS coefficients; textbook linear-model theory",
    "cohort_decision.mean_contrast": "weighted mean of conditional contrast predictions of a retained model; original construction",
    "design.effect_width_reduction": "signed posterior-width reduction se_before - se_after from Gram-matrix arithmetic",
    "decision.functionals": "finite weighted expectation, variance, threshold probability, left-inverse quantile; arithmetic on a supplied law",
    "decision.functional_inverse": "finite grid enumeration of the functional",
    "decision_reliability.ranking_statement": "plain Monte Carlo frequencies with ties split equally",
    "estimate.quadratic_dose": "finite weighted least-squares polynomial reproduction argument",
    "response.joint_panel": "finite-lag Gaussian panel family constructed in the repository",
    "transport.joint_bayesian": "conjugate Gaussian algebra",
    "transport.learned_joint": "conjugate Gaussian algebra",
    "model.discretized_linear_gaussian_scm": "equal-probability-quantile discretization of a linear Gaussian model with a computed bound",
    "temporal.uncertain_initial_state": "exact finite mixture over a supplied initial-state law",
    "inverse.observational_scenarios": "Bayes' rule P(X in scenario | Y event) on a supplied joint law",
    "observation_query.calibrated_probability": "equal-tailed posterior quantile interval of a declared model's probability",
    "counterfactual.whole_path_uncertainty": "accounting of uncertainty sources of a finite Markovian cross-world row",
    "counterfactual.transport_sampled_interval": "delta-method Wald interval of the open record clamped to [0,1]",
    "attribution.structure": "structure player swapping a node's parent set with its refit mechanism",
    "sensitivity.joint_uncertainty": "endpoint percentile bootstrap of a joint sensitivity range",
    "temporal.new_period_refresh": "ordered comparison of declared premises (window-identity bookkeeping)",
}
# No-paper records for which a primary or textbook source plausibly exists and should be
# located before the record is accepted as justified.
SOURCE_CANDIDATES = {
    "ci.knn_dependence": "permutation test on a nearest-neighbour distance statistic; locate the source of the kNN-distance dependence statistic or declare it original",
    "state.incremental_ols": "rank-one / sufficient-statistic least-squares updating; textbook (e.g. recursive least squares) source should be named",
    "state.rolling_mechanism_diag": "declined Page 1954 explicitly; fixed-window max-|partial sum| scan is the record's own judgment",
    "graph.unfold": "unrolling a stationary lagged template into a finite window; graph-theory/folklore, one upstream observed",
    "identify.tiered": "tier-closure on the background closure ADMG; related to tiered background knowledge in the literature",
    "predictive_assessment.distribution_drift": "two-sample Kolmogorov-Smirnov distance and population stability index; textbook/industry statistics",
    "validate.graph_refute": "leave-one-out adjustment-covariate sensitivity; refuter folklore",
    "validate.robustness_refuters": "placebo-treatment, dummy-outcome, random-common-cause and data-subset refuters; the observed upstream library is the origin of the names",
    "validate.target_native_refuters": "target-native refuters; implementation-defined",
    "validate.temporal_mediation": "perturbation refits of mediation regressions",
    "validate.functional": "Bernoulli row-subset refit of a frozen path functional",
    "estimate.bayesian_iv": "joint Gaussian IV model with fixed loading; Bayesian IV literature exists and should be checked",
}


def load_records() -> dict[str, tuple[dict, str]]:
    out = {}
    for path in sorted((REPO_ROOT / "provenance").glob("*.toml")):
        if path.name == "_template.toml":
            continue
        text = path.read_text(encoding="utf-8")
        out[path.stem] = (tomllib.loads(text), text)
    return out


def spec_links() -> dict[str, dict[str, int]]:
    specs = tomllib.loads(
        (REPO_ROOT / "parity/method_specs/specs.toml").read_text(encoding="utf-8")
    )["spec"]
    links: dict[str, dict[str, int]] = defaultdict(lambda: {"admitted": 0, "licensed": 0})
    for spec in specs:
        for stem in spec.get("provenance", []):
            if spec.get("review_status") == "admitted":
                links[stem]["admitted"] += 1
            if spec.get("admission") == "licensed":
                links[stem]["licensed"] += 1
    return links


def comment_text(text: str) -> str:
    return " ".join(
        line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")
    )


def classify(stem: str, record: dict, text: str) -> tuple[str, str]:
    review = record.get("method_review")
    if review:
        return "reviewed", "bounded method_review present (digest-bound by the provenance gate)"
    comment = comment_text(text)
    low = (comment + " " + " ".join(record.get("implementation_deviations", []))).lower()
    papers = record.get("papers", [])
    if papers:
        return (
            "substantive_method",
            "lists primary papers; implementation correspondence not yet read",
        )
    if any(marker in low for marker in CLOSED):
        return "closed_variant", "the record states that no method is implemented and the claim is refused or left open"
    if stem in DERIVED:
        return "derived_in_repository", DERIVED[stem]
    if stem in SOURCE_CANDIDATES:
        return "source_candidate", SOURCE_CANDIDATES[stem]
    if stem.endswith(".bayesian"):
        return (
            "derived_in_repository",
            "posterior mean (conjugate normal or Dirichlet) of the quantity the frequentist counterpart already computes; the conjugate algebra and the unchanged plug-in variance need a derivation check",
        )
    if stem.startswith(CONTRACT_PREFIXES):
        return (
            "wrapper_contract",
            "declaration, binding, lineage or orchestration over other methods; no numerical method of its own",
        )
    return "unclassified", "no-paper record not covered by any rule; needs a reading"


def priority(cls: str, links: dict[str, int], stem: str) -> int:
    if cls == "reviewed":
        return 0
    numeric = cls in ("substantive_method", "derived_in_repository", "source_candidate")
    prefix = stem.split(".")[0]
    core = prefix in {"estimate", "identify", "discovery", "transport", "stats", "ci", "prob", "kernels", "adaptive_inference"}
    if numeric and links["licensed"] and core:
        return 1
    if numeric and links["admitted"]:
        return 2
    if numeric:
        return 3
    return 4


def build() -> str:
    records = load_records()
    links = spec_links()
    paper_key: dict[str, list[str]] = defaultdict(list)
    rows = []
    for stem, (record, text) in records.items():
        cls, why = classify(stem, record, text)
        keys = sorted(
            p.get("doi") or p.get("url") or p["title"] for p in record.get("papers", [])
        )
        group = keys[0] if keys else ""
        if cls != "reviewed":
            for key in keys:
                paper_key[key].append(stem)
        rows.append((stem, record, cls, why, group, links.get(stem, {"admitted": 0, "licensed": 0})))
    lines = [
        "# Triage of provenance records against completed bounded method reviews.",
        "# Generated by scripts/provenance_review_triage.py; a reading aid, not a review.",
        "# `class` is a hypothesis about what review the record needs. `substantive_method` means the",
        "# record lists primary papers and its implementation has not yet been read against them.",
        "# `priority` 1 = numerical records linked by licensed method specifications in the core",
        "# method families; 2 = linked by an admitted specification; 3 = numerical, unlinked;",
        "# 4 = contract, closed variant, bookkeeping or unclassified.",
        "",
    ]
    counts: dict[str, int] = defaultdict(int)
    for stem, record, cls, why, group, link in sorted(rows):
        counts[cls] += 1
        pr = priority(cls, link, stem)
        shared = sorted(
            {s for k in (p.get("doi") or p.get("url") or p["title"] for p in record.get("papers", [])) for s in paper_key.get(k, []) if s != stem}
        )
        lines += [
            "[[record]]",
            f'id = "{stem}"',
            f'class = "{cls}"',
            f"priority = {pr}",
            f"admitted_specs = {link['admitted']}",
            f"licensed_specs = {link['licensed']}",
            f"papers = {len(record.get('papers', []))}",
            f"shares_primary_source_with = {len(shared)}",
            'basis = "' + why.replace('"', "'") + '"',
            "",
        ]
    header = ["# Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))]
    cut = lines.index("")
    return "\n".join(lines[:cut] + header + lines[cut:]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    text = build()
    if args.write:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO_ROOT)}")
        return 0
    if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
        print("provenance review triage is stale: run scripts/provenance_review_triage.py --write")
        return 1
    print("provenance review triage: current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
