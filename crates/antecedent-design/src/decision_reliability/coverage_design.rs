//! The coverage-measurement design for every derived interval and set, as data.
//!
//! Nothing here measures anything. Each cell names the claim, the coordinate grid
//! (candidate-set size, Monte Carlo budget, selection rule, noise coupling), the truth
//! the replicates are scored against, the nominal levels and the boundary probes. The
//! truth is always known by construction: a closed form, an enumerated finite law, or a
//! simulation of [`KnownOptimumDesign`](super::calibration::KnownOptimumDesign) whose
//! true optimum and true values are fixed.
//!
//! A cell is the *whole procedure* (draw, estimate, select, interval), not the interval
//! formula alone.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

/// Where a cell's truth comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruthKind {
    /// A closed form or one-dimensional quadrature of known Gaussian candidates.
    ClosedForm,
    /// An enumerated finite law whose probabilities are exact sums.
    EnumeratedFiniteLaw,
    /// A simulation with a fixed, known true optimum and known true values.
    SimulationKnownOptimum,
}

/// One route's measurement design.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CoverageCell {
    /// Route the cell would license.
    pub route: &'static str,
    /// What is claimed and what is scored.
    pub claim: &'static str,
    /// Candidate-set sizes (considered candidates).
    pub candidate_set_sizes: &'static [u64],
    /// Monte Carlo draws per candidate.
    pub mc_budgets: &'static [u64],
    /// Selection rules.
    pub selection_rules: &'static [&'static str],
    /// Shared-noise correlations of the candidates (zero is separate streams).
    pub noise_correlations: &'static [f64],
    /// Truth family.
    pub truth: TruthKind,
    /// The truth, concretely.
    pub truth_description: &'static str,
    /// Nominal levels.
    pub nominal_levels: &'static [f64],
    /// Fewest independent replicates per grid point.
    pub min_replicates: u64,
    /// What counts as a pass at a grid point.
    pub pass_rule: &'static str,
    /// Points at which the construction is expected to be stressed.
    pub boundary_probes: &'static [&'static str],
}

const SIZES: &[u64] = &[2, 8, 64, 512];
const BUDGETS: &[u64] = &[100, 1_000, 10_000];
const RULES: &[&str] =
    &["maximize_expected_utility", "maximize_threshold_probability", "maximize_cvar"];
const RHOS: &[f64] = &[0.0, 0.5, 0.9];
const LEVELS: &[f64] = &[0.8, 0.9, 0.95];
const PASS: &str = "empirical coverage within the binomial band of the nominal level at the \
                    replicate count, at every point of the stated range; a point below the band \
                    is recorded as a boundary, not averaged away";

/// Every cell of the design, one per derived route.
#[must_use]
pub fn coverage_design() -> Vec<CoverageCell> {
    vec![
        ranking_probability_interval_cell(),
        optimality_confidence_set_cell(),
        selection_corrected_interval_cell(),
        held_out_interval_cell(),
        winners_curse_bound_cell(),
        utility_interval_cell(),
    ]
}

fn ranking_probability_interval_cell() -> CoverageCell {
    CoverageCell {
        route: "ranking_probability_interval",
        claim: "the Wilson interval at the effective sample size contains the probability \
                that a candidate is best, top-k, or better than the baseline under the \
                supplied (known) law",
        candidate_set_sizes: SIZES,
        mc_budgets: BUDGETS,
        selection_rules: &["maximize_expected_utility"],
        noise_correlations: RHOS,
        truth: TruthKind::ClosedForm,
        truth_description: "independent Gaussian candidates: P(best_j) = integral of the \
                            density of j times the product of the other cumulative \
                            distributions, by quadrature; two candidates: \
                            P(A > B) = Phi(delta / (sigma sqrt(2 (1 - rho))))",
        nominal_levels: LEVELS,
        min_replicates: 2_000,
        pass_rule: PASS,
        boundary_probes: &[
            "all true means equal (every best probability 1/K, ties exact)",
            "one dominant candidate (best probability near 1, Wilson boundary)",
            "gap at the Monte Carlo noise scale, delta = sigma / sqrt(n)",
            "effective sample size near the normal-approximation floor (autocorrelated \
             stream, budget 100)",
            "importance weights with Kish effective sample size 0.2 n",
        ],
    }
}

fn optimality_confidence_set_cell() -> CoverageCell {
    CoverageCell {
        route: "optimality_confidence_set",
        claim: "the set contains the true best retained candidate with at least the \
                nominal probability, for both the marginal Bonferroni and the paired \
                comparison constructions",
        candidate_set_sizes: SIZES,
        mc_budgets: BUDGETS,
        selection_rules: RULES,
        noise_correlations: RHOS,
        truth: TruthKind::SimulationKnownOptimum,
        truth_description: "fixed true means with one true optimum (and a tied-optimum \
                            variant); the set is scored on containing every true \
                            optimum, and its size is recorded as a power measure",
        nominal_levels: LEVELS,
        min_replicates: 2_000,
        pass_rule: PASS,
        boundary_probes: &[
            "candidate count 64 against 65 (paired to marginal construction switch)",
            "second best at one standard error below the best",
            "all candidates tied at the optimum",
            "adaptively generated set retained from a known generator: the set is scored \
             against the best retained candidate and, separately, the best considered",
        ],
    }
}

fn selection_corrected_interval_cell() -> CoverageCell {
    CoverageCell {
        route: "selection_corrected_interval",
        claim: "the simultaneous interval contains the true value of whichever candidate \
                the selection rule picked, at the nominal level",
        candidate_set_sizes: SIZES,
        mc_budgets: BUDGETS,
        selection_rules: RULES,
        noise_correlations: RHOS,
        truth: TruthKind::SimulationKnownOptimum,
        truth_description: "known true value of the selected candidate in each replicate; \
                            the ordinary in-sample interval is scored beside it to show \
                            the undercoverage the correction removes",
        nominal_levels: LEVELS,
        min_replicates: 2_000,
        pass_rule: PASS,
        boundary_probes: &[
            "all true means equal (maximal winner's curse)",
            "one candidate far above the rest (selection nearly deterministic)",
            "candidate count 512 at budget 100",
        ],
    }
}

fn held_out_interval_cell() -> CoverageCell {
    CoverageCell {
        route: "held_out_interval",
        claim: "the interval from the independent re-evaluation contains the true value \
                of the selected candidate",
        candidate_set_sizes: SIZES,
        mc_budgets: BUDGETS,
        selection_rules: RULES,
        noise_correlations: RHOS,
        truth: TruthKind::SimulationKnownOptimum,
        truth_description: "known true value of the selected candidate; validation stream \
                            differs from the selection stream in purpose and replicate",
        nominal_levels: LEVELS,
        min_replicates: 2_000,
        pass_rule: PASS,
        boundary_probes: &[
            "sample split with gap 0 on an autocorrelated stream (expected failure)",
            "validation stream already evaluated once (refusal, not coverage)",
        ],
    }
}

fn winners_curse_bound_cell() -> CoverageCell {
    CoverageCell {
        route: "winners_curse_bound",
        claim: "the reported bound is at least the mean upward bias of the in-sample \
                maximum over replicates",
        candidate_set_sizes: SIZES,
        mc_budgets: BUDGETS,
        selection_rules: &["maximize_expected_utility"],
        noise_correlations: RHOS,
        truth: TruthKind::ClosedForm,
        truth_description: "Gaussian candidates with equal true means: the expected bias \
                            is the expected maximum of K standard normals times the \
                            standard error (Tippett table)",
        nominal_levels: &[1.0],
        min_replicates: 5_000,
        pass_rule: "mean realized bias does not exceed the reported bound beyond its \
                    Monte Carlo error at any grid point",
        boundary_probes: &[
            "negatively correlated candidate noise (the exact independent value is not a \
             bound there; the union bound must hold)",
            "padded unseen candidates from a search record",
        ],
    }
}

fn utility_interval_cell() -> CoverageCell {
    CoverageCell {
        route: "utility_interval",
        claim: "the Wald interval at the reported standard error contains the true \
                utility functional of a pre-specified candidate",
        candidate_set_sizes: &[2],
        mc_budgets: BUDGETS,
        selection_rules: &[
            "maximize_expected_utility",
            "maximize_threshold_probability",
            "maximize_cvar",
        ],
        noise_correlations: &[0.0],
        truth: TruthKind::ClosedForm,
        truth_description: "known Gaussian: mean, threshold exceedance Phi, lower-tail \
                            CVaR mu - sigma phi(z_alpha) / alpha",
        nominal_levels: LEVELS,
        min_replicates: 5_000,
        pass_rule: PASS,
        boundary_probes: &[
            "quantile and CVaR at tail mass 0.01 with budget 100 (few tail draws)",
            "autocorrelated stream with batch means",
        ],
    }
}
