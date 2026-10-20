//! Whole-procedure coverage of the published decision-reliability routes against a known
//! optimum: draw per-candidate utilities on a stream, derive the three uncertainty
//! components, and score the reported statement against the truth of the known design.
//!
//! The data-generating process is [`KnownOptimumDesign`] with one leading candidate
//! (`c00000`, true utility `lead * s`) and `k - 1` equal candidates at zero, where `s` is
//! the per-draw standard deviation of the candidate-specific noise (`sd * sqrt(1 - rho)`
//! for a shared stream of correlation `rho`, `sd` for separate streams).
//!
//! Scored statements, one test family per route (the record is keyed by the runtime
//! coordinate through [`calibration_basis`], never by a hand-declared key):
//!
//! * `ranking_probability`: the Wilson interval at the effective sample size for the
//!   probability that the leader is best. Truth is the closed-form quadrature
//!   `integral phi(z) Phi(z + lead)^(k-1) dz` (shared streams only: independently drawn
//!   candidates have no draw index to rank by).
//! * `utility_interval`: the Wald interval of the leader's expected utility. Truth is its
//!   true mean.
//! * `confidence_set`: the Bonferroni confidence set for the best candidate contains the
//!   leader. Conservative by construction, so asserted one-sided.
//! * `selection_corrected`: the simultaneous interval contains the true utility of
//!   whichever candidate the rule selected. Conservative, asserted one-sided.
//!
//! Sample-size grid: the per-candidate Monte Carlo budget is `grid_n(base)`, that is half
//! the base, the base and twice the base. Records are keyed by the candidate-count band
//! (`candidate_count_band`), so the designs below visit every band at several counts:
//!
//! * base 1000 (500, 1000, 2000 draws) at 2, 8 and 64 candidates;
//! * base 2000 (1000, 2000, 4000 draws) at 3, 5, 16, 32 and 100 candidates;
//! * budget sweeps at 16 candidates (bases 500, 8000, 16000 and 32768, reaching 250 to
//!   65536 draws) and at the band extremes (3 and 100 candidates; 32 at the low end);
//! * exact ties (sixteen equal true means) and near ties (a leader a tenth of a noise
//!   unit above fifteen equal candidates) for the routes that use the candidate count.
//!
//! Couplings: shared streams with correlation 0 and 0.5, and separate streams for the
//! routes that do not rank. Named boundaries (recorded under a separate functional so
//! they never gate a route): all candidates equal at 64 candidates (maximal winner's
//! curse), a dominant leader (probability near one) and an autocorrelated-stream stress
//! is deliberately not measured: autocorrelated and importance-weighted carriers stay
//! refused.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::ProbabilityQuantity;
use antecedent_design::decision_reliability::{
    DecisionReliabilityArtifact, DecisionReliabilityExpectation, DerivedUncertainty,
    IntervalRequest, KnownOptimumDesign, NoiseCoupling, PublishedRoute, RankingRequest,
    ReliabilityRequest, StreamId, StreamPurpose, calibration_basis, consume, derive_uncertainty,
    simulate_selection_draws,
};
use antecedent_kernels::{norm_cdf, norm_pdf};
use calibration::{
    Construction, CoverageSide, CoverageTally, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};

const LEVEL: f64 = 0.95;
const BASE_BUDGET: usize = 1000;
const SD: f64 = 1.0;
const LEADER: &str = "c00000";

const SHARED: [(&str, NoiseCoupling); 2] = [
    ("shared_rho_0.0", NoiseCoupling::SharedStream { correlation: 0.0 }),
    ("shared_rho_0.5", NoiseCoupling::SharedStream { correlation: 0.5 }),
];
const ALL: [(&str, NoiseCoupling); 3] = [
    ("separate_streams", NoiseCoupling::SeparateStreams),
    ("shared_rho_0.0", NoiseCoupling::SharedStream { correlation: 0.0 }),
    ("shared_rho_0.5", NoiseCoupling::SharedStream { correlation: 0.5 }),
];

/// Candidate-specific noise standard deviation of a design.
fn own_sd(coupling: NoiseCoupling) -> f64 {
    match coupling {
        NoiseCoupling::SharedStream { correlation } => SD * (1.0 - correlation).sqrt(),
        NoiseCoupling::SeparateStreams => SD,
    }
}

/// Leader's true probability of being best per draw: composite Simpson quadrature of
/// `phi(z) Phi(z + lead)^(k - 1)` over `[-9, 9]`.
fn leader_best_probability(k: usize, lead: f64) -> f64 {
    let steps = 4000_usize;
    let (lo, hi) = (-9.0_f64, 9.0_f64);
    let h = (hi - lo) / steps as f64;
    let f = |z: f64| norm_pdf(z) * norm_cdf(z + lead).powi(i32::try_from(k - 1).unwrap_or(0));
    let mut sum = f(lo) + f(hi);
    for i in 1..steps {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * f(lo + h * i as f64);
    }
    sum * h / 3.0
}

fn leader_design(
    k: usize,
    lead_in_noise_units: f64,
    budget: usize,
    coupling: NoiseCoupling,
) -> KnownOptimumDesign {
    let mut true_means = vec![0.0; k];
    true_means[0] = lead_in_noise_units * own_sd(coupling);
    KnownOptimumDesign { true_means, noise_sd: SD, n_draws: budget, coupling }
}

fn candidate_index(id: &str) -> usize {
    id.trim_start_matches('c').parse().expect("known-optimum candidate id")
}

/// One replicate: the derived uncertainty of a design's selection stream.
fn derive(
    design: &KnownOptimumDesign,
    run_seed: u64,
    replicate: u64,
    ranking: bool,
) -> Option<DerivedUncertainty> {
    let stream = StreamId::new(run_seed, StreamPurpose::Selection, replicate, "known_optimum");
    let draws = simulate_selection_draws(design, &stream).ok()?;
    let request = ReliabilityRequest {
        ranking: ranking.then(RankingRequest::default),
        ..ReliabilityRequest::expected_utility()
    };
    let ctx = ExecutionContext::for_tests(antecedent_core::derive_stream(run_seed, replicate));
    if replicate == 0 {
        // Exercise the real artifact producer and independent consumer on the existing
        // first replicate. No extra draws, stream advances or scored calculations.
        let artifact = DecisionReliabilityArtifact::new(draws.clone(), None, request.clone(), &ctx)
            .expect("first-replicate reliability artifact");
        let bytes = artifact
            .to_bytes("calibration-first-replicate", &ctx)
            .expect("first-replicate artifact export");
        let expected = DecisionReliabilityExpectation {
            draws_digest: Some(draws.digest_hex()),
            candidate_set_digest: Some(draws.candidate_set_digest()),
            // Separate-stream candidates have distinct per-candidate coordinates;
            // their complete stream declarations remain bound by draws_digest.
            selection_stream: match design.coupling {
                NoiseCoupling::SharedStream { .. } => Some(stream),
                NoiseCoupling::SeparateStreams => None,
            },
            utility_unit: Some(draws.utility_unit.clone()),
            ..DecisionReliabilityExpectation::default()
        };
        let verified = consume(&bytes, &expected, &ctx)
            .expect("independent first-replicate artifact consumer");
        assert_eq!(verified, artifact);
        eprintln!(
            "artifact-replay-witness decision_reliability_v1 replicate=0 draws={} digest={}",
            verified.draws.digest_hex(),
            verified.digest,
        );
    }
    derive_uncertainty(&draws, None, &request, &IntervalRequest::marginal(LEVEL), &ctx).ok()
}

/// What one replicate scored: the construction the runtime keyed the interval under and
/// the scope facts, the interval, and the truth.
struct Scored {
    construction: Construction,
    scope: ScopeFacts,
    interval: (f64, f64),
    truth: f64,
}

fn bind_basis(
    route: PublishedRoute,
    derived: &DerivedUncertainty,
    functional_suffix: &str,
) -> Option<(Construction, ScopeFacts)> {
    let basis = calibration_basis(route, &derived.coordinate, LEVEL)?;
    let key = basis.key;
    Some((
        Construction {
            query: key.query,
            graph_class: key.graph_class,
            structure: key.structure,
            modality: key.modality,
            inference: key.inference,
            estimator: key.estimator,
            interval_method: key.interval_method,
            se_kind: key.se_kind,
            dependence: key.dependence,
            posterior: key.posterior,
            functional: format!("{}{functional_suffix}", key.functional),
            identification: key.identification,
            reported_level: key.level,
        },
        ScopeFacts {
            row_count: basis.scope.row_count,
            replicates_ok: basis.scope.replicates_ok,
            posterior_draws: basis.scope.posterior_draws,
            unidentified_mass: basis.scope.unidentified_mass,
        },
    ))
}

fn score(
    route: PublishedRoute,
    design: &KnownOptimumDesign,
    lead_in_noise_units: f64,
    run_seed: u64,
    replicate: u64,
    suffix: &str,
) -> Option<Scored> {
    let k = design.true_means.len();
    let ranking = route == PublishedRoute::RankingProbability;
    let derived = derive(design, run_seed, replicate, ranking)?;
    let (construction, scope) = bind_basis(route, &derived, suffix)?;
    let (interval, truth) = match route {
        PublishedRoute::RankingProbability => {
            let entry = derived
                .numerical
                .probabilities
                .iter()
                .find(|p| p.id == LEADER && matches!(p.quantity, ProbabilityQuantity::Best))?;
            (
                (entry.interval.lower, entry.interval.upper),
                leader_best_probability(k, lead_in_noise_units),
            )
        }
        PublishedRoute::UtilityInterval => {
            let entry = derived.numerical.utilities.iter().find(|u| u.id == LEADER)?;
            ((entry.interval.lower, entry.interval.upper), design.true_means[0])
        }
        PublishedRoute::OptimalityConfidenceSet => {
            // Membership is scored as containment of the truth in a degenerate interval:
            // the unit interval holds 0.5, the shifted one does not.
            let member = derived.selection.confidence_set.members.iter().any(|m| m == LEADER);
            ((if member { 0.0 } else { 0.75 }, 1.0), 0.5)
        }
        PublishedRoute::SelectionCorrectedInterval => {
            let corrected = derived.selection.corrected.as_ref()?;
            let truth = design.true_means[candidate_index(&corrected.selected_id)];
            ((corrected.lower, corrected.upper), truth)
        }
    };
    Some(Scored { construction, scope, interval, truth })
}

fn one_sided_target(route: PublishedRoute) -> &'static str {
    match route {
        PublishedRoute::OptimalityConfidenceSet => {
            "the confidence set contains the true best candidate at least at the nominal level"
        }
        _ => {
            "the simultaneous interval contains the true value of the selected candidate at least at the nominal level"
        }
    }
}

fn run_gated(
    route: PublishedRoute,
    test: &'static str,
    dgp: &'static str,
    k: usize,
    lead: f64,
    base_budget: usize,
    couplings: &[(&'static str, NoiseCoupling)],
) {
    run_gated_at_offset(route, test, dgp, k, lead, base_budget, couplings, 0.0);
}

/// [`run_gated`] with every candidate's true utility shifted by `offset`: coverage is scored
/// against the shifted truths, so an offset far above the noise measures the interval where a
/// one-pass sum of squares would lose every digit of the standard error.
#[allow(clippy::too_many_arguments, reason = "one design is the whole of its coordinates")]
fn run_gated_at_offset(
    route: PublishedRoute,
    test: &'static str,
    dgp: &'static str,
    k: usize,
    lead: f64,
    base_budget: usize,
    couplings: &[(&'static str, NoiseCoupling)],
    offset: f64,
) {
    let budget = grid_n(base_budget);
    let (_, interval_method) = route.record_construction();
    for (cell, &(label, coupling)) in couplings.iter().enumerate() {
        let mut design = leader_design(k, lead, budget, coupling);
        design.true_means.iter_mut().for_each(|mean| *mean += offset);
        // The base-1000 designs keep the seeds they were first measured with.
        let base_salt = if base_budget == BASE_BUDGET { 0 } else { (base_budget as u64) << 20 };
        let run_seed = grid_seed(0x4d00_0000 + base_salt + (k as u64) * 64 + cell as u64);
        let mut tally =
            CoverageTally::for_record(RecordKey { test, dgp, interval: interval_method }, LEVEL)
                .labelled(label);
        let scored = map_replicates(n_sim(), |replicate| {
            score(route, &design, lead, run_seed, replicate, "")
        });
        for outcome in scored {
            match outcome {
                Some(one) => {
                    tally.bind(&one.construction, one.scope);
                    tally.record(Some(one.interval), one.truth);
                }
                None => tally.skip(),
            }
        }
        match route {
            PublishedRoute::RankingProbability | PublishedRoute::UtilityInterval => tally.assert(),
            PublishedRoute::OptimalityConfidenceSet
            | PublishedRoute::SelectionCorrectedInterval => {
                tally.assert_one_sided(CoverageSide::Upper, one_sided_target(route));
            }
        }
    }
}

fn run_named_boundary(
    route: PublishedRoute,
    test: &'static str,
    dgp: &'static str,
    k: usize,
    lead: f64,
    suffix: &'static str,
) {
    let budget = grid_n(BASE_BUDGET);
    let (_, interval_method) = route.record_construction();
    let coupling = NoiseCoupling::SharedStream { correlation: 0.5 };
    let design = leader_design(k, lead, budget, coupling);
    let run_seed = grid_seed(0x4e00_0000 + k as u64);
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: interval_method }, LEVEL)
            .labelled("shared_rho_0.5");
    let scored = map_replicates(n_sim(), |replicate| {
        score(route, &design, lead, run_seed, replicate, suffix)
    });
    for outcome in scored {
        match outcome {
            Some(one) => {
                tally.bind(&one.construction, one.scope);
                tally.record(Some(one.interval), one.truth);
            }
            None => tally.skip(),
        }
    }
    tally.emit_named_boundary();
}

macro_rules! gated {
    ($name:ident, $route:expr, $k:expr, $lead:expr, $couplings:expr, $dgp:literal) => {
        gated_at!($name, $route, $k, $lead, BASE_BUDGET, $couplings, $dgp);
    };
}

macro_rules! gated_at {
    ($name:ident, $route:expr, $k:expr, $lead:expr, $base:expr, $couplings:expr, $dgp:literal) => {
        #[test]
        #[ignore = "calibration: run only at the final measurement step"]
        fn $name() {
            run_gated($route, stringify!($name), $dgp, $k, $lead, $base, &$couplings);
        }
    };
}

// The leader's gap in units of the candidate-specific noise: about even odds against two
// candidates, and still a clear but not certain leader among eight and sixty-four.
gated!(
    ranking_probability_two_candidates_l95,
    PublishedRoute::RankingProbability,
    2,
    0.5,
    SHARED,
    "leader_plus_one_equal_candidate_gaussian_utilities_shared_stream"
);
gated!(
    ranking_probability_eight_candidates_l95,
    PublishedRoute::RankingProbability,
    8,
    1.5,
    SHARED,
    "leader_plus_seven_equal_candidates_gaussian_utilities_shared_stream"
);
gated!(
    ranking_probability_sixty_four_candidates_l95,
    PublishedRoute::RankingProbability,
    64,
    2.5,
    SHARED,
    "leader_plus_sixty_three_equal_candidates_gaussian_utilities_shared_stream"
);
gated!(
    utility_interval_two_candidates_l95,
    PublishedRoute::UtilityInterval,
    2,
    0.5,
    ALL,
    "leader_plus_one_equal_candidate_gaussian_utilities"
);
gated!(
    utility_interval_eight_candidates_l95,
    PublishedRoute::UtilityInterval,
    8,
    1.5,
    ALL,
    "leader_plus_seven_equal_candidates_gaussian_utilities"
);
gated!(
    utility_interval_sixty_four_candidates_l95,
    PublishedRoute::UtilityInterval,
    64,
    2.5,
    ALL,
    "leader_plus_sixty_three_equal_candidates_gaussian_utilities"
);
gated!(
    confidence_set_two_candidates_l95,
    PublishedRoute::OptimalityConfidenceSet,
    2,
    0.5,
    ALL,
    "leader_plus_one_equal_candidate_gaussian_utilities"
);
gated!(
    confidence_set_eight_candidates_l95,
    PublishedRoute::OptimalityConfidenceSet,
    8,
    1.5,
    ALL,
    "leader_plus_seven_equal_candidates_gaussian_utilities"
);
gated!(
    confidence_set_sixty_four_candidates_l95,
    PublishedRoute::OptimalityConfidenceSet,
    64,
    2.5,
    ALL,
    "leader_plus_sixty_three_equal_candidates_gaussian_utilities"
);
gated!(
    selection_corrected_two_candidates_l95,
    PublishedRoute::SelectionCorrectedInterval,
    2,
    0.5,
    ALL,
    "leader_plus_one_equal_candidate_gaussian_utilities"
);
gated!(
    selection_corrected_eight_candidates_l95,
    PublishedRoute::SelectionCorrectedInterval,
    8,
    1.5,
    ALL,
    "leader_plus_seven_equal_candidates_gaussian_utilities"
);
gated!(
    selection_corrected_sixty_four_candidates_l95,
    PublishedRoute::SelectionCorrectedInterval,
    64,
    2.5,
    ALL,
    "leader_plus_sixty_three_equal_candidates_gaussian_utilities"
);

// Candidate counts between and beyond the first designs, one budget grid (1000, 2000, 4000).
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_three_candidates_l95() {
    run_gated(
        PublishedRoute::RankingProbability,
        "ranking_probability_three_candidates_l95",
        "leader_plus_two_equal_candidates_gaussian_utilities_shared_stream",
        3,
        1.0,
        2000,
        &SHARED,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn utility_interval_three_candidates_l95() {
    run_gated(
        PublishedRoute::UtilityInterval,
        "utility_interval_three_candidates_l95",
        "leader_plus_two_equal_candidates_gaussian_utilities",
        3,
        1.0,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_three_candidates_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_three_candidates_l95",
        "leader_plus_two_equal_candidates_gaussian_utilities",
        3,
        1.0,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_three_candidates_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_three_candidates_l95",
        "leader_plus_two_equal_candidates_gaussian_utilities",
        3,
        1.0,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_five_candidates_l95() {
    run_gated(
        PublishedRoute::RankingProbability,
        "ranking_probability_five_candidates_l95",
        "leader_plus_four_equal_candidates_gaussian_utilities_shared_stream",
        5,
        1.25,
        2000,
        &SHARED,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn utility_interval_five_candidates_l95() {
    run_gated(
        PublishedRoute::UtilityInterval,
        "utility_interval_five_candidates_l95",
        "leader_plus_four_equal_candidates_gaussian_utilities",
        5,
        1.25,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_five_candidates_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_five_candidates_l95",
        "leader_plus_four_equal_candidates_gaussian_utilities",
        5,
        1.25,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_five_candidates_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_five_candidates_l95",
        "leader_plus_four_equal_candidates_gaussian_utilities",
        5,
        1.25,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_sixteen_candidates_l95() {
    run_gated(
        PublishedRoute::RankingProbability,
        "ranking_probability_sixteen_candidates_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities_shared_stream",
        16,
        2.0,
        2000,
        &SHARED,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn utility_interval_sixteen_candidates_l95() {
    run_gated(
        PublishedRoute::UtilityInterval,
        "utility_interval_sixteen_candidates_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_sixteen_candidates_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_sixteen_candidates_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_sixteen_candidates_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_sixteen_candidates_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_thirty_two_candidates_l95() {
    run_gated(
        PublishedRoute::RankingProbability,
        "ranking_probability_thirty_two_candidates_l95",
        "leader_plus_thirty_one_equal_candidates_gaussian_utilities_shared_stream",
        32,
        2.25,
        2000,
        &SHARED,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn utility_interval_thirty_two_candidates_l95() {
    run_gated(
        PublishedRoute::UtilityInterval,
        "utility_interval_thirty_two_candidates_l95",
        "leader_plus_thirty_one_equal_candidates_gaussian_utilities",
        32,
        2.25,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_thirty_two_candidates_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_thirty_two_candidates_l95",
        "leader_plus_thirty_one_equal_candidates_gaussian_utilities",
        32,
        2.25,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_thirty_two_candidates_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_thirty_two_candidates_l95",
        "leader_plus_thirty_one_equal_candidates_gaussian_utilities",
        32,
        2.25,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_one_hundred_candidates_l95() {
    run_gated(
        PublishedRoute::RankingProbability,
        "ranking_probability_one_hundred_candidates_l95",
        "leader_plus_ninety_nine_equal_candidates_gaussian_utilities_shared_stream",
        100,
        2.75,
        2000,
        &SHARED,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn utility_interval_one_hundred_candidates_l95() {
    run_gated(
        PublishedRoute::UtilityInterval,
        "utility_interval_one_hundred_candidates_l95",
        "leader_plus_ninety_nine_equal_candidates_gaussian_utilities",
        100,
        2.75,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_one_hundred_candidates_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_one_hundred_candidates_l95",
        "leader_plus_ninety_nine_equal_candidates_gaussian_utilities",
        100,
        2.75,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_one_hundred_candidates_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_one_hundred_candidates_l95",
        "leader_plus_ninety_nine_equal_candidates_gaussian_utilities",
        100,
        2.75,
        2000,
        &ALL,
    );
}

// Budget sweeps: 16 candidates over every route, the band extremes for the routes whose
// correction depends on the candidate count.
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_sixteen_candidates_budget_500_l95() {
    run_gated(
        PublishedRoute::RankingProbability,
        "ranking_probability_sixteen_candidates_budget_500_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities_shared_stream",
        16,
        2.0,
        500,
        &SHARED,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn utility_interval_sixteen_candidates_budget_500_l95() {
    run_gated(
        PublishedRoute::UtilityInterval,
        "utility_interval_sixteen_candidates_budget_500_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        500,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_sixteen_candidates_budget_500_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_sixteen_candidates_budget_500_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        500,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_sixteen_candidates_budget_500_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_sixteen_candidates_budget_500_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        500,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_sixteen_candidates_budget_8000_l95() {
    run_gated(
        PublishedRoute::RankingProbability,
        "ranking_probability_sixteen_candidates_budget_8000_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities_shared_stream",
        16,
        2.0,
        8000,
        &SHARED,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn utility_interval_sixteen_candidates_budget_8000_l95() {
    run_gated(
        PublishedRoute::UtilityInterval,
        "utility_interval_sixteen_candidates_budget_8000_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        8000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_sixteen_candidates_budget_8000_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_sixteen_candidates_budget_8000_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        8000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_sixteen_candidates_budget_8000_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_sixteen_candidates_budget_8000_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        8000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_sixteen_candidates_budget_16000_l95() {
    run_gated(
        PublishedRoute::RankingProbability,
        "ranking_probability_sixteen_candidates_budget_16000_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities_shared_stream",
        16,
        2.0,
        16000,
        &SHARED,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn utility_interval_sixteen_candidates_budget_16000_l95() {
    run_gated(
        PublishedRoute::UtilityInterval,
        "utility_interval_sixteen_candidates_budget_16000_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        16000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_sixteen_candidates_budget_16000_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_sixteen_candidates_budget_16000_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        16000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_sixteen_candidates_budget_16000_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_sixteen_candidates_budget_16000_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        16000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_sixteen_candidates_budget_32768_l95() {
    run_gated(
        PublishedRoute::RankingProbability,
        "ranking_probability_sixteen_candidates_budget_32768_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities_shared_stream",
        16,
        2.0,
        32768,
        &SHARED,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn utility_interval_sixteen_candidates_budget_32768_l95() {
    run_gated(
        PublishedRoute::UtilityInterval,
        "utility_interval_sixteen_candidates_budget_32768_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        32768,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_sixteen_candidates_budget_32768_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_sixteen_candidates_budget_32768_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        32768,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_sixteen_candidates_budget_32768_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_sixteen_candidates_budget_32768_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities",
        16,
        2.0,
        32768,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_three_candidates_budget_500_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_three_candidates_budget_500_l95",
        "leader_plus_two_equal_candidates_gaussian_utilities",
        3,
        1.0,
        500,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_three_candidates_budget_500_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_three_candidates_budget_500_l95",
        "leader_plus_two_equal_candidates_gaussian_utilities",
        3,
        1.0,
        500,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_three_candidates_budget_32768_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_three_candidates_budget_32768_l95",
        "leader_plus_two_equal_candidates_gaussian_utilities",
        3,
        1.0,
        32768,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_three_candidates_budget_32768_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_three_candidates_budget_32768_l95",
        "leader_plus_two_equal_candidates_gaussian_utilities",
        3,
        1.0,
        32768,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_thirty_two_candidates_budget_500_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_thirty_two_candidates_budget_500_l95",
        "leader_plus_thirty_one_equal_candidates_gaussian_utilities",
        32,
        2.25,
        500,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_thirty_two_candidates_budget_500_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_thirty_two_candidates_budget_500_l95",
        "leader_plus_thirty_one_equal_candidates_gaussian_utilities",
        32,
        2.25,
        500,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_one_hundred_candidates_budget_500_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_one_hundred_candidates_budget_500_l95",
        "leader_plus_ninety_nine_equal_candidates_gaussian_utilities",
        100,
        2.75,
        500,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_one_hundred_candidates_budget_500_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_one_hundred_candidates_budget_500_l95",
        "leader_plus_ninety_nine_equal_candidates_gaussian_utilities",
        100,
        2.75,
        500,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_one_hundred_candidates_budget_32768_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_one_hundred_candidates_budget_32768_l95",
        "leader_plus_ninety_nine_equal_candidates_gaussian_utilities",
        100,
        2.75,
        32768,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_one_hundred_candidates_budget_32768_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_one_hundred_candidates_budget_32768_l95",
        "leader_plus_ninety_nine_equal_candidates_gaussian_utilities",
        100,
        2.75,
        32768,
        &ALL,
    );
}

// Numerical precision: sixteen candidates whose utilities sit a billion noise units from zero,
// where the interval is exact only if its standard error is computed without cancellation.
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_sixteen_candidates_large_offset_l95() {
    run_gated_at_offset(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_sixteen_candidates_large_offset_l95",
        "leader_plus_fifteen_equal_candidates_gaussian_utilities_offset_one_billion",
        16,
        2.0,
        2000,
        &ALL,
        1.0e9,
    );
}

// Every candidate count needs a distinct record identity: sharing a test name would emit
// conflicting rows under one ID even when the candidate counts occupy different bands.
gated!(
    selection_corrected_four_candidates_l95,
    PublishedRoute::SelectionCorrectedInterval,
    4,
    2.0,
    ALL,
    "leader_plus_three_equal_candidates_gaussian_utilities"
);
gated!(
    selection_corrected_seven_candidates_l95,
    PublishedRoute::SelectionCorrectedInterval,
    7,
    2.0,
    ALL,
    "leader_plus_six_equal_candidates_gaussian_utilities"
);
gated!(
    selection_corrected_ten_candidates_l95,
    PublishedRoute::SelectionCorrectedInterval,
    10,
    2.0,
    ALL,
    "leader_plus_nine_equal_candidates_gaussian_utilities"
);
gated!(
    selection_corrected_fifty_candidates_l95,
    PublishedRoute::SelectionCorrectedInterval,
    50,
    2.0,
    ALL,
    "leader_plus_forty_nine_equal_candidates_gaussian_utilities"
);

// Exact ties (equal true means) and near ties (the leader a tenth of a noise unit above).
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_sixteen_equal_candidates_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_sixteen_equal_candidates_l95",
        "sixteen_equal_candidates_gaussian_utilities",
        16,
        0.0,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn confidence_set_sixteen_near_tie_candidates_l95() {
    run_gated(
        PublishedRoute::OptimalityConfidenceSet,
        "confidence_set_sixteen_near_tie_candidates_l95",
        "leader_a_tenth_above_fifteen_equal_candidates_gaussian_utilities",
        16,
        0.1,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_sixteen_equal_candidates_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_sixteen_equal_candidates_l95",
        "sixteen_equal_candidates_gaussian_utilities",
        16,
        0.0,
        2000,
        &ALL,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_sixteen_near_tie_candidates_l95() {
    run_gated(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_sixteen_near_tie_candidates_l95",
        "leader_a_tenth_above_fifteen_equal_candidates_gaussian_utilities",
        16,
        0.1,
        2000,
        &ALL,
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn selection_corrected_all_equal_sixty_four_candidates_boundary_l95() {
    run_named_boundary(
        PublishedRoute::SelectionCorrectedInterval,
        "selection_corrected_all_equal_sixty_four_candidates_boundary_l95",
        "sixty_four_equal_candidates_gaussian_utilities_shared_stream",
        64,
        0.0,
        ";boundary=all_equal",
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_dominant_leader_boundary_l95() {
    run_named_boundary(
        PublishedRoute::RankingProbability,
        "ranking_probability_dominant_leader_boundary_l95",
        "dominant_leader_eight_candidates_gaussian_utilities_shared_stream",
        8,
        6.0,
        ";boundary=dominant_leader",
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn ranking_probability_all_equal_boundary_l95() {
    run_named_boundary(
        PublishedRoute::RankingProbability,
        "ranking_probability_all_equal_boundary_l95",
        "eight_equal_candidates_gaussian_utilities_shared_stream",
        8,
        0.0,
        ";boundary=all_equal",
    );
}
