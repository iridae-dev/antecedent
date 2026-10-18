//! Selection and multiplicity corrections of the decision-reliability intervals at
//! arbitrary candidate counts and Monte Carlo budgets, against hand arithmetic.
//!
//! Oracles are alternating `mean +/- 1` series (sample variance `n / (n - 1)`, so the
//! standard error of the mean is exactly `sqrt(1 / (n - 1))`), identical columns for
//! exact ties, closed-form critical values of the stated tail mass and an independent
//! quadrature of the expected maximum of independent normals. No tolerance is tuned to
//! an observed value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    BatchPlan, CANDIDATE_COUNT_BANDS, CandidateCoupling, CandidateDraws, CandidateOrigin,
    ConfidenceSetBasis, DecisionDraws, DrawSegment, Functional, IntervalMultiplicity,
    IntervalRequest, KnownOptimumDesign, McEstimate, NoiseCoupling, NoiseDependence, PairCoupling,
    ProbabilityQuantity, PublishedRoute, RankingRequest, ReliabilityRequest, SourceParadigm,
    StandardErrorMethod, StreamId, StreamPurpose, WithinStreamDependence, calibration_basis,
    candidate_count_band, derive_uncertainty, estimate_difference, simulate_selection_draws,
    winners_curse_bound,
};
use antecedent_kernels::{norm_cdf, norm_inv, norm_pdf};

const COUNTS: [usize; 5] = [3, 5, 16, 32, 100];
const ALPHA: f64 = 0.05;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(13)
}

/// Upper-tail critical value at one-sided tail mass `tail`.
fn z(tail: f64) -> f64 {
    -norm_inv(tail)
}

/// `mean + 1, mean - 1, ...` over `n` draws (`n` even).
fn alternating(mean: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| mean + if i % 2 == 0 { 1.0 } else { -1.0 }).collect()
}

/// Standard error of the mean of [`alternating`].
fn se_alternating(n: usize) -> f64 {
    (1.0 / (n as f64 - 1.0)).sqrt()
}

fn id(j: usize) -> String {
    format!("c{j:03}")
}

/// Candidates on their own streams, declared independent.
fn separate(means: &[f64], n: usize) -> DecisionDraws {
    carrier(means, n, false)
}

/// Candidates on one shared simulation.
fn shared(means: &[f64], n: usize) -> DecisionDraws {
    carrier(means, n, true)
}

fn carrier(means: &[f64], n: usize, one_stream: bool) -> DecisionDraws {
    let candidates = means
        .iter()
        .enumerate()
        .map(|(j, &mean)| {
            let label = if one_stream { "shared".to_owned() } else { format!("candidate/{j}") };
            CandidateDraws {
                id: id(j),
                utilities: alternating(mean, n),
                segment: DrawSegment::whole(
                    StreamId::new(901, StreamPurpose::Selection, 0, label),
                    n,
                ),
            }
        })
        .collect();
    DecisionDraws {
        utility_unit: "utility".into(),
        producer: "test".into(),
        source: SourceParadigm::KnownLaw,
        coupling: if one_stream {
            CandidateCoupling::SharedSimulation
        } else {
            CandidateCoupling::DeclaredIndependent
        },
        within_stream: WithinStreamDependence::Independent,
        weights: None,
        repeated_evaluations: 0,
        origin: CandidateOrigin::Enumerated,
        search_record: None,
        candidates,
    }
}

fn derive(
    draws: &DecisionDraws,
    interval: IntervalRequest,
) -> antecedent_design::decision_reliability::DerivedUncertainty {
    derive_uncertainty(draws, None, &ReliabilityRequest::expected_utility(), &interval, &ctx())
        .unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: expected {expected}, got {actual}");
}

/// `E[max]` of `k` independent standard normals by composite Simpson quadrature of
/// `x k phi(x) Phi(x)^(k - 1)` over `[-10, 10]`, independent of the runtime's tail form.
fn expected_maximum_of_standard_normals(k: usize) -> f64 {
    let steps = 200_000_usize;
    let (low, high) = (-10.0_f64, 10.0_f64);
    let h = (high - low) / steps as f64;
    let power = i32::try_from(k - 1).unwrap();
    let f = |x: f64| x * k as f64 * norm_pdf(x) * norm_cdf(x).powi(power);
    let mut sum = f(low) + f(high);
    for i in 1..steps {
        sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(low + h * i as f64);
    }
    sum * h / 3.0
}

#[test]
fn selection_correction_uses_the_actual_candidate_count() {
    let n = 200;
    let se = se_alternating(n);
    for k in COUNTS {
        let draws = separate(&vec![0.0; k], n);
        let derived = derive(&draws, IntervalRequest::marginal(0.95));
        assert_eq!(derived.coordinate.candidate_set_size, k as u64);
        let corrected = derived.selection.corrected.as_ref().unwrap();
        assert_eq!(corrected.n_considered, k as u64);
        // Two-sided Bonferroni over the k candidates considered.
        let simultaneous = z(ALPHA / (2.0 * k as f64));
        let ordinary = z(ALPHA / 2.0);
        close(corrected.critical_value, simultaneous, 1e-12, "simultaneous critical value");
        close(corrected.standard_error, se, 1e-12, "standard error");
        close(corrected.numerical_half_width, ordinary * se, 1e-12, "ordinary half-width");
        close(
            corrected.selection_extra_half_width,
            (simultaneous - ordinary) * se,
            1e-12,
            "selection extra half-width",
        );
        close(
            corrected.upper - corrected.lower,
            2.0 * simultaneous * se,
            1e-12,
            "corrected interval width",
        );
        // The best-candidate set: paired comparisons divide the level by the k - 1
        // competitors up to 64 candidates; beyond that the dependence-free Bonferroni
        // over all k candidates, two-sided.
        let set = &derived.selection.confidence_set;
        if k <= 64 {
            assert_eq!(set.basis, ConfidenceSetBasis::PairedComparisonsWithBest, "k = {k}");
            close(set.critical_value, z(ALPHA / (k as f64 - 1.0)), 1e-12, "paired critical value");
        } else {
            assert_eq!(set.basis, ConfidenceSetBasis::MarginalBonferroni, "k = {k}");
            close(
                set.critical_value,
                z(ALPHA / (2.0 * k as f64)),
                1e-12,
                "marginal critical value",
            );
        }
        // Exact ties cannot be separated: every candidate stays in the set.
        assert_eq!(set.members.len(), k, "k = {k}");
        assert!(set.excluded.is_empty());
    }
}

#[test]
fn simultaneous_intervals_divide_the_level_by_the_number_reported() {
    let n = 200;
    for k in COUNTS {
        let draws = separate(&vec![0.0; k], n);
        let derived = derive(
            &draws,
            IntervalRequest { level: 0.95, multiplicity: IntervalMultiplicity::Simultaneous },
        );
        let numerical = &derived.numerical;
        let reported = numerical.utilities.len()
            + numerical.differences_vs_decision.len()
            + numerical.probabilities.len();
        assert_eq!(numerical.intervals_reported, reported as u64, "k = {k}");
        assert!(reported >= k, "every candidate has an interval");
        let per_interval = ALPHA / reported as f64;
        for entry in &numerical.utilities {
            close(entry.interval.level, 1.0 - per_interval, 1e-12, "per-interval level");
            close(
                entry.interval.upper - entry.interval.estimate,
                z(per_interval / 2.0) * entry.interval.standard_error,
                1e-12,
                "per-interval half-width",
            );
        }
    }
}

#[test]
fn the_best_set_boundary_moves_with_the_competitor_count() {
    // One leader `gap` above k - 1 equal candidates, each on its own stream. The leader
    // beats a competitor when the gap exceeds z(alpha / (k - 1)) standard errors of the
    // difference, sqrt(2) standard errors of one candidate.
    let n = 200;
    let se_difference = 2.0_f64.sqrt() * se_alternating(n);
    for k in [3_usize, 5, 16, 32] {
        let threshold = z(ALPHA / (k as f64 - 1.0)) * se_difference;
        for (factor, excluded) in [(0.999, false), (1.001, true)] {
            let mut means = vec![0.0; k];
            means[0] = factor * threshold;
            let derived = derive(&separate(&means, n), IntervalRequest::marginal(0.95));
            let set = &derived.selection.confidence_set;
            assert!(set.members.contains(&id(0)), "the leader is always retained");
            assert_eq!(
                set.members.len(),
                if excluded { 1 } else { k },
                "k = {k}, gap = {factor} of the threshold"
            );
        }
    }
}

#[test]
fn exact_ties_select_the_smallest_id_and_share_the_ranking_probability() {
    let n = 200;
    for k in [3_usize, 16, 32] {
        let draws = shared(&vec![0.5; k], n);
        let request = ReliabilityRequest {
            ranking: Some(RankingRequest::default()),
            ..ReliabilityRequest::expected_utility()
        };
        let derived =
            derive_uncertainty(&draws, None, &request, &IntervalRequest::marginal(0.95), &ctx())
                .unwrap();
        assert_eq!(derived.decided_id, id(0), "ties break to the smallest semantic id");
        assert_eq!(derived.selection.confidence_set.members.len(), k);
        let best: Vec<_> = derived
            .numerical
            .probabilities
            .iter()
            .filter(|p| matches!(p.quantity, ProbabilityQuantity::Best))
            .collect();
        assert_eq!(best.len(), k);
        for entry in best {
            close(entry.interval.estimate, 1.0 / k as f64, 1e-12, "tie-shared best probability");
        }
    }
}

#[test]
fn near_ties_are_not_separated_and_keep_the_selection_correction() {
    // A gap of a hundredth of one candidate's standard error: the leader is chosen, no
    // competitor is excluded and the selection correction is unchanged.
    let n = 200;
    let gap = 0.01 * se_alternating(n);
    for k in [5_usize, 16, 100] {
        let mut means = vec![0.0; k];
        means[0] = gap;
        let derived = derive(&separate(&means, n), IntervalRequest::marginal(0.95));
        assert_eq!(derived.decided_id, id(0));
        assert_eq!(derived.selection.confidence_set.members.len(), k);
        let corrected = derived.selection.corrected.as_ref().unwrap();
        close(
            corrected.critical_value,
            z(ALPHA / (2.0 * k as f64)),
            1e-12,
            "critical value at a near tie",
        );
    }
}

#[test]
fn the_winners_curse_bound_is_the_exact_maximum_for_the_candidate_count() {
    let sd = 0.25;
    for k in COUNTS {
        let estimates: Vec<McEstimate> = (0..k)
            .map(|_| McEstimate {
                value: 0.0,
                standard_error: sd,
                effective_sample_size: 400.0,
                n_draws: 400,
                method: StandardErrorMethod::IidSampleVariance,
            })
            .collect();
        let bound =
            winners_curse_bound(&estimates, k as u64, NoiseDependence::IndependentStreams, 0.0)
                .unwrap();
        assert_eq!(bound.n_considered, k as u64);
        let exact = sd * expected_maximum_of_standard_normals(k);
        close(bound.gaussian_expected_maximum.unwrap(), exact, 1e-8 * exact.max(1.0), "E[max]");
        close(bound.union_bound, sd * (2.0 * (k as f64).ln()).sqrt(), 1e-12, "union bound");
        assert!(bound.gaussian_expected_maximum.unwrap() <= bound.union_bound);
        // Unseen candidates are padded at the largest supplied standard error: the same
        // bound as if they had been supplied.
        let supplied = &estimates[..2];
        let padded =
            winners_curse_bound(supplied, k as u64, NoiseDependence::IndependentStreams, 0.0)
                .unwrap();
        assert_eq!(padded.padded_candidates, k as u64 - 2);
        close(
            padded.gaussian_expected_maximum.unwrap(),
            bound.gaussian_expected_maximum.unwrap(),
            1e-12,
            "padded E[max]",
        );
    }
    // Tabulated: the expected maximum of 100 standard normals is 2.5076.
    close(expected_maximum_of_standard_normals(100), 2.507_593_5, 1e-4, "tabulated E[max]");
}

#[test]
fn shared_streams_account_for_common_noise_that_separate_streams_keep() {
    // Two equal candidates with per-draw noise correlation rho: the paired difference has
    // variance 2 sigma^2 (1 - rho); independent streams keep 2 sigma^2.
    let (n, rho) = (40_000_usize, 0.9);
    let theory = |variance: f64| (variance / n as f64).sqrt();
    let difference = |draws: &DecisionDraws, pairing| {
        estimate_difference(
            &draws.candidates[0].utilities,
            &draws.candidates[1].utilities,
            None,
            WithinStreamDependence::Independent,
            BatchPlan::SquareRoot,
            &Functional::ExpectedUtility,
            pairing,
        )
        .unwrap()
    };
    let stream = StreamId::new(77, StreamPurpose::Selection, 0, "streams");
    let design = |coupling| KnownOptimumDesign {
        true_means: vec![0.0, 0.0],
        noise_sd: 1.0,
        n_draws: n,
        coupling,
    };
    let common = simulate_selection_draws(
        &design(NoiseCoupling::SharedStream { correlation: rho }),
        &stream,
    )
    .unwrap();
    let own = simulate_selection_draws(&design(NoiseCoupling::SeparateStreams), &stream).unwrap();
    let paired = difference(&common, PairCoupling::SharedStream);
    let independent = difference(&own, PairCoupling::IndependentStreams);
    close(paired.standard_error / theory(2.0 * (1.0 - rho)), 1.0, 0.05, "shared-stream SE");
    close(independent.standard_error / theory(2.0), 1.0, 0.05, "separate-stream SE");
    // Treating shared draws as independent streams would overstate the error.
    let ignored = difference(&common, PairCoupling::IndependentStreams);
    close(ignored.standard_error / theory(2.0), 1.0, 0.05, "ignored correlation SE");
    assert!(ignored.standard_error > 2.5 * paired.standard_error);
}

#[test]
fn the_standard_error_is_exact_at_extreme_offsets_and_budgets() {
    // A mean of 1e9 against a spread of one: a one-pass sum of squares loses every digit
    // of the variance; the standard error of the alternating series is exact.
    for n in [500_usize, 4_000, 65_536] {
        let k = if n > 4_000 { 3 } else { 16 };
        let derived = derive(&separate(&vec![1.0e9; k], n), IntervalRequest::marginal(0.95));
        assert_eq!(derived.coordinate.mc_budget, n as u64);
        let corrected = derived.selection.corrected.as_ref().unwrap();
        close(corrected.standard_error / se_alternating(n), 1.0, 1e-9, "standard error at offset");
        close(corrected.in_sample, 1.0e9, 1e-6, "in-sample value");
    }
}

#[test]
fn records_are_keyed_by_the_candidate_count_band_not_by_the_count() {
    let n = 200;
    let basis_of = |k: usize| {
        let derived = derive(&separate(&vec![0.0; k], n), IntervalRequest::marginal(0.95));
        calibration_basis(PublishedRoute::SelectionCorrectedInterval, &derived.coordinate, 0.95)
    };
    let functional = |k: usize| basis_of(k).map(|basis| basis.key.functional);
    // Every count of a band shares one key; neighbouring bands do not.
    assert_eq!(functional(3), functional(4));
    assert_eq!(functional(5), functional(16));
    assert_ne!(functional(4), functional(5));
    assert_ne!(functional(16), functional(17));
    assert_eq!(functional(17), functional(64));
    assert_ne!(functional(64), functional(65));
    assert_eq!(functional(65), functional(100));
    assert_eq!(
        functional(32).unwrap(),
        "candidates=17-64;rule=maximize_expected_utility;functional=expected_utility"
    );
    // The budget is the scope, not part of the key.
    assert_eq!(basis_of(5).unwrap().scope.row_count, n as u64);
    // Beyond the last band nothing is measured: no basis, so no route can be licensed.
    assert!(basis_of(101).is_none());
    assert_eq!(candidate_count_band(1), None);
    assert_eq!(candidate_count_band(101), None);
    // The bands tile 2..=100 without gaps or overlap, and none spans the 64 to 65 switch
    // from paired comparisons to the dependence-free best set.
    let mut next = 2;
    for &(low, high) in &CANDIDATE_COUNT_BANDS {
        assert_eq!(low, next);
        assert!(high >= low);
        assert!(!(low <= 64 && high >= 65), "band {low}-{high} spans the construction switch");
        next = high + 1;
    }
    assert_eq!(next, 101);
    for k in 2..=100_u64 {
        let (low, high) = candidate_count_band(k).unwrap();
        assert!((low..=high).contains(&k));
    }
}
