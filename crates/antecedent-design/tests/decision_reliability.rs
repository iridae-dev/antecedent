//! Decision reliability against closed-form, enumerated and known-optimum truth.
//!
//! Every tolerance is predeclared next to the assertion it guards and is derived from
//! the effective sample size of the quantity (a multiple of its theoretical standard
//! error), never tuned to an observed value. All simulation runs at fixed seeds.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    BatchPlan, CalibrationCoordinate, CandidateCoupling, CandidateDraws, CandidateOrigin,
    CandidateSearchRecord, DecisionChoice, DecisionDraws, DecisionReliabilityArtifact,
    DecisionReliabilityExpectation, DrawSegment, Functional, KnownOptimumDesign, NoiseCoupling,
    PairCoupling, PublishedRoute, RankingRequest, ReliabilityError, ReliabilityRequest, RiskTail,
    SelectionRule, SelectionStatus, SelectionUncertainty, SourceParadigm, SourceUncertainty,
    StandardErrorMethod, StreamId, StreamPurpose, ThresholdSide, UncertaintyReporting,
    ValidationKind, ValidationPlan, WithinStreamDependence, assess, consume, estimate,
    estimate_difference, license_route, rank, select_in_sample, simulate_replicate,
    simulate_selection_draws, split_draws, winners_curse_bound,
};
use antecedent_kernels::{norm_cdf, norm_inv, norm_pdf, standard_normal};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn stream(purpose: StreamPurpose, replicate: u64) -> StreamId {
    StreamId::new(20_240, purpose, replicate, "test_stream")
}

/// A shared-simulation carrier of exactly enumerated per-draw utilities.
fn carrier(columns: &[(&str, Vec<f64>)]) -> DecisionDraws {
    let n = columns[0].1.len();
    let shared = DrawSegment::whole(stream(StreamPurpose::Selection, 0), n);
    DecisionDraws {
        utility_unit: "utility".into(),
        producer: "test".into(),
        source: SourceParadigm::KnownLaw,
        coupling: CandidateCoupling::SharedSimulation,
        within_stream: WithinStreamDependence::Independent,
        weights: None,
        repeated_evaluations: 0,
        origin: CandidateOrigin::Enumerated,
        search_record: None,
        candidates: columns
            .iter()
            .map(|(id, utilities)| CandidateDraws {
                id: (*id).to_owned(),
                utilities: utilities.clone(),
                segment: shared.clone(),
            })
            .collect(),
    }
}

fn gaussian_pair(rho: f64, n: usize, coupling_separate: bool) -> DecisionDraws {
    let design = KnownOptimumDesign {
        true_means: vec![1.0, 0.0],
        noise_sd: 1.0,
        n_draws: n,
        coupling: if coupling_separate {
            NoiseCoupling::SeparateStreams
        } else {
            NoiseCoupling::SharedStream { correlation: rho }
        },
    };
    simulate_selection_draws(&design, &stream(StreamPurpose::Selection, 3)).unwrap()
}

// ---- exact enumerated ranking truth ------------------------------------------------

#[test]
fn enumerated_four_state_law_gives_exact_best_top_k_and_baseline_probabilities() {
    // Four equally likely states; the last is a three-way tie.
    let a = vec![3.0, 3.0, 2.0, 2.0];
    let b = vec![2.0, 1.0, 3.0, 2.0];
    let c = vec![1.0, 2.0, 1.0, 2.0];
    let request = RankingRequest {
        baseline: Some("B".into()),
        top_k: vec![2],
        batch_plan: BatchPlan::SquareRoot,
    };
    let forward = carrier(&[("A", a.clone()), ("B", b.clone()), ("C", c.clone())]);
    let reversed = carrier(&[("C", c), ("B", b), ("A", a)]);
    let report = rank(&forward, &request, &ctx()).unwrap();
    let get = |id: &str| report.statements.iter().find(|s| s.id == id).unwrap().clone();
    let tol = 1e-12;
    // P(best): A (1 + 1 + 0 + 1/3)/4, B (0 + 0 + 1 + 1/3)/4, C (0 + 0 + 0 + 1/3)/4.
    assert!((get("A").probability_best.value - 7.0 / 12.0).abs() < tol);
    assert!((get("B").probability_best.value - 1.0 / 3.0).abs() < tol);
    assert!((get("C").probability_best.value - 1.0 / 12.0).abs() < tol);
    let total: f64 = report.statements.iter().map(|s| s.probability_best.value).sum();
    assert!((total - 1.0).abs() < tol, "tie-shared best probabilities sum to one");
    // P(top 2): two slots per state, the tie shares them 2/3 each.
    assert!((get("A").probability_top_k[0].estimate.value - 11.0 / 12.0).abs() < tol);
    assert!((get("B").probability_top_k[0].estimate.value - 2.0 / 3.0).abs() < tol);
    assert!((get("C").probability_top_k[0].estimate.value - 5.0 / 12.0).abs() < tol);
    // P(better than B), ties counting one half.
    let vs = |id: &str| get(id).probability_better_than_baseline.unwrap().value;
    assert!((vs("A") - 0.625).abs() < tol && (vs("C") - 0.375).abs() < tol);
    assert!(get("B").probability_better_than_baseline.is_none());
    assert_eq!(report.best_by_expected_utility, "A");
    // Labelled as an uncalibrated diagnostic with its paradigm and coupling.
    let label = serde_json::to_value(&report.label).unwrap();
    assert_eq!(label["calibration"], "uncalibrated");
    assert_eq!(label["paradigm"], "known_law");
    assert_eq!(label["coupling"], "shared_simulation");
    // Permutation invariance of the whole report.
    assert_eq!(report, rank(&reversed, &request, &ctx()).unwrap());
}

#[test]
fn two_gaussian_candidates_match_the_exact_probability_within_the_ess_band() {
    // A ~ N(1, 1), B ~ N(0, 1): P(A > B) = Phi(1 / sqrt(2 (1 - rho))).
    let n = 20_000;
    for rho in [0.0_f64, 0.75] {
        let draws = gaussian_pair(rho, n, false);
        let exact = norm_cdf(1.0 / (2.0 * (1.0 - rho)).sqrt());
        let request = RankingRequest {
            baseline: Some("c00001".into()),
            top_k: vec![],
            batch_plan: BatchPlan::SquareRoot,
        };
        let report = rank(&draws, &request, &ctx()).unwrap();
        let a = report.statements.iter().find(|s| s.id == "c00000").unwrap();
        let versus = a.probability_better_than_baseline.unwrap();
        // Predeclared band: four reported standard errors; the reported standard error
        // itself must agree with the Bernoulli value sqrt(p (1 - p) / n) within 5%.
        let theory_se = (exact * (1.0 - exact) / n as f64).sqrt();
        assert!((versus.value - exact).abs() < 4.0 * versus.standard_error, "rho={rho}");
        assert!((versus.standard_error / theory_se - 1.0).abs() < 0.05, "rho={rho}");
        assert!((versus.effective_sample_size - n as f64).abs() < 1e-9);
        assert_eq!(versus.method, StandardErrorMethod::IidSampleVariance);
        // With two candidates the best probability is the same event.
        assert!((a.probability_best.value - versus.value).abs() < 1e-12);
    }
}

// ---- common random numbers and effective sample size --------------------------------

#[test]
fn shared_stream_difference_variance_is_smaller_and_matches_one_minus_rho() {
    let n = 20_000;
    let rho = 0.8;
    let functional = Functional::ExpectedUtility;
    let plan = BatchPlan::SquareRoot;
    let dependence = WithinStreamDependence::Independent;
    let shared = gaussian_pair(rho, n, false);
    let separate = gaussian_pair(0.0, n, true);
    let diff = |draws: &DecisionDraws, coupling| {
        estimate_difference(
            &draws.candidates[0].utilities,
            &draws.candidates[1].utilities,
            None,
            dependence,
            plan,
            &functional,
            coupling,
        )
        .unwrap()
    };
    let crn = diff(&shared, PairCoupling::SharedStream);
    let independent = diff(&separate, PairCoupling::IndependentStreams);
    // Theory: Var(mean difference) = 2 (1 - rho) / n with CRN, 2 / n without.
    let se_crn = (2.0 * (1.0 - rho) / n as f64).sqrt();
    let se_independent = (2.0 / n as f64).sqrt();
    assert!((crn.standard_error / se_crn - 1.0).abs() < 0.05);
    assert!((independent.standard_error / se_independent - 1.0).abs() < 0.05);
    assert!(crn.standard_error < 0.5 * independent.standard_error);
    // The design effect is the variance ratio, 1 - rho, and the implied correlation rho.
    let ratio = crn.variance_ratio_vs_independent.unwrap();
    assert!((ratio - (1.0 - rho)).abs() < 0.04, "ratio {ratio}");
    assert!((crn.implied_correlation.unwrap() - rho).abs() < 0.04);
    assert!(independent.variance_ratio_vs_independent.is_none());
    // Effective sample size of the paired series is n; independent streams add variances.
    assert!((crn.effective_sample_size.unwrap() - n as f64).abs() < 1e-9);
    assert!((independent.effective_sample_size.unwrap() / n as f64 - 1.0).abs() < 0.06);
    // The shared simulation is accounted as one stream; separate streams as two.
    let shared_report =
        assess(&shared, None, &ReliabilityRequest::expected_utility(), &ctx()).unwrap();
    let separate_report =
        assess(&separate, None, &ReliabilityRequest::expected_utility(), &ctx()).unwrap();
    assert!(shared_report.monte_carlo_error.streams.shared_noise);
    assert_eq!(shared_report.monte_carlo_error.streams.distinct_streams, 1);
    assert!(!separate_report.monte_carlo_error.streams.shared_noise);
    assert_eq!(separate_report.monte_carlo_error.streams.distinct_streams, 2);
}

fn ar1(phi: f64, n: usize, seed_stream: &StreamId) -> Vec<f64> {
    let mut rng = seed_stream.rng();
    let stationary_sd = (1.0 / (1.0 - phi * phi)).sqrt();
    let mut previous = stationary_sd * standard_normal(&mut rng);
    (0..n)
        .map(|_| {
            previous = phi * previous + standard_normal(&mut rng);
            previous
        })
        .collect()
}

#[test]
fn batch_means_recovers_the_autocorrelation_time_of_an_ar1_stream() {
    // AR(1), phi = 0.9: ESS = n (1 - phi) / (1 + phi) = n / 19; the variance of the
    // mean is sigma^2 (1 + phi) / ((1 - phi) n) with sigma^2 = 1 / (1 - phi^2).
    let (phi, n) = (0.9_f64, 200_000);
    let values = ar1(phi, n, &stream(StreamPurpose::Source, 11));
    let plan = BatchPlan::SquareRoot;
    let dependent = estimate(
        &values,
        None,
        WithinStreamDependence::Autocorrelated,
        plan,
        &Functional::ExpectedUtility,
    )
    .unwrap()
    .estimate;
    let naive = estimate(
        &values,
        None,
        WithinStreamDependence::Independent,
        plan,
        &Functional::ExpectedUtility,
    )
    .unwrap()
    .estimate;
    let sigma2 = 1.0 / (1.0 - phi * phi);
    let theory_se = (sigma2 * (1.0 + phi) / ((1.0 - phi) * n as f64)).sqrt();
    let theory_ess = n as f64 * (1.0 - phi) / (1.0 + phi);
    // Predeclared 20% band: with ~447 batches the batch-variance estimate has relative
    // standard error about sqrt(2 / 447) = 7%, so the band is near three of them.
    assert!((dependent.standard_error / theory_se - 1.0).abs() < 0.20);
    assert!((dependent.effective_sample_size / theory_ess - 1.0).abs() < 0.20);
    assert!(matches!(dependent.method, StandardErrorMethod::BatchMeans { .. }));
    // Ignoring the dependence understates the error by about sqrt(19).
    assert!(dependent.standard_error > 3.0 * naive.standard_error);
    assert!((naive.effective_sample_size - n as f64).abs() < 1e-9);
}

#[test]
fn importance_weights_use_kish_ess_and_the_self_normalized_delta_method() {
    // Weights 1 (even draws) and 3 (odd draws); utility 0 on even, 1 on odd draws.
    let n = 1_000;
    let weights: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 3.0 }).collect();
    let values: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 1))).collect();
    let got = estimate(
        &values,
        Some(&weights),
        WithinStreamDependence::Independent,
        BatchPlan::SquareRoot,
        &Functional::ExpectedUtility,
    )
    .unwrap()
    .estimate;
    // Weighted mean 3 / 4; sum w^2 (y - m)^2 = 500 (0.5625) + 500 (9)(0.0625) = 562.5;
    // sum w = 2000; Kish ESS = (2000)^2 / (500 + 500 * 9) = 800.
    assert!((got.value - 0.75).abs() < 1e-12);
    assert!((got.standard_error - 562.5_f64.sqrt() / 2_000.0).abs() < 1e-12);
    assert!((got.effective_sample_size - 800.0).abs() < 1e-9);
    assert_eq!(got.method, StandardErrorMethod::SelfNormalizedWeights);
    // No weighted order-statistic or influence-function error is licensed.
    for functional in
        [Functional::Quantile { p: 0.5 }, Functional::Cvar { alpha: 0.25, tail: RiskTail::Lower }]
    {
        let err = estimate(
            &values,
            Some(&weights),
            WithinStreamDependence::Independent,
            BatchPlan::SquareRoot,
            &functional,
        )
        .unwrap_err();
        assert_eq!(err.reason_code(), "route_not_supported");
    }
}

// ---- risk measures ------------------------------------------------------------------

#[test]
fn enumerated_cvar_quantile_and_threshold_probability_are_exact() {
    let values = vec![7.0, 1.0, 10.0, 4.0, 2.0, 9.0, 5.0, 3.0, 8.0, 6.0];
    let go = |functional: Functional| {
        estimate(
            &values,
            None,
            WithinStreamDependence::Independent,
            BatchPlan::SquareRoot,
            &functional,
        )
        .unwrap()
        .estimate
        .value
    };
    let lower = |alpha| go(Functional::Cvar { alpha, tail: RiskTail::Lower });
    // Mean of the two smallest; the interpolated 25% tail (1 + 2 + 0.5 * 3) / 2.5.
    assert!((lower(0.2) - 1.5).abs() < 1e-12);
    assert!((lower(0.25) - 1.8).abs() < 1e-12);
    assert!((lower(1.0) - 5.5).abs() < 1e-12);
    assert!((go(Functional::Cvar { alpha: 0.2, tail: RiskTail::Upper }) - 9.5).abs() < 1e-12);
    // Nearest-rank quantile: x_(ceil(n p)).
    assert!((go(Functional::Quantile { p: 0.5 }) - 5.0).abs() < 1e-12);
    assert!((go(Functional::Quantile { p: 0.91 }) - 10.0).abs() < 1e-12);
    // P(U > 6.5) = 4/10; P(U <= 6.5) = 6/10.
    let above = Functional::ThresholdProbability { threshold: 6.5, side: ThresholdSide::Above };
    let below = Functional::ThresholdProbability { threshold: 6.5, side: ThresholdSide::AtOrBelow };
    assert!((go(above) - 0.4).abs() < 1e-12 && (go(below) - 0.6).abs() < 1e-12);
}

fn standard_normal_draws(n: usize, replicate: u64) -> Vec<f64> {
    let mut rng = stream(StreamPurpose::Source, replicate).rng();
    (0..n).map(|_| standard_normal(&mut rng)).collect()
}

/// Asymptotic standard error of the lower-tail `CVaR` estimator for a standard normal:
/// `sd(min(U, q)) / (alpha * sqrt(n))`, with the moments of `min(U, q)` in closed form.
fn cvar_theory_se(alpha: f64, n: usize) -> f64 {
    let q = norm_inv(alpha);
    let phi = norm_pdf(q);
    let m1 = q * (1.0 - alpha) - phi;
    let m2 = alpha - q * phi + q * q * (1.0 - alpha);
    (m2 - m1 * m1).sqrt() / (alpha * (n as f64).sqrt())
}

#[test]
fn cvar_of_a_known_normal_has_the_influence_function_standard_error() {
    let (alpha, n) = (0.1, 200_000);
    let draws = standard_normal_draws(n, 5);
    let lower = estimate(
        &draws,
        None,
        WithinStreamDependence::Independent,
        BatchPlan::SquareRoot,
        &Functional::Cvar { alpha, tail: RiskTail::Lower },
    )
    .unwrap()
    .estimate;
    // CVaR_alpha of N(0, 1) = -phi(Phi^{-1}(alpha)) / alpha.
    let truth = -norm_pdf(norm_inv(alpha)) / alpha;
    let theory_se = cvar_theory_se(alpha, n);
    assert!((lower.value - truth).abs() < 4.0 * theory_se, "{} vs {truth}", lower.value);
    assert!((lower.standard_error / theory_se - 1.0).abs() < 0.03);
    assert_eq!(lower.method, StandardErrorMethod::InfluenceFunction);
    // The upper tail of N(0, 1) mirrors the lower tail.
    let upper = estimate(
        &draws,
        None,
        WithinStreamDependence::Independent,
        BatchPlan::SquareRoot,
        &Functional::Cvar { alpha, tail: RiskTail::Upper },
    )
    .unwrap()
    .estimate;
    assert!((upper.value + truth).abs() < 4.0 * theory_se);
    assert!((upper.standard_error / theory_se - 1.0).abs() < 0.05);
}

#[test]
fn cvar_standard_error_matches_the_across_seed_spread() {
    // 120 independent replicates of n = 2000: the mean reported standard error must
    // equal the empirical standard deviation of the estimator. With 120 replicates the
    // empirical sd has relative standard error 1/sqrt(2 * 119) = 6.5%; the band is 20%.
    let (alpha, n, reps) = (0.1, 2_000, 120);
    let mut values = Vec::new();
    let mut ses = Vec::new();
    for replicate in 0..reps {
        let draws = standard_normal_draws(n, 1_000 + replicate);
        let e = estimate(
            &draws,
            None,
            WithinStreamDependence::Independent,
            BatchPlan::SquareRoot,
            &Functional::Cvar { alpha, tail: RiskTail::Lower },
        )
        .unwrap()
        .estimate;
        values.push(e.value);
        ses.push(e.standard_error);
    }
    let mean = values.iter().sum::<f64>() / reps as f64;
    let empirical =
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (reps as f64 - 1.0)).sqrt();
    let reported = ses.iter().sum::<f64>() / reps as f64;
    assert!((reported / empirical - 1.0).abs() < 0.20, "{reported} vs {empirical}");
}

#[test]
fn maritz_jarrett_quantile_error_matches_the_normal_theory_value() {
    // The bootstrap standard error of a sample quantile converges at the slow rate
    // n^(-1/4) (relative error near 5% here), so the band is 25%; the formula itself is
    // pinned exactly by the enumerated-bootstrap test beside the implementation.
    let n = 200_001;
    let draws = standard_normal_draws(n, 9);
    for p in [0.5_f64, 0.9] {
        let e = estimate(
            &draws,
            None,
            WithinStreamDependence::Independent,
            BatchPlan::SquareRoot,
            &Functional::Quantile { p },
        )
        .unwrap()
        .estimate;
        let q = norm_inv(p);
        let theory_se = (p * (1.0 - p)).sqrt() / (norm_pdf(q) * (n as f64).sqrt());
        assert!((e.value - q).abs() < 4.0 * theory_se, "p={p}");
        assert!((e.standard_error / theory_se - 1.0).abs() < 0.25, "p={p}");
        assert_eq!(e.method, StandardErrorMethod::MaritzJarrett);
    }
}

// ---- selection: winner's curse with a known true optimum ----------------------------

#[test]
fn in_sample_maximum_is_biased_up_and_held_out_value_is_unbiased() {
    // 20 candidates with identical true utility 0. Estimates have standard error
    // 1 / sqrt(100) = 0.1, so the selected in-sample value overstates the truth by
    // E max of 20 N(0, 0.1^2) = 1.8675 * 0.1 = 0.1867.
    let design = KnownOptimumDesign {
        true_means: vec![0.0; 20],
        noise_sd: 1.0,
        n_draws: 100,
        coupling: NoiseCoupling::SharedStream { correlation: 0.0 },
    };
    let reps = 300_u64;
    let (mut in_sample, mut held_out, mut bound) = (0.0, 0.0, 0.0);
    for replicate in 0..reps {
        let outcome = simulate_replicate(&design, 99, replicate, &ctx()).unwrap();
        in_sample += outcome.in_sample_bias;
        held_out += outcome.held_out_bias;
        bound += outcome.winners_curse_bound;
    }
    let r = reps as f64;
    let (in_sample, held_out, bound) = (in_sample / r, held_out / r, bound / r);
    // Predeclared: the in-sample bias lies in [0.17, 0.205] (the exact value 0.1867
    // with a mean-bias standard error of about 0.003), and the held-out mean bias is
    // within four standard errors, 4 * 0.1 / sqrt(300) = 0.023, of zero.
    assert!((0.17..=0.205).contains(&in_sample), "in-sample bias {in_sample}");
    assert!(held_out.abs() < 0.023, "held-out bias {held_out}");
    // The bound holds on average and is not vacuous (within a factor 1.5).
    assert!(in_sample <= bound + 0.015, "bound {bound} below bias {in_sample}");
    assert!(bound < 1.5 * in_sample, "bound {bound} is vacuous against {in_sample}");
}

#[test]
fn a_clear_optimum_is_selected_and_its_in_sample_value_is_nearly_unbiased() {
    let mut means = vec![0.0; 12];
    means[7] = 1.0;
    let design = KnownOptimumDesign {
        true_means: means,
        noise_sd: 1.0,
        n_draws: 400,
        coupling: NoiseCoupling::SharedStream { correlation: 0.5 },
    };
    let reps = 100_u64;
    let (mut correct, mut in_sample, mut held_out) = (0_u64, 0.0, 0.0);
    for replicate in 0..reps {
        let outcome = simulate_replicate(&design, 5, replicate, &ctx()).unwrap();
        assert_eq!(outcome.true_optimum_index, 7);
        correct += u64::from(outcome.selection_correct);
        in_sample += outcome.in_sample_bias;
        held_out += outcome.held_out_bias;
    }
    // The optimum leads by 1.0 against a per-estimate error of 0.05: 20 sigma.
    assert_eq!(correct, reps);
    // With selection certain the bias is the error of one estimate: mean 0, se 0.005.
    assert!((in_sample / reps as f64).abs() < 0.02);
    assert!((held_out / reps as f64).abs() < 0.02);
}

#[test]
fn sample_splitting_selects_on_one_segment_and_validates_on_a_disjoint_one() {
    let design = KnownOptimumDesign {
        true_means: vec![0.0; 10],
        noise_sd: 1.0,
        n_draws: 200,
        coupling: NoiseCoupling::SharedStream { correlation: 0.0 },
    };
    let request = ReliabilityRequest {
        validation: ValidationPlan::SplitSelectionDraws { selection_fraction: 0.5, gap: 10 },
        ..ReliabilityRequest::expected_utility()
    };
    let reps = 200_u64;
    let (mut in_sample, mut held_out) = (0.0, 0.0);
    for replicate in 0..reps {
        let draws = simulate_selection_draws(&design, &stream(StreamPurpose::Selection, replicate))
            .unwrap();
        let report = assess(&draws, None, &request, &ctx()).unwrap();
        let SelectionUncertainty::Selected(assessment) = &report.selection_uncertainty else {
            panic!("a selection was made");
        };
        assert_eq!(assessment.status, SelectionStatus::SampleSplitValidated);
        let held = assessment.held_out.as_ref().unwrap();
        let record = &held.record;
        assert_eq!(record.kind, ValidationKind::SplitOfSelectionStream);
        assert_eq!(record.selection_segments.len(), 1);
        let selection = &record.selection_segments[0];
        assert_eq!((selection.start, selection.len), (0, 100));
        assert_eq!((record.validation_segment.start, record.validation_segment.len), (110, 90));
        assert_eq!(selection.stream, record.validation_segment.stream);
        in_sample += assessment.in_sample.value;
        held_out += held.held_out.estimate.value;
    }
    // True utility is 0 for every candidate. Selection on 100 draws: E max of ten
    // N(0, 0.1^2) = 1.5388 * 0.1 = 0.154 (se of the mean 0.0035); validation on 90
    // independent draws is unbiased (se 0.1 / sqrt(0.9 * 200) = 0.0075 per mean).
    let r = reps as f64;
    assert!((0.13..=0.18).contains(&(in_sample / r)), "in-sample {}", in_sample / r);
    assert!((held_out / r).abs() < 0.035, "held-out {}", held_out / r);
}

#[test]
fn winners_curse_bound_uses_the_search_record_count_and_the_dependence() {
    let se = |s: f64| antecedent_design::decision_reliability::McEstimate {
        value: 0.0,
        standard_error: s,
        effective_sample_size: 100.0,
        n_draws: 100,
        method: StandardErrorMethod::IidSampleVariance,
    };
    let two = [se(1.0), se(1.0)];
    let independent = winners_curse_bound(
        &two,
        2,
        antecedent_design::decision_reliability::NoiseDependence::IndependentStreams,
        0.0,
    )
    .unwrap();
    // E max of two N(0, 1) = 1 / sqrt(pi).
    let exact = 1.0 / std::f64::consts::PI.sqrt();
    assert!((independent.bound - exact).abs() < 1e-8);
    assert!((independent.union_bound - (2.0 * 2.0_f64.ln()).sqrt()).abs() < 1e-12);
    // Without a dependence statement only the dependence-free union bound is valid.
    let unassessed = winners_curse_bound(
        &two,
        2,
        antecedent_design::decision_reliability::NoiseDependence::Unassessed,
        0.0,
    )
    .unwrap();
    assert!((unassessed.bound - unassessed.union_bound).abs() < 1e-12);
    assert!(unassessed.bound > independent.bound);
    // A nonnegative covariance in the observed sample does not establish the
    // population covariance ordering required by Sudakov-Fernique. Keep the
    // empirical diagnostic, but use the dependence-free bound.
    let sample_positive = winners_curse_bound(
        &two,
        2,
        antecedent_design::decision_reliability::NoiseDependence::SampleCovariancesNonNegative,
        0.0,
    )
    .unwrap();
    assert_eq!(
        sample_positive.bound_kind,
        antecedent_design::decision_reliability::BoundKind::UnionBound
    );
    assert!((sample_positive.bound - sample_positive.union_bound).abs() < 1e-12);
    // 100 candidates were considered though two are retained: the bound grows and the
    // padding is reported.
    let padded = winners_curse_bound(
        &two,
        100,
        antecedent_design::decision_reliability::NoiseDependence::IndependentStreams,
        0.0,
    )
    .unwrap();
    assert_eq!(padded.padded_candidates, 98);
    assert!((padded.bound - 2.507_593_596_6).abs() < 1e-6, "E max of 100 normals");
    assert!((padded.adjusted_value + padded.bound).abs() < 1e-12);
}

// ---- adaptive generation and dependence refusals ------------------------------------

fn adaptive_carrier(record: Option<CandidateSearchRecord>) -> DecisionDraws {
    let design = KnownOptimumDesign {
        true_means: vec![0.0, 0.2, 0.4],
        noise_sd: 1.0,
        n_draws: 200,
        coupling: NoiseCoupling::SharedStream { correlation: 0.3 },
    };
    let mut draws =
        simulate_selection_draws(&design, &stream(StreamPurpose::Selection, 0)).unwrap();
    draws.origin = CandidateOrigin::AdaptivelyGenerated;
    draws.search_record = record;
    draws
}

fn record(n_evaluated: u64, budget: u64) -> CandidateSearchRecord {
    CandidateSearchRecord {
        generator: "random_restart_hill_climb@1".into(),
        budget,
        n_evaluated,
        selection_rule: SelectionRule::MaximizeExpectedUtility,
        generator_stream: Some(stream(StreamPurpose::Source, 77)),
    }
}

#[test]
fn adaptively_generated_candidates_need_a_search_record() {
    let request = ReliabilityRequest {
        ranking: Some(RankingRequest::default()),
        ..ReliabilityRequest::expected_utility()
    };
    let err = assess(&adaptive_carrier(None), None, &request, &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "candidate_search_record_required");
    let refusal = err.to_refusal();
    assert!(refusal.detail.starts_with("decision_reliability."));
    assert!(refusal.remedy.is_some(), "the refusal states what would license the request");
    // Ranking alone refuses the same way.
    let err = rank(&adaptive_carrier(None), &RankingRequest::default(), &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "candidate_search_record_required");

    // With the record the evaluation runs; the bound counts every considered candidate.
    let with_record =
        assess(&adaptive_carrier(Some(record(500, 1_000))), None, &request, &ctx()).unwrap();
    let SelectionUncertainty::Selected(assessment) = &with_record.selection_uncertainty else {
        panic!("selected");
    };
    assert_eq!(assessment.n_considered, 500);
    let curse = assessment.winners_curse.as_ref().unwrap();
    assert_eq!((curse.n_supplied, curse.padded_candidates), (3, 497));
    let retained_only = assess(
        &{
            let mut d = adaptive_carrier(Some(record(3, 1_000)));
            d.origin = CandidateOrigin::Enumerated;
            d
        },
        None,
        &ReliabilityRequest::expected_utility(),
        &ctx(),
    )
    .unwrap();
    let SelectionUncertainty::Selected(small) = &retained_only.selection_uncertainty else {
        panic!("selected");
    };
    assert!(curse.bound > 2.0 * small.winners_curse.as_ref().unwrap().bound);
    // The ranking is labelled as over the retained set of an adaptive search.
    let label = serde_json::to_value(&with_record.ranking.unwrap().label).unwrap();
    assert_eq!(label["scope"], "adaptive_retained_set");
    assert_eq!(label["n_considered"], 500);
}

#[test]
fn an_inconsistent_search_record_is_refused() {
    // Fewer evaluated than retained, and more evaluated than the budget allows.
    for bad in [record(2, 1_000), record(1_001, 1_000)] {
        let err = adaptive_carrier(Some(bad)).validate().unwrap_err();
        assert!(matches!(err, ReliabilityError::SearchRecordInconsistent { .. }));
        assert_eq!(err.reason_code(), "invalid_argument");
    }
}

#[test]
fn unknown_or_contradicted_dependence_is_refused_with_registered_codes() {
    let base = || carrier(&[("A", vec![1.0, 2.0, 3.0, 4.0]), ("B", vec![2.0, 1.0, 4.0, 3.0])]);
    let request = ReliabilityRequest::expected_utility();
    let mut unknown = base();
    unknown.coupling = CandidateCoupling::Unknown;
    let err = assess(&unknown, None, &request, &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "sampling_dependence_unknown");
    let mut serial = base();
    serial.within_stream = WithinStreamDependence::Unknown;
    let err = assess(&serial, None, &request, &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "sampling_dependence_unknown");
    // Declared one shared simulation, but the candidates sit on different streams.
    let mut mismatch = base();
    mismatch.candidates[1].segment = DrawSegment::whole(stream(StreamPurpose::Selection, 1), 4);
    let err = assess(&mismatch, None, &request, &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "joint_law_required");
    // Declared independent, but both candidates read the same draws.
    let mut overlapping = base();
    overlapping.coupling = CandidateCoupling::DeclaredIndependent;
    let err = assess(&overlapping, None, &request, &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "joint_law_required");
    // Index-pairing unrelated streams as if jointly drawn is the same refusal.
    let err = estimate_difference(
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        None,
        WithinStreamDependence::Independent,
        BatchPlan::SquareRoot,
        &Functional::ExpectedUtility,
        PairCoupling::SharedStream,
    )
    .unwrap_err();
    assert_eq!(err.reason_code(), "joint_law_required");
}

#[test]
fn validation_must_be_independent_of_the_selection_draws() {
    let design = KnownOptimumDesign {
        true_means: vec![0.0, 0.5, 1.0],
        noise_sd: 1.0,
        n_draws: 200,
        coupling: NoiseCoupling::SharedStream { correlation: 0.0 },
    };
    let selection =
        simulate_selection_draws(&design, &stream(StreamPurpose::Selection, 0)).unwrap();
    let request = ReliabilityRequest {
        validation: ValidationPlan::ProvidedStream {
            kind: ValidationKind::IndependentSimulationStream,
            gap: 0,
        },
        ..ReliabilityRequest::expected_utility()
    };
    // The same stream coordinates: refused.
    let err = assess(&selection, Some(&selection.clone()), &request, &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "validation_stream_not_independent");
    assert!(err.to_refusal().remedy.is_some());
    // A different replicate coordinate is a different stream: accepted, and recorded.
    let validation =
        simulate_selection_draws(&design, &stream(StreamPurpose::Validation, 0)).unwrap();
    let report = assess(&selection, Some(&validation), &request, &ctx()).unwrap();
    let SelectionUncertainty::Selected(assessment) = &report.selection_uncertainty else {
        panic!("selected");
    };
    assert_eq!(assessment.status, SelectionStatus::HeldOutValidated);
    let record = &assessment.held_out.as_ref().unwrap().record;
    assert_eq!(record.selection_segments[0].stream.purpose, StreamPurpose::Selection);
    assert_eq!(record.validation_segment.stream.purpose, StreamPurpose::Validation);
    // A shifted overlap of the selection stream is refused too.
    let mut shifted = validation.clone();
    for c in &mut shifted.candidates {
        c.segment =
            DrawSegment { stream: stream(StreamPurpose::Selection, 0), start: 100, len: 200 };
    }
    let err = assess(&selection, Some(&shifted), &request, &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "validation_stream_not_independent");
    // A serially dependent stream cannot be split without a gap.
    let mut chain = selection.clone();
    chain.within_stream = WithinStreamDependence::Autocorrelated;
    let err = split_draws(&chain, 0.5, 0).unwrap_err();
    assert_eq!(err.reason_code(), "validation_stream_not_independent");
    assert!(split_draws(&chain, 0.5, 20).is_ok());
}

// ---- the three components stay separate; closed routes ------------------------------

#[test]
fn the_three_components_are_separate_fields_and_a_combined_number_is_refused() {
    let design = KnownOptimumDesign {
        true_means: vec![0.0, 0.3],
        noise_sd: 2.0,
        n_draws: 500,
        coupling: NoiseCoupling::SharedStream { correlation: 0.5 },
    };
    let mut draws =
        simulate_selection_draws(&design, &stream(StreamPurpose::Selection, 4)).unwrap();
    // Declared posterior draws: the utility spread is source uncertainty.
    draws.source = SourceParadigm::Posterior;
    let report = assess(&draws, None, &ReliabilityRequest::expected_utility(), &ctx()).unwrap();
    let SourceUncertainty::Spread { candidates, .. } = &report.source_uncertainty else {
        panic!("posterior draws carry a spread");
    };
    let spread = &candidates[1];
    // Spread is a property of the law (sd 2.0), not of the draw count: it is about 2,
    // while the Monte Carlo standard error of the same candidate is 2 / sqrt(500).
    assert!((spread.standard_deviation - 2.0).abs() < 0.2);
    let mc = &report.monte_carlo_error.candidates[1].estimate;
    // The source spread is the empirical law's population SD; the Monte Carlo
    // error uses the Bessel-corrected sample variance on the same 500 draws.
    assert!((mc.standard_error - spread.standard_deviation / 499_f64.sqrt()).abs() < 1e-9);
    assert!(spread.standard_deviation > 20.0 * mc.standard_error);
    // Known law: no source uncertainty is reported.
    draws.source = SourceParadigm::KnownLaw;
    let known = assess(&draws, None, &ReliabilityRequest::expected_utility(), &ctx()).unwrap();
    assert_eq!(known.source_uncertainty, SourceUncertainty::KnownLaw);
    // Combined reporting refuses at the request and at the report.
    let combined = ReliabilityRequest {
        reporting: UncertaintyReporting::Combined,
        ..ReliabilityRequest::expected_utility()
    };
    let err = assess(&draws, None, &combined, &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "uncertainty_components_not_combined");
    assert_eq!(
        report.combined_uncertainty().unwrap_err().reason_code(),
        "uncertainty_components_not_combined"
    );
    // The in-sample value is labelled; the unadjusted maximum is never presented bare.
    let SelectionUncertainty::Selected(assessment) = &report.selection_uncertainty else {
        panic!("selected");
    };
    let json = serde_json::to_value(&**assessment).unwrap();
    assert_eq!(json["in_sample_label"], "in_sample");
    assert_eq!(json["status"], "winners_curse_bounded");
}

#[test]
fn every_published_ranking_or_interval_route_is_closed_at_its_coordinate() {
    // 100 draws per candidate is below the measured range (500 to 2000): no current
    // coverage record covers it, so every route refuses with the typed reason.
    let draws = gaussian_pair(0.5, 100, false);
    let report = assess(&draws, None, &ReliabilityRequest::expected_utility(), &ctx()).unwrap();
    let coordinate: &CalibrationCoordinate = &report.coordinate;
    assert_eq!(coordinate.candidate_set_size, 2);
    assert_eq!(coordinate.mc_budget, 100);
    assert_eq!(coordinate.selection_rule, "maximize_expected_utility");
    assert_eq!(report.closed_routes.len(), PublishedRoute::ALL.len());
    for route in PublishedRoute::ALL {
        let err = license_route(route, coordinate).unwrap_err();
        assert_eq!(err.reason_code(), "cell_not_licensed");
        let refusal = err.to_refusal();
        assert!(refusal.detail.starts_with("decision_reliability."));
        assert!(refusal.offending.unwrap().contains("candidate_set_size=2"));
    }
    assert!(report.closed_routes.iter().all(|r| r.code == "cell_not_licensed"));
}

#[test]
fn measured_routes_respect_current_attestation_and_report_their_closures() {
    use antecedent_design::decision_reliability::{calibration_basis, record_functional};
    use antecedent_io_core::CalibrationStatus;
    use antecedent_io_core::calibration::{
        RECORD_NOT_ATTESTING, calibration_slot, record_attests_current_code,
    };
    use antecedent_io_core::coverage_records_data::RECORDS;

    // These coordinates have measured records. A measured record licenses a route
    // only while its governing statistical surface is currently attested.
    let draws = gaussian_pair(0.5, 1000, false);
    let report = assess(&draws, None, &ReliabilityRequest::expected_utility(), &ctx()).unwrap();
    assert_eq!(report.coordinate.mc_budget, 1000);
    for route in PublishedRoute::ALL {
        let banded = calibration_basis(route, &report.coordinate, 0.95).unwrap();
        let mut exact = banded.clone();
        exact.key.functional = record_functional(&report.coordinate);
        let slots = [calibration_slot(&banded), calibration_slot(&exact)];
        assert!(
            slots.iter().any(|slot| slot.record_id.is_some()),
            "{} lacks a measured record",
            route.as_str()
        );
        for slot in &slots {
            if slot.reason.as_deref() == Some(RECORD_NOT_ATTESTING) {
                let record = RECORDS
                    .iter()
                    .find(|record| Some(record.id) == slot.record_id.as_deref())
                    .unwrap();
                assert!(!record_attests_current_code(record));
            }
        }
        let should_open = slots.iter().any(|slot| slot.status == CalibrationStatus::Calibrated);
        let result = license_route(route, &report.coordinate);
        assert_eq!(result.is_ok(), should_open, "{} attestation decision", route.as_str());
        assert_eq!(report.closed_routes.iter().any(|closed| closed.route == route), !should_open);
        if let Err(error) = result {
            assert_eq!(error.reason_code(), "cell_not_licensed");
        }
    }
    // A different nominal level and serial dependence remain outside the records.
    let at_ninety = antecedent_design::decision_reliability::license_route_at(
        PublishedRoute::RankingProbability,
        &report.coordinate,
        0.9,
    )
    .unwrap_err();
    assert_eq!(at_ninety.reason_code(), "cell_not_licensed");
    let mut dependent = report.coordinate.clone();
    dependent.within_stream =
        antecedent_design::decision_reliability::WithinStreamDependence::Autocorrelated;
    assert_eq!(
        license_route(PublishedRoute::UtilityInterval, &dependent).unwrap_err().reason_code(),
        "cell_not_licensed"
    );
}

#[test]
fn pre_specified_decisions_have_no_selection_effect_and_adaptive_ones_cannot_claim_it() {
    let draws = gaussian_pair(0.0, 400, false);
    let request = ReliabilityRequest {
        decision: DecisionChoice::PreSpecified("c00001".into()),
        ..ReliabilityRequest::expected_utility()
    };
    let report = assess(&draws, None, &request, &ctx()).unwrap();
    assert_eq!(report.decided_id, "c00001");
    assert_eq!(report.selection_uncertainty, SelectionUncertainty::NotSelected);
    let adaptive = adaptive_carrier(Some(record(10, 20)));
    let err = assess(&adaptive, None, &request, &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "invalid_argument");
    let missing = ReliabilityRequest {
        decision: DecisionChoice::PreSpecified("absent".into()),
        ..ReliabilityRequest::expected_utility()
    };
    let absent = assess(&draws, None, &missing, &ctx()).unwrap_err();
    assert_eq!(absent.reason_code(), "invalid_argument");
}

#[test]
fn in_sample_selection_breaks_ties_by_semantic_id_and_ignores_listing_order() {
    let tied = [("b", vec![1.0, 2.0, 3.0, 4.0]), ("a", vec![1.0, 2.0, 3.0, 4.0])];
    let forward = carrier(&tied);
    let mut reversed = tied.clone();
    reversed.reverse();
    let backward = carrier(&reversed);
    let pick = |d: &DecisionDraws| {
        select_in_sample(d, &Functional::ExpectedUtility, BatchPlan::SquareRoot).unwrap()
    };
    assert_eq!(pick(&forward).selected_id, "a");
    assert_eq!(pick(&forward), pick(&backward));
}

#[test]
fn cancellation_is_a_typed_refusal_not_a_partial_result() {
    let draws = gaussian_pair(0.0, 200, false);
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let err =
        assess(&draws, None, &ReliabilityRequest::expected_utility(), &cancelled).unwrap_err();
    assert_eq!(err, ReliabilityError::Cancelled);
    assert_eq!(err.reason_code(), "cancelled_no_claim");
}

// ---- artifact -----------------------------------------------------------------------

fn sealed_artifact() -> (DecisionReliabilityArtifact, DecisionReliabilityExpectation) {
    let design = KnownOptimumDesign {
        true_means: vec![0.2, 0.0, 0.4],
        noise_sd: 1.0,
        n_draws: 300,
        coupling: NoiseCoupling::SharedStream { correlation: 0.4 },
    };
    let selection =
        simulate_selection_draws(&design, &stream(StreamPurpose::Selection, 0)).unwrap();
    let validation =
        simulate_selection_draws(&design, &stream(StreamPurpose::Validation, 0)).unwrap();
    let request = ReliabilityRequest {
        validation: ValidationPlan::ProvidedStream {
            kind: ValidationKind::IndependentSimulationStream,
            gap: 0,
        },
        ranking: Some(RankingRequest {
            baseline: Some("c00001".into()),
            top_k: vec![2],
            batch_plan: BatchPlan::SquareRoot,
        }),
        ..ReliabilityRequest::expected_utility()
    };
    let expected = DecisionReliabilityExpectation {
        draws_digest: Some(selection.digest_hex()),
        candidate_set_digest: Some(selection.candidate_set_digest()),
        selection_stream: Some(stream(StreamPurpose::Selection, 0)),
        validation_stream: Some(stream(StreamPurpose::Validation, 0)),
        utility_unit: Some("utility".into()),
        decided_id: None,
    };
    let artifact =
        DecisionReliabilityArtifact::new(selection, Some(validation), request, &ctx()).unwrap();
    (artifact, expected)
}

#[test]
fn the_artifact_round_trips_and_is_recomputed_from_the_draws() {
    let (artifact, expected) = sealed_artifact();
    let bytes = artifact.to_bytes("decision-reliability-1", &ctx()).unwrap();
    let consumed = consume(&bytes, &expected, &ctx()).unwrap();
    assert_eq!(consumed, artifact);
    // Deterministic encoding.
    assert_eq!(bytes, artifact.to_bytes("decision-reliability-1", &ctx()).unwrap());
    // Bit-for-bit: the consumer's recomputation equals the stored report.
    let recomputed =
        assess(&consumed.draws, consumed.validation.as_ref(), &consumed.request, &ctx()).unwrap();
    assert_eq!(recomputed, consumed.report);
    // The decision is the true optimum here (index 2, true utility 0.4 against 0.2).
    assert_eq!(consumed.report.decided_id, "c00002");
}

#[test]
fn semantic_mutations_of_the_artifact_are_refused() {
    let (artifact, expected) = sealed_artifact();
    let bytes = artifact.to_bytes("decision-reliability-2", &ctx()).unwrap();
    let default = DecisionReliabilityExpectation::default();
    let reseal = |forged: &mut DecisionReliabilityArtifact| {
        forged.digest = forged.recomputed_digest().unwrap();
    };
    let message = |forged: &DecisionReliabilityArtifact,
                   expected: &DecisionReliabilityExpectation| {
        forged.verify(expected, &ctx()).unwrap_err().to_string()
    };

    // One utility changed: the digest no longer holds; resealed, the report no longer
    // recomputes.
    let mut forged = DecisionReliabilityArtifact::from_bytes(&bytes).unwrap();
    forged.draws.candidates[0].utilities[0] += 1.0;
    assert!(message(&forged, &default).contains("identity_mismatch"));
    reseal(&mut forged);
    assert!(message(&forged, &default).contains("report_mismatch"));

    // A claimed value edited.
    let mut forged = DecisionReliabilityArtifact::from_bytes(&bytes).unwrap();
    forged.report.monte_carlo_error.decision.standard_error *= 0.5;
    reseal(&mut forged);
    assert!(message(&forged, &default).contains("report_mismatch"));

    // The decision named differently, or a held-out claim removed.
    let mut forged = DecisionReliabilityArtifact::from_bytes(&bytes).unwrap();
    forged.report.decided_id = "c00001".into();
    reseal(&mut forged);
    assert!(message(&forged, &default).contains("report_mismatch"));
    let mut forged = DecisionReliabilityArtifact::from_bytes(&bytes).unwrap();
    forged.validation = None;
    forged.request.validation = ValidationPlan::NoValidation;
    reseal(&mut forged);
    assert!(message(&forged, &default).contains("report_mismatch"));

    // A stream coordinate changed (selection claimed to be another replicate), and a
    // declaration changed (independent claimed where the draws share a stream).
    let mut forged = DecisionReliabilityArtifact::from_bytes(&bytes).unwrap();
    for c in &mut forged.draws.candidates {
        c.segment.stream.replicate = 9;
    }
    reseal(&mut forged);
    assert!(message(&forged, &default).contains("report_mismatch"));
    let mut forged = DecisionReliabilityArtifact::from_bytes(&bytes).unwrap();
    forged.draws.coupling = CandidateCoupling::DeclaredIndependent;
    reseal(&mut forged);
    assert!(message(&forged, &default).contains("declared_independent_but_draws_overlap"));

    // A consistently resealed forgery (draws and report both regenerated) is caught by
    // an independently retained expectation.
    let mut other = sealed_artifact().0;
    other.draws.candidates[0].utilities[0] += 1.0;
    let rebuilt = DecisionReliabilityArtifact::new(
        other.draws.clone(),
        other.validation.clone(),
        other.request.clone(),
        &ctx(),
    )
    .unwrap();
    assert!(rebuilt.verify(&default, &ctx()).is_ok(), "internally consistent");
    assert!(message(&rebuilt, &expected).contains("draws_digest_mismatch"));
    // Wrong retained streams.
    let wrong = DecisionReliabilityExpectation {
        selection_stream: Some(stream(StreamPurpose::Selection, 5)),
        ..expected.clone()
    };
    assert!(message(&artifact, &wrong).contains("selection_stream_mismatch"));
    let wrong = DecisionReliabilityExpectation {
        validation_stream: Some(stream(StreamPurpose::Validation, 5)),
        ..expected
    };
    assert!(message(&artifact, &wrong).contains("validation_stream_mismatch"));

    // A damaged container is refused before any recomputation.
    let mut damaged = bytes.clone();
    let middle = damaged.len() / 2;
    damaged[middle] ^= 0xFF;
    assert!(DecisionReliabilityArtifact::from_bytes(&damaged).is_err());
    // A different artifact kind is not decoded.
    assert!(DecisionReliabilityArtifact::from_bytes(b"not an artifact").is_err());
}

#[test]
fn stream_identities_are_semantic_coordinates() {
    let a = stream(StreamPurpose::Selection, 0);
    assert_eq!(a.digest_hex(), stream(StreamPurpose::Selection, 0).digest_hex());
    assert_ne!(a.digest_hex(), stream(StreamPurpose::Validation, 0).digest_hex());
    assert_ne!(a.digest_hex(), stream(StreamPurpose::Selection, 1).digest_hex());
    // Draws are a function of the coordinates only: regenerating gives identical bits,
    // and another replicate gives different draws.
    let design = KnownOptimumDesign {
        true_means: vec![0.0, 1.0],
        noise_sd: 1.0,
        n_draws: 50,
        coupling: NoiseCoupling::SharedStream { correlation: 0.2 },
    };
    let first = simulate_selection_draws(&design, &a).unwrap();
    let again = simulate_selection_draws(&design, &a).unwrap();
    let other = simulate_selection_draws(&design, &stream(StreamPurpose::Selection, 1)).unwrap();
    assert_eq!(first, again);
    assert_ne!(first.candidates[0].utilities, other.candidates[0].utilities);
    // Each candidate's stream under separate streams is its own coordinate.
    let separate = KnownOptimumDesign { coupling: NoiseCoupling::SeparateStreams, ..design };
    let drawn = simulate_selection_draws(&separate, &a).unwrap();
    assert_ne!(drawn.candidates[0].segment.stream, drawn.candidates[1].segment.stream);
    assert_eq!(drawn.coupling, CandidateCoupling::DeclaredIndependent);
}

#[test]
fn a_validation_stream_already_used_is_not_a_clean_hold_out() {
    let design = KnownOptimumDesign {
        true_means: vec![0.0, 0.5],
        noise_sd: 1.0,
        n_draws: 100,
        coupling: NoiseCoupling::SharedStream { correlation: 0.0 },
    };
    let selection =
        simulate_selection_draws(&design, &stream(StreamPurpose::Selection, 0)).unwrap();
    let mut validation =
        simulate_selection_draws(&design, &stream(StreamPurpose::Validation, 0)).unwrap();
    let request = ReliabilityRequest {
        validation: ValidationPlan::ProvidedStream {
            kind: ValidationKind::IndependentSimulationStream,
            gap: 0,
        },
        ..ReliabilityRequest::expected_utility()
    };
    let clean = assess(&selection, Some(&validation), &request, &ctx()).unwrap();
    assert_eq!(clean.dependence.validation_independent_of_selection, Some(true));
    assert_eq!(clean.dependence.repeated_evaluations, 0);
    validation.repeated_evaluations = 1;
    let err = assess(&selection, Some(&validation), &request, &ctx()).unwrap_err();
    assert_eq!(err.reason_code(), "validation_stream_not_independent");
}

#[test]
fn rank_statements_on_independently_drawn_candidates_are_refused_by_the_propagation_rule() {
    // Declared-independent candidates on separate streams have no joint law: a rank or
    // better-than probability is a nonlinear functional of it, and the rule never pairs
    // independent marginals by index.
    let separate = gaussian_pair(0.0, 200, true);
    assert_eq!(separate.coupling, CandidateCoupling::DeclaredIndependent);
    let error = rank(&separate, &RankingRequest::default(), &ctx()).unwrap_err();
    assert!(matches!(
        error,
        ReliabilityError::JointLawRequired { reason: "nonlinear_needs_aligned_joint_law" }
    ));
    assert_eq!(error.reason_code(), "joint_law_required");

    // The mean difference of the same candidates is a linear contrast: independence of
    // distinct streams licenses it (variances add), so the carrier still assesses.
    assert!(assess(&separate, None, &ReliabilityRequest::expected_utility(), &ctx()).is_ok());

    // One shared simulation is an aligned joint law and ranks.
    let shared = gaussian_pair(0.5, 200, false);
    assert!(rank(&shared, &RankingRequest::default(), &ctx()).is_ok());
}
