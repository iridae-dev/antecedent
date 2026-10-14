//! Derived intervals and confidence sets against hand-calculated truth.
//!
//! Oracles are hand arithmetic on alternating `mean +/- 1` series (whose sample mean,
//! sample variance and standard error are exact) and enumerated 0/1 series; critical
//! values are `norm_inv` of the stated tail mass. No tolerance is tuned to an observed
//! value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    BatchPlan, CandidateCoupling, CandidateDraws, CandidateOrigin, CandidateSearchRecord,
    ConfidenceSetBasis, DecisionDraws, DerivedStanding, DrawSegment, IntervalMultiplicity,
    IntervalRequest, KnownOptimumDesign, NoiseCoupling, NumericalIntervalMethod, PublishedRoute,
    RankingRequest, ReliabilityError, ReliabilityRequest, SelectionRule, SourceParadigm,
    SourceUncertainty, StreamId, StreamPurpose, ValidationKind, ValidationPlan,
    WithinStreamDependence, coverage_design, derive_uncertainty, simulate_selection_draws,
};
use antecedent_design::decision_reliability::ProbabilityQuantity;
use antecedent_kernels::{NORMAL_Q975, norm_inv};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

/// Upper-tail critical value at one-sided tail mass `tail`.
fn z(tail: f64) -> f64 {
    -norm_inv(tail)
}

fn series_stream(purpose: StreamPurpose, label: &str) -> StreamId {
    StreamId::new(555, purpose, 0, label)
}

/// `mean + 1, mean - 1, ...` over `n` draws (`n` even): sample mean `mean`, sample
/// variance `n / (n - 1)`, so the standard error of the mean is `sqrt(1 / (n - 1))`.
fn alternating(mean: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| mean + if i % 2 == 0 { 1.0 } else { -1.0 }).collect()
}

fn se_alternating(n: usize) -> f64 {
    (1.0 / (n as f64 - 1.0)).sqrt()
}

fn carrier_of(
    columns: &[(String, Vec<f64>)],
    purpose: StreamPurpose,
    coupling: CandidateCoupling,
    origin: CandidateOrigin,
    record: Option<CandidateSearchRecord>,
) -> DecisionDraws {
    let n = columns[0].1.len();
    let shared = DrawSegment::whole(series_stream(purpose, "shared"), n);
    DecisionDraws {
        utility_unit: "utility".into(),
        producer: "test".into(),
        source: SourceParadigm::KnownLaw,
        coupling,
        within_stream: WithinStreamDependence::Independent,
        weights: None,
        repeated_evaluations: 0,
        origin,
        search_record: record,
        candidates: columns
            .iter()
            .map(|(id, utilities)| CandidateDraws {
                id: id.clone(),
                utilities: utilities.clone(),
                segment: if coupling == CandidateCoupling::SharedSimulation {
                    shared.clone()
                } else {
                    DrawSegment::whole(series_stream(purpose, id), n)
                },
            })
            .collect(),
    }
}

fn independent(columns: &[(&str, Vec<f64>)], purpose: StreamPurpose) -> DecisionDraws {
    let owned: Vec<(String, Vec<f64>)> =
        columns.iter().map(|(id, u)| ((*id).to_owned(), u.clone())).collect();
    carrier_of(
        &owned,
        purpose,
        CandidateCoupling::DeclaredIndependent,
        CandidateOrigin::Enumerated,
        None,
    )
}

fn three(n: usize) -> DecisionDraws {
    independent(
        &[("A", alternating(10.0, n)), ("B", alternating(9.9, n)), ("C", alternating(0.0, n))],
        StreamPurpose::Selection,
    )
}

fn expected_utility_request() -> ReliabilityRequest {
    ReliabilityRequest::expected_utility()
}

// ---- numerical component ------------------------------------------------------------

#[test]
fn wilson_interval_at_the_effective_sample_size_matches_hand_arithmetic() {
    // A beats B in exactly 50 of 100 draws, no ties: every probability is 1/2, the
    // series is iid so the effective sample size is 100.
    let n = 100;
    let a: Vec<f64> = (0..n).map(|i| if i < 50 { 1.0 } else { 0.0 }).collect();
    let b: Vec<f64> = (0..n).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect();
    let owned = vec![("A".to_owned(), a), ("B".to_owned(), b)];
    let draws = carrier_of(
        &owned,
        StreamPurpose::Selection,
        CandidateCoupling::SharedSimulation,
        CandidateOrigin::Enumerated,
        None,
    );
    let request = ReliabilityRequest {
        ranking: Some(RankingRequest {
            baseline: Some("B".into()),
            top_k: vec![],
            batch_plan: BatchPlan::SquareRoot,
        }),
        ..expected_utility_request()
    };
    let derived =
        derive_uncertainty(&draws, None, &request, &IntervalRequest::marginal(0.95), &ctx())
            .unwrap();
    assert_eq!(derived.standing, DerivedStanding::DerivedUnmeasured);
    // Wilson, p = 0.5, n = 100, z = 1.959964: centre 0.5, half-width 0.09617.
    let best = derived
        .numerical
        .probabilities
        .iter()
        .find(|p| p.id == "A" && p.quantity == ProbabilityQuantity::Best)
        .unwrap();
    assert_eq!(best.interval.method, NumericalIntervalMethod::WilsonEffectiveSample);
    assert!((best.interval.effective_sample_size - 100.0).abs() < 1e-12);
    assert!((best.interval.lower - 0.403_83).abs() < 1e-4, "{}", best.interval.lower);
    assert!((best.interval.upper - 0.596_17).abs() < 1e-4, "{}", best.interval.upper);
    // Same series, so the better-than-baseline interval is identical.
    let versus = derived
        .numerical
        .probabilities
        .iter()
        .find(|p| p.id == "A" && p.quantity == ProbabilityQuantity::BetterThanBaseline)
        .unwrap();
    assert!((versus.interval.lower - best.interval.lower).abs() < 1e-12);
    // Wald interval of the expected utility: se = sqrt(0.25 * 100 / 99 / 100).
    let se = (0.25_f64 * 100.0 / 99.0 / 100.0).sqrt();
    let utility = derived.numerical.utilities.iter().find(|u| u.id == "A").unwrap();
    assert_eq!(utility.interval.method, NumericalIntervalMethod::WaldNormal);
    assert!((utility.interval.lower - (0.5 - NORMAL_Q975 * se)).abs() < 1e-9);
    assert!((utility.interval.upper - (0.5 + NORMAL_Q975 * se)).abs() < 1e-9);
    assert!((derived.numerical.minimum_effective_sample_size - 100.0).abs() < 1e-12);
    // The law was declared known: no source spread is reported.
    assert_eq!(derived.source, SourceUncertainty::KnownLaw);
}

#[test]
fn simultaneous_multiplicity_widens_every_numerical_interval_by_bonferroni() {
    let draws = three(100);
    let marginal =
        derive_uncertainty(&draws, None, &expected_utility_request(), &IntervalRequest::marginal(0.95), &ctx())
            .unwrap();
    let simultaneous = derive_uncertainty(
        &draws,
        None,
        &expected_utility_request(),
        &IntervalRequest { level: 0.95, multiplicity: IntervalMultiplicity::Simultaneous },
        &ctx(),
    )
    .unwrap();
    // Three candidate intervals and two differences to the decision: five intervals.
    assert_eq!(simultaneous.numerical.intervals_reported, 5);
    let se = se_alternating(100);
    let a = simultaneous.numerical.utilities.iter().find(|u| u.id == "A").unwrap();
    assert!((a.interval.level - 0.99).abs() < 1e-12);
    let half = z(0.05 / 5.0 / 2.0) * se;
    assert!((a.interval.upper - (10.0 + half)).abs() < 1e-9);
    let narrow = marginal.numerical.utilities.iter().find(|u| u.id == "A").unwrap();
    assert!((narrow.interval.upper - (10.0 + NORMAL_Q975 * se)).abs() < 1e-9);
    assert!(a.interval.upper > narrow.interval.upper);
}

#[test]
fn common_random_numbers_narrow_the_difference_interval() {
    let design = |coupling| KnownOptimumDesign {
        true_means: vec![1.0, 0.0],
        noise_sd: 1.0,
        n_draws: 4_000,
        coupling,
    };
    let stream = StreamId::new(91, StreamPurpose::Selection, 0, "crn");
    let request = expected_utility_request();
    let interval = IntervalRequest::marginal(0.9);
    let shared = derive_uncertainty(
        &simulate_selection_draws(&design(NoiseCoupling::SharedStream { correlation: 0.9 }), &stream)
            .unwrap(),
        None,
        &request,
        &interval,
        &ctx(),
    )
    .unwrap();
    let separate = derive_uncertainty(
        &simulate_selection_draws(&design(NoiseCoupling::SeparateStreams), &stream).unwrap(),
        None,
        &request,
        &interval,
        &ctx(),
    )
    .unwrap();
    assert!(shared.numerical.shared_noise && !separate.numerical.shared_noise);
    let width = |d: &antecedent_design::decision_reliability::DerivedUncertainty| {
        let i = &d.numerical.differences_vs_decision[0].interval;
        i.upper - i.lower
    };
    // Variance ratio is 1 - rho = 0.1, so the standard error ratio is about 0.32.
    assert!(width(&shared) < 0.5 * width(&separate));
}

// ---- selection component ------------------------------------------------------------

#[test]
fn confidence_set_excludes_the_clearly_worse_candidate_with_the_paired_critical_value() {
    let draws = three(100);
    let derived = derive_uncertainty(
        &draws,
        None,
        &expected_utility_request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    let set = &derived.selection.confidence_set;
    assert_eq!(set.basis, ConfidenceSetBasis::PairedComparisonsWithBest);
    // A and B differ by 0.1 against a paired standard error of sqrt(2) * 0.1005 = 0.142,
    // far inside 1.96 standard errors; C is 10 below.
    assert_eq!(set.members, vec!["A".to_owned(), "B".to_owned()]);
    assert_eq!(set.excluded, vec!["C".to_owned()]);
    // One-sided tail alpha / (K - 1) = 0.025.
    assert!((set.critical_value - NORMAL_Q975).abs() < 1e-9);
    assert!((set.level - 0.95).abs() < 1e-12);
    assert_eq!(set.unretained_candidates, 0);
}

#[test]
fn many_candidates_use_the_dependence_free_marginal_construction() {
    // 70 candidates: 69 at zero and one at ten. Past the paired-comparison cap.
    let mut columns: Vec<(String, Vec<f64>)> =
        (0..69).map(|j| (format!("c{j:02}"), alternating(0.0, 100))).collect();
    columns.push(("c69".to_owned(), alternating(10.0, 100)));
    let draws = carrier_of(
        &columns,
        StreamPurpose::Selection,
        CandidateCoupling::DeclaredIndependent,
        CandidateOrigin::Enumerated,
        None,
    );
    let derived = derive_uncertainty(
        &draws,
        None,
        &expected_utility_request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    let set = &derived.selection.confidence_set;
    assert_eq!(set.basis, ConfidenceSetBasis::MarginalBonferroni);
    // Two-sided Bonferroni over 70: tail 0.05 / 140. 10 - 3.4 * 0.1005 is above 0 + 3.4 *
    // 0.1005, so only the dominant candidate remains.
    assert!((set.critical_value - z(0.05 / 140.0)).abs() < 1e-9);
    assert_eq!(set.members, vec!["c69".to_owned()]);
    assert_eq!(set.excluded.len(), 69);
}

#[test]
fn selection_corrected_interval_reports_the_extra_width_the_selection_costs() {
    let draws = three(100);
    let derived = derive_uncertainty(
        &draws,
        None,
        &expected_utility_request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    let corrected = derived.selection.corrected.as_ref().unwrap();
    let se = se_alternating(100);
    assert_eq!(corrected.selected_id, "A");
    assert_eq!(corrected.n_considered, 3);
    assert!((corrected.in_sample - 10.0).abs() < 1e-12);
    assert!((corrected.standard_error - se).abs() < 1e-12);
    let (ordinary, simultaneous) = (z(0.025), z(0.05 / 6.0));
    assert!((corrected.numerical_half_width - ordinary * se).abs() < 1e-9);
    assert!(
        (corrected.selection_extra_half_width - (simultaneous - ordinary) * se).abs() < 1e-9
    );
    assert!(corrected.selection_extra_half_width > 0.0);
    assert!((corrected.lower - (10.0 - simultaneous * se)).abs() < 1e-9);
    assert!((corrected.upper - (10.0 + simultaneous * se)).abs() < 1e-9);
    // Independent equal noise: exact expected maximum of three standard normals,
    // 0.8462843753 times the common standard error.
    let bound = corrected.expected_bias_bound.unwrap();
    assert!((bound - 0.846_284_375_3 * se).abs() < 1e-6);
    assert!(corrected.held_out.is_none());
}

#[test]
fn held_out_interval_comes_from_the_independent_stream_and_is_purely_numerical() {
    let selection = three(100);
    let validation = independent(
        &[("A", alternating(9.8, 100)), ("B", alternating(9.9, 100)), ("C", alternating(0.0, 100))],
        StreamPurpose::Validation,
    );
    let request = ReliabilityRequest {
        validation: ValidationPlan::ProvidedStream {
            kind: ValidationKind::IndependentSimulationStream,
            gap: 0,
        },
        ..expected_utility_request()
    };
    let derived = derive_uncertainty(
        &selection,
        Some(&validation),
        &request,
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    let corrected = derived.selection.corrected.as_ref().unwrap();
    let held = corrected.held_out.unwrap();
    let se = se_alternating(100);
    assert!((held.estimate - 9.8).abs() < 1e-12);
    assert!((held.lower - (9.8 - NORMAL_Q975 * se)).abs() < 1e-9);
    assert!((held.upper - (9.8 + NORMAL_Q975 * se)).abs() < 1e-9);
    // The in-sample value (10) lies above the held-out one: realized optimism 0.2.
    assert!(corrected.in_sample > held.estimate);
}

#[test]
fn an_adaptive_search_enters_the_critical_value_but_not_the_confidence_set() {
    let columns = vec![
        ("A".to_owned(), alternating(10.0, 100)),
        ("B".to_owned(), alternating(9.9, 100)),
        ("C".to_owned(), alternating(0.0, 100)),
    ];
    let record = CandidateSearchRecord {
        generator: "random_search/1".into(),
        budget: 40,
        n_evaluated: 40,
        selection_rule: SelectionRule::MaximizeExpectedUtility,
        generator_stream: None,
    };
    let adaptive = carrier_of(
        &columns,
        StreamPurpose::Selection,
        CandidateCoupling::DeclaredIndependent,
        CandidateOrigin::AdaptivelyGenerated,
        Some(record),
    );
    let derived = derive_uncertainty(
        &adaptive,
        None,
        &expected_utility_request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    assert_eq!(derived.selection.confidence_set.unretained_candidates, 37);
    let corrected = derived.selection.corrected.as_ref().unwrap();
    assert_eq!(corrected.n_considered, 40);
    assert!((corrected.critical_value - z(0.05 / 80.0)).abs() < 1e-9);
    // Without the record the same candidates are refused outright.
    let unrecorded = carrier_of(
        &columns,
        StreamPurpose::Selection,
        CandidateCoupling::DeclaredIndependent,
        CandidateOrigin::AdaptivelyGenerated,
        None,
    );
    let error = derive_uncertainty(
        &unrecorded,
        None,
        &expected_utility_request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap_err();
    assert!(matches!(error, ReliabilityError::SearchRecordRequired { candidates: 3 }));
    assert_eq!(error.reason_code(), "candidate_search_record_required");
}

#[test]
fn a_pre_specified_decision_has_no_selection_corrected_interval() {
    let request = ReliabilityRequest {
        decision: antecedent_design::decision_reliability::DecisionChoice::PreSpecified("B".into()),
        ..expected_utility_request()
    };
    let derived = derive_uncertainty(
        &three(100),
        None,
        &request,
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    assert_eq!(derived.decided_id, "B");
    assert!(derived.selection.corrected.is_none());
    // The numerical component still stands for the pre-specified decision.
    assert_eq!(derived.numerical.utilities.len(), 3);
}

// ---- refusals -----------------------------------------------------------------------

#[test]
fn an_effective_sample_size_below_the_floor_refuses_with_a_typed_detail() {
    // Ten iid draws: the normal approximation is not derived at an effective size of ten.
    let error = derive_uncertainty(
        &three(10),
        None,
        &expected_utility_request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ReliabilityError::EffectiveSampleSizeTooSmall { effective_sample_size, minimum, .. }
            if (effective_sample_size - 10.0).abs() < 1e-12 && (minimum - 30.0).abs() < 1e-12
    ));
    let refusal = error.to_refusal();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "decision_reliability.effective_sample_size_too_small");
    assert!(refusal.remedy.is_some());
}

#[test]
fn levels_outside_the_derived_range_and_undeclared_dependence_are_refused() {
    for level in [0.3, 0.99999, f64::NAN] {
        let error = derive_uncertainty(
            &three(100),
            None,
            &expected_utility_request(),
            &IntervalRequest::marginal(level),
            &ctx(),
        )
        .unwrap_err();
        assert!(matches!(error, ReliabilityError::InvalidInput { slot: "interval_level", .. }));
    }
    let mut unknown = three(100);
    unknown.coupling = CandidateCoupling::Unknown;
    let error = derive_uncertainty(
        &unknown,
        None,
        &expected_utility_request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.reason_code(), "sampling_dependence_unknown");
}

#[test]
fn a_derivation_is_unmeasured_so_every_route_and_the_combined_number_stay_closed() {
    let derived = derive_uncertainty(
        &three(100),
        None,
        &expected_utility_request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    for route in PublishedRoute::ALL {
        let error = derived.publish(route).unwrap_err();
        assert_eq!(error.reason_code(), "cell_not_licensed");
        assert!(error.to_refusal().offending.unwrap().contains("candidate_set_size=3"));
    }
    assert_eq!(
        derived.combined_uncertainty().unwrap_err().reason_code(),
        "uncertainty_components_not_combined"
    );
    assert!(derived.assumptions.iter().any(|a| a.contains("treated_as_known")));
}

// ---- fresh consumer -----------------------------------------------------------------

#[test]
fn a_fresh_consumer_rederives_and_refuses_edited_draws_or_edited_intervals() {
    let draws = three(100);
    let request = expected_utility_request();
    let derived = derive_uncertainty(
        &draws,
        None,
        &request,
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    derived.verify(&draws, None, &request, &ctx()).unwrap();
    // One utility changed at unchanged shape: the stored derivation no longer recomputes.
    let mut edited = draws.clone();
    edited.candidates[0].utilities[0] += 0.5;
    assert!(derived.verify(&edited, None, &request, &ctx()).is_err());
    // A claimed interval edited after the fact.
    let mut forged = derived.clone();
    forged.selection.corrected.as_mut().unwrap().lower += 1.0;
    assert!(forged.verify(&draws, None, &request, &ctx()).is_err());
    // A changed request level is part of the derivation.
    let mut other_level = derived.clone();
    other_level.request.level = 0.9;
    assert!(other_level.verify(&draws, None, &request, &ctx()).is_err());
}

// ---- measurement design -------------------------------------------------------------

#[test]
fn the_coverage_design_names_a_truth_and_a_grid_for_every_derived_route() {
    let design = coverage_design();
    let routes: Vec<&str> = design.iter().map(|c| c.route).collect();
    for expected in [
        "ranking_probability_interval",
        "optimality_confidence_set",
        "selection_corrected_interval",
        "held_out_interval",
        "winners_curse_bound",
        "utility_interval",
    ] {
        assert!(routes.contains(&expected), "missing {expected}");
    }
    let mut unique = routes.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), routes.len());
    for cell in &design {
        assert!(!cell.candidate_set_sizes.is_empty() && !cell.mc_budgets.is_empty());
        assert!(!cell.selection_rules.is_empty() && !cell.boundary_probes.is_empty());
        assert!(!cell.truth_description.is_empty() && !cell.pass_rule.is_empty());
        assert!(cell.min_replicates >= 2_000);
        assert!(cell.nominal_levels.iter().all(|l| *l > 0.0 && *l <= 1.0));
    }
    let selection = design.iter().find(|c| c.route == "selection_corrected_interval").unwrap();
    assert!(selection.candidate_set_sizes.contains(&64));
    assert!(selection.mc_budgets.contains(&1_000));
}
