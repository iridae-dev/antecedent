//! Facade integration of derived decision-reliability uncertainty: three separate
//! components for a hand-enumerated law, the producer connection, the coverage design as
//! data and the refusals each carries.
//!
//! Hand oracle: the four-state law of the point-diagnostic tests repeated ten times
//! (forty draws, serially independent, shared simulation, law declared known):
//! * A = (3, 3, 2, 2), B = (2, 1, 3, 2), C = (1, 2, 1, 2): utilities 5/2, 2, 3/2.
//! * Sample variance of A is 40 * (1/4) / 39, so its standard error is `0.5 / sqrt(39)`.
//! * A - B is (1, 2, -1, 0) repeated: mean 1/2, population variance 5/4, standard error
//!   `sqrt(1.25 / 39)`; A - C is (2, 1, 1, 0): mean 1, population variance 1/2.
//! * P(best): A 7/12, B 1/3, C 1/12; its Wilson interval at effective sample size 40 is
//!   computed below from the closed form.
//! * Normal quantiles: `z(0.975) = 1.959964`; the Bonferroni quantile for three
//!   candidates at the 95% level, `z(1 - 0.05 / 6) = 2.39398`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::decision_reliability::{
    CandidateCoupling, CandidateDraws, CandidateOrigin, CohortUtilityDraws, ConfidenceSetBasis,
    DecisionDraws, DecisionReliabilityExpectation, DerivedStanding, DrawSegment,
    IntervalMultiplicity, IntervalRequest, JointLawDeclaration, JointLawStanding,
    OffPolicyReturnDraws, PointOnlyOutput, PreparedDecisionReliability, ProbabilityQuantity,
    ProducerKind, ProducerOutput, PublishedRoute, RankingRequest, ReliabilityRequest, RolloutNoise,
    RolloutUtilityDraws, SelectionUncertainty, SourceParadigm, SourceUncertainty, StreamId,
    StreamPurpose, TruthKind, WithinStreamDependence, consume_derived_uncertainty,
    decision_reliability_coverage_design, derive_decision_uncertainty, derive_producer_uncertainty,
    publish_derived_route, reach_decision_carrier,
};
use antecedent_core::ExecutionContext;

const SE_A: f64 = 0.080_064_076_902_543_43; // 0.5 / sqrt(39)
const Z_975: f64 = 1.959_963_984_540_054;
const Z_BONFERRONI_3: f64 = 2.393_980_0;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn repeated(pattern: [f64; 4], times: usize) -> Vec<f64> {
    (0..times).flat_map(|_| pattern).collect()
}

fn law(times: usize) -> DecisionDraws {
    let n = 4 * times;
    let segment = DrawSegment::whole(StreamId::new(1, StreamPurpose::Selection, 0, "states"), n);
    let candidate = |id: &str, pattern: [f64; 4]| CandidateDraws {
        id: id.to_owned(),
        utilities: repeated(pattern, times),
        segment: segment.clone(),
    };
    DecisionDraws {
        utility_unit: "utility".into(),
        producer: "enumerated_finite_law".into(),
        source: SourceParadigm::KnownLaw,
        coupling: CandidateCoupling::SharedSimulation,
        within_stream: WithinStreamDependence::Independent,
        weights: None,
        repeated_evaluations: 0,
        origin: CandidateOrigin::Enumerated,
        search_record: None,
        candidates: vec![
            candidate("A", [3.0, 3.0, 2.0, 2.0]),
            candidate("B", [2.0, 1.0, 3.0, 2.0]),
            candidate("C", [1.0, 2.0, 1.0, 2.0]),
        ],
    }
}

fn request() -> ReliabilityRequest {
    ReliabilityRequest {
        ranking: Some(RankingRequest::default()),
        ..ReliabilityRequest::expected_utility()
    }
}

fn wilson(p: f64, n: f64, z: f64) -> (f64, f64) {
    let denominator = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denominator;
    let half = z / denominator * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
    (centre - half, centre + half)
}

#[test]
fn derived_components_match_the_enumerated_law_and_stay_separate() {
    let prepared = PreparedDecisionReliability::assess(law(10), None, request(), &ctx()).unwrap();
    let derived = prepared.derive_uncertainty(&IntervalRequest::marginal(0.95), &ctx()).unwrap();
    assert_eq!(derived.decided_id, "A");
    assert_eq!(derived.standing, DerivedStanding::DerivedUnmeasured);
    assert!(!derived.assumptions.is_empty());
    // Component one is carried over unchanged from the report.
    assert_eq!(derived.source, SourceUncertainty::KnownLaw);
    assert_eq!(derived.source, prepared.report().source_uncertainty);
    // Component two: Wald interval for A at the reported effective sample size.
    let numerical = &derived.numerical;
    assert_eq!(numerical.multiplicity, IntervalMultiplicity::Marginal);
    assert_eq!(numerical.intervals_reported, 3 + 2 + 3);
    let a = numerical.utilities.iter().find(|u| u.id == "A").unwrap();
    assert!((a.interval.estimate - 2.5).abs() < 1e-12);
    assert!((a.interval.standard_error - SE_A).abs() < 1e-9);
    assert!((a.interval.lower - (2.5 - Z_975 * SE_A)).abs() < 1e-6);
    assert!((a.interval.upper - (2.5 + Z_975 * SE_A)).abs() < 1e-6);
    // The paired difference A - B has the paired standard error, not the sum of two.
    let ab = numerical.differences_vs_decision.iter().find(|d| d.id == "B").unwrap();
    assert!((ab.interval.estimate - 0.5).abs() < 1e-12);
    assert!((ab.interval.standard_error - (1.25_f64 / 39.0).sqrt()).abs() < 1e-9);
    // Wilson interval of P(A best) = 7/12 at effective sample size 40.
    let best = numerical
        .probabilities
        .iter()
        .find(|p| p.id == "A" && p.quantity == ProbabilityQuantity::Best)
        .unwrap();
    let (lower, upper) = wilson(7.0 / 12.0, 40.0, Z_975);
    assert!((best.interval.estimate - 7.0 / 12.0).abs() < 1e-12);
    assert!((best.interval.lower - lower).abs() < 1e-6);
    assert!((best.interval.upper - upper).abs() < 1e-6);
    // Component three: the paired confidence set excludes both competitors, and the
    // corrected interval costs extra width over the ordinary one.
    let set = &derived.selection.confidence_set;
    assert_eq!(set.basis, ConfidenceSetBasis::PairedComparisonsWithBest);
    assert_eq!(set.members, vec!["A".to_owned()]);
    assert_eq!(set.excluded, vec!["B".to_owned(), "C".to_owned()]);
    assert!((set.critical_value - Z_975).abs() < 1e-6);
    assert_eq!(set.unretained_candidates, 0);
    let corrected = derived.selection.corrected.as_ref().unwrap();
    assert_eq!(corrected.selected_id, "A");
    assert_eq!(corrected.n_considered, 3);
    assert!((corrected.critical_value - Z_BONFERRONI_3).abs() < 1e-4);
    assert!((corrected.numerical_half_width - Z_975 * SE_A).abs() < 1e-6);
    assert!(corrected.selection_extra_half_width > 0.0);
    assert!(
        (corrected.upper - corrected.lower - 2.0 * Z_BONFERRONI_3 * SE_A).abs() < 1e-4,
        "the corrected interval is the simultaneous critical value times the standard error"
    );
    // The report's own selection assessment is untouched and still in-sample.
    assert!(matches!(prepared.report().selection_uncertainty, SelectionUncertainty::Selected(_)));
}

#[test]
fn derived_uncertainty_is_never_published_or_combined() {
    let prepared = PreparedDecisionReliability::assess(law(10), None, request(), &ctx()).unwrap();
    let derived = prepared.derive_uncertainty(&IntervalRequest::marginal(0.9), &ctx()).unwrap();
    for route in PublishedRoute::ALL {
        let refusal = publish_derived_route(&derived, route).unwrap_err();
        assert_eq!(refusal.code, "cell_not_licensed");
        assert_eq!(refusal.detail, "decision_reliability.cell_not_licensed");
    }
    assert_eq!(
        derived.combined_uncertainty().unwrap_err().reason_code(),
        "uncertainty_components_not_combined"
    );
}

#[test]
fn derivation_refuses_a_small_effective_sample_and_an_invalid_level() {
    // Four draws: effective sample size 4 is below the normal-approximation floor.
    let refusal = derive_decision_uncertainty(
        &law(1),
        None,
        &request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "decision_reliability.effective_sample_size_too_small");
    // A level outside [0.5, 0.9999].
    for level in [0.3, 1.0, f64::NAN] {
        let refusal = derive_decision_uncertainty(
            &law(10),
            None,
            &request(),
            &IntervalRequest::marginal(level),
            &ctx(),
        )
        .unwrap_err();
        assert_eq!(refusal.code, "invalid_argument");
    }
    // Undeclared dependence still refuses before any interval exists.
    let mut unknown = law(10);
    unknown.within_stream = WithinStreamDependence::Unknown;
    let refusal = derive_decision_uncertainty(
        &unknown,
        None,
        &request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.code, "sampling_dependence_unknown");
}

#[test]
fn simultaneous_multiplicity_widens_every_numerical_interval() {
    let marginal = derive_decision_uncertainty(
        &law(10),
        None,
        &request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    let simultaneous = derive_decision_uncertainty(
        &law(10),
        None,
        &request(),
        &IntervalRequest { level: 0.95, multiplicity: IntervalMultiplicity::Simultaneous },
        &ctx(),
    )
    .unwrap();
    let width = |d: &antecedent::analysis::decision_reliability::DerivedUncertainty| {
        let a = d.numerical.utilities.iter().find(|u| u.id == "A").unwrap();
        a.interval.upper - a.interval.lower
    };
    assert!(width(&simultaneous) > width(&marginal));
    // Eight reported intervals: the per-interval level is 1 - 0.05 / 8.
    let a = simultaneous.numerical.utilities.iter().find(|u| u.id == "A").unwrap();
    assert!((a.interval.level - (1.0 - 0.05 / 8.0)).abs() < 1e-12);
    // The Bonferroni half-width is z(1 - 0.05 / 16) * SE_A, z = 2.734368786533177 (scipy ppf).
    assert!((width(&simultaneous) - 2.0 * 2.734_368_786_533_177 * SE_A).abs() < 1e-5);
    assert!((width(&marginal) - 2.0 * Z_975 * SE_A).abs() < 1e-6);
}

#[test]
fn a_fresh_consumer_rederives_from_the_carried_draws() {
    let prepared = PreparedDecisionReliability::assess(law(10), None, request(), &ctx()).unwrap();
    let derived = prepared.derive_uncertainty(&IntervalRequest::marginal(0.95), &ctx()).unwrap();
    let bytes = prepared.export("derived-facade", &ctx()).unwrap();
    let expected = DecisionReliabilityExpectation {
        draws_digest: Some(law(10).digest_hex()),
        ..DecisionReliabilityExpectation::default()
    };
    let consumed =
        consume_derived_uncertainty(&bytes, &expected, &IntervalRequest::marginal(0.95), &ctx())
            .unwrap();
    assert_eq!(consumed, derived);
    prepared.verify_derived(&derived, &ctx()).unwrap();
    // A stored derivation whose interval was edited does not recompute.
    let mut edited = derived;
    edited.selection.corrected.as_mut().unwrap().lower -= 0.25;
    let refusal = prepared.verify_derived(&edited, &ctx()).unwrap_err();
    assert_eq!(refusal.detail, "decision_reliability.invalid_input");
    // A consumer that retained another candidate-set digest refuses the artifact.
    let wrong = DecisionReliabilityExpectation {
        decided_id: Some("B".into()),
        ..DecisionReliabilityExpectation::default()
    };
    let refusal =
        consume_derived_uncertainty(&bytes, &wrong, &IntervalRequest::marginal(0.95), &ctx())
            .unwrap_err();
    assert_eq!(refusal.detail, "decision_reliability.artifact", "{refusal:?}");
    assert!(refusal.message.contains("decision_mismatch"), "{refusal:?}");
}

fn cohort_output(joint_law: JointLawDeclaration) -> CohortUtilityDraws {
    CohortUtilityDraws {
        source_identity: "cohort-source".into(),
        utility_unit: "utility".into(),
        run_seed: 7,
        purpose: StreamPurpose::Selection,
        replicate: 0,
        label: "posterior".into(),
        source: SourceParadigm::Posterior,
        joint_law,
        within_stream: WithinStreamDependence::Independent,
        actions: vec![
            ("treat".into(), repeated([3.0, 3.0, 2.0, 2.0], 10)),
            ("wait".into(), repeated([1.0, 2.0, 1.0, 2.0], 10)),
        ],
        reported_means: vec![("treat".into(), 2.5), ("wait".into(), 1.5)],
        max_standard_errors: 4.0,
    }
}

#[test]
fn cohort_utilities_reach_the_carrier_with_semantic_stream_identity() {
    let carrier = reach_decision_carrier(ProducerOutput::Cohort(cohort_output(
        JointLawDeclaration::OneSharedLaw,
    )))
    .unwrap();
    assert_eq!(carrier.producer(), ProducerKind::CohortMeanDecision);
    assert_eq!(carrier.joint_law(), JointLawStanding::OneSharedLaw);
    let draws = carrier.draws();
    assert_eq!(draws.producer, "cohort_decision");
    assert!(
        draws
            .candidates
            .iter()
            .all(|c| c.segment.stream.label == "cohort_decision/cohort-source/posterior"
                && c.segment.stream.run_seed == 7)
    );
    let derived = derive_producer_uncertainty(
        &carrier,
        None,
        &request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    assert_eq!(derived.decided_id, "treat");
    // treat - wait is (2, 1, 1, 0): mean 1.
    let wait = derived.numerical.differences_vs_decision.iter().find(|d| d.id == "wait").unwrap();
    assert!((wait.interval.estimate - 1.0).abs() < 1e-12);
    assert!((wait.interval.standard_error - (0.5_f64 / 39.0).sqrt()).abs() < 1e-9);
    // The same draws assessed through the producer export a consumable artifact.
    let prepared =
        PreparedDecisionReliability::assess_producer(&carrier, None, request(), &ctx()).unwrap();
    assert_eq!(prepared.report().decided_id, "treat");
}

#[test]
fn point_only_and_undeclared_producer_outputs_refuse_with_typed_reasons() {
    let cohort_point = reach_decision_carrier(ProducerOutput::PointOnly(PointOnlyOutput {
        producer: ProducerKind::CohortMeanDecision,
        identity: "cohort-source".into(),
    }))
    .unwrap_err();
    assert_eq!(cohort_point.code, "joint_law_required");
    assert_eq!(cohort_point.detail, "decision_reliability.producer_point_only");
    let rollout_point = reach_decision_carrier(ProducerOutput::PointOnly(PointOnlyOutput {
        producer: ProducerKind::SequentialRollout,
        identity: "model".into(),
    }))
    .unwrap_err();
    assert_eq!(rollout_point.code, "diagnostic_not_available");
    let off_policy_point = reach_decision_carrier(ProducerOutput::PointOnly(PointOnlyOutput {
        producer: ProducerKind::SequentialOffPolicy,
        identity: "log".into(),
    }))
    .unwrap_err();
    assert_eq!(off_policy_point.code, "diagnostic_not_available");
    let undeclared = reach_decision_carrier(ProducerOutput::Cohort(cohort_output(
        JointLawDeclaration::Undeclared,
    )))
    .unwrap_err();
    assert_eq!(undeclared.code, "sampling_dependence_unknown");
    let separate = reach_decision_carrier(ProducerOutput::Cohort(cohort_output(
        JointLawDeclaration::SeparateMarginals,
    )))
    .unwrap_err();
    assert_eq!(separate.code, "joint_law_required");
    // Draws that disagree with the point value the cohort route reported.
    let mut disagreeing = cohort_output(JointLawDeclaration::OneSharedLaw);
    disagreeing.reported_means = vec![("treat".into(), 9.0)];
    let refusal = reach_decision_carrier(ProducerOutput::Cohort(disagreeing)).unwrap_err();
    assert_eq!(refusal.detail, "decision_reliability.producer_mean_disagrees");
}

#[test]
fn rollout_noise_decides_whether_ranking_statements_are_licensed() {
    let rollout = |noise: RolloutNoise| RolloutUtilityDraws {
        model_identity: "model".into(),
        stream_scheme: "keyed".into(),
        seed: 3,
        utility_unit: "utility".into(),
        purpose: StreamPurpose::Selection,
        replicate: 0,
        noise,
        policies: vec![
            ("p".into(), repeated([3.0, 3.0, 2.0, 2.0], 10)),
            ("q".into(), repeated([2.0, 1.0, 3.0, 2.0], 10)),
        ],
        reported_means: vec![("p".into(), 2.5)],
        max_standard_errors: 4.0,
    };
    let keyed =
        reach_decision_carrier(ProducerOutput::Rollout(rollout(RolloutNoise::ImplementationKeyed)))
            .unwrap();
    assert!(matches!(keyed.joint_law(), JointLawStanding::MarginalOnly { .. }));
    // Functional estimates and the selection component are derived without ranking.
    let derived = derive_producer_uncertainty(
        &keyed,
        None,
        &ReliabilityRequest::expected_utility(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    assert!(derived.numerical.probabilities.is_empty());
    // Ranking is refused on the marginal-only carrier.
    let refusal = derive_producer_uncertainty(
        &keyed,
        None,
        &request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.code, "joint_law_required");
    let refusal =
        PreparedDecisionReliability::assess_producer(&keyed, None, request(), &ctx()).unwrap_err();
    assert_eq!(refusal.code, "joint_law_required");
    // Structural exogenous noise is one shared law and licenses ranking statements.
    let structural =
        reach_decision_carrier(ProducerOutput::Rollout(rollout(RolloutNoise::StructuralExogenous)))
            .unwrap();
    let derived = derive_producer_uncertainty(
        &structural,
        None,
        &request(),
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap();
    assert!(!derived.numerical.probabilities.is_empty());
}

#[test]
fn off_policy_returns_carry_the_expected_utility_only() {
    let output = ProducerOutput::OffPolicy(OffPolicyReturnDraws {
        model_identity: "model".into(),
        log_identity: "log".into(),
        run_seed: 5,
        utility_unit: "return".into(),
        purpose: StreamPurpose::Selection,
        replicate: 0,
        policies: vec![
            ("p".into(), repeated([3.0, 3.0, 2.0, 2.0], 10)),
            ("q".into(), repeated([1.0, 2.0, 1.0, 2.0], 10)),
        ],
        reported_means: Vec::new(),
        max_standard_errors: 4.0,
    });
    let carrier = reach_decision_carrier(output).unwrap();
    assert_eq!(carrier.producer(), ProducerKind::SequentialOffPolicy);
    let refusal = derive_producer_uncertainty(
        &carrier,
        None,
        &ReliabilityRequest {
            functional: antecedent::analysis::decision_reliability::Functional::Cvar {
                alpha: 0.5,
                tail: antecedent::analysis::decision_reliability::RiskTail::Lower,
            },
            ..ReliabilityRequest::expected_utility()
        },
        &IntervalRequest::marginal(0.95),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
}

#[test]
fn the_coverage_design_is_data_and_measures_nothing() {
    let design = decision_reliability_coverage_design();
    let routes: Vec<&str> = design.iter().map(|cell| cell.route).collect();
    for expected in [
        "ranking_probability_interval",
        "optimality_confidence_set",
        "selection_corrected_interval",
        "held_out_interval",
        "winners_curse_bound",
        "utility_interval",
    ] {
        assert!(routes.contains(&expected), "{expected} missing from {routes:?}");
    }
    for cell in &design {
        assert!(cell.min_replicates >= 2_000, "{}", cell.route);
        assert!(!cell.boundary_probes.is_empty(), "{}", cell.route);
        assert!(!cell.nominal_levels.is_empty(), "{}", cell.route);
        assert!(!cell.candidate_set_sizes.is_empty(), "{}", cell.route);
    }
    let ranking = design.iter().find(|c| c.route == "ranking_probability_interval").unwrap();
    assert_eq!(ranking.truth, TruthKind::ClosedForm);
    let set = design.iter().find(|c| c.route == "optimality_confidence_set").unwrap();
    assert_eq!(set.truth, TruthKind::SimulationKnownOptimum);
}
