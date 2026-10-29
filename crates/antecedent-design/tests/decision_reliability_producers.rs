//! The producer connection: cohort-decision and sequential-evaluation outputs reach the
//! reliability carrier only as per-draw utilities from a declared joint law, and refuse
//! with a typed reason otherwise.
//!
//! Oracle. The cohort contrast draws are `x = (2.0, 2.4, 2.2, 2.3)` with mean `2.225`;
//! the utility of `treat` is `4 x - 1.5`, so its draws are `(6.5, 8.1, 7.3, 7.7)` with
//! mean `7.4` and sample variance `16 * 0.0875 / 3`, standard error `0.34157`; `wait` is
//! identically zero.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    CohortUtilityDraws, Functional, IntervalRequest, JointLawDeclaration, JointLawStanding,
    OffPolicyReturnDraws, PointOnlyOutput, ProducerKind, ProducerOutput, RankingRequest,
    ReliabilityError, ReliabilityRequest, RolloutNoise, RolloutUtilityDraws, SourceParadigm,
    StreamPurpose, WithinStreamDependence, reach_carrier,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn cohort_input() -> CohortUtilityDraws {
    let contrast = [2.0_f64, 2.4, 2.2, 2.3];
    CohortUtilityDraws {
        source_identity: "cohort-src-1".into(),
        utility_unit: "mmHg".into(),
        run_seed: 41,
        purpose: StreamPurpose::Selection,
        replicate: 0,
        label: "posterior".into(),
        source: SourceParadigm::Posterior,
        joint_law: JointLawDeclaration::OneSharedLaw,
        within_stream: WithinStreamDependence::Independent,
        actions: vec![
            ("treat".into(), contrast.iter().map(|x| 4.0 * x - 1.5).collect()),
            ("wait".into(), vec![0.0; 4]),
        ],
        reported_means: vec![("treat".into(), 7.4), ("wait".into(), 0.0)],
        max_standard_errors: 4.0,
    }
}

fn rollout_input(noise: RolloutNoise) -> RolloutUtilityDraws {
    RolloutUtilityDraws {
        model_identity: "model-1".into(),
        stream_scheme: "seed_rollout_time_role".into(),
        seed: 9,
        utility_unit: "qaly".into(),
        purpose: StreamPurpose::Selection,
        replicate: 0,
        noise,
        policies: vec![
            ("keep".into(), vec![3.0, 1.0, 2.0, 2.0]),
            ("switch".into(), vec![1.0, 1.0, 2.0, 0.0]),
        ],
        reported_means: vec![("keep".into(), 2.0), ("switch".into(), 1.0)],
        max_standard_errors: 4.0,
    }
}

fn off_policy_input() -> OffPolicyReturnDraws {
    OffPolicyReturnDraws {
        model_identity: "model-1".into(),
        log_identity: "log-7".into(),
        run_seed: 0,
        utility_unit: "qaly".into(),
        purpose: StreamPurpose::Selection,
        replicate: 0,
        policies: vec![
            ("keep".into(), vec![4.0, 0.0, 2.0, 2.0]),
            ("switch".into(), vec![1.0, 3.0, 0.0, 0.0]),
        ],
        reported_means: vec![("keep".into(), 2.0), ("switch".into(), 1.0)],
        max_standard_errors: 4.0,
    }
}

fn with_ranking() -> ReliabilityRequest {
    ReliabilityRequest {
        ranking: Some(RankingRequest::default()),
        ..ReliabilityRequest::expected_utility()
    }
}

// ---- point-only outputs refuse with a typed reason ---------------------------------

#[test]
fn a_point_only_cohort_mean_refuses_for_want_of_a_joint_law() {
    let error = reach_carrier(ProducerOutput::PointOnly(PointOnlyOutput {
        producer: ProducerKind::CohortMeanDecision,
        identity: "cohort-src-1".into(),
    }))
    .unwrap_err();
    assert!(matches!(
        error,
        ReliabilityError::ProducerPointOnly { producer: ProducerKind::CohortMeanDecision, .. }
    ));
    let refusal = error.to_refusal();
    assert_eq!(refusal.code, "joint_law_required");
    assert_eq!(refusal.detail, "decision_reliability.producer_point_only");
    assert!(refusal.offending.unwrap().contains("an_affine_mean_contrast_without_a_law"));
    assert!(refusal.remedy.unwrap().contains("aligned joint law"));
}

#[test]
fn point_only_sequential_estimates_refuse_because_no_per_draw_utilities_were_kept() {
    for (producer, held) in [
        (ProducerKind::SequentialRollout, "without_rollout_utilities"),
        (ProducerKind::SequentialOffPolicy, "without_trajectory_returns"),
    ] {
        let error = reach_carrier(ProducerOutput::PointOnly(PointOnlyOutput {
            producer,
            identity: "estimate-1".into(),
        }))
        .unwrap_err();
        let refusal = error.to_refusal();
        assert_eq!(refusal.code, "diagnostic_not_available");
        assert_eq!(refusal.detail, "decision_reliability.producer_point_only");
        assert!(refusal.offending.unwrap().contains(held));
        assert!(refusal.remedy.is_some());
    }
}

// ---- cohort decision ----------------------------------------------------------------

#[test]
fn cohort_utilities_from_one_law_reach_the_carrier_and_reproduce_the_reported_mean() {
    let carrier = reach_carrier(ProducerOutput::Cohort(cohort_input())).unwrap();
    assert_eq!(carrier.producer(), ProducerKind::CohortMeanDecision);
    assert_eq!(carrier.joint_law(), JointLawStanding::OneSharedLaw);
    let draws = carrier.draws();
    assert_eq!(draws.producer, "cohort_decision");
    // The stream is named by the producing source, not by a position.
    assert!(draws.candidates[0].segment.stream.label.contains("cohort-src-1"));
    assert!(draws.candidates.iter().all(|c| c.segment == draws.candidates[0].segment));
    let report = carrier.assess(None, &with_ranking(), &ctx()).unwrap();
    assert_eq!(report.decided_id, "treat");
    assert!((report.monte_carlo_error.decision.value - 7.4).abs() < 1e-12);
    let se = (16.0_f64 * 0.0875 / 3.0 / 4.0).sqrt();
    assert!((report.monte_carlo_error.decision.standard_error - se).abs() < 1e-12);
    // treat beats wait in every draw: it is best with probability one.
    let ranking = report.ranking.unwrap();
    let treat = ranking.statements.iter().find(|s| s.id == "treat").unwrap();
    assert!((treat.probability_best.value - 1.0).abs() < 1e-12);
    // Four draws: the intervals are not derived (effective sample size 4 < 30).
    let refused = carrier
        .derive_uncertainty(
            None,
            &ReliabilityRequest::expected_utility(),
            &IntervalRequest::marginal(0.9),
            &ctx(),
        )
        .unwrap_err();
    assert_eq!(refused.to_refusal().detail, "decision_reliability.effective_sample_size_too_small");
}

#[test]
fn cohort_draws_that_disagree_with_the_reported_point_are_refused() {
    let mut input = cohort_input();
    // 12.0 is 4.6 above the draws' mean, about 13 standard errors (tolerance is 4).
    input.reported_means = vec![("treat".into(), 12.0)];
    let error = reach_carrier(ProducerOutput::Cohort(input)).unwrap_err();
    match &error {
        ReliabilityError::ProducerMeanDisagrees { id, reported, draws_mean, .. } => {
            assert_eq!(id, "treat");
            assert!((reported - 12.0).abs() < 1e-12 && (draws_mean - 7.4).abs() < 1e-12);
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(error.reason_code(), "distribution_meaning_mismatch");
    // The check is optional: no reported values, no comparison.
    let mut unchecked = cohort_input();
    unchecked.reported_means.clear();
    assert!(reach_carrier(ProducerOutput::Cohort(unchecked)).is_ok());
}

#[test]
fn cohort_utilities_without_a_declared_shared_law_or_with_a_known_law_are_refused() {
    let mut separate = cohort_input();
    separate.joint_law = JointLawDeclaration::SeparateMarginals;
    assert_eq!(
        reach_carrier(ProducerOutput::Cohort(separate)).unwrap_err().reason_code(),
        "joint_law_required"
    );
    let mut undeclared = cohort_input();
    undeclared.joint_law = JointLawDeclaration::Undeclared;
    assert_eq!(
        reach_carrier(ProducerOutput::Cohort(undeclared)).unwrap_err().reason_code(),
        "sampling_dependence_unknown"
    );
    let mut known = cohort_input();
    known.source = SourceParadigm::KnownLaw;
    let error = reach_carrier(ProducerOutput::Cohort(known)).unwrap_err();
    assert!(matches!(error, ReliabilityError::InvalidInput { slot: "source", .. }));
    let mut blank = cohort_input();
    blank.source_identity = " ".into();
    let error = reach_carrier(ProducerOutput::Cohort(blank)).unwrap_err();
    assert!(matches!(
        &error,
        ReliabilityError::InvalidInput { slot: "source_identity", message } if message == "must be named"
    ));
}

#[test]
fn changed_inputs_at_unchanged_shape_change_the_carrier_identity() {
    let base = reach_carrier(ProducerOutput::Cohort(cohort_input())).unwrap();
    let mut edited = cohort_input();
    edited.actions[0].1[2] += 0.25;
    edited.reported_means.clear();
    let edited = reach_carrier(ProducerOutput::Cohort(edited)).unwrap();
    assert_ne!(base.draws().digest_hex(), edited.draws().digest_hex());
    let mut other_source = cohort_input();
    other_source.source_identity = "cohort-src-2".into();
    let other_source = reach_carrier(ProducerOutput::Cohort(other_source)).unwrap();
    assert_ne!(base.draws().digest_hex(), other_source.draws().digest_hex());
    let mut other_replicate = cohort_input();
    other_replicate.replicate = 1;
    let other_replicate = reach_carrier(ProducerOutput::Cohort(other_replicate)).unwrap();
    assert_ne!(base.draws().digest_hex(), other_replicate.draws().digest_hex());
}

// ---- sequential rollouts and off-policy returns ------------------------------------

#[test]
fn implementation_keyed_rollouts_report_functionals_but_refuse_ranking() {
    let carrier =
        reach_carrier(ProducerOutput::Rollout(rollout_input(RolloutNoise::ImplementationKeyed)))
            .unwrap();
    assert!(matches!(carrier.joint_law(), JointLawStanding::MarginalOnly { .. }));
    assert_eq!(carrier.draws().producer, "sequential_evaluation");
    assert!(carrier.draws().candidates[0].segment.stream.label.contains("model-1"));
    // Mean values and the shared-noise difference error are valid under any coupling.
    let report = carrier.assess(None, &ReliabilityRequest::expected_utility(), &ctx()).unwrap();
    assert_eq!(report.decided_id, "keep");
    assert!((report.monte_carlo_error.decision.value - 2.0).abs() < 1e-12);
    // Ranking is a statement about a joint law the implementation noise does not supply.
    let error = carrier.assess(None, &with_ranking(), &ctx()).unwrap_err();
    assert!(matches!(error, ReliabilityError::JointLawRequired { .. }));
    assert_eq!(error.reason_code(), "joint_law_required");
    let direct = carrier.rank(&RankingRequest::default(), &ctx()).unwrap_err();
    assert!(matches!(direct, ReliabilityError::JointLawRequired { .. }));
}

#[test]
fn structurally_shared_rollout_noise_licenses_ranking_to_the_carrier() {
    let carrier =
        reach_carrier(ProducerOutput::Rollout(rollout_input(RolloutNoise::StructuralExogenous)))
            .unwrap();
    assert_eq!(carrier.joint_law(), JointLawStanding::OneSharedLaw);
    let ranking = carrier.rank(&RankingRequest::default(), &ctx()).unwrap();
    // keep = (3, 1, 2, 2), switch = (1, 1, 2, 0): keep is strictly better in draws 0 and
    // 3, tied in 1 and 2; best probability (1 + 0.5 + 0.5 + 1) / 4 = 0.75.
    let keep = ranking.statements.iter().find(|s| s.id == "keep").unwrap();
    assert!((keep.probability_best.value - 0.75).abs() < 1e-12);
}

#[test]
fn off_policy_contributions_carry_the_expected_utility_only() {
    let carrier = reach_carrier(ProducerOutput::OffPolicy(off_policy_input())).unwrap();
    assert!(matches!(carrier.joint_law(), JointLawStanding::MarginalOnly { .. }));
    let report = carrier.assess(None, &ReliabilityRequest::expected_utility(), &ctx()).unwrap();
    assert_eq!(report.decided_id, "keep");
    assert!((report.monte_carlo_error.decision.value - 2.0).abs() < 1e-12);
    // Contributions are not outcomes: ranking and a quantile of them are refused.
    assert_eq!(
        carrier.assess(None, &with_ranking(), &ctx()).unwrap_err().reason_code(),
        "joint_law_required"
    );
    let quantile = ReliabilityRequest {
        functional: Functional::Quantile { p: 0.5 },
        ..ReliabilityRequest::expected_utility()
    };
    let error = carrier.assess(None, &quantile, &ctx()).unwrap_err();
    assert!(matches!(error, ReliabilityError::RouteNotSupported { .. }));
    assert_eq!(error.reason_code(), "route_not_supported");
}

#[test]
fn a_rollout_reported_value_the_draws_do_not_reproduce_is_refused() {
    let mut input = rollout_input(RolloutNoise::StructuralExogenous);
    input.reported_means = vec![("keep".into(), 5.0)];
    let error = reach_carrier(ProducerOutput::Rollout(input)).unwrap_err();
    assert!(matches!(error, ReliabilityError::ProducerMeanDisagrees { .. }));
    let mut unknown_id = rollout_input(RolloutNoise::StructuralExogenous);
    unknown_id.reported_means = vec![("absent".into(), 1.0)];
    assert!(matches!(
        reach_carrier(ProducerOutput::Rollout(unknown_id)).unwrap_err(),
        ReliabilityError::InvalidInput { slot: "reported_means", .. }
    ));
}
