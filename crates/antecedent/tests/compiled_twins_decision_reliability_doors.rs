//! Compiled decision-reliability side doors and the direct kernel calls they replace give the
//! same result bit for bit, and each compiled route reproduces a hand-derived value that an
//! existing fixture already states.
//!
//! Carrier ranking calls `ProducerCarrier::rank` (joint-law check, then `rank`). Assessment
//! calls `assess`; its seal calls `DecisionReliabilityArtifact::new`, which recomputes.
//! Derivation calls `derive_uncertainty` or `ProducerCarrier::derive_uncertainty` and returns
//! the whole `DerivedUncertainty`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use antecedent::analysis::decision_reliability_programs::{
    compile_decision_reliability_assessment, compile_decision_reliability_derivation,
    compile_producer_carrier_assessment, compile_producer_carrier_derivation,
};
use antecedent::analysis::input_decision_programs::{
    TerminalError, compile_producer_carrier_ranking,
};
use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    BatchPlan, CandidateCoupling, CandidateDraws, CandidateOrigin, DecisionDraws,
    DecisionReliabilityArtifact, DerivedUncertainty, DrawSegment, IntervalRequest,
    JointLawStanding, ProducerOutput, RankingRequest, ReliabilityError, ReliabilityRequest,
    RolloutNoise, RolloutUtilityDraws, SourceParadigm, SourceUncertainty, StreamId, StreamPurpose,
    WithinStreamDependence, assess, derive_uncertainty, rank, reach_carrier,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn same<T: std::fmt::Debug>(compiled: &T, direct: &T) {
    assert_eq!(
        without_lease_ids(&format!("{compiled:?}")),
        without_lease_ids(&format!("{direct:?}"))
    );
}

fn without_lease_ids(text: &str) -> String {
    const LEASE: &str = "AdmissionLease { id: ";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(LEASE) {
        out.push_str(&rest[..at + LEASE.len()]);
        rest = &rest[at + LEASE.len()..];
        let digits = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        out.push('_');
        rest = &rest[digits..];
    }
    out.push_str(rest);
    out
}

fn routed(route: &str, unit: &str, lowered_route: &str, view: &antecedent::planner::PlanView) {
    assert_eq!(lowered_route, route);
    let support = view.unit_support().expect("the route is bound to a unit");
    assert_eq!(support.unit(), unit);
}

fn columns(length: usize) -> Vec<(&'static str, Vec<f64>)> {
    let pattern = |values: [f64; 4]| values.into_iter().cycle().take(length).collect();
    vec![
        ("A", pattern([3.0, 3.0, 2.0, 2.0])),
        ("B", pattern([2.0, 1.0, 3.0, 2.0])),
        ("C", pattern([1.0, 2.0, 1.0, 2.0])),
    ]
}

fn draws(length: usize) -> DecisionDraws {
    let shared =
        DrawSegment::whole(StreamId::new(1, StreamPurpose::Selection, 0, "states"), length);
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
        candidates: columns(length)
            .into_iter()
            .map(|(id, utilities)| CandidateDraws {
                id: id.to_owned(),
                utilities,
                segment: shared.clone(),
            })
            .collect(),
    }
}

fn rollout(
    noise: RolloutNoise,
    length: usize,
) -> antecedent_design::decision_reliability::ProducerCarrier {
    reach_carrier(ProducerOutput::Rollout(RolloutUtilityDraws {
        model_identity: "model".into(),
        stream_scheme: "keyed".into(),
        seed: 3,
        utility_unit: "utility".into(),
        purpose: StreamPurpose::Selection,
        replicate: 0,
        noise,
        policies: columns(length)
            .into_iter()
            .map(|(id, utilities)| (id.to_owned(), utilities))
            .collect(),
        reported_means: Vec::new(),
        max_standard_errors: 4.0,
    }))
    .unwrap()
}

fn ranking_request() -> RankingRequest {
    RankingRequest { baseline: Some("B".into()), top_k: vec![2], batch_plan: BatchPlan::SquareRoot }
}

fn reliability_request() -> ReliabilityRequest {
    ReliabilityRequest {
        ranking: Some(RankingRequest::default()),
        ..ReliabilityRequest::expected_utility()
    }
}

#[test]
fn compiled_producer_carrier_ranking_equals_rank() {
    let ctx = ctx();
    let carrier = rollout(RolloutNoise::StructuralExogenous, 4);
    let request = ranking_request();
    let direct = carrier.rank(&request, &ctx).unwrap();
    let program = compile_producer_carrier_ranking(carrier, request, &ctx).unwrap();
    routed(
        "DecisionDrawsRanking",
        "decision_reliability.ranking_statement",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    // P(A best) = (1 + 1 + 0 + 1/3) / 4 = 7/12 with the tie shared.
    let a = compiled.statements.iter().find(|s| s.id == "A").unwrap();
    assert!((a.probability_best.value - 7.0 / 12.0).abs() < 1e-12);
    assert_eq!(compiled.best_by_expected_utility, "A");
    // The draws-only ranking of the same utilities is the same report; the program is not.
    let plain =
        rank(rollout(RolloutNoise::StructuralExogenous, 4).draws(), &ranking_request(), &ctx)
            .unwrap();
    assert_eq!(compiled, plain);
}

#[test]
fn carrier_ranking_identity_reads_joint_law() {
    let ctx = ctx();
    let shared = rollout(RolloutNoise::StructuralExogenous, 4);
    let marginal = rollout(RolloutNoise::ImplementationKeyed, 4);
    assert_eq!(shared.joint_law(), JointLawStanding::OneSharedLaw);
    assert_eq!(shared.draws().digest_hex(), marginal.draws().digest_hex());
    let request = ranking_request();
    let shared_id =
        compile_producer_carrier_ranking(shared, request.clone(), &ctx).unwrap().identity();
    let marginal_program =
        compile_producer_carrier_ranking(marginal, request.clone(), &ctx).unwrap();
    assert_ne!(shared_id, marginal_program.identity());
    let direct = rollout(RolloutNoise::ImplementationKeyed, 4).rank(&request, &ctx).unwrap_err();
    match marginal_program.execute(&ctx).unwrap_err() {
        TerminalError::Kernel(error) => assert_eq!(error, direct),
        other => panic!("compiler refusal: {other:?}"),
    }
    assert!(matches!(direct, ReliabilityError::JointLawRequired { .. }));
}

#[test]
fn compiled_decision_reliability_assessment_equals_assess_and_seal_recomputes() {
    let ctx = ctx();
    let draws = draws(4);
    let request = reliability_request();
    let direct = assess(&draws, None, &request, &ctx).unwrap();
    let program =
        compile_decision_reliability_assessment(draws.clone(), None, request.clone(), &ctx)
            .unwrap();
    routed(
        "DecisionReliabilityAssessment",
        "decision_reliability.sampled_decision",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    assert_eq!(compiled.decided_id, "A");
    assert!((compiled.monte_carlo_error.decision.value - 2.5).abs() < 1e-12);
    assert_eq!(compiled.source_uncertainty, SourceUncertainty::KnownLaw);
    let ranking = compiled.ranking.as_ref().unwrap();
    let best =
        |id: &str| ranking.statements.iter().find(|s| s.id == id).unwrap().probability_best.value;
    assert!((best("A") - 7.0 / 12.0).abs() < 1e-12);
    assert!((best("B") - 1.0 / 3.0).abs() < 1e-12);
    assert!((best("C") - 1.0 / 12.0).abs() < 1e-12);
    let sealed = program.seal(&ctx).unwrap();
    let direct_artifact = DecisionReliabilityArtifact::new(draws, None, request, &ctx).unwrap();
    assert_eq!(sealed, direct_artifact);
    same(&sealed, &direct_artifact);
}

#[test]
fn compiled_producer_carrier_assessment_equals_assess() {
    let ctx = ctx();
    let carrier = rollout(RolloutNoise::StructuralExogenous, 4);
    let request = reliability_request();
    let direct = carrier.assess(None, &request, &ctx).unwrap();
    let program = compile_producer_carrier_assessment(carrier, None, request, &ctx).unwrap();
    routed(
        "DecisionReliabilityAssessment",
        "decision_reliability.sampled_decision",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    assert_eq!(compiled.decided_id, "A");
}

// A has twenty utilities of 2 and twenty of 3. Its mean is 2.5 and its
// standard error is 0.5 / sqrt(39). The selected-value interval spends alpha
// over both tails of all three candidates, giving z(1 - 0.05/6).
fn check_three_candidate_selection(result: &DerivedUncertainty) {
    let set = &result.selection.confidence_set;
    assert_eq!(set.members, ["A"]);
    assert_eq!(set.excluded, ["B", "C"]);
    let selected = result.selection.corrected.as_ref().unwrap();
    assert_eq!(selected.n_considered, 3);
    let half_width = 2.393_979_799_818_51 * 0.5 / 39.0_f64.sqrt();
    assert!((selected.lower - (2.5 - half_width)).abs() < 1e-8);
    assert!((selected.upper - (2.5 + half_width)).abs() < 1e-8);
}

#[test]
fn compiled_decision_reliability_derivation_equals_derive_uncertainty() {
    let ctx = ctx();
    // Forty draws: the derivation refuses an effective sample size below 30.
    let draws = draws(40);
    let request = reliability_request();
    let interval = IntervalRequest::marginal(0.95);
    let direct = derive_uncertainty(&draws, None, &request, &interval, &ctx).unwrap();
    let program =
        compile_decision_reliability_derivation(draws, None, request, interval, &ctx).unwrap();
    routed(
        "DecisionReliabilityDerivation",
        "decision_reliability.derived_numerical_interval",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    check_three_candidate_selection(&compiled);
    assert_eq!(compiled.decided_id, "A");
    let value = compiled.numerical.utilities.iter().find(|row| row.id == "A").unwrap();
    assert!((value.interval.estimate - 2.5).abs() < 1e-12);
}

#[test]
fn compiled_producer_carrier_derivation_equals_derive_uncertainty() {
    let ctx = ctx();
    let carrier = rollout(RolloutNoise::StructuralExogenous, 40);
    let request = ReliabilityRequest::expected_utility();
    let interval = IntervalRequest::marginal(0.95);
    let direct = carrier.derive_uncertainty(None, &request, &interval, &ctx).unwrap();
    let program =
        compile_producer_carrier_derivation(carrier, None, request, interval, &ctx).unwrap();
    routed(
        "DecisionReliabilityDerivation",
        "decision_reliability.derived_numerical_interval",
        program.lowered().route(),
        &program.plan_view(),
    );
    let compiled = program.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    same(&compiled, &direct);
    check_three_candidate_selection(&compiled);
    assert_eq!(compiled.decided_id, "A");
}
